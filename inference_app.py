import io
import re
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

# Import from existing module
from module import (
    fetch_recent_daily_history,
    fetch_snapshot,
    fetch_last_completed_close,
    build_today_estimate,
)

# -------------------------------------------------------------------
# App + logging setup
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("l1_inference")

st.set_page_config(
    page_title="Cascade Trader — L1 Inference",
    layout="wide",
)
st.title("Cascade Trader — L1 Inference & Limit Orders (Auto-arch loader)")

# -------------------------------------------------------------------
# Flexible Level-1 model (accepts channels tuple)
# -------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, pdrop=0.1):
        super().__init__()
        pad = (k - 1) * d // 2
        self.conv = nn.Conv1d(
            c_in, c_out,
            kernel_size=k,
            dilation=d,
            padding=pad,
        )
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(pdrop)
        self.res = (c_in == c_out)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out + x if self.res else out


class Level1ScopeCNN(nn.Module):
    """
    Flexible L1 CNN.

    channels: tuple of out-channels per block, e.g. (32, 64, 128)
    in_features: number of input features (channels) to first conv
    """

    def __init__(
        self,
        in_features=12,
        channels=(32, 64, 128),
        kernel_sizes=(5, 3, 3),
        dilations=(1, 2, 4),
        dropout=0.1,
    ):
        super().__init__()

        chs = [in_features] + list(channels)
        blocks = []

        for i in range(len(channels)):
            k = kernel_sizes[min(i, len(kernel_sizes) - 1)]
            d = dilations[min(i, len(dilations) - 1)]
            blocks.append(
                ConvBlock(
                    chs[i],
                    chs[i + 1],
                    k=k,
                    d=d,
                    pdrop=dropout,
                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
        self.head = nn.Linear(chs[-1], 1)

    @property
    def embedding_dim(self):
        return int(self.blocks[-1].conv.out_channels)

    def forward(self, x):
        z = self.blocks(x)
        z = self.proj(z)
        z_pool = z.mean(dim=-1)
        logit = self.head(z_pool)
        return logit, z_pool


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T

    def load_state(self, state_dict):
        try:
            self.load_state_dict(state_dict)
        except Exception:
            logger.warning("Temperature scaler load failed.")


# -------------------------------------------------------------------
# Feature engineering (same as training)
# -------------------------------------------------------------------

def compute_engineered_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = (
        df["volume"].astype(float)
        if "volume" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    ret1 = c.pct_change().fillna(0.0)
    f["ret1"] = ret1
    f["logret1"] = np.log1p(ret1.replace(-1, -0.999999))

    tr = (h - l).clip(lower=0)
    f["tr"] = tr.fillna(0.0)
    f["atr"] = tr.rolling(14, min_periods=1).mean().fillna(0.0)

    for w in windows:
        f[f"rmean_{w}"] = c.pct_change(w).fillna(0.0)
        f[f"vol_{w}"] = ret1.rolling(w).std().fillna(0.0)
        f[f"tr_mean_{w}"] = tr.rolling(w).mean().fillna(0.0)
        f[f"vol_z_{w}"] = (
            v.rolling(w).mean()
            - v.rolling(max(1, w * 3)).mean()
        ).fillna(0.0)
        f[f"mom_{w}"] = (c - c.rolling(w).mean()).fillna(0.0)

        roll_max = c.rolling(w).max().fillna(method="bfill")
        roll_min = c.rolling(w).min().fillna(method="bfill")
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f"chanpos_{w}"] = ((c - roll_min) / denom).fillna(0.5)

    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    n_rows, n_feat = features.shape
    X = np.zeros((len(indices), seq_len, n_feat), dtype=features.dtype)

    for i, t in enumerate(indices):
        t = int(t)
        t0 = t - seq_len + 1

        if t0 < 0:
            pad_count = -t0
            pad = np.repeat(features[[0]], pad_count, axis=0)
            seq = np.vstack([pad, features[0 : t + 1]])
        else:
            seq = features[t0 : t + 1]

        if seq.shape[0] < seq_len:
            pad_needed = seq_len - seq.shape[0]
            pad = np.repeat(seq[[0]], pad_needed, axis=0)
            seq = np.vstack([pad, seq])

        X[i] = seq[-seq_len:]

    return X


# -------------------------------------------------------------------
# Checkpoint robust loader helpers
# -------------------------------------------------------------------

def _is_state_dict_like(d: dict) -> bool:
    if not isinstance(d, dict):
        return False

    keys = list(d.keys())
    for k in keys[:20]:
        if any(
            sub in k
            for sub in (
                "conv.weight",
                "bn.weight",
                "head.weight",
                "proj.weight",
                "blocks.0.conv.weight",
            )
        ):
            return True

    vals = list(d.values())[:10]
    if all(isinstance(v, (torch.Tensor, np.ndarray)) for v in vals):
        return True

    return False


def extract_state_dict(container):
    if container is None:
        return None, {}

    if isinstance(container, dict) and _is_state_dict_like(container):
        return container, {}

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "model_state",
        "model_weights",
    ):
        if (
            isinstance(container, dict)
            and key in container
            and _is_state_dict_like(container[key])
        ):
            extras = {k: v for k, v in container.items() if k != key}
            return container[key], extras

    if isinstance(container, dict):
        for k, v in container.items():
            if isinstance(v, dict) and _is_state_dict_like(v):
                extras = {kk: vv for kk, vv in container.items() if kk != k}
                return v, extras

    return None, {}


def strip_module_prefix(state):
    new = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new[nk] = v
    return new


_conv_key_re = re.compile(r"blocks.(\d+).conv.weight")


def infer_arch_from_state(state):
    """
    Infer:
      - in_features (input channels to first conv)
      - channels tuple (out-channels per block)
    """
    blocks = {}

    for k, v in state.items():
        m = _conv_key_re.search(k)
        if m and hasattr(v, "shape"):
            idx = int(m.group(1))
            out_ch = int(v.shape[0])
            in_ch = int(v.shape[1])
            blocks[idx] = (out_ch, in_ch)

    if not blocks:
        return None, None

    ordered = [blocks[i] for i in sorted(blocks.keys())]
    channels = [b[0] for b in ordered]
    in_features = ordered[0][1]

    return int(in_features), tuple(int(x) for x in channels)


def load_checkpoint_bytes_safe(raw_bytes: bytes):
    buf = io.BytesIO(raw_bytes)

    try:
        return torch.load(buf, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.info("torch.load failed (full): %s", e)

    buf.seek(0)
    try:
        return torch.load(buf, map_location="cpu", weights_only=True)
    except Exception as e:
        logger.info("torch.load failed (weights_only): %s", e)

    buf.seek(0)
    import pickle

    try:
        return pickle.loads(buf.read())
    except Exception as e:
        logger.exception("All checkpoint load attempts failed")
        raise RuntimeError("Failed to load checkpoint") from e


# -------------------------------------------------------------------
# Sidebar configuration
# -------------------------------------------------------------------

st.sidebar.header("Config")

seq_len = st.sidebar.slider("Sequence length", 8, 256, 64, step=8)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 2.0) / 100.0
tp_mult = st.sidebar.slider("TP ATR multiplier", 1.0, 5.0, 1.0)
sl_mult = st.sidebar.slider("SL ATR multiplier", 0.5, 3.0, 1.0)
account_balance = st.sidebar.number_input("Account balance ($)", value=10000.0)

include_today = st.sidebar.checkbox("Include Today's Estimate", value=True)
ckpt = st.sidebar.file_uploader(
    "Upload L1 checkpoint (.pt / .pth / .bin)",
    type=["pt", "pth", "bin"],
)

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

for k in ("market_df", "l1_model", "scaler_seq", "temp_scaler"):
    if k not in st.session_state:
        st.session_state[k] = None

# -------------------------------------------------------------------
# Fetch Gold data
# -------------------------------------------------------------------

if st.button("Fetch latest Gold (GC=F)"):
    try:
        symbol = "GC=F"
        df_raw = fetch_recent_daily_history(symbol, 365, st.session_state)

        today = datetime.utcnow().date()

        if include_today and today.weekday() < 5:
            if df_raw.empty or df_raw.index[-1] != today:
                try:
                    yesterday_close = fetch_last_completed_close(
                        symbol, st.session_state
                    )
                except Exception:
                    yesterday_close = None

                snapshot = fetch_snapshot(symbol, st.session_state)
                est = build_today_estimate(
                    yesterday_close, snapshot, st.session_state
                )
                est.name = today

                df = pd.concat([df_raw, pd.DataFrame([est])], axis=0)
            else:
                df = df_raw.copy()
        else:
            df = df_raw[df_raw.index != today].copy()

        if df.empty:
            st.error("No data returned")
        else:
            st.session_state.market_df = df
            st.success(f"Fetched {len(df)} bars")
            st.dataframe(df.tail(10))

    except Exception as e:
        st.error(f"Fetch failed: {e}")

# -------------------------------------------------------------------
# Load checkpoint & build matching model
# -------------------------------------------------------------------

if ckpt is not None:
    try:
        raw = ckpt.read()
        loaded = load_checkpoint_bytes_safe(raw)

        state_dict, extras = extract_state_dict(loaded)

        if state_dict is None:
            if isinstance(loaded, nn.Module):
                st.session_state.l1_model = loaded
                st.success("Loaded model object directly from checkpoint.")
            else:
                st.error("No usable state_dict found in checkpoint.")
        else:
            state_dict = strip_module_prefix(state_dict)
            inferred_in, inferred_channels = infer_arch_from_state(state_dict)

            if inferred_in is None:
                inferred_in = 12
                inferred_channels = (32, 64, 128)
                st.warning("Architecture inference failed; using defaults.")

            st.info(
                f"Inferred in_features={inferred_in}, "
                f"channels={inferred_channels}"
            )

            model = Level1ScopeCNN(
                in_features=inferred_in,
                channels=inferred_channels,
            )

            try:
                model.load_state_dict(state_dict, strict=True)
                st.success("Loaded state_dict (strict=True)")
            except Exception as e:
                st.warning(f"strict=True failed: {e}")
                model.load_state_dict(state_dict, strict=False)
                st.success("Loaded state_dict (strict=False)")

            model.eval()
            st.session_state.l1_model = model

            scaler = None
            if isinstance(loaded, dict):
                scaler = loaded.get("scaler_seq") or loaded.get("scaler")

            if scaler is not None:
                st.session_state.scaler_seq = scaler
                st.success("Loaded scaler from checkpoint")
            else:
                st.warning("No scaler found in checkpoint")

            temp_state = None
            if isinstance(loaded, dict):
                temp_state = loaded.get("temp_scaler_state")

            if temp_state is not None:
                ts = TemperatureScaler()
                try:
                    ts.load_state_dict(temp_state)
                    st.session_state.temp_scaler = ts
                    st.success("Loaded temperature scaler")
                except Exception:
                    st.warning("Temperature scaler load failed")

    except Exception as e:
        st.error(f"Checkpoint load failed: {e}")

# -------------------------------------------------------------------
# Run inference & propose limit order
# -------------------------------------------------------------------

if st.button("Run L1 inference & propose limit order"):
    if st.session_state.market_df is None:
        st.error("No market data. Fetch first.")
    elif st.session_state.l1_model is None:
        st.error("No model loaded.")
    else:
        df = st.session_state.market_df.copy()
        feats = compute_engineered_features(df)

        seq_cols = ["open", "high", "low", "close", "volume"]
        micro_cols = ["ret1", "tr", "vol_5", "mom_5", "chanpos_10"]

        feat_df = pd.concat(
            [
                df[seq_cols].astype(float),
                feats[[c for c in micro_cols if c in feats.columns]],
            ],
            axis=1,
        ).fillna(0.0)

        X = feat_df.values.astype("float32")

        scaler = st.session_state.scaler_seq
        if scaler is None:
            st.warning("Fitting temporary scaler (fallback only).")
            scaler = StandardScaler().fit(X)

        Xs = scaler.transform(X)

        Xseq = to_sequences(
            Xs,
            np.array([len(Xs) - 1]),
            seq_len,
        )
        xb = torch.tensor(Xseq.transpose(0, 2, 1))

        model = st.session_state.l1_model
        model.eval()

        with torch.no_grad():
            logit, _ = model(xb)

            if st.session_state.temp_scaler is not None:
                logit = st.session_state.temp_scaler(logit)

            prob = float(torch.sigmoid(logit).cpu().numpy()[0, 0])

        st.subheader("L1 result")
        st.write(f"Probability (buy): **{prob:.4f}**")

        atr = feats["atr"].iloc[-1]
        entry = float(df["close"].iloc[-1])
        sl = entry - atr * sl_mult
        tp = entry + atr * tp_mult

        risk_amount = account_balance * risk_pct
        stop_dist = abs(entry - sl)
        size = risk_amount / stop_dist if stop_dist > 0 else 0.0

        st.subheader("Proposed limit order (LONG)")
        st.json(
            {
                "entry": round(entry, 6),
                "stop_loss": round(sl, 6),
                "take_profit": round(tp, 6),
                "atr": float(atr),
                "position_size": float(size),
                "risk_amount_usd": float(risk_amount),
                "probability": float(prob),
            }
        )

st.caption(
    "Checkpoint loader auto-inferrs L1 architecture. "
    "If you still see key mismatch warnings, paste loaded.keys() and we’ll adapt."
)