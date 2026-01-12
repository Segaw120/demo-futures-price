# Standard library imports
import io
import re
import logging
from datetime import datetime, timedelta

# Third-party imports
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Local imports
from module import (
    fetch_recent_daily_history,
    fetch_snapshot,
    fetch_last_completed_close,
    build_today_estimate,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("l1_inference")

st.set_page_config(page_title="Cascade Trader — L1 Inference", layout="wide")
st.title("Cascade Trader — L1 Inference & Limit Orders (Auto-arch loader)")

# Flexible Level1 model (accepts channels tuple)
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, pdrop=0.1):
        super().__init__()
        pad = (k - 1) * d // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
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
    Flexible L1 CNN which accepts a channels tuple, kernel_sizes and dilations.
    channels: tuple of out-channels per block, e.g. (32,64,128)
    in_features: number of input features (channels) to the first conv
    """
    def __init__(self, in_features=12, channels=(32, 64, 128), kernel_sizes=(5, 3, 3), dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        chs = [in_features] + list(channels)
        blocks = []
        for i in range(len(channels)):
            k = kernel_sizes[min(i, len(kernel_sizes)-1)]
            d = dilations[min(i, len(dilations)-1)]
            blocks.append(ConvBlock(chs[i], chs[i+1], k=k, d=d, pdrop=dropout))
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

    def load_state(self, st_dict):
        try:
            self.load_state_dict(st_dict)
        except Exception:
            logger.warning("Temp scaler load failed.")

# Feature engineering (same as training)
def compute_engineered_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """
    Compute engineered features from financial data.

    Parameters:
    df (pd.DataFrame): Input data frame with financial data.
    windows (tuple): Windows for rolling calculations.

    Returns:
    pd.DataFrame: DataFrame with engineered features.
    """
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)
    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret1'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f['tr'] = tr.fillna(0.0)
    f['atr'] = tr.rolling(14, min_periods=1).mean().fillna(0.0)
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean() - v.rolling(max(1, w*3)).mean()).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Convert features into sequences for model input.

    Parameters:
    features (np.ndarray): Array of features.
    indices (np.ndarray): Array of indices.
    seq_len (int): Length of sequences.

    Returns:
    np.ndarray: Array of sequences.
    """
    Nrows, F = features.shape
    X = np.zeros((len(indices), seq_len, F), dtype=features.dtype)
    for i, t in enumerate(indices):
        t = int(t)
        t0 = t - seq_len + 1
        if t0 < 0:
            pad_count = -t0
            pad = np.repeat(features[[0]], pad_count, axis=0)
            seq = np.vstack([pad, features[0:t+1]])
        else:
            seq = features[t0:t+1]
        if seq.shape[0] < seq_len:
            pad_needed = seq_len - seq.shape[0]
            pad = np.repeat(seq[[0]], pad_needed, axis=0)
            seq = np.vstack([pad, seq])
        X[i] = seq[-seq_len:]
    return X

# Checkpoint robust loader helpers
def _is_state_dict_like(d: dict) -> bool:
    """
    Check if a dictionary is likely a state dict.

    Parameters:
    d (dict): Dictionary to check.

    Returns:
    bool: True if the dictionary is likely a state dict.
    """
    if not isinstance(d, dict):
        return False
    keys = list(d.keys())
    for k in keys[:20]:
        if any(sub in k for sub in ("conv.weight", "bn.weight", "head.weight", "proj.weight", "blocks.0.conv.weight")):
            return True
    vals = list(d.values())[:10]
    if all(isinstance(v, (torch.Tensor, np.ndarray)) for v in vals):
        return True
    return False

def extract_state_dict(container):
    """
    Extract state dict from a container.

    Parameters:
    container: Container to extract state dict from.

    Returns:
    tuple: (state_dict, extras)
    """
    if container is None:
        return None, {}
    if isinstance(container, dict) and _is_state_dict_like(container):
        return container, {}
    for key in ("model_state_dict", "state_dict", "model", "model_state", "model_weights"):
        if isinstance(container, dict) and key in container and _is_state_dict_like(container[key]):
            extras = {k: v for k, v in container.items() if k != key}
            return container[key], extras
    if isinstance(container, dict):
        for k, v in container.items():
            if isinstance(v, dict) and _is_state_dict_like(v):
                extras = {kk: vv for kk, vv in container.items() if kk != k}
                return v, extras
    return None, {}

def strip_module_prefix(state):
    """
    Strip 'module.' prefix from state dict keys.

    Parameters:
    state (dict): State dict.

    Returns:
    dict: State dict with 'module.' prefix stripped.
    """
    new = {}
    for k, v in state.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new[nk] = v
    return new

_conv_key_re = re.compile(r"blocks.(\d+).conv.weight")

def infer_arch_from_state(state):
    """
    Infer model architecture from state dict.

    Parameters:
    state (dict): State dict.

    Returns:
    tuple: (in_features, channels_list)
    """
    blocks = {}
    for k, v in state.items():
        m = _conv_key_re.search(k)
        if m and hasattr(v, "shape"):
            idx = int(m.group(1))
            out_ch = int(v.shape[0])
            in_ch = int(v.shape[1])
            blocks[idx] = (out_ch, in_ch, tuple(v.shape))
    if not blocks:
        for k, v in state.items():
            if ".conv.weight" in k and hasattr(v, "shape"):
                parts = k.split(".")
                try:
                    idx = int(parts[1]) if parts[0] == 'blocks' else None
                except Exception:
                    idx = None
                out_ch = int(v.shape[0])
                in_ch = int(v.shape[1])
                if idx is None:
                    blocks[0] = (out_ch, in_ch, tuple(v.shape))
                else:
                    blocks[idx] = (out_ch, in_ch, tuple(v.shape))
    if not blocks:
        return None, None
    ordered = [blocks[i] for i in sorted(blocks.keys())]
    channels = [b[0] for b in ordered]
    in_features = ordered[0][1]
    return int(in_features), tuple(int(x) for x in channels)

def load_checkpoint_bytes_safe(raw_bytes: bytes):
    """
    Load checkpoint from bytes safely.

    Parameters:
    raw_bytes (bytes): Bytes to load checkpoint from.

    Returns:
    object: Loaded object.
    """
    buf = io.BytesIO(raw_bytes)
    try:
        obj = torch.load(buf, map_location="cpu", weights_only=False)
        return obj
    except Exception as e:
        logger.info("torch.load direct failed: %s", e)
        buf.seek(0)
        try:
            obj = torch.load(buf, map_location="cpu", weights_only=True)
            return obj
        except Exception as e2:
            buf.seek(0)
            import pickle
            try:
                obj = pickle.loads(buf.read())
                return obj
            except Exception as e3:
                logger.exception("All checkpoint load attempts failed")
                raise RuntimeError(f"Failed to load checkpoint: {e3}") from e3

# Sidebar configuration
st.sidebar.header("Config")
seq_len = st.sidebar.slider("Sequence length", 8, 256, 64, step=8)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 2.0) / 100.0
tp_mult = st.sidebar.slider("TP ATR multiplier", 1.0, 5.0, 1.0)
sl_mult = st.sidebar.slider("SL ATR multiplier", 0.5, 3.0, 1.0)
account_balance = st.sidebar.number_input("Account balance ($)", value=10000.0)

# Toggle for Today's Data
include_today = st.sidebar.checkbox("Include Today's Estimate", value=True)

ckpt = st.sidebar.file_uploader("Upload L1 checkpoint (.pt/.pth/.bin)", type=["pt", "pth", "bin"])

# Session state
if "market_df" not in st.session_state:
    st.session_state.market_df = None
if "l1_model" not in st.session_state:
    st.session_state.l1_model = None
if "scaler_seq" not in st.session_state:
    st.session_state.scaler_seq = None
if "temp_scaler" not in st.session_state:
    st.session_state.temp_scaler = None

# Fetch Gold data
if st.button("Fetch latest Gold (GC=F)"):
    try:
        symbol = "GC=F"
        # 1. Fetch 365 days history using our module
        df_raw = fetch_recent_daily_history(
            symbol, 365, st.session_state
        )

        # 2. Logic to append today's estimate if toggle is ON
        today = datetime.utcnow().date()

        if include_today and today.weekday() < 5:
            # If today is a weekday and we want to include it...
            # Check if today is already in history (often is if market is open/settled)
            if df_raw.empty or df_raw.index[-1] != today:
                # Need to fetch estimate
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

                df = pd.concat(
                    [df_raw, pd.DataFrame([est])],
                    axis=0,
                )
            else:
                # Today is already in history, just use it
                df = df_raw.copy()
        else:
            # Toggle is OFF: Explicitly exclude today if it's in the history
            # This ensures we only see "yesterday's" settled data
            df = df_raw[df_raw.index != today].copy()

        if df.empty:
            st.error("No data returned")
        else:
            st.session_state.market_df = df
            st.success(f"Fetched {len(df)} bars")
            st.dataframe(df.tail(10))

    except Exception as e:
        st.error(f"Fetch failed: {e}")

# Load checkpoint and build model to match checkpoint architecture
if ckpt is not None:
    try:
        raw = ckpt.read()
        loaded = load_checkpoint_bytes_safe(raw)
        state_dict, extras = extract_state_dict(loaded)
        if state_dict is None:
            # maybe the file contains a nn.Module object directly
            if isinstance(loaded, nn.Module):
                st.session_state.l1_model = loaded
                st.success("Loaded L1 as module object from checkpoint.")
            else:
                st.error("Could not find state_dict inside checkpoint. Try saving state_dict for model only.")
        else:
            # remove 'module.' prefix if present
            state_dict = strip_module_prefix(state_dict)
            inferred_in, inferred_channels = infer_arch_from_state(state_dict)
            if inferred_in is None or inferred_channels is None:
                st.warning("Could not infer architecture from checkpoint; falling back to default channels (32,64,128) and in_features=12")
                inferred_in = inferred_in or 12
                inferred_channels = inferred_channels or (32, 64, 128)
            st.info(f"Inferred in_features={inferred_in}, channels={inferred_channels}")
            # instantiate model matching inferred channels
            model = Level1ScopeCNN(in_features=inferred_in, channels=inferred_channels)
            # load state_dict; prefer strict=True if shapes match exactly, else strict=False
            # attempt strict=True first to surface errors (so we can fallback)
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=True)
                st.success("Loaded state_dict with strict=True")
            except Exception as e_strict:
                st.warning(f"strict=True failed: {e_strict}. Trying strict=False to load compatible params.")
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                st.success("Loaded state_dict with strict=False (some params may be missing/unexpected)")
            model.eval()
            st.session_state.l1_model = model
            # load scaler from extras or loaded dict if present (best-effort)
            scaler_candidate = None
            if isinstance(loaded, dict):
                if "scaler_seq" in loaded:
                    scaler_candidate = loaded["scaler_seq"]
                elif "scaler" in loaded:
                    scaler_candidate = loaded["scaler"]
                elif "scaler_seq.pkl" in loaded:
                    scaler_candidate = loaded["scaler_seq.pkl"]
            if not scaler_candidate and isinstance(extras, dict):
                scaler_candidate = extras.get("scaler_seq") or extras.get("scaler")
            if scaler_candidate is not None:
                st.session_state.scaler_seq = scaler_candidate
                st.success("Loaded scaler from checkpoint (best-effort)")
            else:
                st.warning("No sequence scaler found in checkpoint; a temporary StandardScaler will be fit at inference (not recommended).")
            # temp scaler
            temp_state = None
            if isinstance(loaded, dict) and "temp_scaler_state" in loaded:
                temp_state = loaded["temp_scaler_state"]
            elif isinstance(extras, dict) and "temp_scaler_state" in extras:
                temp_state = extras.get("temp_scaler_state")
            if temp_state is not None:
                ts = TemperatureScaler()
                try:
                    ts.load_state_dict(temp_state)
                    st.session_state.temp_scaler = ts
                    st.success("Loaded temperature scaler state (best-effort)")
                except Exception:
                    st.warning("Failed to load temperature scaler state (shape mismatch).")
    except Exception as e:
        st.error(f"Failed to load L1 checkpoint: {e}")

# Run inference and propose limit order
if st.button("Run L1 inference & propose limit order"):
    if st.session_state.market_df is None:
        st.error("No market data. Fetch first.")
    elif st.session_state.l1_model is None:
        st.error("No model loaded. Upload checkpoint.")
    else:
        df = st.session_state.market_df.copy()
        feats = compute_engineered_features(df)
        seq_cols = ['open', 'high', 'low', 'close', 'volume']
        micro_cols = ['ret1', 'tr', 'vol_5', 'mom_5', 'chanpos_10']
        use_cols = [c for c in seq_cols + micro_cols if c in list(df.columns) + list(feats.columns)]
        feat_seq_df = pd.concat([df[seq_cols].astype(float), feats[[c for c in micro_cols if c in feats.columns]]], axis=1)[use_cols].fillna(0.0)
        X_all = feat_seq_df.values.astype('float32')
        scaler = st.session_state.scaler_seq
        if scaler is None:
            st.warning("No scaler in checkpoint — fitting temporary StandardScaler on the fetched data (this is only a fallback).")
            scaler = StandardScaler().fit(X_all)
        X_scaled = scaler.transform(X_all)
        last_idx = np.array([len(X_scaled)-1], dtype=int)
        Xseq = to_sequences(X_scaled, last_idx, seq_len=seq_len)
        xb = torch.tensor(Xseq.transpose(0, 2, 1), dtype=torch.float32)
        model = st.session_state.l1_model
        model.eval()
        with torch.no_grad():
            logit, emb = model(xb)
            # apply temp scaler if available
            if st.session_state.temp_scaler is not None:
                try:
                    logit_np = logit.cpu().numpy().reshape(-1, 1)
                    temp = st.session_state.temp_scaler
                    scaled = temp(torch.tensor(logit_np)).cpu().numpy().reshape(-1)
                    prob = float(1.0 / (1.0 + np.exp(-scaled))[0])
                except Exception:
                    prob = float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0])
            else:
                prob = float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0])
        st.subheader("L1 result")
        st.write(f"Probability (buy): {prob:.4f}")
        # compute ATR and limit order
        atr = feats['atr'].iloc[-1] if 'atr' in feats.columns else (df['high']-df['low']).rolling(14, min_periods=1).mean().iloc[-1]
        entry = float(df['close'].iloc[-1])
        sl = float(entry - atr * sl_mult)
        tp = float(entry + atr * tp_mult)
        # position sizing
        risk_amount = account_balance * risk_pct
        stop_distance = abs(entry - sl)
        size = risk_amount / stop_distance if stop_distance > 0 else 0.0
        st.subheader("Proposed limit order (LONG)")
        st.json({
            "entry": round(entry, 6),
            "stop_loss": round(sl, 6),
            "take_profit": round(tp, 6),
            "atr": float(atr),
            "position_size": float(size),
            "risk_amount_usd": float(risk_amount),
            "probability": float(prob)
        })

st.caption("This loader inspects the checkpoint, infers the L1 channels and input width, builds a matching model, and loads weights. If you still see warnings about missing/unexpected keys, paste list(loaded.keys()) here and I can adapt further.")