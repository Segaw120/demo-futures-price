# backtest_full.py
import io
import re
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

from module import fetch_recent_daily_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("l1_backtest")

st.set_page_config(page_title="Cascade Trader — L1 Backtester", layout="wide")
st.title("Cascade Trader — L1 Backtesting Engine")

# ---------------------------
# Model definition (UNCHANGED)
# ---------------------------
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
    def __init__(self, in_features=12, channels=(32,64,128)):
        super().__init__()
        chs = [in_features] + list(channels)
        blocks = []
        for i in range(len(channels)):
            blocks.append(ConvBlock(chs[i], chs[i+1], k=3, d=1))
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
        self.head = nn.Linear(chs[-1], 1)

    def forward(self, x):
        z = self.blocks(x)
        z = self.proj(z)
        z = z.mean(dim=-1)
        return self.head(z), z


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))
    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T

# ---------------------------
# Feature engineering (UNCHANGED)
# ---------------------------
def compute_engineered_features(df):
    f = pd.DataFrame(index=df.index)
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    ret1 = c.pct_change().fillna(0.0)
    f["ret1"] = ret1
    tr = (h - l).clip(lower=0)
    f["atr"] = tr.rolling(14, min_periods=1).mean().fillna(0.0)

    f["mom_5"] = (c - c.rolling(5).mean()).fillna(0.0)
    f["vol_5"] = ret1.rolling(5).std().fillna(0.0)

    return f.fillna(0.0)


def to_sequences(arr, idx, seq_len):
    out = []
    for i in idx:
        start = max(0, i - seq_len + 1)
        seq = arr[start:i+1]
        if len(seq) < seq_len:
            pad = np.repeat(seq[[0]], seq_len - len(seq), axis=0)
            seq = np.vstack([pad, seq])
        out.append(seq)
    return np.array(out)


# ---------------------------
# Robust checkpoint helpers (copied from inference loader)
# ---------------------------
def _is_state_dict_like(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    keys = list(d.keys())
    for k in keys[:20]:
        if any(sub in k for sub in ("conv.weight","bn.weight","head.weight","proj.weight","blocks.0.conv.weight")):
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
    for key in ("model_state_dict","state_dict","model","model_state","model_weights","l1_state_dict"):
        if isinstance(container, dict) and key in container and _is_state_dict_like(container[key]):
            extras = {k:v for k,v in container.items() if k != key}
            return container[key], extras
    if isinstance(container, dict):
        for k,v in container.items():
            if isinstance(v, dict) and _is_state_dict_like(v):
                extras = {kk:vv for kk,vv in container.items() if kk != k}
                return v, extras
    return None, {}

def strip_module_prefix(state):
    new = {}
    for k,v in state.items():
        nk = k
        if isinstance(k, str) and k.startswith("module."):
            nk = k[len("module."):]
        new[nk] = v
    return new

_conv_key_re = re.compile(r"blocks\.(\d+)\.conv\.weight")
def infer_arch_from_state(state):
    blocks = {}
    for k,v in state.items():
        m = _conv_key_re.search(k)
        if m and hasattr(v, "shape"):
            idx = int(m.group(1))
            out_ch = int(v.shape[0])
            in_ch = int(v.shape[1])
            blocks[idx] = (out_ch, in_ch, tuple(v.shape))
    if not blocks:
        for k,v in state.items():
            if ".conv.weight" in k and hasattr(v, "shape"):
                parts = k.split(".")
                try:
                    idx = int(parts[1]) if parts[0]=='blocks' else None
                except Exception:
                    idx = None
                out_ch = int(v.shape[0]); in_ch = int(v.shape[1])
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

def load_l1_from_checkpoint_bytes(raw_bytes: bytes):
    """
    Robust loader for L1 checkpoints.
    Returns: model, scaler_candidate (or None), temp_scaler (or None), loaded_obj, extras
    """
    loaded = load_checkpoint_bytes_safe(raw_bytes)
    state_dict, extras = extract_state_dict(loaded)

    # If loader returned an nn.Module directly
    if state_dict is None:
        if isinstance(loaded, nn.Module):
            model = loaded
            model.eval()
            return model, None, None, loaded, {}
        raise RuntimeError("Could not find state_dict inside checkpoint. Provide a state_dict or Module.")

    # strip prefix and infer architecture
    state_dict = strip_module_prefix(state_dict)
    inferred_in, inferred_channels = infer_arch_from_state(state_dict)
    if inferred_in is None or inferred_channels is None:
        inferred_in = inferred_in or 12
        inferred_channels = inferred_channels or (32,64,128)

    # instantiate and load
    model = Level1ScopeCNN(in_features=inferred_in, channels=inferred_channels)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    # scaler candidate best-effort
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

    # temp scaler
    temp_state = None
    if isinstance(loaded, dict) and "temp_scaler_state" in loaded:
        temp_state = loaded["temp_scaler_state"]
    elif isinstance(extras, dict) and "temp_scaler_state" in extras:
        temp_state = extras.get("temp_scaler_state")

    temp_scaler = None
    if temp_state is not None:
        ts = TemperatureScaler()
        try:
            ts.load_state_dict(temp_state)
            temp_scaler = ts
        except Exception:
            temp_scaler = None

    return model, scaler_candidate, temp_scaler, loaded, extras

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Backtest Config")

symbol = st.sidebar.text_input("Symbol", value="GC=F")
start_date = st.sidebar.date_input("Start Date", date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2024, 1, 1))

seq_len = st.sidebar.slider("Sequence Length", 16, 128, 64, step=8)

buy_threshold = st.sidebar.slider("BUY prob ≥", 0.50, 0.95, 0.60)
sell_threshold = st.sidebar.slider("SELL prob ≥ (short prob)", 0.50, 0.05, 0.40)

sl_mult = st.sidebar.slider("SL ATR Mult", 0.5, 3.0, 1.0)
tp_mult = st.sidebar.slider("TP ATR Mult", 1.0, 5.0, 1.5)

account_balance = st.sidebar.number_input("Account Balance", 1000.0, value=10000.0)
risk_pct = st.sidebar.slider("Risk % per trade", 0.1, 5.0, 1.0) / 100.0

ckpt = st.sidebar.file_uploader("Upload model.pt", type=["pt", "pth", "bin"])

# Session state
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "temp_scaler" not in st.session_state:
    st.session_state.temp_scaler = None
if "loaded_ckpt" not in st.session_state:
    st.session_state.loaded_ckpt = None

# ---------------------------
# Load model (robust)
# ---------------------------
if ckpt:
    raw = ckpt.read()
    try:
        model, scaler_candidate, temp_scaler, loaded_obj, extras = load_l1_from_checkpoint_bytes(raw)
        st.session_state.model = model
        if scaler_candidate is not None:
            st.session_state.scaler = scaler_candidate
            st.success("Loaded scaler from checkpoint (best-effort).")
        else:
            st.warning("No scaler in checkpoint — will fit StandardScaler on data at inference.")
        if temp_scaler is not None:
            st.session_state.temp_scaler = temp_scaler
            st.success("Loaded temperature scaler state (best-effort).")
        st.session_state.loaded_ckpt = loaded_obj
        st.success("Model loaded (best-effort).")
        if isinstance(loaded_obj, dict):
            st.write("Checkpoint keys (sample):", list(loaded_obj.keys())[:40])
        if extras:
            st.write("Extras keys (sample):", list(extras.keys())[:40])
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        logger.exception("Checkpoint load failed")

# ---------------------------
# Run backtest
# ---------------------------
if st.button("Run Backtest"):
    if st.session_state.model is None:
        st.error("Upload model first")
        st.stop()

    df = fetch_recent_daily_history(symbol, 3000)
    if df.empty:
        st.error("No data returned from fetch_recent_daily_history")
        st.stop()

    # slice to requested date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    if df.empty:
        st.error("No bars in requested date range")
        st.stop()

    feats = compute_engineered_features(df)

    Xdf = pd.concat(
        [df[["open","high","low","close","volume"]], feats],
        axis=1
    ).astype("float32")

    # prefer scaler from checkpoint
    if st.session_state.scaler is not None:
        scaler = st.session_state.scaler
    else:
        scaler = StandardScaler().fit(Xdf.values)
    Xscaled = scaler.transform(Xdf.values)

    trades = []
    equity = account_balance

    # candidate loop
    for i in range(seq_len-1, len(df)-1):
        Xseq = to_sequences(Xscaled, [i], seq_len)
        xb = torch.tensor(Xseq.transpose(0,2,1))

        with torch.no_grad():
            logit, _ = st.session_state.model(xb)
            # apply temp scaler if available
            if st.session_state.temp_scaler is not None:
                try:
                    logit_np = logit.cpu().numpy().reshape(-1,1)
                    temp = st.session_state.temp_scaler
                    scaled = temp(torch.tensor(logit_np)).cpu().numpy().reshape(-1)
                    pvals = 1.0 / (1.0 + np.exp(-scaled))
                    prob = float(pvals[0])
                except Exception:
                    prob = float(torch.sigmoid(logit).item())
            else:
                prob = float(torch.sigmoid(logit).item())

        price = float(df["close"].iloc[i])
        atr = float(feats["atr"].iloc[i])

        chosen_side = None
        chosen_conf = None
        short_prob = 1.0 - prob

        if (prob >= buy_threshold) and (prob >= short_prob):
            chosen_side = "LONG"
            chosen_conf = prob
        elif (short_prob >= sell_threshold) and (short_prob > prob):
            chosen_side = "SHORT"
            chosen_conf = short_prob

        if chosen_side is None:
            continue

        entry = price
        if chosen_side == "LONG":
            sl = entry - atr * sl_mult
            tp = entry + atr * tp_mult
        else:
            sl = entry + atr * sl_mult
            tp = entry - atr * tp_mult

        # simulate forward until TP/SL or end
        exit_price = None
        exit_reason = "timeout"
        for j in range(i+1, len(df)):
            hi = float(df["high"].iloc[j])
            lo = float(df["low"].iloc[j])
            if chosen_side == "LONG":
                if hi >= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    exit_idx = j
                    break
                if lo <= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    exit_idx = j
                    break
            else:
                if lo <= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    exit_idx = j
                    break
                if hi >= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    exit_idx = j
                    break

        if exit_price is None:
            # final exit at last close
            exit_idx = len(df)-1
            exit_price = float(df["close"].iloc[exit_idx])
            if chosen_side == "LONG":
                pnl_frac = (exit_price - entry) / entry
            else:
                pnl_frac = (entry - exit_price) / entry
            exit_reason = "timeout"
        else:
            if chosen_side == "LONG":
                pnl_frac = (exit_price - entry) / entry
            else:
                pnl_frac = (entry - exit_price) / entry

        # sizing
        risk_amt = equity * risk_pct
        stop_distance = abs(entry - sl)
        position_size = (risk_amt / stop_distance) if stop_distance > 0 else 0.0
        usd_pnl = position_size * pnl_frac * entry  # approx

        equity += usd_pnl

        trades.append({
            "entry_date": df.index[i].date(),
            "entry_idx": i,
            "side": chosen_side,
            "confidence": round(chosen_conf, 4),
            "entry_px": round(entry, 6),
            "sl_px": round(sl, 6),
            "tp_px": round(tp, 6),
            "exit_date": df.index[exit_idx].date(),
            "exit_idx": exit_idx,
            "exit_px": round(exit_price, 6),
            "pnl_frac": round(pnl_frac, 6),
            "usd_pnl": round(usd_pnl, 2),
            "duration_days": (df.index[exit_idx] - df.index[i]).days,
            "exit_reason": exit_reason
        })

    trades_df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)

    st.subheader("Backtest Results")
    st.write(f"Total Trades: {len(trades_df)}")
    st.write(f"Net PnL (USD): {trades_df['usd_pnl'].sum():.2f}")
    st.write(f"Final Equity: {equity:.2f}")
    if not trades_df.empty:
        st.write(f"Win Rate: {(trades_df['pnl_frac'] > 0).mean()*100:.1f}%")
    st.dataframe(trades_df.tail(100))

    # download
    if not trades_df.empty:
        csv = trades_df.to_csv(index=False).encode()
        st.download_button("Download trades CSV", data=csv, file_name="backtest_trades.csv", mime="text/csv")