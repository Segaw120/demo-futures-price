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
            # OPTION 1 FIX: kernel_size = 5 (matches training checkpoint)
            blocks.append(ConvBlock(chs[i], chs[i+1], k=5, d=1))
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
        nk = k[len("module."):] if isinstance(k, str) and k.startswith("module.") else k
        new[nk] = v
    return new


_conv_key_re = re.compile(r"blocks\.(\d+)\.conv\.weight")
def infer_arch_from_state(state):
    blocks = {}
    for k,v in state.items():
        m = _conv_key_re.search(k)
        if m and hasattr(v, "shape"):
            idx = int(m.group(1))
            blocks[idx] = (int(v.shape[0]), int(v.shape[1]))
    if not blocks:
        return None, None
    ordered = [blocks[i] for i in sorted(blocks.keys())]
    return ordered[0][1], tuple(o[0] for o in ordered)


def load_checkpoint_bytes_safe(raw_bytes):
    buf = io.BytesIO(raw_bytes)
    try:
        return torch.load(buf, map_location="cpu", weights_only=False)
    except Exception:
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=True)


def load_l1_from_checkpoint_bytes(raw_bytes):
    loaded = load_checkpoint_bytes_safe(raw_bytes)
    state_dict, extras = extract_state_dict(loaded)
    state_dict = strip_module_prefix(state_dict)

    inferred_in, inferred_channels = infer_arch_from_state(state_dict)
    inferred_in = inferred_in or 12
    inferred_channels = inferred_channels or (32,64,128)

    model = Level1ScopeCNN(in_features=inferred_in, channels=inferred_channels)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, None, None, loaded, extras

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Backtest Config")

symbol = st.sidebar.text_input("Symbol", value="GC=F")
start_date = st.sidebar.date_input("Start Date", date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2024, 1, 1))
seq_len = st.sidebar.slider("Sequence Length", 16, 128, 64, step=8)

buy_threshold = st.sidebar.slider("BUY prob ≥", 0.50, 0.95, 0.60)
sell_threshold = st.sidebar.slider("SELL prob ≥", 0.50, 0.05, 0.40)

sl_mult = st.sidebar.slider("SL ATR Mult", 0.5, 3.0, 1.0)
tp_mult = st.sidebar.slider("TP ATR Mult", 1.0, 5.0, 1.5)

account_balance = st.sidebar.number_input("Account Balance", 1000.0, value=10000.0)
risk_pct = st.sidebar.slider("Risk % per trade", 0.1, 5.0, 1.0) / 100.0

ckpt = st.sidebar.file_uploader("Upload model.pt", type=["pt","pth","bin"])


# ---------------------------
# Load model
# ---------------------------
if ckpt:
    model, _, _, _, _ = load_l1_from_checkpoint_bytes(ckpt.read())
    st.session_state.model = model
    st.success("Model loaded")


# ---------------------------
# Run backtest
# ---------------------------
if st.button("Run Backtest"):
    model = st.session_state.model
    df = fetch_recent_daily_history(symbol, 3000)
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    feats = compute_engineered_features(df)
    Xdf = pd.concat([df[["open","high","low","close","volume"]], feats], axis=1).astype("float32")
    scaler = StandardScaler().fit(Xdf.values)
    Xscaled = scaler.transform(Xdf.values)

    trades = []
    equity = account_balance

    for i in range(seq_len-1, len(df)-1):
        Xseq = to_sequences(Xscaled, [i], seq_len)
        xb = torch.tensor(Xseq.transpose(0,2,1))

        with torch.no_grad():
            logit, _ = model(xb)
            prob = torch.sigmoid(logit).item()

        short_prob = 1.0 - prob
        price = df["close"].iloc[i]
        atr = feats["atr"].iloc[i]

        side = None
        if prob >= buy_threshold and prob >= short_prob:
            side = "LONG"
        elif short_prob >= sell_threshold and short_prob > prob:
            side = "SHORT"
        else:
            continue

        entry = price
        sl = entry - atr*sl_mult if side=="LONG" else entry + atr*sl_mult
        tp = entry + atr*tp_mult if side=="LONG" else entry - atr*tp_mult

        for j in range(i+1, len(df)):
            hi, lo = df["high"].iloc[j], df["low"].iloc[j]
            if side=="LONG" and (hi>=tp or lo<=sl):
                exit_px = tp if hi>=tp else sl
                break
            if side=="SHORT" and (lo<=tp or hi>=sl):
                exit_px = tp if lo<=tp else sl
                break
        else:
            exit_px = df["close"].iloc[-1]

        pnl = (exit_px-entry) if side=="LONG" else (entry-exit_px)
        equity += pnl

        trades.append({
            "date": df.index[i],
            "side": side,
            "entry": entry,
            "exit": exit_px,
            "pnl": pnl,
            "equity": equity
        })

    st.dataframe(pd.DataFrame(trades))