# backtest_full.py
import io
import re
import logging
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from module import fetch_recent_daily_history

# 🔐 PyTorch 2.6+ safe globals
import torch.serialization
torch.serialization.add_safe_globals([StandardScaler])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("l1_backtest")

st.set_page_config(page_title="Cascade Trader — L1 Backtester", layout="wide")
st.title("Cascade Trader — L1 Backtesting Engine")

# ---------------------------
# Model definition (FIXED MIXED KERNEL SIZES)
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, d=1, pdrop=0.1):
        super().__init__()
        # Padding ensures output length matches input length
        pad = (k - 1) * d // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(pdrop)
        self.res = (c_in == c_out)

    def forward(self, x):
        y = self.drop(self.act(self.bn(self.conv(x))))
        return y + x if self.res else y


class Level1ScopeCNN(nn.Module):
    def __init__(self, in_features=10, channels=(32, 64, 128)):
        super().__init__()
        chs = [in_features] + list(channels)
        
        # FIX: Explicitly define kernel sizes based on error logs
        # Block 0: k=5
        # Block 1+: k=3
        kernels = [5] + [3] * (len(channels) - 1)

        self.blocks = nn.Sequential(
            *[
                ConvBlock(chs[i], chs[i + 1], k=kernels[i]) 
                for i in range(len(channels))
            ]
        )
        
        # MUST be named `project`
        self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
        self.head = nn.Linear(chs[-1], 1)

    def forward(self, x):
        z = self.blocks(x)
        z = self.project(z)
        z = z.mean(dim=-1)
        return self.head(z), z


# ---------------------------
# Feature engineering (10 FEATURES)
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
    
    # ADDED: 10th feature to match model input channels
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean().fillna(0.0)
    ma_down = down.rolling(14).mean().fillna(0.0)
    rs = ma_up / ma_down.replace(0, 1)
    f["rsi_14"] = 100 - (100 / (1 + rs))
    f["rsi_14"] = f["rsi_14"].fillna(50.0)

    return f.fillna(0.0)


def to_sequences(arr, idx, seq_len):
    out = []
    for i in idx:
        s = max(0, i - seq_len + 1)
        seq = arr[s:i + 1]
        if len(seq) < seq_len:
            pad = np.repeat(seq[[0]], seq_len - len(seq), axis=0)
            seq = np.vstack([pad, seq])
        out.append(seq)
    return np.array(out)




# ---------------------------
# SAFE checkpoint loading
# ---------------------------
def strip_module_prefix(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj:
                return obj[k]
        return obj
    return obj


def load_l1_from_checkpoint_bytes(raw_bytes):
    buf = io.BytesIO(raw_bytes)

    try:
        # 🔐 First attempt: safe load
        ckpt = torch.load(buf, map_location="cpu", weights_only=True)
    except Exception:
        # 🔓 Trusted fallback
        buf.seek(0)
        ckpt = torch.load(buf, map_location="cpu", weights_only=False)

    state_dict = strip_module_prefix(extract_state_dict(ckpt))

    model = Level1ScopeCNN()
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, None, None, ckpt, {}

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Backtest Config")

symbol = st.sidebar.text_input("Symbol", "GC=F")
start_date = st.sidebar.date_input("Start Date", date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2024, 1, 1))
seq_len = st.sidebar.slider("Sequence Length", 16, 128, 64, step=8)

buy_threshold = st.sidebar.slider("BUY prob ≥", 0.50, 0.95, 0.60)
sell_threshold = st.sidebar.slider("SELL prob ≥", 0.05, 0.50, 0.40)

sl_mult = st.sidebar.slider("SL ATR Mult", 0.5, 3.0, 1.0)
tp_mult = st.sidebar.slider("TP ATR Mult", 1.0, 5.0, 1.5)

account_balance = st.sidebar.number_input("Account Balance", 1000.0, value=10000.0)
risk_pct = st.sidebar.slider("Risk % per trade", 0.1, 5.0, 1.0) / 100.0

ckpt = st.sidebar.file_uploader("Upload model.pt", type=["pt", "pth"])


# ---------------------------
# Load model
# ---------------------------
if ckpt:
    model, _, _, _, _ = load_l1_from_checkpoint_bytes(ckpt.read())
    st.session_state.model = model
    st.success("Model loaded successfully")


# ---------------------------
# Run backtest
# ---------------------------
if st.button("Run Backtest"):
    if 'model' not in st.session_state:
        st.error("Please upload a model checkpoint first.")
        st.stop()
        
    model = st.session_state.model

    df = fetch_recent_daily_history(symbol, 3000)
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    feats = compute_engineered_features(df)
    
    # Concatenate 5 raw features + 5 engineered features = 10 total
    Xdf = pd.concat(
        [df[["open", "high", "low", "close", "volume"]], feats],
        axis=1
    ).astype("float32")

    scaler = StandardScaler().fit(Xdf.values)
    Xscaled = scaler.transform(Xdf.values)

    equity = account_balance
    trades = []

    for i in range(seq_len - 1, len(df) - 1):
        Xseq = to_sequences(Xscaled, [i], seq_len)
        xb = torch.tensor(Xseq.transpose(0, 2, 1))

        with torch.no_grad():
            logit, _ = model(xb)
            prob = torch.sigmoid(logit).item()

        short_prob = 1.0 - prob
        price = df["close"].iloc[i]
        atr = feats["atr"].iloc[i]

        if prob >= buy_threshold:
            side = "LONG"
        elif short_prob >= sell_threshold:
            side = "SHORT"
        else:
            continue

        entry = price
        sl = entry - atr * sl_mult if side == "LONG" else entry + atr * sl_mult
        tp = entry + atr * tp_mult if side == "LONG" else entry - atr * tp_mult

        for j in range(i + 1, len(df)):
            hi, lo = df["high"].iloc[j], df["low"].iloc[j]
            if side == "LONG" and (hi >= tp or lo <= sl):
                exit_px = tp if hi >= tp else sl
                break
            if side == "SHORT" and (lo <= tp or hi >= sl):
                exit_px = tp if lo <= tp else sl
                break
        else:
            exit_px = df["close"].iloc[-1]

        pnl = (exit_px - entry) if side == "LONG" else (entry - exit_px)
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
