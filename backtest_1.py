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
# Sidebar controls
# ---------------------------
st.sidebar.header("Backtest Config")

symbol = st.sidebar.text_input("Symbol", value="GC=F")
start_date = st.sidebar.date_input("Start Date", date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2024, 1, 1))

seq_len = st.sidebar.slider("Sequence Length", 16, 128, 64, step=8)

buy_threshold = st.sidebar.slider("BUY prob ≥", 0.50, 0.95, 0.60)
sell_threshold = st.sidebar.slider("SELL prob ≤", 0.50, 0.05, 0.40)

sl_mult = st.sidebar.slider("SL ATR Mult", 0.5, 3.0, 1.0)
tp_mult = st.sidebar.slider("TP ATR Mult", 1.0, 5.0, 1.5)

account_balance = st.sidebar.number_input("Account Balance", 1000.0, value=10000.0)
risk_pct = st.sidebar.slider("Risk % per trade", 0.1, 5.0, 1.0) / 100.0

ckpt = st.sidebar.file_uploader("Upload model.pt", type=["pt", "pth"])

# Session state
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None

# ---------------------------
# Load model
# ---------------------------
if ckpt:
    raw = ckpt.read()
    state = torch.load(io.BytesIO(raw), map_location="cpu")
    model = Level1ScopeCNN()
    model.load_state_dict(state, strict=False)
    model.eval()
    st.session_state.model = model
    st.success("Model loaded")


# ---------------------------
# Run backtest
# ---------------------------
if st.button("Run Backtest"):
    if st.session_state.model is None:
        st.error("Upload model first")
        st.stop()

    df = fetch_recent_daily_history(symbol, 3000)
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    feats = compute_engineered_features(df)

    Xdf = pd.concat(
        [df[["open","high","low","close","volume"]], feats],
        axis=1
    ).astype("float32")

    scaler = StandardScaler().fit(Xdf.values)
    Xscaled = scaler.transform(Xdf.values)

    trades = []
    equity = account_balance

    for i in range(seq_len, len(df)-1):
        Xseq = to_sequences(Xscaled, [i], seq_len)
        xb = torch.tensor(Xseq.transpose(0,2,1))

        with torch.no_grad():
            logit, _ = st.session_state.model(xb)
            prob = torch.sigmoid(logit).item()

        price = df["close"].iloc[i]
        atr = feats["atr"].iloc[i]

        direction = None
        if prob >= buy_threshold:
            direction = "LONG"
            entry = price
            sl = entry - atr * sl_mult
            tp = entry + atr * tp_mult
        elif prob <= sell_threshold:
            direction = "SHORT"
            entry = price
            sl = entry + atr * sl_mult
            tp = entry - atr * tp_mult

        if direction is None:
            continue

        # simulate next bars
        exit_price = None
        for j in range(i+1, len(df)):
            hi = df["high"].iloc[j]
            lo = df["low"].iloc[j]

            if direction == "LONG":
                if lo <= sl:
                    exit_price = sl
                    break
                if hi >= tp:
                    exit_price = tp
                    break
            else:
                if hi >= sl:
                    exit_price = sl
                    break
                if lo <= tp:
                    exit_price = tp
                    break

        if exit_price is None:
            continue

        risk_amt = equity * risk_pct
        pos_size = risk_amt / abs(entry - sl)
        pnl = pos_size * (exit_price - entry)
        if direction == "SHORT":
            pnl = -pnl

        equity += pnl

        trades.append({
            "date": df.index[i],
            "direction": direction,
            "entry": entry,
            "exit": exit_price,
            "pnl": pnl,
            "equity": equity,
            "prob": prob
        })

    trades_df = pd.DataFrame(trades)

    st.subheader("Backtest Results")
    st.write(f"Total Trades: {len(trades_df)}")
    st.write(f"Net PnL: {trades_df['pnl'].sum():.2f}")
    st.write(f"Final Equity: {equity:.2f}")
    st.write(f"Win Rate: {(trades_df['pnl'] > 0).mean()*100:.1f}%")

    st.dataframe(trades_df.tail(50))