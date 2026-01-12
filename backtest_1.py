import os
import io
import math
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# sklearn
from sklearn.preprocessing import StandardScaler

# yahooquery
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

# xgboost (optional for L2)
try:
    import xgboost as xgb
except Exception:
    xgb = None

# torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    nn = None
    optim = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cascade_backtester")

# Streamlit config
st.set_page_config(page_title="Cascade Model Backtester — Gold Futures", layout="wide")
st.title("🎯 Cascade Model Backtester — Gold Futures (GC=F)")

# ============================================================================
# UTILITY: Ensure unique index
# ============================================================================
def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate index entries and sort."""
    if df is None or df.empty:
        return df
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()

# ============================================================================
# FEATURE ENGINEERING (exact replica from training)
# ============================================================================
def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    """Compute engineered features from OHLCV - exact replica from training."""
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
    
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean() - v.rolling(max(1,w*3)).mean()).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
    
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def _true_range(high, low, close):
    """Compute true range for ATR calculation."""
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Build sequences ending at each index t: [t-seq_len+1, ..., t]
    Returns shape [N, seq_len, F]
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

# ============================================================================
# MODEL ARCHITECTURE DEFINITIONS (must match training)
# ============================================================================
if torch is not None:
    class ConvBlock(nn.Module):
        def __init__(self, c_in, c_out, k, d, pdrop):
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
            if self.res:
                out = out + x
            return out

    class Level1ScopeCNN(nn.Module):
        def __init__(self, in_features=12, channels=(32,64,128), 
                     kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1):
            super().__init__()
            chs = [in_features] + list(channels)
            blocks = []
            for i in range(len(channels)):
                k = kernel_sizes[min(i, len(kernel_sizes)-1)]
                d = dilations[min(i, len(dilations)-1)]
                blocks.append(ConvBlock(chs[i], chs[i+1], k, d, dropout))
            self.blocks = nn.Sequential(*blocks)
            self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
            self.head = nn.Linear(chs[-1], 1)
        
        @property
        def embedding_dim(self):
            return int(self.blocks[-1].conv.out_channels)
        
        def forward(self, x):
            z = self.blocks(x)
            z = self.project(z)
            z_pool = z.mean(dim=-1)
            logit = self.head(z_pool)
            return logit, z_pool

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, out_dim=1, dropout=0.1):
            super().__init__()
            layers = []
            last = in_dim
            for h in hidden:
                layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
                last = h
            layers += [nn.Linear(last, out_dim)]
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.net(x)

    class Level3ShootMLP(nn.Module):
        def __init__(self, in_dim, hidden=(128,64), dropout=0.1, use_regression_head=True):
            super().__init__()
            self.backbone = MLP(in_dim, list(hidden), out_dim=128, dropout=dropout)
            self.cls_head = nn.Linear(128, 1)
            self.reg_head = nn.Linear(128, 1) if use_regression_head else None
        
        def forward(self, x):
            h = self.backbone(x)
            logit = self.cls_head(h)
            ret = self.reg_head(h) if self.reg_head is not None else None
            return logit, ret

    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_temp = nn.Parameter(torch.zeros(1))
        
        def forward(self, logits):
            T = torch.exp(self.log_temp)
            return logits / T
        
        def transform(self, logits: np.ndarray) -> np.ndarray:
            """Transform logits with temperature scaling."""
            with torch.no_grad():
                device = next(self.parameters()).device
                logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
                scaled = self.forward(logits_t).cpu().numpy()
            return scaled.reshape(-1)

# ============================================================================
# MODEL LOADER
# ============================================================================
def load_cascade_model(pt_path: str, device: str = "auto") -> Dict[str, Any]:
    """
    Load a complete cascade model from .pt file.
    Returns dict with all model components.
    """
    if torch is None:
        raise RuntimeError("PyTorch not available")
    
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(pt_path, map_location=dev)
    
    # Extract configuration
    seq_len = checkpoint.get('seq_len', 64)
    feat_windows = checkpoint.get('feat_windows', (5, 10, 20))
    tab_feature_names = checkpoint.get('tab_feature_names', [])
    
    # Load L1 model
    l1_config = checkpoint.get('config', {})
    in_features = l1_config.get('in_features', 12)
    channels = l1_config.get('channels', [32, 64, 128])
    
    l1_model = Level1ScopeCNN(
        in_features=in_features,
        channels=tuple(channels),
        kernel_sizes=(5, 3, 3),
        dilations=(1, 2, 4),
        dropout=0.1
    ).to(dev)
    
    # Load state dicts
    if 'l1_state_dict' in checkpoint:
        l1_model.load_state_dict(checkpoint['l1_state_dict'])
    elif 'model_state_dict' in checkpoint:
        l1_model.load_state_dict(checkpoint['model_state_dict'])
    
    l1_model.eval()
    
    # Load temperature scaler for L1
    l1_temp = TemperatureScaler().to(dev)
    if 'l1_temp_state_dict' in checkpoint:
        l1_temp.load_state_dict(checkpoint['l1_temp_state_dict'])
    elif 'temp_scaler_state' in checkpoint:
        l1_temp.load_state_dict(checkpoint['temp_scaler_state'])
    
    # Load scalers
    scaler_seq = checkpoint.get('scaler_seq', StandardScaler())
    scaler_tab = checkpoint.get('scaler_tab', StandardScaler())
    
    # Load L2 if present (optional for backtester)
    l2_model = None
    l2_backend = checkpoint.get('l2_backend', 'mlp')
    
    if 'l2_state_dict' in checkpoint:
        # MLP backend
        l2_in_dim = checkpoint['l2_state_dict']['net.0.weight'].shape[1]
        l2_model = MLP(l2_in_dim, [128, 64], out_dim=1, dropout=0.1).to(dev)
        l2_model.load_state_dict(checkpoint['l2_state_dict'])
        l2_model.eval()
    
    # Load L3 if present
    l3_model = None
    l3_temp = TemperatureScaler().to(dev)
    
    if 'l3_state_dict' in checkpoint:
        l3_in_dim = checkpoint['l3_state_dict']['backbone.net.0.weight'].shape[1]
        l3_model = Level3ShootMLP(l3_in_dim, hidden=(128, 64), dropout=0.1).to(dev)
        l3_model.load_state_dict(checkpoint['l3_state_dict'])
        l3_model.eval()
        
        if 'l3_temp_state_dict' in checkpoint:
            l3_temp.load_state_dict(checkpoint['l3_temp_state_dict'])
    
    metadata = checkpoint.get('metadata', {})
    
    return {
        'l1_model': l1_model,
        'l1_temp': l1_temp,
        'l2_model': l2_model,
        'l2_backend': l2_backend,
        'l3_model': l3_model,
        'l3_temp': l3_temp,
        'scaler_seq': scaler_seq,
        'scaler_tab': scaler_tab,
        'seq_len': seq_len,
        'feat_windows': feat_windows,
        'tab_feature_names': tab_feature_names,
        'device': dev,
        'metadata': metadata
    }

# ============================================================================
# TRADE GENERATION (exact replica from training)
# ============================================================================
def generate_candidates_and_labels(
    bars: pd.DataFrame,
    lookback: int = 64,
    k_tp: float = 3.0,
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars: int = 60,
    direction: str = "long"
) -> pd.DataFrame:
    """
    Generate trade candidates with TP/SL based on ATR.
    This is the EXACT same function from training.
    """
    if bars is None or bars.empty:
        return pd.DataFrame()
    
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    
    for col in ("high", "low", "close"):
        if col not in bars.columns:
            raise KeyError(f"Missing column {col}")
    
    # Compute ATR
    bars["tr"] = _true_range(bars["high"], bars["low"], bars["close"])
    bars["atr"] = bars["tr"].rolling(atr_window, min_periods=1).mean()
    
    records = []
    n = len(bars)
    
    for i in range(lookback, n):
        t = bars.index[i]
        entry_px = float(bars["close"].iat[i])
        atr_val = float(bars["atr"].iat[i])
        
        if atr_val <= 0 or math.isnan(atr_val):
            continue
        
        # Calculate SL and TP based on direction
        if direction == "long":
            sl_px = entry_px - k_sl * atr_val
            tp_px = entry_px + k_tp * atr_val
        else:  # short/sell
            sl_px = entry_px + k_sl * atr_val
            tp_px = entry_px - k_tp * atr_val
        
        # Simulate forward to find outcome
        end_i = min(i + max_bars, n - 1)
        label = 0
        hit_i = end_i
        hit_px = float(bars["close"].iat[end_i])
        
        for j in range(i + 1, end_i + 1):
            hi = float(bars["high"].iat[j])
            lo = float(bars["low"].iat[j])
            
            if direction == "long":
                if hi >= tp_px:
                    label, hit_i, hit_px = 1, j, tp_px
                    break
                if lo <= sl_px:
                    label, hit_i, hit_px = 0, j, sl_px
                    break
            else:  # short
                if lo <= tp_px:
                    label, hit_i, hit_px = 1, j, tp_px
                    break
                if hi >= sl_px:
                    label, hit_i, hit_px = 0, j, sl_px
                    break
        
        end_t = bars.index[hit_i]
        
        # Calculate realized return
        if direction == "long":
            ret_val = (hit_px - entry_px) / entry_px
        else:
            ret_val = (entry_px - hit_px) / entry_px
        
        dur_min = (end_t - t).total_seconds() / 60.0
        
        records.append({
            'candidate_time': t,
            'entry_price': entry_px,
            'atr': float(atr_val),
            'sl_price': float(sl_px),
            'tp_price': float(tp_px),
            'end_time': end_t,
            'label': int(label),
            'duration': float(dur_min),
            'realized_return': float(ret_val),
            'direction': direction
        })
    
    return pd.DataFrame(records)

# ============================================================================
# MODEL INFERENCE ON CANDIDATES
# ============================================================================
def predict_on_candidates(
    model_components: Dict[str, Any],
    bars: pd.DataFrame,
    candidate_indices: np.ndarray
) -> pd.DataFrame:
    """
    Run cascade inference on candidate indices.
    Returns DataFrame with p1, p2, p3 predictions.
    """
    dev = model_components['device']
    l1_model = model_components['l1_model']
    l1_temp = model_components['l1_temp']
    l2_model = model_components['l2_model']
    l3_model = model_components['l3_model']
    l3_temp = model_components['l3_temp']
    scaler_seq = model_components['scaler_seq']
    scaler_tab = model_components['scaler_tab']
    seq_len = model_components['seq_len']
    feat_windows = model_components['feat_windows']
    tab_feature_names = model_components['tab_feature_names']
    
    # Compute features
    eng = compute_engineered_features(bars, windows=feat_windows)
    
    # Sequence features
    seq_cols = ['open', 'high', 'low', 'close', 'volume']
    micro_cols = ['ret1', 'tr', 'vol_5', 'mom_5', 'chanpos_10']
    use_cols = [c for c in seq_cols + micro_cols if c in list(bars.columns) + list(eng.columns)]
    
    feat_seq_df = pd.concat([
        bars[seq_cols].astype(float),
        eng[[c for c in micro_cols if c in eng.columns]]
    ], axis=1)[use_cols].fillna(0.0)
    
    feat_tab_df = eng[tab_feature_names].fillna(0.0)
    
    # Scale features
    X_seq_all_scaled = scaler_seq.transform(feat_seq_df.values)
    X_tab_all_scaled = scaler_tab.transform(feat_tab_df.values)
    
    # Build sequences
    Xseq = to_sequences(X_seq_all_scaled, candidate_indices, seq_len=seq_len)
    
    # L1 inference
    l1_model.eval()
    logits = []
    embeds = []
    batch = 256
    
    with torch.no_grad():
        for i in range(0, len(Xseq), batch):
            sub = Xseq[i:i+batch]
            xb = torch.tensor(sub.transpose(0,2,1), dtype=torch.float32, device=dev)
            logit, emb = l1_model(xb)
            logits.append(logit.detach().cpu().numpy())
            embeds.append(emb.detach().cpu().numpy())
    
    l1_logits = np.concatenate(logits, axis=0).reshape(-1, 1)
    l1_emb = np.concatenate(embeds, axis=0)
    
    # Temperature scaling
    l1_logits_scaled = l1_temp.transform(l1_logits.reshape(-1, 1)).reshape(-1)
    p1 = 1.0 / (1.0 + np.exp(-l1_logits_scaled))
    
    # L2 inference (if available)
    p2 = np.zeros_like(p1)
    if l2_model is not None:
        X_l2 = np.hstack([l1_emb, X_tab_all_scaled[candidate_indices]])
        
        l2_model.eval()
        batch = 2048
        probs = []
        
        with torch.no_grad():
            for i in range(0, len(X_l2), batch):
                xb = torch.tensor(X_l2[i:i+batch], dtype=torch.float32, device=dev)
                logit = l2_model(xb)
                p = torch.sigmoid(logit).cpu().numpy().reshape(-1)
                probs.append(p)
        
        p2 = np.concatenate(probs, axis=0)
    
    # L3 inference (if available)
    p3 = np.zeros_like(p1)
    if l3_model is not None and l2_model is not None:
        X_l3 = np.hstack([l1_emb, X_tab_all_scaled[candidate_indices]])
        
        l3_model.eval()
        batch = 2048
        l3_logits_list = []
        
        with torch.no_grad():
            for i in range(0, len(X_l3), batch):
                xb = torch.tensor(X_l3[i:i+batch], dtype=torch.float32, device=dev)
                logit, _ = l3_model(xb)
                l3_logits_list.append(logit.detach().cpu().numpy())
        
        l3_logits = np.concatenate(l3_logits_list, axis=0).reshape(-1, 1)
        l3_logits_scaled = l3_temp.transform(l3_logits.reshape(-1, 1)).reshape(-1)
        p3 = 1.0 / (1.0 + np.exp(-l3_logits_scaled))
    
    return pd.DataFrame({
        't': candidate_indices,
        'p1': p1,
        'p2': p2,
        'p3': p3
    })

# ============================================================================
# TRADE EXECUTION LOGIC (replicated from training)
# ============================================================================
def generate_trade_from_prediction(
    close: float,
    atr: float,
    sl_mult: float,
    tp_mult: float,
    p_long: float,
    side: str
) -> Dict[str, Any]:
    """
    Faithful replication of training-time trade generation.
    
    Args:
        close: Current close price
        atr: Current ATR value
        sl_mult: Stop loss multiplier
        tp_mult: Take profit multiplier
        p_long: Model probability for LONG
        side: "LONG" or "SELL"
    
    Returns:
        Trade dictionary with entry, SL, TP, and confidence
    """
    if side == "LONG":
        entry = close
        sl = entry - sl_mult * atr
        tp = entry + tp_mult * atr
        confidence = p_long
    
    elif side == "SELL":
        entry = close
        sl = entry + sl_mult * atr
        tp = entry - tp_mult * atr
        confidence = 1.0 - p_long
    
    else:
        raise ValueError("side must be LONG or SELL")
    
    return {
        "side": side,
        "entry": float(entry),
        "stop_loss": float(sl),
        "take_profit": float(tp),
        "confidence": float(confidence),
        "atr": float(atr)
    }

# ============================================================================
# BACKTEST SIMULATION
# ============================================================================
def simulate_backtest(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
    confidence_threshold: float = 0.55,
    max_holding_bars: int = 60
) -> pd.DataFrame:
    """
    Simulate trades based on predictions and track outcomes.
    
    Returns DataFrame with trade results.
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    
    # Merge predictions with candidates
    merged = candidates.copy()
    pred_dict = predictions.set_index('t')['p3'].to_dict()
    merged['predicted_prob'] = merged.index.map(lambda i: pred_dict.get(i, 0.0))
    
    # Filter by confidence threshold
    merged = merged[merged['predicted_prob'] >= confidence_threshold].copy()
    
    if merged.empty:
        return pd.DataFrame()
    
    trades = []
    
    for idx, row in merged.iterrows():
        entry_time = pd.to_datetime(row['candidate_time'])
        entry_px = float(row['entry_price'])
        sl_px = float(row['sl_price'])
        tp_px = float(row['tp_price'])
        direction = row.get('direction', 'long')
        predicted_prob = float(row['predicted_prob'])
        
        if entry_time not in bars.index:
            continue
        
        # Get forward-looking segment
        segment = bars.loc[entry_time:].head(max_holding_bars)
        
        if segment.empty:
            continue
        
        # Track trade outcome
        exit_time = None
        exit_px = None
        pnl = None
        outcome = 'timeout'
        
        for t, bar in segment.iterrows():
            if t == entry_time:
                continue
            
            hi = float(bar['high'])
            lo = float(bar['low'])
            
            if direction == 'long':
                # Check TP hit
                if hi >= tp_px:
                    exit_time = t
                    exit_px = tp_px
                    pnl = (tp_px - entry_px) / entry_px
                    outcome = 'tp_hit'
                    break
                # Check SL hit
                if lo <= sl_px:
                    exit_time = t
                    exit_px = sl_px
                    pnl = (sl_px - entry_px) / entry_px
                    outcome = 'sl_hit'
                    break
            else:  # short/sell
                # Check TP hit
                if lo <= tp_px:
                    exit_time = t
                    exit_px = tp_px
                    pnl = (entry_px - tp_px) / entry_px
                    outcome = 'tp_hit'
                    break
                # Check SL hit
                if hi >= sl_px:
                    exit_time = t
                    exit_px = sl_px
                    pnl = (entry_px - sl_px) / entry_px
                    outcome = 'sl_hit'
                    break
        
        # If no exit, close at last bar
        if exit_time is None:
            last_bar = segment.iloc[-1]
            exit_time = last_bar.name
            exit_px = float(last_bar['close'])
            
            if direction == 'long':
                pnl = (exit_px - entry_px) / entry_px
            else:
                pnl = (entry_px - exit_px) / entry_px
        
        duration_bars = len(bars.loc[entry_time:exit_time]) - 1
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_px,
            'exit_price': exit_px,
            'sl_price': sl_px,
            'tp_price': tp_px,
            'direction': direction,
            'pnl': float(pnl),
            'pnl_pct': float(pnl * 100),
            'outcome': outcome,
            'duration_bars': int(duration_bars),
            'predicted_prob': predicted_prob,
            'actual_label': int(row.get('label', 0))
        })
    
    return pd.DataFrame(trades)

# ============================================================================
# BACKTEST METRICS
# ============================================================================
def compute_backtest_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive backtest metrics."""
    if trades is None or trades.empty:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'total_pnl': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0
        }
    
    total_trades = len(trades)
    wins = (trades['pnl'] > 0).sum()
    losses = (trades['pnl'] <= 0).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    
    avg_pnl = float(trades['pnl'].mean())
    median_pnl = float(trades['pnl'].median())
    total_pnl = float(trades['pnl'].sum())
    
    # Cumulative PnL for drawdown
    cum_pnl = trades['pnl'].cumsum()
    running_max = cum_pnl.expanding().max()
    drawdown = cum_pnl - running_max
    max_drawdown = float(drawdown.min())
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    if trades['pnl'].std() > 0:
        sharpe = (avg_pnl / trades['pnl'].std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Profit factor
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    # Outcome breakdown
    tp_hits = (trades['outcome'] == 'tp_hit').sum()
    sl_hits = (trades['outcome'] == 'sl_hit').sum()
    timeouts = (trades['outcome'] == 'timeout').sum()
    
    # Average duration
    avg_duration = float(trades['duration_bars'].mean())
    
    return {
        'total_trades': int(total_trades),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(win_rate),
        'avg_pnl': float(avg_pnl),
        'median_pnl': float(median_pnl),
        'total_pnl': float(total_pnl),
        'total_pnl_pct': float(total_pnl * 100),
        'max_drawdown': float(max_drawdown),
        'sharpe_ratio': float(sharpe),
        'profit_factor': float(profit_factor),
        'tp_hits': int(tp_hits),
        'sl_hits': int(sl_hits),
        'timeouts': int(timeouts),
        'avg_duration_bars': float(avg_duration),
        'start_date': trades['entry_time'].min(),
        'end_date': trades['exit_time'].max()
    }

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.markdown("""
This backtester loads a trained Cascade model (.pt file) and backtests it on Gold futures (GC=F) 
using the **exact same trade generation logic** from training.

### How it works:
1. Upload your trained .pt model file
2. Select backtest date range
3. Configure trade parameters (SL/TP multipliers, confidence threshold)
4. Run backtest and analyze results
""")

st.sidebar.header("📁 Model Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Cascade Model (.pt file)",
    type=['pt'],
    help="Upload the .pt file exported from training"
)

st.sidebar.header("📅 Backtest Period")
start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.today() - timedelta(days=180)
)
end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.today()
)

interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1h", "15m", "5m"],
    index=0,
    help="Data timeframe (daily recommended for Gold futures)"
)

st.sidebar.header("⚙️ Trade Parameters")
k_sl = st.sidebar.number_input(
    "Stop Loss Multiplier (ATR)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="SL = entry ± (k_sl × ATR)"
)

k_tp = st.sidebar.number_input(
    "Take Profit Multiplier (ATR)",
    min_value=0.1,
    max_value=10.0,
    value=3.0,
    step=0.1,
    help="TP = entry ± (k_tp × ATR)"
)

atr_window = st.sidebar.number_input(
    "ATR Window",
    min_value=5,
    max_value=50,
    value=14,
    step=1
)

max_holding_bars = st.sidebar.number_input(
    "Max Holding Period (bars)",
    min_value=10,
    max_value=200,
    value=60,
    step=10
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (p3)",
    min_value=0.0,
    max_value=1.0,
    value=0.55,
    step=0.05,
    help="Minimum p3 probability to take trade"
)

st.sidebar.header("🎮 Actions")
run_backtest_btn = st.sidebar.button("🚀 Run Backtest", type="primary")

# Session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_components' not in st.session_state:
    st.session_state.model_components = None

# ============================================================================
# LOAD MODEL
# ============================================================================
if uploaded_file is not None and not st.session_state.model_loaded:
    try:
        with st.spinner("Loading model..."):
            # Save uploaded file temporarily
            temp_path = f"temp_model_{uuid.uuid4()}.pt"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load model
            model_components = load_cascade_model(temp_path, device="auto")
            
            # Clean up temp file
            os.remove(temp_path)
            
            st.session_state.model_components = model_components
            st.session_state.model_loaded = True
            
            st.success("✅ Model loaded successfully!")
            
            # Display model info
            st.subheader("Model Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sequence Length", model_components['seq_len'])
            with col2:
                st.metric("Feature Windows", str(model_components['feat_windows']))
            with col3:
                st.metric("Device", str(model_components['device']))
            
            metadata = model_components.get('metadata', {})
            if metadata:
                st.json(metadata)
    
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        logger.exception("Model loading error")

# ============================================================================
# RUN BACKTEST
# ============================================================================
if run_backtest_btn:
    if not st.session_state.model_loaded:
        st.error("⚠️ Please upload a model first!")
    else:
        try:
            with st.spinner("Running backtest..."):
                model_components = st.session_state.model_components
                
                # 1. Fetch data
                st.info("📊 Fetching Gold futures data from Yahoo...")
                
                if YahooTicker is None:
                    st.error("yahooquery not installed. Please install: pip install yahooquery")
                else:
                    ticker = YahooTicker("GC=F")
                    raw_data = ticker.history(
                        start=start_date.isoformat(),
                        end=end_date.isoformat(),
                        interval=interval
                    )
                    
                    if raw_data is None or (isinstance(raw_data, pd.DataFrame) and raw_data.empty):
                        st.error("No data returned from Yahoo. Try different dates or interval.")
                    else:
                        # Process data
                        if isinstance(raw_data.index, pd.MultiIndex):
                            bars = raw_data.reset_index(level=0, drop=True)
                        else:
                            bars = raw_data
                        
                        bars.index = pd.to_datetime(bars.index)
                        bars = bars.sort_index()
                        bars.columns = [c.lower() for c in bars.columns]
                        
                        if 'close' not in bars.columns and 'adjclose' in bars.columns:
                            bars['close'] = bars['adjclose']
                        
                        bars = ensure_unique_index(bars)
                        
                        st.success(f"✅ Fetched {len(bars)} bars")
                        
                        # 2. Generate candidates (LONG direction as per training)
                        st.info("🎯 Generating trade candidates...")
                        
                        candidates = generate_candidates_and_labels(
                            bars=bars,
                            lookback=model_components['seq_len'],
                            k_tp=k_tp,
                            k_sl=k_sl,
                            atr_window=atr_window,
                            max_bars=max_holding_bars,
                            direction="long"  # Training was on LONG
                        )
                        
                        if candidates.empty:
                            st.warning("No candidates generated. Try different parameters.")
                        else:
                            st.success(f"✅ Generated {len(candidates)} candidates")
                            
                            # 3. Map candidate times to bar indices
                            bar_idx_map = {t: i for i, t in enumerate(bars.index)}
                            candidate_indices = []
                            
                            for t in candidates['candidate_time']:
                                t0 = pd.Timestamp(t)
                                if t0 in bar_idx_map:
                                    candidate_indices.append(bar_idx_map[t0])
                                else:
                                    locs = bars.index[bars.index <= t0]
                                    if len(locs) > 0:
                                        candidate_indices.append(bar_idx_map[locs[-1]])
                                    else:
                                        candidate_indices.append(0)
                            
                            candidate_indices = np.array(candidate_indices, dtype=int)
                            
                            # 4. Run model predictions
                            st.info("🧠 Running model inference...")
                            
                            predictions = predict_on_candidates(
                                model_components=model_components,
                                bars=bars,
                                candidate_indices=candidate_indices
                            )
                            
                            st.success(f"✅ Generated {len(predictions)} predictions")
                            
                            # 5. Simulate backtest
                            st.info("📈 Simulating trades...")
                            
                            trades = simulate_backtest(
                                candidates=candidates.reset_index(),
                                predictions=predictions,
                                bars=bars,
                                confidence_threshold=confidence_threshold,
                                max_holding_bars=max_holding_bars
                            )
                            
                            if trades.empty:
                                st.warning(f"No trades met confidence threshold of {confidence_threshold:.2f}")
                            else:
                                # 6. Compute metrics
                                metrics = compute_backtest_metrics(trades)
                                
                                # Store results
                                st.session_state.backtest_results = {
                                    'trades': trades,
                                    'metrics': metrics,
                                    'candidates': candidates,
                                    'predictions': predictions,
                                    'bars': bars
                                }
                                
                                # Display results
                                st.success("✅ Backtest complete!")
                                
                                # === METRICS DASHBOARD ===
                                st.header("📊 Backtest Results")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Trades", metrics['total_trades'])
                                    st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
                                
                                with col2:
                                    st.metric("Total PnL", f"{metrics['total_pnl_pct']:.2f}%")
                                    st.metric("Avg PnL", f"{metrics['avg_pnl']*100:.2f}%")
                                
                                with col3:
                                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                                    st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                                
                                with col4:
                                    st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
                                    st.metric("Avg Duration", f"{metrics['avg_duration_bars']:.0f} bars")
                                
                                # Outcome breakdown
                                st.subheader("Outcome Breakdown")
                                outcome_col1, outcome_col2, outcome_col3 = st.columns(3)
                                
                                with outcome_col1:
                                    st.metric("TP Hits", metrics['tp_hits'])
                                with outcome_col2:
                                    st.metric("SL Hits", metrics['sl_hits'])
                                with outcome_col3:
                                    st.metric("Timeouts", metrics['timeouts'])
                                
                                # === EQUITY CURVE ===
                                st.subheader("📈 Equity Curve")
                                
                                fig_equity, ax_equity = plt.subplots(figsize=(12, 6))
                                cum_pnl = trades['pnl'].cumsum()
                                ax_equity.plot(trades['exit_time'], cum_pnl, linewidth=2, color='#1f77b4')
                                ax_equity.fill_between(trades['exit_time'], 0, cum_pnl, alpha=0.3, color='#1f77b4')
                                ax_equity.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                                ax_equity.set_xlabel('Date')
                                ax_equity.set_ylabel('Cumulative PnL')
                                ax_equity.set_title('Cumulative PnL Over Time')
                                ax_equity.grid(True, alpha=0.3)
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig_equity)
                                
                                # === PNL DISTRIBUTION ===
                                st.subheader("📊 PnL Distribution")
                                
                                fig_dist, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(12, 4))
                                
                                # Histogram
                                ax_hist.hist(trades['pnl_pct'], bins=30, edgecolor='black', alpha=0.7, color='#2ca02c')
                                ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
                                ax_hist.set_xlabel('PnL (%)')
                                ax_hist.set_ylabel('Frequency')
                                ax_hist.set_title('PnL Distribution')
                                ax_hist.legend()
                                ax_hist.grid(True, alpha=0.3)
                                
                                # Box plot
                                ax_box.boxplot([trades['pnl_pct']], vert=True, patch_artist=True)
                                ax_box.set_ylabel('PnL (%)')
                                ax_box.set_title('PnL Box Plot')
                                ax_box.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig_dist)
                                
                                # === TRADES TABLE ===
                                st.subheader("📋 Trade Log")
                                
                                # Display controls
                                show_all = st.checkbox("Show all trades", value=False)
                                
                                if show_all:
                                    st.dataframe(trades, use_container_width=True)
                                else:
                                    st.dataframe(trades.head(50), use_container_width=True)
                                    st.caption(f"Showing first 50 of {len(trades)} trades")
                                
                                # Download button
                                csv = trades.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Trade Log (CSV)",
                                    data=csv,
                                    file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                                # === MODEL PREDICTION ANALYSIS ===
                                st.subheader("🎯 Model Prediction Analysis")
                                
                                # Prediction vs Actual
                                pred_actual = trades[['predicted_prob', 'actual_label', 'pnl']].copy()
                                pred_actual['correct'] = ((pred_actual['predicted_prob'] >= 0.5) == (pred_actual['actual_label'] == 1))
                                
                                accuracy = pred_actual['correct'].mean()
                                st.metric("Prediction Accuracy", f"{accuracy:.1%}")
                                
                                # Probability bins
                                fig_prob, ax_prob = plt.subplots(figsize=(12, 5))
                                
                                bins = np.linspace(0, 1, 11)
                                bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
                                
                                pred_actual['prob_bin'] = pd.cut(pred_actual['predicted_prob'], bins=bins, labels=bin_labels)
                                
                                bin_stats = pred_actual.groupby('prob_bin').agg({
                                    'pnl': ['mean', 'count']
                                }).reset_index()
                                bin_stats.columns = ['prob_bin', 'avg_pnl', 'count']
                                
                                ax_prob.bar(bin_stats['prob_bin'], bin_stats['avg_pnl'], alpha=0.7, color='#ff7f0e')
                                ax_prob.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                                ax_prob.set_xlabel('Predicted Probability Bin')
                                ax_prob.set_ylabel('Average PnL')
                                ax_prob.set_title('Average PnL by Predicted Probability')
                                plt.xticks(rotation=45)
                                ax_prob.grid(True, alpha=0.3)
                                
                                # Annotate counts
                                for i, row in bin_stats.iterrows():
                                    ax_prob.text(i, row['avg_pnl'], f"n={int(row['count'])}", 
                                               ha='center', va='bottom', fontsize=8)
                                
                                plt.tight_layout()
                                st.pyplot(fig_prob)
        
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            logger.exception("Backtest error")
            st.exception(e)

# ============================================================================
# SIDEBAR: Model Info
# ============================================================================
if st.session_state.model_loaded:
    st.sidebar.success("✅ Model Loaded")
    
    if st.sidebar.button("ℹ️ View Model Details"):
        st.sidebar.json(st.session_state.model_components.get('metadata', {}))

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(f"""
**Cascade Model Backtester** | 
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
Data: Yahoo Finance (GC=F) | 
Model: Uploaded .pt file
""")
