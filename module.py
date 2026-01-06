import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from yahooquery import Ticker
except Exception:
    Ticker = None

# -------------------------
# JSON-structured logging (shared)
# -------------------------
LOGGER_NAME = "mini_gold_dashboard"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

def st_json_log(level: str, action: str, details: dict, session_state=None):
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "action": action,
        "details": details,
    }

    if session_state is not None:
        if "logs" not in session_state:
            session_state["logs"] = []
        session_state["logs"].append(entry)

    try:
        logger.info(json.dumps(entry))
    except Exception:
        logger.info(entry)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# -------------------------
# Helpers
# -------------------------
def _ensure_ticker():
    if Ticker is None:
        raise RuntimeError("yahooquery is not installed")

# -------------------------
# Fetching functions
# -------------------------
def fetch_recent_daily_history(symbol: str, lookback_days: int, session_state=None) -> pd.DataFrame:
    _ensure_ticker()
    t = Ticker(symbol)

    st_json_log(
        "info",
        "fetch_recent_daily_history.start",
        {"symbol": symbol, "lookback_days": lookback_days},
        session_state,
    )

    raw = t.history(period=f"{lookback_days}d", interval="1d")

    if raw is None or raw.empty:
        st_json_log("warn", "fetch_recent_daily_history.empty", {"symbol": symbol}, session_state)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=0, drop=True)

    raw.index = [pd.to_datetime(x).replace(tzinfo=None).date() for x in raw.index]
    raw.index.name = "date"

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in raw.columns:
            raw[c] = np.nan

    df = raw[["open", "high", "low", "close", "volume"]].copy()

    st_json_log(
        "info",
        "fetch_recent_daily_history.done",
        {"rows_returned": len(df)},
        session_state,
    )
    return df


def last_n_weekdays(df: pd.DataFrame, n: int, session_state=None) -> pd.DataFrame:
    if df.empty:
        return df

    df2 = df[df.index.to_series().apply(lambda d: pd.Timestamp(d).weekday() < 5)]
    out = df2.tail(n).copy()

    st_json_log(
        "info",
        "last_n_weekdays.selected",
        {"requested": n, "selected": len(out)},
        session_state,
    )
    return out


def fetch_snapshot(symbol: str, session_state=None) -> dict:
    _ensure_ticker()
    t = Ticker(symbol)

    snap = t.price.get(symbol, {}) or {}

    result = {
        "price": snap.get("regularMarketPrice"),
        "high": snap.get("regularMarketDayHigh"),
        "low": snap.get("regularMarketDayLow"),
        "volume": snap.get("regularMarketVolume"),
    }

    st_json_log(
        "info",
        "fetch_snapshot.done",
        {"snapshot_keys": list(result.keys()), "price": result.get("price")},
        session_state,
    )
    return result


def fetch_last_completed_close(symbol: str, session_state=None) -> float:
    _ensure_ticker()
    t = Ticker(symbol)

    hist = t.history(period="7d", interval="1d")

    if hist is None or hist.empty:
        raise RuntimeError("No history returned to determine last completed close")

    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.reset_index(level=0, drop=True)

    hist.index = [pd.to_datetime(x).replace(tzinfo=None).date() for x in hist.index]
    hist_valid = hist[hist["close"].notna()]

    if hist_valid.empty:
        raise RuntimeError("No valid completed close in history")

    last_close = float(hist_valid["close"].iloc[-1])

    st_json_log(
        "info",
        "fetch_last_completed_close.done",
        {"last_close": last_close},
        session_state,
    )
    return last_close


def build_today_estimate(yesterday_close: float, snapshot: dict, session_state=None) -> pd.Series:
    row = {
        "open": float(yesterday_close) if yesterday_close is not None else np.nan,
        "high": snapshot.get("high"),
        "low": snapshot.get("low"),
        "close": snapshot.get("price"),
        "volume": snapshot.get("volume"),
        "is_estimated": True,
    }

    st_json_log(
        "info",
        "build_today_estimate",
        {"open": row["open"], "close": row["close"]},
        session_state,
    )

    return pd.Series(row)
