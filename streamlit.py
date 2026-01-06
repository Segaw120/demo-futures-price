from datetime import datetime
import streamlit as st
import pandas as pd

from module import (
    fetch_recent_daily_history,
    last_n_weekdays,
    fetch_snapshot,
    fetch_last_completed_close,
    build_today_estimate,
)

SYMBOL_DEFAULT = "GC=F"

st.set_page_config(page_title="Mini Gold Dashboard", layout="wide")
st.title("Mini Gold: 7 weekday bars + today's estimate")

if "logs" not in st.session_state:
    st.session_state["logs"] = []

# Sidebar
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Yahoo Symbol", SYMBOL_DEFAULT)
lookback_days = st.sidebar.number_input("Calendar lookback", 7, 30, 14)
n_weekdays = st.sidebar.number_input("Last N weekdays", 1, 10, 7)
include_today = st.sidebar.checkbox("Include Today's Estimate", value=True)

if st.button("Fetch latest"):
    try:
        df_raw = fetch_recent_daily_history(
            symbol, int(lookback_days), st.session_state
        )
        df_weekdays = last_n_weekdays(
            df_raw, int(n_weekdays), st.session_state
        )

        today = datetime.utcnow().date()
        
        # Only fetch/calculate estimate if the toggle is ON
        if include_today and today.weekday() < 5:
            if df_weekdays.empty or df_weekdays.index[-1] != today:
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

                df_display = pd.concat(
                    [df_weekdays, pd.DataFrame([est])],
                    axis=0,
                )
            else:
                df_display = df_weekdays.copy()
        else:
            # If toggle is OFF, exclude today if it exists in the history
            df_display = df_weekdays[df_weekdays.index != today].copy()

        if "is_estimated" not in df_display.columns:
            df_display["is_estimated"] = False

        st.subheader("Price Data")
        st.dataframe(
            df_display[
                ["open", "high", "low", "close", "volume", "is_estimated"]
            ]
        )

    except Exception as e:
        st.error(f"Fetch failed: {e}")

# Logs
st.sidebar.subheader("Logs")
if st.sidebar.button("Clear logs"):
    st.session_state["logs"] = []

st.sidebar.write(f"Entries: {len(st.session_state['logs'])}")
if st.session_state["logs"]:
    st.sidebar.json(st.session_state["logs"][-50:])

st.subheader("JSON Logs (last 50)")
if st.session_state["logs"]:
    st.json(st.session_state["logs"][-50:])
