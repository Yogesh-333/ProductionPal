import os
import pandas as pd
import joblib
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import logging
import datetime

# Set up logging
log_path = os.path.join(os.path.dirname(__file__), '..', 'productionpal_dashboard.log')
logger = logging.getLogger("ProductionPalDashboard")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)

logger.info("Streamlit dashboard started.")

# Paths and features
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'motor_health_model.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'CSV_Fault_Data', 'lines_stream.csv')
FEATURES = ["Accelerometer 1 (m/s^2)", "Accelerometer 2 (m/s^2)", "Accelerometer 3 (m/s^2)"]

STATE_MAP = {
    0: ("Healthy", "✅", "green", "All systems normal"),
    1: ("Rot. Misalignment", "↩️", "orange", "Check for alignment issues"),
    2: ("Rot. Unbalance", "📉", "red", "Unbalance detected, inspect soon"),
    3: ("Faulty Bearing", "⚙️", "red", "Bearing fault, needs urgent attention")
}

# For attention: only these codes matter
ATTENTION_CODES = [2, 3]

st.set_page_config("ProductionPal: Real-Time Motor Health", layout="wide")
st.title("⚡ ProductionPal Dashboard")
st_autorefresh(interval=2000, key="data-refresh")

model = joblib.load(MODEL_PATH)
logger.info("Loaded health model for predictions.")

if not os.path.exists(CSV_PATH):
    logger.error("Live data file not found.")
    st.warning("Live data file not found. Please start the sensor_mocker.py script!")
    st.stop()

# Read sensor data, filter columns needed
required_cols = FEATURES + ['line_id']
df = pd.read_csv(CSV_PATH, usecols=lambda c: c in required_cols)
lines = [1, 2, 3, 4]

# Status legend
with st.expander("Status Legend", expanded=True):
    st.markdown("**Status Codes Explained:**")
    for _, (text, icon, color, hint) in STATE_MAP.items():
        st.markdown(f"<span style='color:{color}; font-size:20px'>{icon} <b>{text}:</b></span> {hint}", unsafe_allow_html=True)

# Prepare the dashboard cards
cols = st.columns(4)
needs_attention = []

for i, line_id in enumerate(lines):
    dline = df[df["line_id"] == line_id]
    if len(dline) > 0:
        latest = dline.iloc[-1][FEATURES]
        x = pd.DataFrame([latest])
        pred = model.predict(x)[0]
        pred_text, pred_icon, pred_color, pred_hint = STATE_MAP[pred]
        if pred in ATTENTION_CODES:
            alert_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            needs_attention.append((line_id, pred_text, pred_icon, pred_color, pred_hint, alert_time))
        with cols[i % 4]:
            st.markdown(f"### Line {line_id} <span style='font-size:24px'>{pred_icon}</span>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background-color:{pred_color};padding:10px 6px;border-radius:8px;text-align:center;'>"
                f"<b>{pred_text}</b></div>", unsafe_allow_html=True)
            st.caption(pred_hint)
            st.line_chart(dline[FEATURES].iloc[-10:])
    else:
        with cols[i % 4]:
            st.markdown(f"### Line {line_id}")
            st.warning("No recent data available")

logger.info("Checked current statuses for all lines.")

# Needs Attention area (only Rot. Unbalance and Faulty Bearing)
st.markdown("## 🚨 **Needs Attention: Unbalance or Faulty Bearing Only**")
if needs_attention:
    for line_id, text, icon, color, hint, alert_time in needs_attention:
        st.markdown(
            f"<div style='background-color:{color};padding:12px 10px;border-radius:12px;margin-bottom:6px;'>"
            f"<b>Line {line_id}</b> | <span style='font-size:28px'>{icon}</span> <b>{text}</b> <br>"
            f"<span style='font-size:14px;'>{hint} — <b>{alert_time}</b></span></div>",
            unsafe_allow_html=True
        )
else:
    st.success("No Rot. Unbalance or Faulty Bearing detected on any line.")

logger.info("Updated Needs Attention for attention codes [2, 3].")

st.info("Dashboard auto-updates every 2 seconds. Run sensor_mocker.py for live data.")
