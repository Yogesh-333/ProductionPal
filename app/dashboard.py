import os
import pandas as pd
import joblib
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import logging

# Set up logging
log_path = os.path.join(os.path.dirname(__file__), '..', 'productionpal_dashboard.log')
logger = logging.getLogger("ProductionPalDashboard")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Also log to console (stdout)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# LOG: Dashboard start
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

st.set_page_config("ProductionPal: Real-Time Motor Health", layout="wide")
st.title("⚡ ProductionPal Multi-Line Health Dashboard")
st_autorefresh(interval=2000, key="data-refresh")

# Load model
model = joblib.load(MODEL_PATH)
logger.info("Loaded health model for predictions.")

# Check live data file
if not os.path.exists(CSV_PATH):
    logger.error("Live data file not found.")
    st.warning("Live data file not found. Please start the sensor_mocker.py script!")
    st.stop()

# Read sensor data, filter columns needed
required_cols = FEATURES + ['line_id']
df = pd.read_csv(CSV_PATH, usecols=lambda c: c in required_cols)
lines = [1, 2, 3, 4]

# Show status legend expanding area
with st.expander("Status Legend", expanded=True):
    st.markdown("**Status Codes Explained:**")
    for _, (text, icon, color, hint) in STATE_MAP.items():
        st.markdown(f"<span style='color:{color}; font-size:20px'>{icon} <b>{text}:</b></span> {hint}", unsafe_allow_html=True)

# Prepare display cards for each line
needs_attention = {}
cols = st.columns(4)

for i, line_id in enumerate(lines):
    dline = df[df["line_id"] == line_id]
    if len(dline) > 0:
        latest = dline.iloc[-1][FEATURES]
        x = pd.DataFrame([latest])
        pred = model.predict(x)[0]
        pred_text, pred_icon, pred_color, pred_hint = STATE_MAP[pred]
        if pred != 0:
            needs_attention[line_id] = (pred_text, pred_icon, pred_color, pred_hint)
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

logger.info("Checked status and rendered dashboard tiles.")

# Persistent Needs Attention panel
st.markdown("## 🚨 **Needs Attention**")
shown_attention = False
for line_id in lines:
    if line_id in needs_attention:
        text, icon, color, hint = needs_attention[line_id]
        st.markdown(
            f"<div style='background-color:{color};padding:12px 10px;border-radius:12px;margin-bottom:6px;'>"
            f"Line <b>{line_id}</b>: <span style='font-size:28px'>{icon}</span> <b>{text}</b> &mdash; {hint}</div>",
            unsafe_allow_html=True
        )
        shown_attention = True
    else:
        st.markdown(
            f"<div style='background-color:#2ecc40;padding:10px 10px;border-radius:10px;margin-bottom:6px;'>"
            f"Line <b>{line_id}</b>: <span style='font-size:24px'>✅</span> <b>All Good</b></div>",
            unsafe_allow_html=True
        )

# One last info log
logger.info("Dashboard attention area updated. Lines needing attention: %s", list(needs_attention.keys()))

st.info("Dashboard auto-updates every 2 seconds. Run sensor_mocker.py for live data.")
