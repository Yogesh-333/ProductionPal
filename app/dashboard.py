import os
import pandas as pd
import joblib
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import logging
import datetime

# Setup logging: file and stdout (Docker logs)
log_path = os.path.join(os.path.dirname(__file__), '..', 'productionpal_dashboard.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProductionPalDashboard")

logger.info("Streamlit dashboard started.")

# Paths and features
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'motor_health_model.pkl')
LABELMAP_PATH = os.path.join(BASE_DIR, 'models', 'label_map.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'CSV_Fault_Data', 'lines_stream.csv')

FEATURES = ["Accelerometer 1 (m/s^2)", "Accelerometer 2 (m/s^2)", "Accelerometer 3 (m/s^2)"]

# Load label map (ensure model export and app use same Python & sklearn/joblib versions)
label_map = joblib.load(LABELMAP_PATH)

def friendly_name(label):
    if label.startswith('H_H'):
        return "Healthy"
    elif label.startswith('R_M'):
        return "Rot. Misalignment"
    elif label.startswith('R_U'):
        return "Rot. Unbalance"
    elif label.startswith('F_B'):
        return "Faulty Bearing"
    return None  # Only include mapped ones

# Build STATE_MAP excluding unknown statuses
STATE_MAP = {}
ATTENTION_CODES = []
for code, label in label_map.items():
    name = friendly_name(label)
    if name is None:
        continue
    icon, color, hint = "❓", "gray", "Unknown status"
    if name == "Healthy":
        icon, color, hint = "✅", "green", "All systems normal"
    elif name == "Rot. Misalignment":
        icon, color, hint = "↩️", "orange", "Check for alignment issues"
    elif name == "Rot. Unbalance":
        icon, color, hint = "📉", "red", "Unbalance detected, inspect soon"
        ATTENTION_CODES.append(code)
    elif name == "Faulty Bearing":
        icon, color, hint = "⚙️", "red", "Bearing fault, needs urgent attention"
        ATTENTION_CODES.append(code)
    STATE_MAP[code] = (name, icon, color, hint)

st.set_page_config("ProductionPal: Real-Time Motor Health", layout="wide")
st.title("⚡ ProductionPal Dashboard")
st_autorefresh(interval=2000, key="data-refresh")

model = joblib.load(MODEL_PATH)
logger.info("Loaded model and label map.")

if not os.path.exists(CSV_PATH):
    logger.error("Live data file not found.")
    st.warning("Live data file not found. Please start sensor_mocker.py!")
    st.stop()

df = pd.read_csv(CSV_PATH, usecols=lambda c: c in FEATURES + ['line_id'])
lines = [1, 2, 3, 4]

with st.expander("Status Legend", expanded=True):
    st.markdown("**Status Codes Explained:**")
    for _, (text, icon, color, hint) in STATE_MAP.items():
        st.markdown(f"<span style='color:{color}; font-size:20px'>{icon} <b>{text}:</b></span> {hint}", unsafe_allow_html=True)

cols = st.columns(4)

if "attention_list" not in st.session_state:
    st.session_state.attention_list = {}

for i, line_id in enumerate(lines):
    dline = df[df["line_id"] == line_id]
    if len(dline) > 0:
        latest = dline.iloc[-1][FEATURES]
        x = pd.DataFrame([latest])
        pred = model.predict(x)[0]
        text, icon, color, hint = STATE_MAP.get(pred, ("Unknown", "❓", "gray", "Unknown"))

        # Add or update attention list with timestamp
        if pred in ATTENTION_CODES:
            st.session_state.attention_list[line_id] = (text, icon, color, hint, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        with cols[i % 4]:
            st.markdown(f"### Line {line_id} <span style='font-size:24px'>{icon}</span>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:{color};padding:10px 6px;border-radius:8px;text-align:center;'><b>{text}</b></div>", unsafe_allow_html=True)
            st.caption(hint)
            st.line_chart(dline[FEATURES].iloc[-10:])
    else:
        with cols[i % 4]:
            st.markdown(f"### Line {line_id}")
            st.warning("No recent data available")

logger.info("Rendered dashboard lines status.")

st.markdown("## 🚨 **Needs Attention: Rot. Unbalance & Faulty Bearing**")

if st.session_state.attention_list:
    remove_keys = []
    for line_id, (text, icon, color, hint, alert_time) in st.session_state.attention_list.items():
        st.markdown(
            f"<div style='background-color:{color};padding:12px 10px;border-radius:12px;margin-bottom:6px;'>"
            f"<b>Line {line_id}</b> | <span style='font-size:28px'>{icon}</span> <b>{text}</b> <br>"
            f"<span style='font-size:14px;'>{hint} — <b>{alert_time}</b></span></div>",
            unsafe_allow_html=True
        )
        if st.button(f"Mark Line {line_id} Fixed", key=f"fix_{line_id}"):
            remove_keys.append(line_id)
    for k in remove_keys:
        del st.session_state.attention_list[k]
else:
    st.success("No Rot. Unbalance or Faulty Bearing detected.")

logger.info("Updated Needs Attention panel with persistence.")

st.info("Dashboard auto-updates every 2 seconds. Run sensor_mocker.py for live data.")
