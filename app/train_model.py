import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq 
import joblib
import logging

# --- SECRETS & CONFIGURATION ---
# We use os.getenv to read variables injected by Docker
DB_USER = os.getenv("DB_USERNAME", "default_user")
DB_PASS = os.getenv("DB_PASSWORD", "default_pass")
DB_HOST = os.getenv("DB_HOSTNAME", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

EXP_NAME = os.getenv("EXPERIMENT_NAME", "ProductionPal_Default")
EXP_VERSION = os.getenv("EXPERIMENT_VERSION", "1.0.0")

# Hyperparameters
N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", 40))
EXPECTED_ACCURACY = float(os.getenv("EXPECTED_ACCURACY", 0.85))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 1))

# Feature Names (Comma separated string in Env Var)
DEFAULT_FEATS = "Accelerometer 1 (m/s^2),Accelerometer 2 (m/s^2),Accelerometer 3 (m/s^2)"
FEATURE_LIST = os.getenv("FEATURE_NAMES", DEFAULT_FEATS).split(',')
# -------------------------------

# --- 1. CONFIGURATION AND PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'CSV_Fault_Data', '2_CSV_Data_Files')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'motor_health_model.pkl')
LABELMAP_PATH = os.path.join(BASE_DIR, 'models', 'label_map.pkl')

# Sensor columns (Indices 0-4)
SENSOR_COLS = {
    0: "Acc1_Vibration", 
    1: "Mic_Acoustic", 
    2: "Acc2_Vibration", 
    3: "Acc3_Vibration", 
    4: "Temp_Integrated"
}
NUM_SENSORS = len(SENSOR_COLS)
FS = 42000 
N_SAMPLES_EXPECTED = 420000 

# CSV Indices to Load: 0, 1, 2, 3, 4
CSV_INDICES_TO_LOAD = list(range(NUM_SENSORS)) 

# Setup logging
log_path = os.path.join(BASE_DIR, 'productionpal_training.log')
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProductionPalModelTrain")

# Log Configuration (Secrets Management Verification)
logger.info(f"--- Experiment: {EXP_NAME} v{EXP_VERSION} ---")
logger.info(f"DB Configuration: User={DB_USER}, Host={DB_HOST}:{DB_PORT}")
logger.info(f"Training Config: n_estimators={N_ESTIMATORS}, epochs={NUM_EPOCHS}")

dfs = []
feature_names = [f.strip() for f in FEATURE_LIST]

# Data Loading Loop
for condition in ['1_Unloaded_Condition', '2_Loaded_Condition']:
    folder = os.path.join(DATA_DIR, condition)
    if not os.path.exists(folder):
        continue
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            fpath = os.path.join(folder, fname)
            try:
                # Optimized loading: usecols + skiprows
                df = pd.read_csv(fpath, usecols=[0, 1, 2], names=feature_names, header=None, skiprows=1)
                label = "_".join(fname.split('_')[0:2])
                df['label'] = label
                dfs.append(df.iloc[::1000].reset_index(drop=True))
            except Exception as e:
                logger.error(f"Skipping {fname}: {e}")

if not dfs:
    logger.error("No data loaded. Creating dummy model for Docker build process.")
    # Create dummy data so build doesn't fail if data folder is empty
    X = pd.DataFrame(np.random.rand(100, 3), columns=feature_names)
    y = np.random.randint(0, 2, 100)
    label_map = {0: "H_H", 1: "F_B"}
else:
    data = pd.concat(dfs, ignore_index=True)
    data['class'] = data['label'].astype('category').cat.codes
    X = data[feature_names]
    y = data['class']
    label_map = dict(enumerate(data['label'].astype('category').cat.categories))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
joblib.dump(label_map, LABELMAP_PATH)

logger.info(f"Model saved to {MODEL_PATH}")
logger.info(f"Training Accuracy: {score:.4f} (Target: {EXPECTED_ACCURACY})")
print(f"Training Accuracy: {score:.4f}")