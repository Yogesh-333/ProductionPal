import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'CSV_Fault_Data', '2_CSV_Data_Files')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'motor_health_model.pkl')
LABELMAP_PATH = os.path.join(BASE_DIR, 'models', 'label_map.pkl')

# Setup logging
log_path = os.path.join(BASE_DIR, 'productionpal_training.log')
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("ProductionPalModelTrain")

feature_names = ["Accelerometer 1 (m/s^2)", "Accelerometer 2 (m/s^2)", "Accelerometer 3 (m/s^2)"]
dfs = []
labels = []

for condition in ['1_Unloaded_Condition', '2_Loaded_Condition']:
    folder = os.path.join(DATA_DIR, condition)
    logger.info(f"Processing folder: {folder}")
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            fpath = os.path.join(folder, fname)
            try:
                df = pd.read_csv(fpath, usecols=feature_names)
                label = "_".join(fname.split('_')[0:2])  # e.g., H_H, R_U, F_B
                df['label'] = label
                dfs.append(df.iloc[::1000].reset_index(drop=True))  # Downsample
                logger.info(f"Loaded {fname} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Failed loading {fname}: {e}")

if not dfs:
    logger.error("No data files loaded. Check dataset folders.")
    raise RuntimeError("No data files loaded.")

data = pd.concat(dfs, ignore_index=True)
data['class'] = data['label'].astype('category').cat.codes

X = data[feature_names]
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=40, random_state=42)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
logger.info(f"Model trained with test accuracy: {score:.4f}")
print(f"Test accuracy: {score:.4f}")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
logger.info(f"Model saved to {MODEL_PATH}")

label_map = dict(enumerate(data['label'].astype('category').cat.categories))
joblib.dump(label_map, LABELMAP_PATH)
logger.info(f"Label map saved to {LABELMAP_PATH}")
