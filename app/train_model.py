import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'CSV_Fault_Data', '2_CSV_Data_Files', '1_Unloaded_Condition')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'motor_health_model.pkl')

# Get a few healthy, misalignment, unbalance, and faulty examples
csvs = [
    ('H_H_1_0.csv', 0),   # Healthy
    ('R_M_1_0.csv', 1),   # Rotor misalignment
    ('R_U_1_0.csv', 2),   # Rotor unbalance
    ('F_B_1_0.csv', 3),   # Faulty bearing
]

dfs = []
for fname, label in csvs:
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)
    # Minimal cleaning: remove obviously non-sensor columns if present
    for drop_col in ['Microphone (V)', 'Temperature (Celsius)', 'Microphone (V)', 'Temperature (Celsius)']:
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
    df['class'] = label
    # Downsample: every 1000th row to keep memory lower
    dfs.append(df.iloc[::1000].reset_index(drop=True))

data = pd.concat(dfs, ignore_index=True)

FEATURES = [
    "Accelerometer 1 (m/s^2)", 
    "Accelerometer 2 (m/s^2)", 
    "Accelerometer 3 (m/s^2)",
]
X = data[FEATURES]
y = data['class']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=40, random_state=42)
clf.fit(X_train, y_train)
print("Test accuracy:", clf.score(X_test, y_test))

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print("Model saved to", MODEL_PATH)
