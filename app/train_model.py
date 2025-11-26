import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq 
import joblib
import logging
import mlflow
import mlflow.sklearn

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

# MLflow Experiment Setup 
mlflow.set_tracking_uri("http://localhost:5000") 
mlflow.set_experiment("ProductionPal_MotorHealth")

# --- 2. FEATURE ENGINEERING FUNCTIONS ---

def extract_time_features(data):
    return {
        'mean': data.mean(), 
        'std': data.std(), 
        'rms': np.sqrt(np.mean(data**2)),
        'peak_to_peak': data.max() - data.min(), 
        'skewness': skew(data), 
        'kurtosis': kurtosis(data)
    }

def extract_frequency_features(data, fs=FS, n_samples=N_SAMPLES_EXPECTED):
    yf = rfft(data.values)
    yf_magnitude = np.abs(yf)
    xf = rfftfreq(n_samples, 1/fs)
    # Start from index 1 to ignore DC component
    peak_index = np.argmax(yf_magnitude[1:]) + 1 
    peak_magnitude = yf_magnitude[peak_index]
    peak_frequency = xf[peak_index]
    
    return {'fft_peak_mag': peak_magnitude, 'fft_peak_freq': peak_frequency}

def feature_engineer_file(df, fname):
    file_features = {}
    current_n_samples = df.shape[0]
    
    for col_index, col_name in SENSOR_COLS.items(): 
        series = df.iloc[:, col_index]
        
        time_feats = extract_time_features(series)
        for key, val in time_feats.items():
            file_features[f'{col_name}_{key}'] = val
            
        if 'Vibration' in col_name or 'Acoustic' in col_name:
            freq_feats = extract_frequency_features(series, n_samples=current_n_samples)
            for key, val in freq_feats.items():
                file_features[f'{col_name}_{key}'] = val
        
        if 'Temp' in col_name:
             file_features[f'{col_name}_max'] = series.max()
             
    return file_features

# --- 3. DATA LOADING, CLEANING, AND AGGREGATION ---

def load_and_engineer_data():
    all_rows = []
    
    print("\n--- DEBUG: Checking Data Paths ---")
    print(f"Project Base Directory: {BASE_DIR}")
    print(f"Data Source Directory: {DATA_DIR}")
    print("---------------------------------")
    
    DTYPE_MAP = {i: np.float32 for i in CSV_INDICES_TO_LOAD} 
    
    if not os.path.exists(DATA_DIR):
        logger.error(f"FATAL: DATA_DIR not found at {DATA_DIR}")
        return pd.DataFrame() 

    for condition in ['1_Unloaded_Condition', '2_Loaded_Condition']:
        folder = os.path.join(DATA_DIR, condition)
        logger.info(f"Processing folder: {folder}")
        print(f"Checking condition folder: {folder}")

        if not os.path.exists(folder):
            logger.warning(f"Condition folder not found: {folder}. Skipping.")
            continue
            
        for fname in os.listdir(folder):
            if fname.endswith('.csv'):
                fpath = os.path.join(folder, fname)
                try:
                    # 1. Load Data
                    df = pd.read_csv(fpath, 
                                     header=None,           
                                     skiprows=1,            
                                     usecols=CSV_INDICES_TO_LOAD,
                                     dtype=DTYPE_MAP,
                                     engine='c') 
                    
                    # 2. Aggressive Cleaning
                    df = df.dropna()
                    
                    # 4. Feature Engineering
                    features = feature_engineer_file(df, fname)
                    
                    # 5. Label Extraction (The Fix: Split by underscore)
                    parts = fname.split('_')
                    # e.g., B_R_1_0.csv -> parts[0]=B, parts[1]=R
                    health_label = f"{parts[0]}_{parts[1]}"
                    
                    features['label'] = health_label
                    all_rows.append(features)
                    
                    logger.info(f"SUCCESS: Engineered features for {fname} (Label: {health_label})")
                    print(f"Processed: {fname} (Samples: {df.shape[0]})")
                
                except Exception as e:
                    logger.error(f"FAILED processing {fname}: {e}")
                    print(f"CRASH: Failed to process {fname} due to error: {e}")

    if not all_rows:
        logger.error("No data files successfully processed.")

    return pd.DataFrame(all_rows)

# --- 4. MODEL TRAINING AND MLFLOW LOGGING ---

if __name__ == "__main__":
    
    N_ESTIMATORS = 80
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    logger.info("--- Starting Feature Engineering and Training Pipeline ---")
    
    try:
        data = load_and_engineer_data()
        
        if data.empty:
            print("\nFATAL ERROR: Data loading failed. Check logs and printed paths.")
            logger.error("Training halted because feature DataFrame is empty.")
            exit()
            
        logger.info(f"Final feature DataFrame shape: {data.shape}")
        
    except RuntimeError as e:
        logger.error(f"Pipeline stopped: {e}")
        exit()

    # --- Data Preparation for Model ---
    data['class'] = data['label'].astype('category').cat.codes
    feature_cols = [col for col in data.columns if col not in ['label', 'class']]
    
    X = data[feature_cols]
    y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=TEST_SIZE,
                                                        stratify=y, 
                                                        random_state=RANDOM_STATE)
    
    label_map = dict(enumerate(data['label'].astype('category').cat.categories))
    
    # --- MLFLOW RUN ---
    with mlflow.start_run(run_name="RF_Aggregated_Features_V1"):
        
        # Log Hyperparameters
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("test_split_ratio", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        
        # Train Model
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        
        # Evaluate Model
        score = clf.score(X_test, y_test)
        
        # Log Metrics
        mlflow.log_metric("test_accuracy", score)
        
        # Save and Log Artifacts
        mlflow.sklearn.log_model(clf, "model")
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(label_map, LABELMAP_PATH)

        mlflow.log_artifact(LABELMAP_PATH, "label_map_artifact")
        
        logger.info(f"Model trained with test accuracy: {score:.4f}")
        print(f"Test accuracy: {score:.4f}")

    logger.info("--- Training Pipeline Finished ---")