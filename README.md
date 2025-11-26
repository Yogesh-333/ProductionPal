# ⚡ ProductionPal: Electric Motor Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Managed-blue?style=for-the-badge&logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**ProductionPal** is an end-to-end Machine Learning Operations (MLOps) project designed to predict the health status of electric motors. Using vibration, acoustic, and temperature sensor data, the system classifies motors into various health states (Healthy, Bearing Faults, Rotor Faults, etc.) to enable predictive maintenance.

## 🚀 Project Overview

This project implements a complete data pipeline:

1.  **Data Ingestion:** Processing high-frequency time-series data (42 kHz) from the **UOEMD-VAFCVS** dataset.
2.  **Feature Engineering:** Extracting robust features from both Time Domain (RMS, Kurtosis) and Frequency Domain (FFT Peak & Frequency).
3.  **Model Training:** Training a **Random Forest Classifier** to categorize motor health states.
4.  **MLOps Integration:** Using **MLflow** for experiment tracking, metric logging, and model versioning.
5.  **Real-Time Simulation:** A Streamlit dashboard (planned) to visualize motor health in real-time.

## 📂 Repository Structure

```text
ProductionPal/
├── app/
│   ├── train_model.py          # Main training pipeline with MLflow & Feature Engineering
│   ├── dashboard.py            # (Planned) Streamlit Real-time Dashboard
│   └── sensor_mocker.py        # (Planned) Simulates live sensor data stream
├── data/
│   └── CSV_Fault_Data/         # Raw UOEMD-VAFCVS Dataset
├── models/
│   ├── motor_health_model.pkl  # Trained Random Forest Model
│   └── label_map.pkl           # Mapping of Class IDs to Health Labels (e.g., H_H)
├── mlruns/                     # MLflow tracking data (Local)
├── mlruns.db                   # MLflow SQLite backend
├── requirements.txt            # Project dependencies
└── README.md                   # Project Documentation

```
## 🛠️ Tech Stack

- **Language:** Python 3.9+  
- **Data Processing:** Pandas, NumPy, SciPy (Signal Processing / FFT)  
- **Machine Learning:** Scikit-learn (Random Forest)  
- **MLOps:** MLflow (Experiment Tracking & Registry)  
- **Version Control:** Git & Git LFS (Large File Storage for models)  
- **Visualization:** Streamlit (Dashboard)

---

## 📊 Dataset

The project uses the **University of Ottawa Electric Motor Dataset (UOEMD-VAFCVS)**.

- **Sensors:** 3 Accelerometers, 1 Microphone, 1 Temperature Sensor  
- **Sampling Rate:** 42,000 Hz  
- **Conditions:** Variable speeds and loads  
- **Labels:** Encoded in filenames (e.g., `B_R_1_0.csv` = Bowed Rotor)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/ProductionPal.git
cd ProductionPal
```

### 2. Set up Virtual Environment

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate

```

## ✅ **3. Install Dependencies**

```bash
pip install pandas numpy scikit-learn joblib scipy mlflow streamlit


```

## ✅ **4. Configure Git LFS (Important)**


Since the trained model files are large, this project uses Git Large File Storage.

```bash
git lfs install
git lfs pull


```

##  🏃‍♂️  **How to Run the Pipeline**

### ✅ Step 1: Start the MLflow Server

We use a local MLflow server with a SQLite backend to track all experiments.

```bash
mlflow server \
    --backend-store-uri sqlite:///mlruns.db \
    --default-artifact-root ./mlruns_artifacts \
    --host 0.0.0.0 \
    --port 5000


```

## ✅ **Step 2: Train the Model**


```bash
python app/train_model.py


```

## ✅ **Step 3: Run the Dashboard (Coming Soon)**


```bash
streamlit run app/dashboard.py


```

##  🧠  **Feature Engineering**


To handle high-dimensional raw data (~420k samples per file), raw signals are aggregated into a compact feature vector.

| Domain | Features | Purpose |
|--------|----------|---------|
| Time Domain | Mean, Std Dev, RMS, Peak-to-Peak, Skewness, Kurtosis | Captures signal intensity and distribution shape |
| Frequency Domain | FFT Peak Magnitude, FFT Peak Frequency | Identifies dominant fault frequencies |

## 📈 MLOps Workflow

- Every training run is automatically logged to MLflow  
- **Parameters Logged:** `n_estimators`, `test_split_ratio`, `random_state`  
- **Metrics Logged:** `test_accuracy`  
- **Artifacts:** Trained model + label mapping dictionary  
- Full experiment reproducibility is guaranteed
## 🤝 Contributing

1. Fork the project  
2. Create your feature branch  
   ```bash
   git checkout -b feature/NewFeature
    ```
    
## 📄 License

Distributed under the **MIT License**.  
See `LICENSE` for more information.
