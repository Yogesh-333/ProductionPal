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