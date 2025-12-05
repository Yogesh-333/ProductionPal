# ProductionPal API Documentation

## Overview
This document outlines the API endpoints and configuration parameters for the **ProductionPal** MLOps system. The system consists of a Gateway Service, a Streamlit Dashboard, and an MLflow Tracking Server.

## 🔗 Services

| Service | Port | Description |
| :--- | :--- | :--- |
| **Gateway App** | `8000` | Central entry point and API for triggering training. |
| **Dashboard** | `8501` | Real-time monitoring and visualization. |
| **MLflow UI** | `5000` | Experiment tracking and model registry. |

---

## 📡 Gateway API
**Base URL:** `http://localhost:8000`

### 1. Trigger Model Training
Triggers a new execution of the ML training pipeline. The pipeline loads data, preprocesses it, trains a Random Forest model, and logs everything to MLflow.

- **Endpoint:** `/train`
- **Method:** `POST`
- **Content-Type:** `application/json` (No body required)

**Response (Success - 200 OK):**
```json
{
  "status": "success",
  "run_id": "764a8523..."
}
```

**Response (Error - 500 Internal Server Error):**
```json
{
  "status": "error",
  "error": "Error details..."
}
```

### 2. Model Serving (Prediction)
Serves predictions using the latest trained model.

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Content-Type:** `application/json`

**Request Body:**
```json
{
  "features": [0.02, 0.05, 0.01]
}
```
*(Order: Accelerometer 1, Accelerometer 2, Accelerometer 3)*

**Response:**
```json
{
  "status": "success",
  "prediction_code": 0,
  "prediction_label": "H_H"
}
```

### 3. Landing Page
Renders the HTML landing page with links to all services.

- **Endpoint:** `/`
- **Method:** `GET`

---

## ⚙️ Configuration (Environment Variables)
The training pipeline is configured via environment variables. These can be passed to the Docker container.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `EXPERIMENT_NAME` | `ProductionPal_Default` | Name of the MLflow experiment. |
| `EXPERIMENT_VERSION` | `1.0.0` | Version tag for the run. |
| `RF_N_ESTIMATORS` | `40` | Number of trees in the Random Forest. |
| `EXPECTED_ACCURACY` | `0.85` | Target accuracy for logging purposes. |
| `NUM_EPOCHS` | `1` | Number of training epochs (simulated). |
| `FEATURE_NAMES` | *(See Code)* | Comma-separated list of sensor feature names. |
| `DB_USERNAME` | `default_user` | Database username (if applicable). |
| `DB_PASSWORD` | `default_pass` | Database password. |
| `DB_HOSTNAME` | `localhost` | Database host address. |
| `DB_PORT` | `5432` | Database port. |

## 🚀 Usage Example
To run the container with custom settings:

```bash
docker run -p 8501:8501 -p 5000:5000 -p 8000:8000 \
    -e RF_N_ESTIMATORS="100" \
    -e EXPECTED_ACCURACY="0.95" \
    yogeshkumar333/productionpal:assignment
```
