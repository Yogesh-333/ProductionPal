import os
import subprocess
import logging
from flask import Flask, render_template, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Gateway")

@app.route('/')
def home():
    logger.info("API Call: GET / (Home Page) started")
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    logger.info("Triggering training run...")
    try:
        # Run training script as a subprocess
        # We assume run_all.py or Docker sets CWD correctly, but we'll use absolute path just in case
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(current_dir, 'train_model.py')
        root_dir = os.path.dirname(current_dir)
        
        # Determine command based on OS (python vs python3)
        # In this env it's 'python'
        result = subprocess.run(
            ['python', train_script], 
            cwd=root_dir, 
            capture_output=True, 
            text=True
        )

        if result.returncode == 0:
            # Parse output for Run ID if needed, or just return success
            output_lines = result.stdout.split('\n')
            run_id = "Unknown"
            for line in output_lines:
                if "MLflow Run ID:" in line:
                    run_id = line.split(":")[-1].strip()
            
            logger.info(f"Training successful. Run ID: {run_id}")
            return jsonify({"status": "success", "run_id": run_id})
        else:
            logger.error(f"Training failed: {result.stderr}")
            return jsonify({"status": "error", "error": result.stderr}), 500

    except Exception as e:
        logger.error(f"Exception during training: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        from flask import request
        import joblib
        import pandas as pd
        import numpy as np

        logger.info("API Call: POST /predict started")
        
        # Load Model and Label Map
        # Note: In a real production setting, we'd load this once globally or use MLflow Model Serving
        # For this assignment, lazy loading ensures we get the latest trained model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        model_path = os.path.join(base_dir, 'models', 'motor_health_model.pkl')
        label_map_path = os.path.join(base_dir, 'models', 'label_map.pkl')

        if not os.path.exists(model_path):
            return jsonify({"status": "error", "error": "Model not trained yet."}), 404

        model = joblib.load(model_path)
        label_map = joblib.load(label_map_path)

        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"status": "error", "error": "Missing 'features' in request body"}), 400
        
        # Expecting features as list of lists or dict
        features = data['features']
        # Convert to DataFrame if needed or list
        # We assume input is matching the training feature set: [Acc1, Acc2, Acc3]
        prediction = model.predict([features])
        pred_code = prediction[0]
        
        # Map code to label
        # label_map is {0: 'H_H', 1: 'F_B', ...}
        pred_label = label_map.get(pred_code, f"Unknown Code {pred_code}")

        logger.info(f"Prediction: {pred_label} (Code: {pred_code})")
        return jsonify({
            "status": "success",
            "prediction_code": int(pred_code),
            "prediction_label": pred_label
        })

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    # Hosted on 0.0.0.0 to be accessible from outside container
    app.run(host='0.0.0.0', port=8000)
