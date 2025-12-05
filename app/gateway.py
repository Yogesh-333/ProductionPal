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

if __name__ == '__main__':
    # Hosted on 0.0.0.0 to be accessible from outside container
    app.run(host='0.0.0.0', port=8000)
