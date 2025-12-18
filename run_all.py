import subprocess
import time
import os
import signal
import sys

def run_app():
    print("--- 🚀 Starting ProductionPal Integrated Services ---")
    
    processes = []

    # 1. Start Sensor Mocker
    print("✅ Starting Sensor Mocker...")
    p_mocker = subprocess.Popen(["python", "sensor_mocker.py"])
    processes.append(p_mocker)

    # 2. Start MLflow UI
    # Using --host 0.0.0.0 to ensure Docker accessibility
    print("✅ Starting MLflow UI...")
    p_mlflow = subprocess.Popen(
        ["mlflow", "ui", "--backend-store-uri", "mlruns", "--host", "0.0.0.0", "--port", "5000"]
    )
    processes.append(p_mlflow)

    # 3. Start Streamlit Dashboard
    print("✅ Starting Streamlit Dashboard...")
    p_streamlit = subprocess.Popen(
        ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
    )
    processes.append(p_streamlit)

    # 4. Start Gateway App (Flask)
    print("✅ Starting Gateway App...")
    p_gateway = subprocess.Popen(["python", "app/gateway.py"])
    processes.append(p_gateway)

    print("\n🎉 All services are running!")
    print("   👉 Gateway (Landing Page): http://localhost:8000")
    print("   👉 Dashboard:             http://localhost:8501")
    print("   👉 MLflow UI:             http://localhost:5000")
    print("   (Press Ctrl+C to stop)")

    try:
        # Wait for any process to exit (or indefinite wait)
        p_gateway.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        for p in processes:
            p.terminate()
        sys.exit(0)
        
if __name__ == "__main__":
    run_app()