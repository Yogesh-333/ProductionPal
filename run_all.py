import subprocess
import time
import os

def run_app():
    print("--- 🚀 Starting ProductionPal Container Services ---")
    
    # 1. Start Sensor Mocker in the background
    mocker_process = subprocess.Popen(["python", "sensor_mocker.py"])
    print(f"✅ Sensor Mocker started (PID: {mocker_process.pid})")

    # 2. Start Streamlit Dashboard
    print("✅ Starting Streamlit Dashboard...")
    dashboard_cmd = ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
    
    try:
        subprocess.run(dashboard_cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping services...")
        mocker_process.terminate()
        
if __name__ == "__main__":
    run_app()