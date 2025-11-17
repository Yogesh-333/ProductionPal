import subprocess
import threading

def run_sensor_mocker():
    subprocess.run(["python", "sensor_mocker.py"])

def run_dashboard():
    subprocess.run(["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"])

if __name__ == "__main__":
    t1 = threading.Thread(target=run_sensor_mocker)
    t2 = threading.Thread(target=run_dashboard)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
