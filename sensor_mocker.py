import pandas as pd
import numpy as np
import os
import time
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'CSV_Fault_Data', 'lines_stream.csv')
HEALTHY_PATH = os.path.join(BASE_DIR, 'data', 'CSV_Fault_Data', '2_CSV_Data_Files', '1_Unloaded_Condition', 'H_H_1_0.csv')

# Set up logging
log_path = os.path.join(BASE_DIR, 'productionpal_sensor.log')
logger = logging.getLogger("ProductionPalSensorMocker")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

FEATURES = ["Accelerometer 1 (m/s^2)", "Accelerometer 2 (m/s^2)", "Accelerometer 3 (m/s^2)"]
COLS = FEATURES + ['line_id']

healthy_df = pd.read_csv(HEALTHY_PATH, usecols=FEATURES)
healthy_df = healthy_df.iloc[::1000].reset_index(drop=True)

logger.info("Sensor mocker started.")

while True:
    rows = []
    for _ in range(20):
        record = healthy_df.sample(1).iloc[0].to_dict()
        record['line_id'] = np.random.randint(1, 5)
        rows.append(record)

    fault_lines = np.random.choice([1, 2, 3, 4], size=1, replace=False)
    for line_id in range(1, 5):
        if line_id in fault_lines:
            record = {f: healthy_df[FEATURES].mean()[f] + np.random.normal(0.4, 0.1) for f in FEATURES}
        else:
            record = healthy_df.sample(1).iloc[0].to_dict()
        record['line_id'] = line_id
        rows.append(record)

    df = pd.DataFrame(rows, columns=COLS)
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_PATH, header=True, index=False)
    logger.info(f"Appended {len(rows)} rows: majority healthy, 1-4 per line with possible fault(s).")
    time.sleep(2)
