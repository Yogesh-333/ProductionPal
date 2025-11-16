import pandas as pd
import numpy as np
import os
import time
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'CSV_Fault_Data', 'lines_stream.csv')
HEALTHY_PATH = os.path.join(BASE_DIR, 'data', 'CSV_Fault_Data', '2_CSV_Data_Files', '1_Unloaded_Condition', 'H_H_1_0.csv')

log_path = os.path.join(BASE_DIR, 'productionpal_sensor.log')
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("ProductionPalSensorMocker")

FEATURES = ["Accelerometer 1 (m/s^2)", "Accelerometer 2 (m/s^2)", "Accelerometer 3 (m/s^2)"]
COLS = FEATURES + ['line_id']

healthy_df = pd.read_csv(HEALTHY_PATH, usecols=FEATURES)
healthy_df = healthy_df.iloc[::1000].reset_index(drop=True)

logger.info("Sensor mocker started.")

while True:
    rows = []
    # 20 real healthy samples randomly assigned to any line 1-4
    for _ in range(20):
        record = healthy_df.sample(1).iloc[0].to_dict()
        record['line_id'] = np.random.randint(1, 5)
        rows.append(record)

    # Add 4 rows, one per line, with one randomly injected fault
    fault_lines = np.random.choice([1, 2, 3, 4], size=1, replace=False)
    for line_id in range(1, 5):
        if line_id in fault_lines:
            record = healthy_df.mean().to_dict()
            for f in FEATURES:
                record[f] += np.random.normal(0.4, 0.1)
        else:
            record = healthy_df.sample(1).iloc[0].to_dict()
        record['line_id'] = line_id
        rows.append(record)

    df = pd.DataFrame(rows, columns=COLS)
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_PATH, header=True, index=False)
    logger.info(f"Appended {len(rows)} rows: mostly healthy with occasional fault.")
    time.sleep(2)
