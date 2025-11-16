# Use official Python runtime image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app folder
COPY app ./app

# Copy models and data folders to the right place (one level up from /app)
COPY models ./models
COPY data ./data

# Copy other scripts to /app
COPY sensor_mocker.py .
COPY productionpal_dashboard.log .
COPY productionpal_sensor.log .
COPY productionpal_training.log .

# Expose Streamlit default port
EXPOSE 8501
RUN python app/train_model.py
# Run Streamlit dashboard by default
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
