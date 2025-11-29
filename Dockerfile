# 1. Base Image
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Application Code & Scripts
COPY app/ ./app/
COPY sensor_mocker.py .
COPY run_all.py .

# 5. Create directories for data and models
RUN mkdir -p data/CSV_Fault_Data/2_CSV_Data_Files models

# 6. Copy data (Required for training to succeed)
COPY data ./data

# 7. Define Build-Time Environment Variables
ENV DB_USERNAME="admin_build"
ENV EXPERIMENT_NAME="Docker_Build"

# 8. TRAIN MODEL DURING BUILD
# Since MLflow is removed, this simply trains sklearn model and saves .pkl files
RUN python app/train_model.py

# 9. Expose Port
EXPOSE 8501

# 10. Run the orchestrator script
CMD ["python", "run_all.py"]