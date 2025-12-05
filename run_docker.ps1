# Helper script to run ProductionPal with the assignment-specific environment variables

Write-Host "--- 🐳 Starting ProductionPal in Docker ---"

# 1. Build the image (optional, ensure it exists)
Write-Host "Building Docker image..."
docker build -t yogeshkumar333/productionpal:assignment .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed. Please ensure Docker Desktop is running." -ForegroundColor Red
    exit 1
}

# 2. Run the container with all Environment Variables
Write-Host "Running container..."
docker run -p 8501:8501 -p 5000:5000 -p 8000:8000 `
    -e DB_USERNAME="admin" `
    -e DB_PASSWORD="secret_password_123" `
    -e DB_HOSTNAME="production-db-server" `
    -e DB_PORT="5432" `
    -e EXPERIMENT_NAME="Assignment_Submission" `
    -e EXPERIMENT_VERSION="1.0.Final" `
    -e RF_N_ESTIMATORS="100" `
    -e EXPECTED_ACCURACY="0.95" `
    -e NUM_EPOCHS="10" `
    -e FEATURE_NAMES="Accelerometer 1 (m/s^2),Accelerometer 2 (m/s^2),Accelerometer 3 (m/s^2)" `
    yogeshkumar333/productionpal:assignment

# Note: Added -p 5000:5000 and -p 8000:8000 to expose MLflow and Gateway as well
