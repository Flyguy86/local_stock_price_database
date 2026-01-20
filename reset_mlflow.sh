#!/bin/bash
# Reset MLflow - Clean all tracking data and artifacts

set -e

echo "🧹 Resetting MLflow database and artifacts..."

# Stop MLflow service if running
echo "Stopping MLflow service..."
docker-compose stop mlflow 2>/dev/null || true

# Remove MLflow backend database
echo "Removing MLflow backend database..."
rm -f ./data/mlflow/backend/mlflow.db
rm -f ./data/mlflow/backend/mlflow.db-shm
rm -f ./data/mlflow/backend/mlflow.db-wal

# Remove all artifact directories (experiments and models)
echo "Removing MLflow artifacts..."
rm -rf ./data/mlflow/artifacts/*

# Recreate directory structure
echo "Recreating directory structure..."
mkdir -p ./data/mlflow/backend
mkdir -p ./data/mlflow/artifacts

# Restart MLflow service
echo "Starting MLflow service..."
docker-compose up -d mlflow

echo "✅ MLflow reset complete!"
echo "📊 MLflow UI will be available at http://localhost:5000"
echo ""
echo "Note: Allow a few seconds for the service to initialize."
