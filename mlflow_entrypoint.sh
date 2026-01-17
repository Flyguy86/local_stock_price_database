#!/bin/bash
set -e

echo "Starting MLflow server..."
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:////mlflow/backend/mlflow.db \
  --default-artifact-root /mlflow/artifacts &

MLFLOW_PID=$!

# Wait for MLflow to be ready
echo "Waiting for MLflow server to start..."
for i in {1..30}; do
  if curl -f http://localhost:5000/health 2>/dev/null; then
    echo "MLflow server is ready!"
    break
  fi
  echo "Waiting for MLflow... ($i/30)"
  sleep 2
done

# Run migration
echo "Running Ray checkpoint migration..."
python3 /app/migrate_ray_checkpoints_to_mlflow.py || echo "Migration completed with warnings"

# Keep MLflow running
echo "MLflow server running on port 5000"
wait $MLFLOW_PID
