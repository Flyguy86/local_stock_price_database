#!/bin/bash
# Check MLflow database directly

echo "Checking MLflow database..."
echo ""

# Check if database file exists
if [ -f "./data/mlflow/backend/mlflow.db" ]; then
    echo "✅ Database file exists: ./data/mlflow/backend/mlflow.db"
    ls -lh ./data/mlflow/backend/mlflow.db
    echo ""
else
    echo "❌ Database file NOT found!"
    echo "Looking for mlflow directories..."
    find ./data -name "mlflow*" -type d
    exit 1
fi

# Query database
echo "Querying registered models..."
docker-compose exec mlflow sqlite3 /mlflow/backend/mlflow.db "SELECT name, COUNT(*) as versions FROM model_versions GROUP BY name;" 2>/dev/null || \
docker-compose exec mlflow sh -c "cd /mlflow/backend && ls -la" 2>/dev/null

echo ""
echo "Querying model version count..."
docker-compose exec mlflow sqlite3 /mlflow/backend/mlflow.db "SELECT COUNT(*) FROM model_versions;" 2>/dev/null

echo ""
echo "Checking if mlflow container can access the database..."
docker-compose exec mlflow ls -la /mlflow/backend/

echo ""
echo "Testing MLflow Python client from ray_orchestrator..."
docker-compose exec ray_orchestrator python3 -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.tracking.MlflowClient('http://mlflow:5000')
models = list(client.search_registered_models())
print(f'Found {len(models)} registered models')
for m in models[:3]:
    print(f'  - {m.name}')
"
