#!/bin/bash
# Check which database MLflow server is actually using

echo "Checking MLflow server configuration..."
echo ""

echo "1. MLflow server command line:"
docker-compose exec mlflow ps aux | grep mlflow || echo "Process not found"

echo ""
echo "2. Environment variables:"
docker-compose exec mlflow env | grep -i mlflow

echo ""
echo "3. Files in /mlflow/backend/:"
docker-compose exec mlflow ls -la /mlflow/backend/

echo ""
echo "4. Find all .db files in container:"
docker-compose exec mlflow find /mlflow -name "*.db" -ls

echo ""
echo "5. Test direct query inside container:"
docker-compose exec mlflow python3 -c "
import mlflow
from mlflow.tracking import MlflowClient

# Test with full path
mlflow.set_tracking_uri('sqlite:////mlflow/backend/mlflow.db')
client = MlflowClient('sqlite:////mlflow/backend/mlflow.db')

print('Direct SQLite query:')
models = list(client.search_registered_models())
print(f'  Found {len(models)} registered models')
for m in models[:3]:
    print(f'    - {m.name}')

# Test with server URI
print()
print('Via server (http://localhost:5000):')
mlflow.set_tracking_uri('http://localhost:5000')
client2 = MlflowClient('http://localhost:5000')
models2 = list(client2.search_registered_models())
print(f'  Found {len(models2)} registered models')
for m in models2[:3]:
    print(f'    - {m.name}')
"
