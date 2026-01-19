#!/bin/bash
echo "Testing MLflow server's actual database..."
echo ""

# Query MLflow REST API directly
echo "1. MLflow REST API (what ray_orchestrator sees):"
curl -s http://localhost:5000/api/2.0/mlflow/registered-models/list | python3 -m json.tool

echo ""
echo ""
echo "2. Direct Python query inside MLflow container:"
docker-compose exec mlflow python3 -c "
from mlflow.tracking import MlflowClient

# Query via HTTP (what the server returns)
print('Via HTTP server:')
client_http = MlflowClient('http://localhost:5000')
models_http = list(client_http.search_registered_models())
print(f'  Found {len(models_http)} models')

# Query via direct SQLite
print()
print('Via direct SQLite:')
client_sqlite = MlflowClient('sqlite:////mlflow/backend/mlflow.db')
models_sqlite = list(client_sqlite.search_registered_models())
print(f'  Found {len(models_sqlite)} models')
for m in models_sqlite[:3]:
    print(f'    - {m.name}')
"

echo ""
echo ""
echo "3. Check what backend-store-uri MLflow server actually started with:"
docker-compose logs mlflow | grep -i "backend-store-uri\|Starting MLflow" | head -5
