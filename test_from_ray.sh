#!/bin/bash
echo "Testing MLflow connection FROM ray_orchestrator container..."
echo ""

docker-compose exec ray_orchestrator python3 -c "
import mlflow
from mlflow.tracking import MlflowClient
import os

tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
print(f'Tracking URI: {tracking_uri}')
print()

# Test connection
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri)

print('1. Search registered models:')
try:
    models = list(client.search_registered_models())
    print(f'   Found {len(models)} registered model names')
    for m in models:
        print(f'     - {m.name}')
        print(f'       Description: {m.description}')
        print(f'       Tags: {m.tags}')
except Exception as e:
    print(f'   ERROR: {e}')
    import traceback
    traceback.print_exc()

print()
print('2. Search model versions:')
try:
    if models:
        for rm in models:
            versions = list(client.search_model_versions(f\"name='{rm.name}'\"))
            print(f'   Model: {rm.name}')
            print(f'   Versions: {len(versions)}')
            for v in versions[:5]:
                print(f'     - v{v.version} (stage: {v.current_stage})')
except Exception as e:
    print(f'   ERROR: {e}')

print()
print('3. List experiments:')
try:
    experiments = client.search_experiments()
    print(f'   Found {len(experiments)} experiments')
    for exp in experiments[:5]:
        print(f'     - [{exp.experiment_id}] {exp.name}')
except Exception as e:
    print(f'   ERROR: {e}')
"

echo ""
echo "Testing network connectivity:"
docker-compose exec ray_orchestrator curl -s http://mlflow:5000/health || echo "Cannot reach MLflow"
