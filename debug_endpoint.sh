#!/bin/bash
echo "Testing the /mlflow/models endpoint that the webpage uses..."
echo ""

echo "1. Direct curl to endpoint:"
response=$(curl -s http://localhost:8265/mlflow/models)
echo "$response" | python3 -m json.tool

echo ""
echo ""
echo "2. Check ray_orchestrator logs for the API call:"
docker-compose logs ray_orchestrator | grep -A 10 "📡 /mlflow/models called" | tail -20

echo ""
echo ""
echo "3. Verify the code is actually updated in the container:"
docker-compose exec ray_orchestrator grep -A 5 "Use REST API directly" /app/ray_orchestrator/mlflow_integration.py | head -10

echo ""
echo ""
echo "4. Test if MLflowTracker class works in the container:"
docker-compose exec ray_orchestrator python3 -c "
import sys
sys.path.insert(0, '/app')
from ray_orchestrator.mlflow_integration import MLflowTracker

tracker = MLflowTracker('http://mlflow:5000')
models = tracker.get_registered_models()
print(f'MLflowTracker returned {len(models)} models')
if models:
    print(f'First model: {models[0]}')
"
