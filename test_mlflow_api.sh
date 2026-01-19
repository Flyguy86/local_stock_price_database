#!/bin/bash
# Test MLflow API endpoints

echo "Testing /mlflow/models endpoint..."
echo ""

# Test the ray_orchestrator endpoint
echo "1. Ray Orchestrator endpoint (http://localhost:8265/mlflow/models):"
curl -s http://localhost:8265/mlflow/models | python3 -m json.tool | head -50

echo ""
echo ""
echo "2. Check ray_orchestrator logs for MLflow requests:"
docker-compose logs --tail=20 ray_orchestrator | grep -i mlflow

echo ""
echo ""
echo "3. Check if ray_orchestrator has the updated code:"
docker-compose exec ray_orchestrator grep -A 5 "search_model_versions" /app/ray_orchestrator/mlflow_integration.py | head -10
