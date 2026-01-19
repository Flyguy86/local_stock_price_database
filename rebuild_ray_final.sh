#!/bin/bash
echo "Rebuilding ray_orchestrator with updated MLflow code..."
docker-compose up -d --build --force-recreate ray_orchestrator

echo ""
echo "Waiting for ray_orchestrator to start..."
sleep 8

echo ""
echo "Testing /mlflow/models endpoint..."
curl -s http://localhost:8265/mlflow/models | python3 -m json.tool | head -50

echo ""
echo ""
echo "Check ray_orchestrator logs for debugging:"
docker-compose logs --tail=30 ray_orchestrator | grep -E "(📡|🔍|MLflow|models)"

echo ""
echo "Now refresh the Model Registry page in your browser"
