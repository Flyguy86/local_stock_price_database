#!/bin/bash
echo "Checking ray_orchestrator logs for /mlflow/models calls..."
echo ""

# Watch for the next API call
docker-compose logs --tail=100 ray_orchestrator | grep -E "(mlflow|models|📡|🔍)" || echo "No matching logs found"

echo ""
echo ""
echo "Now manually trigger the API:"
curl -s http://localhost:8265/mlflow/models

echo ""
echo ""
echo "Check logs again:"
docker-compose logs --tail=50 ray_orchestrator
