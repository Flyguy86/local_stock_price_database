#!/bin/bash
echo "Rebuilding and restarting ray_orchestrator..."
docker-compose up -d --build --force-recreate ray_orchestrator

echo ""
echo "Waiting for ray_orchestrator to start..."
sleep 5

echo ""
echo "Testing /mlflow/models endpoint..."
curl -s http://localhost:8265/mlflow/models | python3 -m json.tool | head -30

echo ""
echo ""
echo "Check browser DevTools console for:"
echo "  - 'Received models data'"
echo "  - 'Models count: XX'"
echo ""
echo "Then refresh the Model Registry page"
