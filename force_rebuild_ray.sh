#!/bin/bash
echo "Force rebuilding ray_orchestrator..."
docker-compose stop ray_orchestrator
docker-compose rm -f ray_orchestrator
docker-compose build --no-cache ray_orchestrator
docker-compose up -d ray_orchestrator

echo ""
echo "Waiting for startup..."
sleep 8

echo ""
echo "Testing endpoint:"
curl -s http://localhost:8265/mlflow/models | python3 -m json.tool | head -50

echo ""
echo ""
echo "Check logs for REST API call:"
docker-compose logs --tail=50 ray_orchestrator | grep -E "(REST API|🔍|Found)"
