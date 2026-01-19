#!/bin/bash
echo "Testing MLflow REST API endpoints..."
echo ""

echo "1. Health check:"
curl -s http://localhost:5000/health
echo ""

echo ""
echo "2. List registered models (REST API):"
curl -s "http://localhost:5000/api/2.0/mlflow/registered-models/list" 
echo ""

echo ""
echo "3. Search registered models (REST API):"
curl -s -X GET "http://localhost:5000/api/2.0/mlflow/registered-models/search" \
  -H "Content-Type: application/json" \
  -d '{}'
echo ""

echo ""
echo "4. Check experiments (REST API):"
curl -s "http://localhost:5000/api/2.0/mlflow/experiments/search" | python3 -m json.tool | head -20
