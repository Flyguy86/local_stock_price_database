#!/bin/bash
set -e

echo "Starting Ray head node with Prometheus metrics..."
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --metrics-export-port=8080 \
    --num-cpus=0 \
    --block &

# Wait for Ray to be ready
echo "Waiting for Ray to initialize..."
sleep 5

# Check Ray status
ray status || echo "Ray status check failed, continuing anyway..."

echo "Starting Ray Orchestrator application..."
exec python -m ray_orchestrator.main
