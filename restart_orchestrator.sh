#!/bin/bash
# Restart the orchestrator service to pick up code changes

echo "Restarting ray_orchestrator service..."
docker compose restart ray_orchestrator

echo "Waiting for service to be ready..."
sleep 5

echo "Checking service health..."
docker compose ps ray_orchestrator

echo ""
echo "✅ Orchestrator restarted!"
echo "Access the Results Viewer at your Codespaces URL on port 8265:"
echo "   https://<your-codespace>-8265.app.github.dev/backtest/results_viewer"
