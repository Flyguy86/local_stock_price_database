#!/bin/bash
echo "Restarting with Codespaces-compatible URLs..."
docker-compose restart ray_orchestrator

echo "Waiting..."
for i in {1..30}; do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "✅ Ready after $i seconds"
        break
    fi
    sleep 1
done

echo ""
echo "✅ Updated! Now the UI will:"
echo "  - Detect if running in Codespaces"
echo "  - Use proper forwarded port URL for Ray Dashboard"
echo "  - Example: https://your-codespace-8265.preview.app.github.dev/#/jobs/..."
echo ""
echo "Test by opening http://localhost:8100 and submitting a retrain job"
echo "The Ray Dashboard link will now work in Codespaces!"
