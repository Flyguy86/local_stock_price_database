#!/bin/bash
echo "🔄 Restarting with form-based configuration..."
docker-compose restart ray_orchestrator

echo "⏳ Waiting..."
for i in {1..30}; do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "✅ Ready after $i seconds"
        break
    fi
    sleep 1
done

echo ""
echo "✅ UI Updated!"
echo ""
echo "The feature selection dialog now has a form with:"
echo "  - Primary Ticker (default: GOOGL)"
echo "  - Context Symbols (optional, comma-separated)"
echo "  - Start/End Dates (date pickers)"
echo "  - Train/Test/Step Months (number inputs)"
echo "  - Number of Trials (how many hyperparameter combinations to try)"
echo ""
echo "These values will be used directly instead of trying to extract from parent config!"
echo ""
echo "Open: http://localhost:8100"
echo "Then: Rank Models → Select for Retrain → Fill in the form → Train"
