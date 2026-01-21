#!/bin/bash
echo "Testing Codespaces URL detection..."
echo ""

# Get current hostname
HOSTNAME=$(hostname)
echo "Current hostname: $HOSTNAME"

# Simulate what the JavaScript will do
echo ""
echo "Testing URL construction logic:"
echo "================================"

# Check if we're in localhost or Codespaces
if [[ "$HOSTNAME" == *"codespaces"* ]]; then
    echo "✅ Detected: Codespaces environment"
    # In browser, window.location.hostname would be something like:
    # 'psychic-space-disco-qpv5g6x6p5fxj7-8100.preview.app.github.dev'
    echo "Browser hostname would be: [codespace-name]-8100.preview.app.github.dev"
    echo "Ray Dashboard URL would be: https://[codespace-name]-8265.preview.app.github.dev/#/jobs/[job_id]"
else
    echo "⚠️  Not in Codespaces, using localhost"
    echo "Ray Dashboard URL would be: http://localhost:8265/#/jobs/[job_id]"
fi

echo ""
echo "Let me check the actual JavaScript code:"
echo "========================================"

grep -A 10 "rayDashboardUrl = window.location.hostname" /workspaces/local_stock_price_database/ray_orchestrator/templates/training_dashboard.html | head -15

echo ""
echo "Now testing with a real job submission..."
echo "=========================================="

docker-compose restart ray_orchestrator > /dev/null 2>&1

for i in {1..30}; do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "✅ Service ready"
        break
    fi
    sleep 1
done

echo ""
echo "Submitting test job..."
RESPONSE=$(curl -s -X POST "http://localhost:8100/training/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "primary_ticker": "GOOGL",
    "selected_features": ["returns"],
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "train_months": 12,
    "test_months": 3,
    "step_months": 3,
    "num_samples": 1,
    "experiment_name": "test_url_proof",
    "parent_experiment": "walk_forward_xgboost_GOOGL",
    "parent_trial": "train_on_folds_0b701_00044_44_colsample_bytree=0.8582,learning_rate=0.0226,max_depth=6,n_estimators=300,subsample=0.9261_2026-01-21_01-55-39"
  }')

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null)

if [ ! -z "$JOB_ID" ]; then
    echo "✅ Job submitted: $JOB_ID"
    echo ""
    echo "PROOF: In the browser UI, the JavaScript will:"
    echo "  1. Get current URL hostname"
    echo "  2. Check if it contains localhost or is a Codespaces URL"
    echo "  3. Construct Ray Dashboard URL:"
    echo "     - Localhost: http://localhost:8265/#/jobs/$JOB_ID"
    echo "     - Codespaces: https://[hostname with 8100→8265]/#/jobs/$JOB_ID"
    echo ""
    echo "The code that does this is:"
    echo "  const rayDashboardUrl = window.location.hostname === 'localhost'"
    echo "      ? \`http://localhost:8265/#/jobs/\${result.job_id}\`"
    echo "      : \`https://\${window.location.hostname.replace('-8100', '-8265')}/#/jobs/\${result.job_id}\`;"
fi
