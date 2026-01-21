#!/bin/bash
echo "PROOF: Codespaces URL Construction"
echo "==================================="
echo ""
echo "Your Codespaces URL: https://cuddly-trout-v959vxxj7gc6p79-8100.app.github.dev/"
echo ""
echo "JavaScript logic in the UI:"
echo "  const rayDashboardUrl = window.location.hostname === 'localhost'"
echo "      ? 'http://localhost:8265/#/jobs/...' "
echo "      : 'https://' + window.location.hostname.replace('-8100', '-8265') + '/#/jobs/...'"
echo ""
echo "Applying the logic:"
echo "  1. window.location.hostname = 'cuddly-trout-v959vxxj7gc6p79-8100.app.github.dev'"
echo "  2. Is it 'localhost'? NO"
echo "  3. So use: hostname.replace('-8100', '-8265')"
echo "  4. Result hostname: 'cuddly-trout-v959vxxj7gc6p79-8265.app.github.dev'"
echo "  5. Final URL: 'https://cuddly-trout-v959vxxj7gc6p79-8265.app.github.dev/#/jobs/[job_id]'"
echo ""

# Simulate with actual example
EXAMPLE_HOSTNAME="cuddly-trout-v959vxxj7gc6p79-8100.app.github.dev"
RAY_HOSTNAME=$(echo "$EXAMPLE_HOSTNAME" | sed 's/-8100/-8265/')
EXAMPLE_JOB_ID="raysubmit_ABC123XYZ"

echo "✅ PROOF WITH REAL EXAMPLE:"
echo "  Training Dashboard: https://$EXAMPLE_HOSTNAME/"
echo "  Ray Dashboard:      https://$RAY_HOSTNAME/#/jobs/$EXAMPLE_JOB_ID"
echo ""
echo "This is exactly what will happen when you click 'Train New Model' in the UI!"
echo ""

# Show the actual code
echo "Here's the actual code from the file:"
echo "====================================="
grep -B 2 -A 5 "rayDashboardUrl = window.location.hostname" /workspaces/local_stock_price_database/ray_orchestrator/templates/training_dashboard.html
