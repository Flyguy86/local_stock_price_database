#!/bin/bash
# Quick backtest with trading simulation

MODEL_VERSION=${1:-1}  # Default to version 1
TICKER=${2:-AAPL}      # Default to AAPL
START=${3:-2024-01-01}
END=${4:-2024-12-31}

echo "Running trading backtest:"
echo "  Model: walk_forward_elasticnet_unknown v${MODEL_VERSION}"
echo "  Ticker: ${TICKER}"
echo "  Period: ${START} to ${END}"
echo ""

docker-compose exec ray_orchestrator python /app/backtest_model.py \
  --checkpoint models:/walk_forward_elasticnet_unknown/${MODEL_VERSION} \
  --ticker ${TICKER} \
  --start-date ${START} \
  --end-date ${END} \
  --initial-capital 100000 \
  --position-size-pct 0.95 \
  --slippage-pct 0.001 \
  --commission-per-share 0.005 \
  --prediction-threshold 0.001

echo ""
echo "✅ Results saved to: /app/data/backtest_results/"
echo ""
echo "To view equity curve:"
echo "  docker-compose exec ray_orchestrator cat /app/data/backtest_results/equity_curve_${TICKER}_${START}_${END}.csv"
