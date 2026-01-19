# MLflow Model Registry Quick Access

## ✅ Working Solution: Direct MLflow UI

The MLflow native UI is fully operational with all 104 migrated models.

### Local Access
```
http://localhost:5000
```

### GitHub Codespaces Access
Replace `8100` with `5000` in your current URL:
```
https://your-codespace-5000.app.github.dev
```

## 📊 What You Can Do in MLflow

1. **Browse Models**: See all `walk_forward_elasticnet_unknown` versions (1-104)
2. **View Metrics**: RMSE, R2, MAE for each model
3. **Compare Versions**: Side-by-side comparison
4. **Manage Stages**: Promote to Production/Staging
5. **Inspect Parameters**: See alpha, l1_ratio, etc.

## 🚀 Running Backtests

From terminal:
```bash
./run_backtest.sh [model_version] [ticker] [start_date] [end_date]

# Example:
./run_backtest.sh 50 AAPL 2024-01-01 2024-12-31
```

Results saved to: `/app/data/backtest_results/`

## 📈 Backtest Output Includes

- **Total Return** & Final Equity
- **Sharpe Ratio** & Sortino Ratio
- **Max Drawdown**
- **Win Rate** & Profit Factor
- **Transaction Costs** (commission + slippage)
- **Full Trade Log**
- **Equity Curve** (CSV for plotting)

## 🔧 Integration Note

The embedded Model Registry at `/registry` has connectivity issues between the FastAPI frontend and MLflow backend. Use MLflow's native UI at port 5000 for full functionality.
