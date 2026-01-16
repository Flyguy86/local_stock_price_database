# Quick Start: Train Your First Trading Bot

## 1. Start Services

```bash
docker-compose up -d
```

Services running:
- API (ingestion): http://localhost:8600
- Feature Builder: http://localhost:8500  
- Ray Orchestrator: http://localhost:8100
- Ray Dashboard: http://localhost:8265
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## 2. Verify Data

```bash
curl "http://localhost:8100/streaming/status"
```

Should show your parquet files are available.

## 3. Train a Model (Full Pipeline)

```bash
curl -X POST "http://localhost:8100/train/walk_forward" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "context_symbols": ["QQQ"],
    "start_date": "2024-01-01",
    "end_date": "2024-06-30",
    "train_months": 3,
    "test_months": 1,
    "algorithm": "elasticnet",
    "num_samples": 50,
    "windows": [50, 200]
  }'
```

This will:
1. Generate walk-forward folds (Jan-Mar train → Apr test, etc.)
2. Calculate technical indicators per fold (SMA, RSI, MACD, etc.)
3. Run 50 hyperparameter trials
4. Test each trial across all folds
5. Return the best configuration

## 4. Monitor Progress

- **Ray Dashboard**: http://localhost:8265
  - See live trials running
  - View resource utilization
  - Monitor task progress

- **Check Logs**:
  ```bash
  docker logs -f ray_orchestrator
  ```

## 5. Results

After completion, check logs for:
- Best hyperparameters (alpha, l1_ratio, etc.)
- Performance metrics (RMSE, MAE, R²)
- Per-fold breakdown

Example output:
```
Best config: {"alpha": 0.0234, "l1_ratio": 0.45}
Best test RMSE: 0.001856
Best test R²: 0.7234
Num folds: 5
```

## What's Happening Under the Hood

### Walk-Forward Validation Ensures No Look-Ahead

**Traditional (WRONG)**:
```
[All Data] → Calculate SMA-200 → Split Train/Test
❌ Test data influenced by future prices in the SMA calculation
```

**Walk-Forward (CORRECT)**:
```
Fold 1: [Jan-Mar] → Calculate SMA-200 → Train → [Apr] → Test
Fold 2: [Feb-Apr] → Calculate SMA-200 → Train → [May] → Test
✅ Each fold's SMA calculated independently
```

### Each Trial Tested Across Time

```
Trial 1 (alpha=0.01):
  Fold 1 (Jan-Mar→Apr): RMSE = 0.0234
  Fold 2 (Feb-Apr→May): RMSE = 0.0198  
  Fold 3 (Mar-May→Jun): RMSE = 0.0212
  → Average RMSE = 0.0215

Trial 2 (alpha=0.1):
  Fold 1: RMSE = 0.0189
  Fold 2: RMSE = 0.0201
  Fold 3: RMSE = 0.0195
  → Average RMSE = 0.0195 ← BEST!
```

## Next Steps

1. **Experiment with Algorithms**:
   - `elasticnet` - Good for linear patterns
   - `randomforest` - Good for non-linear patterns

2. **Add Context Symbols**:
   ```json
   "context_symbols": ["QQQ", "VIX", "SPY"]
   ```

3. **Multi-Timeframe Features**:
   ```json
   "resampling_timeframes": ["5min", "15min", "1H"]
   ```

4. **Increase Trials**:
   ```json
   "num_samples": 200
   ```

5. **Enable GPU Acceleration**:
   - Set `"num_gpus": 1.0` in walk_forward endpoint
   - Requires GPU-enabled instance (Vast.ai, etc.)

6. **Deploy Best Model**:
   - Use model artifacts from Ray checkpoints
   - Set up paper trading simulation
   - Monitor live performance

## Troubleshooting

**No data found:**
```bash
# Check if ingestion ran
curl "http://localhost:8600/status"

# Verify parquet files
docker exec ray_orchestrator ls -la /app/data/parquet/
```

**Ray not initialized:**
```bash
# Restart ray_orchestrator
docker-compose restart ray_orchestrator
```

**Out of memory:**
- Reduce `num_samples`
- Reduce date range
- Increase docker memory limit
