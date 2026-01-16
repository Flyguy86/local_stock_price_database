# Ray Orchestrator Examples

This directory contains example scripts demonstrating the complete ML trading bot pipeline.

## Complete Workflow

### 1. Walk-Forward Preprocessing → 2. Hyperparameter Tuning → 3. Deployment

---

## Step 1: Walk-Forward Preprocessing (RECOMMENDED)

**The balanced way to preprocess trading data with no look-ahead bias.**

Walk-forward preprocessing ensures that technical indicators (SMA, RSI, MACD) are calculated **independently** for each fold's train and test data. This prevents the common mistake where an SMA calculated on the entire dataset "leaks" information from the test period into the training period.

### How It Works

```
Fold 1: Train Jan-Mar (SMA calculated only on Jan-Mar) → Test Apr
Fold 2: Train Feb-Apr (SMA calculated only on Feb-Apr) → Test May  
Fold 3: Train Mar-May (SMA calculated only on Mar-May) → Test Jun
```

Each fold's indicators **reset** at the fold boundary. The SMA-200 at the start of the test period will have ~200 NaN values as it warms up, which is correct behavior.

### Run via Python

```bash
# Inside the container
docker exec -it ray_orchestrator python -m ray_orchestrator.examples.streaming_example
```

### Run via API

**1. Preview Data**
```bash
curl -X POST "http://localhost:8100/streaming/preview?symbols=AAPL&limit=10"
```

**2. Run Walk-Forward Preprocessing (RECOMMENDED)**
```bash
curl -X POST "http://localhost:8100/streaming/walk_forward" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "context_symbols": ["QQQ", "VIX"],
    "start_date": "2024-01-01",
    "end_date": "2024-06-30",
    "train_months": 3,
    "test_months": 1,
    "step_months": 1,
    "windows": [50, 200],
    "resampling_timeframes": ["5min", "15min"],
    "output_base_path": "/app/data/walk_forward_folds",
    "num_gpus": 0.0
  }'
```

**3. Run Simple Preprocessing (Not Recommended for Backtesting)**
```bash
curl -X POST "http://localhost:8100/streaming/preprocess" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "output_path": "/app/data/preprocessed_parquet",
    "market_hours_only": true,
    "rolling_windows": [5, 10, 20]
  }'
```

**4. Check Status**
```bash
curl "http://localhost:8100/streaming/status"
```

## GPU Acceleration

To enable GPU acceleration, set `"num_gpus": 1.0` in the API request or when running programmatically:

```python
for fold in preprocessor.create_walk_forward_pipeline(
    symbols=["AAPL"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    windows=[50, 200],
    num_gpus=1.0,  # Enable GPU
    actor_pool_size=2
):
    # Each fold processed with GPU acceleration
    pass
```

## Multi-Timeframe Features

Enable multi-timeframe resampling to create features like `close_5min`, `sma50_15min`:

```python
resampling_timeframes=["5min", "15min", "1H"]
```

This resamples 1-minute bars to higher timeframes and calculates indicators on each, giving your model access to multi-scale patterns.

## Context Features

Add context symbols like QQQ (Nasdaq) and VIX (volatility) to create relative features:

```python
context_symbols=["QQQ", "VIX"]
```

This allows calculating features like:
- `relative_sma = sma50_AAPL / sma50_QQQ`
- Correlation with market volatility

## Output Structure

Walk-forward preprocessing saves data to:
```
/app/data/walk_forward_folds/
  fold_1/
    train/
      part-0.parquet
    test/
      part-0.parquet
  fold_2/
    train/
      part-0.parquet
    test/
      part-0.parquet
  ...
```

Each fold is self-contained with properly calculated indicators.

---

## Step 2: Train Models with Walk-Forward Validation

Once you have the preprocessing pipeline set up, train models with hyperparameter tuning:

### Run via Python

```bash
# Inside the container
docker exec -it ray_orchestrator python -m ray_orchestrator.examples.training_example
```

### Run via API

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
    "step_months": 1,
    "algorithm": "elasticnet",
    "num_samples": 50,
    "windows": [50, 200],
    "resampling_timeframes": ["5min", "15min"]
  }'
```

### What This Does

1. **Generates Folds**: Creates time-based train/test splits
   ```
   Fold 1: Train Jan-Mar → Test Apr
   Fold 2: Train Feb-Apr → Test May
   Fold 3: Train Mar-May → Test Jun
   ```

2. **Preprocesses Each Fold**: Calculates indicators independently per fold

3. **Tests Each Hyperparameter Config**: Every trial is evaluated across ALL folds
   ```
   Trial 1 (alpha=0.01, l1_ratio=0.5):
     Fold 1: RMSE=0.0234
     Fold 2: RMSE=0.0198
     Fold 3: RMSE=0.0212
     Average: RMSE=0.0215
   
   Trial 2 (alpha=0.1, l1_ratio=0.3):
     Fold 1: RMSE=0.0189
     Fold 2: RMSE=0.0201
     Fold 3: RMSE=0.0195
     Average: RMSE=0.0195 ← BEST!
   ```

4. **Selects Best Config**: Based on average performance across all folds

### Available Algorithms

- `elasticnet` - ElasticNet (combines L1/L2 regularization)
- `ridge` - Ridge regression (L2 only)
- `lasso` - Lasso regression (L1 only)
- `randomforest` - Random Forest regressor

### Custom Search Space

You can define custom hyperparameter spaces:

```json
{
  "algorithm": "elasticnet",
  "param_space": {
    "alpha": {"type": "loguniform", "min": 0.0001, "max": 1.0},
    "l1_ratio": {"type": "uniform", "min": 0.0, "max": 1.0}
  }
}
```

---

## Step 3: Monitor Results

Check Ray Dashboard for real-time training progress:
- **Ray Dashboard**: http://localhost:8265
- **Grafana Metrics**: http://localhost:3000
- **API Status**: http://localhost:8100/status

### View Best Results

The training endpoint will log:
- Best hyperparameters
- Average metrics across folds
- Per-fold breakdown

---

## Complete Example

```python
from ray_orchestrator.trainer import create_walk_forward_trainer

# Create trainer
trainer = create_walk_forward_trainer()

# Run complete pipeline
results = trainer.run_walk_forward_tuning(
    symbols=["AAPL"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    algorithm="elasticnet",
    num_samples=100,
    context_symbols=["QQQ", "VIX"],
    windows=[50, 200],
    resampling_timeframes=["5min", "15min"]
)

# Get best model config
best = results.get_best_result()
print(f"Best config: {best.config}")
print(f"Best RMSE: {best.metrics['test_rmse']:.6f}")
```

---

## Why This Approach?

✅ **No Look-Ahead Bias**: Indicators calculated per-fold  
✅ **Temporal Robustness**: Tested across multiple time periods  
✅ **GPU Acceleration**: Fast preprocessing with Ray Data  
✅ **Distributed Tuning**: Parallel hyperparameter search  
✅ **Production Ready**: Best config selected via proper validation
