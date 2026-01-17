# Feature Engineering - Single Source of Truth

## Problem Statement

When training ML models and then backtesting them on new data, **feature engineering must be identical**. Any discrepancy between training and backtesting leads to:
- ❌ Misleading performance metrics
- ❌ Different feature distributions
- ❌ Model predictions not matching expected behavior
- ❌ Inability to reproduce results

## Solution: Ray Data Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│         StreamingPreprocessor.calculate_indicators_gpu  │
│         (Single Source of Truth for Feature Calc)      │
└──────────────────┬──────────────────┬───────────────────┘
                   │                  │
         ┌─────────▼────────┐  ┌──────▼────────┐
         │  Training        │  │  Backtesting  │
         │  Pipeline        │  │  Pipeline     │
         └──────────────────┘  └───────────────┘
                   │                  │
         ┌─────────▼────────┐  ┌──────▼────────┐
         │  Walk-Forward    │  │  New Ticker   │
         │  Folds           │  │  Date Range   │
         │  (GOOGL)         │  │  (AAPL)       │
         └──────────────────┘  └───────────────┘
```

### Implementation

**Training Path** (`ray_orchestrator/streaming.py`):
```python
preprocessor = StreamingPreprocessor(loader)

for fold in preprocessor.create_walk_forward_pipeline(...):
    # Features calculated via:
    fold = preprocessor.process_fold_with_gpu(fold)
    # Which calls:
    batch = preprocessor.calculate_indicators_gpu(batch)
    # Result: Features with v3.1 normalization
```

**Backtesting Path** (`backtest_model.py`):
```python
loader = BarDataLoader(parquet_dir="/app/data/parquet")
preprocessor = StreamingPreprocessor(loader)

def process_batch(batch):
    # EXACT SAME CODE PATH as training:
    df = preprocessor.calculate_indicators_gpu(
        df,
        windows=[50, 200],      # Match training
        zscore_window=200,       # Match training
        drop_warmup=True
    )
    return df

feature_ds = primary_ds.map_batches(process_batch, batch_format="pandas")
```

### Key Guarantees

1. **Same Feature Calculations**
   - Stochastic: Raw → simple norm → z-score (Phase 1, 3, 4)
   - RSI: Raw → simple norm → z-score
   - MACD, Bollinger, ATR, OBV, SMAs, EMAs, Volatility, Volume
   - All use identical `_rolling_zscore()`, `_calculate_rsi()`, etc.

2. **Same Version**
   - Training saves: `FEATURE_ENGINEERING_VERSION = "v3.1"`
   - Backtest checks: `preprocessor.feature_engineering_version`
   - Mismatch = warning

3. **Same Context Features**
   - Beta-60 calculation identical
   - VIX regime detection identical
   - Relative features (close_ratio, sma_ratio) identical

4. **Same Ray Data Processing**
   - Both use `map_batches()` with pandas format
   - Same batch sizes and parallelism
   - Same null handling and edge cases

## Verification

### Check Feature Consistency

```python
# In training (from checkpoint metadata):
{
  "model_info": {
    "feature_engineering_version": "v3.1"
  }
}

# In backtesting (printed during feature generation):
# Feature engineering version: v3.1
```

### Compare Feature Distributions

```python
# Load training fold features
train_fold = load_fold_from_disk(fold_id=1)
train_stats = train_fold.describe()

# Generate backtest features
backtest_df = generate_features(ticker="AAPL", ...)
backtest_stats = backtest_df.describe()

# Compare distributions
print(train_stats['rsi_14_norm'].mean())    # e.g., 0.002
print(backtest_stats['rsi_14_norm'].mean()) # e.g., 0.001 (similar)
```

### Test End-to-End

```bash
# 1. Train model on GOOGL
curl -X POST http://localhost:8265/train/walk-forward \
  -d '{"symbol": "GOOGL", "algorithm": "elasticnet", ...}'

# 2. Backtest on AAPL (uses same feature pipeline)
curl -X POST http://localhost:8265/backtest/simulate \
  -d '{"model_name": "...", "ticker": "AAPL", ...}'

# 3. Check feature version in results
cat /app/data/backtest_results/{job_id}_results.json
```

## Best Practices

### DO ✅
- Always use `StreamingPreprocessor` for feature generation
- Match `windows`, `zscore_window` parameters between train/backtest
- Verify `feature_engineering_version` matches
- Use Ray Data `map_batches()` for consistency
- Log feature stats for comparison

### DON'T ❌
- Create custom feature calculation code for backtesting
- Use pandas-only processing (bypasses Ray Data)
- Skip context symbols in backtesting if used in training
- Change normalization parameters between train/test
- Mix feature engineering versions

## Code References

| Component | File | Function |
|-----------|------|----------|
| Feature Calculation | `ray_orchestrator/streaming.py` | `calculate_indicators_gpu()` |
| Training Pipeline | `ray_orchestrator/streaming.py` | `create_walk_forward_pipeline()` |
| Backtest Features | `backtest_model.py` | `generate_features()` |
| Version Constant | `ray_orchestrator/streaming.py` | `FEATURE_ENGINEERING_VERSION` |
| Context Features | `ray_orchestrator/streaming.py` | `_calculate_context_features()` |

## Troubleshooting

### Features Don't Match

**Symptom**: Backtest performance drastically different from training

**Check**:
1. Feature engineering version mismatch?
2. Missing context symbols in backtest?
3. Different `windows` or `zscore_window` parameters?
4. Date filtering removing warm-up period?

**Fix**:
```python
# Verify parameters match training
preprocessor.calculate_indicators_gpu(
    batch,
    windows=[50, 200],     # Must match training
    zscore_window=200,     # Must match training
    drop_warmup=True       # Must match training
)
```

### Version Mismatch Error

**Symptom**: `Feature engineering version: v3.0 (expected v3.1)`

**Fix**: Retrain model with latest feature pipeline, or update backtest to use v3.0 checkpoint

### Missing Features

**Symptom**: Model expects `close_ratio_QQQ` but backtest doesn't have it

**Fix**: Add context symbols to backtest:
```python
context_symbols = ["QQQ", "SPY", "VIX"]  # Must match training
```

## Summary

**One Pipeline, All Use Cases**:
- `StreamingPreprocessor.calculate_indicators_gpu()` is the single source of truth
- Training uses it via `create_walk_forward_pipeline()`
- Backtesting uses it via `generate_features()` in `backtest_model.py`
- Feature version tracking ensures reproducibility
- Ray Data ensures consistent distributed processing

This architecture guarantees that **what you train is what you backtest** - no surprises!
