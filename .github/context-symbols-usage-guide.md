# Context Symbols Usage Guide

Quick reference for using context symbols in training and preprocessing.

---

## API Examples

### 1. Walk-Forward Preprocessing with Context Symbols

```bash
curl -X POST http://localhost:8000/streaming/walk_forward \
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
    "output_base_path": "/app/data/walk_forward_folds",
    "num_gpus": 0.0,
    "actor_pool_size": 4
  }'
```

**Expected Behavior**:
- Loads AAPL as primary symbol
- Loads QQQ and VIX as context symbols
- Merges context features with `_QQQ` and `_VIX` suffixes
- Creates features like:
  - `close_QQQ`, `rsi_14_QQQ`, `macd_QQQ`
  - `close_VIX`, `vix_zscore`, `high_vix_regime`
  - `close_ratio_QQQ`, `beta_60_QQQ`, `residual_return_QQQ`

### 2. Training with Context Symbols (Legacy Path)

```bash
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL,QQQ,VIX",
    "algorithm": "elasticnet",
    "target_col": "close",
    "lookforward": 1,
    "timeframe": "1m",
    "split_ratio": 0.8,
    "data_options": "{\"windows\": [50, 200]}",
    "p_val_thresh": 0.05
  }'
```

**Note**: This uses the legacy `training_service/data.py` path which already supports context symbols via comma-separated symbol list.

### 3. Walk-Forward Training with Context Symbols

```bash
curl -X POST http://localhost:8000/train/walk_forward \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "context_symbols": ["QQQ", "VIX"],
    "start_date": "2024-01-01",
    "end_date": "2024-06-30",
    "train_months": 3,
    "test_months": 1,
    "algorithm": "elasticnet",
    "num_samples": 50,
    "windows": [50, 200]
  }'
```

---

## Python Examples

### Direct Pipeline Usage

```python
from ray_orchestrator.streaming import StreamingPreprocessor, BarDataLoader

# Initialize
loader = BarDataLoader(parquet_dir="/app/data/parquet")
preprocessor = StreamingPreprocessor(loader)

# Create pipeline with context symbols
for fold in preprocessor.create_walk_forward_pipeline(
    symbols=["AAPL"],
    context_symbols=["QQQ", "VIX"],
    start_date="2024-01-01",
    end_date="2024-06-30",
    train_months=3,
    test_months=1,
    windows=[50, 200],
    num_gpus=0.0
):
    # fold.train_ds and fold.test_ds now include context features
    train_df = fold.train_ds.to_pandas()
    
    # Check context features
    context_cols = [c for c in train_df.columns if c.endswith('_QQQ') or c.endswith('_VIX')]
    print(f"Fold {fold.fold_id}: {len(context_cols)} context features")
    print(f"Sample: {context_cols[:5]}")
```

### Loading Preprocessed Data with Context

```python
from ray_orchestrator.data import load_fold_from_disk
import pandas as pd

# Load preprocessed fold
train_df, test_df = load_fold_from_disk(
    fold_id=0,
    symbol="AAPL",
    fold_base_dir="/app/data/walk_forward_folds"
)

# Identify context features
all_cols = train_df.columns.tolist()
context_cols = [c for c in all_cols if any(c.endswith(f"_{sym}") for sym in ["QQQ", "VIX", "SPY"])]
primary_cols = [c for c in all_cols if c not in context_cols and c != 'ts']

print(f"Primary features: {len(primary_cols)}")
print(f"Context features: {len(context_cols)}")
print(f"Total: {len(all_cols)}")
```

---

## Common Context Symbols

### Market Indices
- **QQQ**: Nasdaq-100 ETF (tech-heavy market proxy)
- **SPY**: S&P 500 ETF (broad market proxy)
- **DIA**: Dow Jones Industrial Average ETF
- **IWM**: Russell 2000 ETF (small-cap proxy)

### Volatility Indicators
- **VIX**: CBOE Volatility Index (fear gauge)
- **VIXY**: VIX Short-Term Futures ETF

### Sector ETFs
- **XLF**: Financial Select Sector
- **XLE**: Energy Select Sector
- **XLK**: Technology Select Sector
- **XLV**: Health Care Select Sector

---

## Context Features Generated

### Basic Context Features
For each context symbol (e.g., `QQQ`):
- **Price Data**: `open_QQQ`, `high_QQQ`, `low_QQQ`, `close_QQQ`
- **Volume**: `volume_QQQ`, `volume_ratio_QQQ`
- **Indicators**: `rsi_14_QQQ`, `macd_QQQ`, `sma_50_QQQ`, `ema_12_QQQ`
- **Volatility**: `atr_14_QQQ`, `bollinger_width_QQQ`, `volatility_20_QQQ`

### Cross-Sectional Features
Created by `_calculate_context_features()`:
- **Relative Strength**: `close_ratio_QQQ`, `close_ratio_QQQ_zscore`
- **Beta**: `beta_60_QQQ` (rolling 60-bar beta)
- **Residual Returns**: `residual_return_QQQ`, `residual_return_QQQ_zscore`
- **SMA Ratios**: `sma_50_ratio_QQQ`, `sma_200_ratio_QQQ`

### VIX-Specific Features
When VIX is used as context:
- `vix_zscore`: Z-score of VIX level (is VIX elevated?)
- `high_vix_regime`: Binary flag (VIX > 20)
- `vix_spike`: VIX jumped > 1 std dev
- `vix_log_return`: Fear velocity

---

## Verification Steps

### 1. Check Preprocessing Logs

```bash
# Watch merge progress
docker-compose logs -f ray-orchestrator | grep "Context merge"

# Expected output:
# INFO: Merging 2 context symbols: ['QQQ', 'VIX']
# INFO:   Merged QQQ: 87653 context rows → 45 features
# INFO:   Merged VIX: 87653 context rows → 45 features
# INFO: Context merge complete for AAPL: 90 context features added
```

### 2. Verify Saved Parquet Files

```python
import pandas as pd
import duckdb

# Query parquet files
df = duckdb.query("""
    SELECT *
    FROM read_parquet('/app/data/walk_forward_folds/AAPL/fold_000/train/**/*.parquet')
    LIMIT 10
""").df()

# List context columns
context_cols = [c for c in df.columns if '_QQQ' in c or '_VIX' in c]
print(f"Context features: {len(context_cols)}")
print(context_cols)
```

### 3. Check Training Logs

```bash
# Watch feature importance
docker-compose logs -f training-service | grep -A 20 "Top 15 Features"

# Expected output:
# === Top 15 Features for Grid Model 1 ===
#    1. close_log_return                       =   0.234567
#    2. rsi_14_QQQ                             =   0.189432 [CONTEXT]
#    3. vix_zscore                             =   0.134567 [CONTEXT]
#    ...
# Context Features: 47/183 (25.7%)
```

### 4. Validate Timestamp Alignment

```python
# Check for future leakage
df = pd.read_parquet('/app/data/walk_forward_folds/AAPL/fold_000/train/')

# Ensure timestamps are sorted
assert df['ts'].is_monotonic_increasing, "Timestamps out of order!"

# Check for unexpected NaNs in context features
context_cols = [c for c in df.columns if c.endswith('_QQQ')]
nan_pct = df[context_cols].isna().sum().sum() / (len(df) * len(context_cols)) * 100
print(f"NaN percentage in context features: {nan_pct:.2f}%")

# Should be very low (<5%) after forward-fill
assert nan_pct < 5.0, "Too many NaNs in context features!"

# Check for infinite values
numeric_cols = df.select_dtypes(include=['number']).columns
inf_count = np.isinf(df[numeric_cols]).sum().sum()
print(f"Infinite values: {inf_count}")
assert inf_count == 0, "Found infinite values in data!"
```

### 5. Monitor Validation Logs

```bash
# Watch data quality validation
docker-compose logs -f ray-orchestrator | grep "Data Quality Validation"

# Expected output:
# INFO: Data Quality Validation [after_context_merge] for AAPL:
#         Rows: 87,653
#         Columns: 210
#         NaN cells: 456,234 (2.48%)
#         Threshold: 5.0%
# INFO: ✅ Validation [after_context_merge] PASSED for AAPL

# Check for validation failures
docker-compose logs -f ray-orchestrator | grep "VALIDATION FAILED"
```

---

## Troubleshooting

### Issue: Validation fails with high NaN percentage

**Symptoms**:
```
ERROR: VALIDATION FAILED [after_context_merge]: NaN percentage 8.23% exceeds threshold 5.0%
Columns with >10% NaN (top 10):
  - close_VIX: 9,234/87,653 (10.5%)
  - rsi_14_VIX: 8,901/87,653 (10.2%)
```

**Solutions**:
1. **Check context symbol data coverage**:
   ```python
   # Verify VIX has data for your date range
   import duckdb
   vix_data = duckdb.query("""
       SELECT MIN(ts) as first_date, MAX(ts) as last_date, COUNT(*) as rows
       FROM read_parquet('/app/data/parquet/VIX/**/*.parquet')
   """).df()
   print(vix_data)
   ```

2. **Use different context symbols** with better coverage
3. **Adjust date range** to match context symbol availability
4. **Increase threshold** (if acceptable for your use case):
   ```python
   # In streaming.py, increase allow_nan_threshold
   self._validate_data_quality(..., allow_nan_threshold=0.10)  # 10%
   ```

### Issue: Infinite values detected

**Symptoms**:
```
WARNING: Columns with infinite values: {'beta_60_QQQ': 234}
INFO: Replaced 234 infinite values with NaN in beta_60_QQQ
```

**Cause**: Division by zero or near-zero in calculations (e.g., beta, ratios)

**Solution**: Already handled automatically - infinite values are replaced with NaN and will be:
- Imputed during training (mean/median)
- Potentially dropped if column has too many NaNs
- Tracked in dropped columns report

### Issue: No context features in saved files

**Symptoms**:
```python
context_cols = [c for c in df.columns if '_QQQ' in c]
print(len(context_cols))  # Returns 0
```

**Solution**:
1. Check preprocessing logs for merge confirmation
2. Verify context symbols exist in parquet directory
3. Ensure date ranges overlap between primary and context symbols

### Issue: Row count drops during merge

**Symptoms**:
```
INFO:   Merged QQQ: 87653 → 45231 rows  # Unexpected drop
```

**Solution**:
1. Check for timestamp mismatches
2. Verify timezone consistency (all should be US/Eastern)
3. Look for gaps in context symbol data

### Issue: Context features have high NaN percentage

**Symptoms**:
```
NaN percentage in context features: 45.23%  # Too high
```

**Solution**:
1. Check if context symbols have complete data coverage
2. Verify forward-fill is working (should reduce NaNs)
3. Consider using different context symbols with better coverage

---

## Best Practices

### 1. Symbol Selection
- **Market Proxy**: Always include QQQ or SPY for beta calculation
- **Volatility**: Include VIX for regime detection
- **Sector Alignment**: Add relevant sector ETF (e.g., XLK for tech stocks)

### 2. Feature Engineering
- Focus on **relative** features (ratios, beta) over absolute prices
- Use **z-scores** for regime detection
- Calculate **residual returns** to identify alpha

### 3. Performance
- Limit to 2-3 context symbols per model (avoid feature explosion)
- Use p-value pruning to remove low-signal context features
- Monitor feature importance to validate context symbol usefulness

### 4. Validation
- Always verify timestamp alignment after merge
- Check NaN percentages in context features
- Review feature importance to ensure context is informative

---

## Example Workflow

```bash
# Step 1: Preprocess with context
curl -X POST http://localhost:8000/streaming/walk_forward \
  -d '{"symbols": ["AAPL"], "context_symbols": ["QQQ", "VIX"], ...}'

# Step 2: Verify preprocessing
docker-compose logs ray-orchestrator | grep "Context merge complete"

# Step 3: Check saved files
python -c "import pandas as pd; df = pd.read_parquet('...'); print([c for c in df.columns if '_QQQ' in c][:10])"

# Step 4: Train model
curl -X POST http://localhost:8000/train/walk_forward \
  -d '{"symbols": ["AAPL"], "context_symbols": ["QQQ", "VIX"], ...}'

# Step 5: Review feature importance
docker-compose logs training-service | grep -A 20 "Top 15 Features"

# Step 6: Check dropped columns summary
docker-compose logs training-service | grep -A 30 "FEATURE DROPPING SUMMARY"
```

---

## See Also

- [Ray Data Preprocessing Review](.github/ray-data-preprocessing-review.md)
- [Implementation Summary](.github/context-symbols-implementation-summary.md)
- [Feature Consistency Guide](FEATURE_CONSISTENCY_GUIDE.md)
