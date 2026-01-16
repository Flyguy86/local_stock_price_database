# Walk-Forward Fold-Based Training Architecture

## Overview

All models in the Ray orchestrator **MUST** use pre-processed walk-forward folds for training. Direct loading of raw parquet data for training is **DEPRECATED** and will lead to look-ahead bias.

## Why Fold-Based Training?

### The Problem with On-the-Fly Feature Calculation

When you calculate features (MA, RSI, Bollinger Bands, etc.) on the entire dataset and then split train/test:

```python
# ❌ WRONG - Causes Look-Ahead Bias
df = load_all_data("AAPL", "2024-01-01", "2024-12-31")
df['ma_50'] = df['close'].rolling(50).mean()  # Uses future data!
train = df[:80%]  # Train Jan-Sept
test = df[80%:]   # Test Oct-Dec
```

**Problem**: The MA_50 at September 1st includes data from September 2-50, which includes test period data. Your model is "peeking" into the future.

### The Solution: Walk-Forward Fold Isolation

Each fold calculates features **independently** on only its training window:

```python
# ✅ CORRECT - Fold Isolation
Fold 1:
  Train: Jan-Mar (MA calculated on Jan-Mar only)
  Test: Apr

Fold 2:
  Train: Feb-Apr (MA calculated on Feb-Apr only)  
  Test: May

Fold 3:
  Train: Mar-May (MA calculated on Mar-May only)
  Test: Jun
```

**Result**: Each fold's features are calculated without any knowledge of future data.

## Architecture Changes

### 1. Data Configuration

**Added** to [ray_orchestrator/config.py](ray_orchestrator/config.py):

```python
class DataSettings(BaseSettings):
    # Source data (raw OHLCV bars)
    parquet_dir: Path = Path("/app/data/parquet")  # For preprocessing only
    
    # Pre-processed walk-forward folds (REQUIRED for training)
    walk_forward_folds_dir: Path = Path("/app/data/walk_forward_folds")  # ← NEW
    
    # Model outputs
    models_dir: Path = Path("/app/data/models/ray")
```

### 2. Data Loading Functions

**Added** to [ray_orchestrator/data.py](ray_orchestrator/data.py):

#### `load_fold_from_disk(fold_id, symbol)` ✅ **USE THIS**
Loads a pre-processed fold with isolated features.

```python
train_df, test_df = load_fold_from_disk(fold_id=0, symbol="AAPL")
# train_df has features calculated ONLY on training window
# test_df has features calculated using parameters from training window
```

#### `get_available_folds(symbol)` ✅ **CHECK BEFORE TRAINING**
Returns list of fold IDs available for a symbol.

```python
fold_ids = get_available_folds("AAPL")  # [0, 1, 2, 3, 4, ...]
```

#### `load_symbol_data_pandas(symbol)` ❌ **DEPRECATED FOR TRAINING**
Now shows warning. Only use for:
- Generating folds (preprocessing)
- Visualization
- Non-ML analytics

### 3. New API Endpoints

#### `GET /folds/list/{symbol}`
Check available folds for a symbol:

```bash
curl http://localhost:8100/folds/list/AAPL
```

Response:
```json
{
  "symbol": "AAPL",
  "num_folds": 10,
  "folds": [
    {"fold_id": 0, "train_start": "2024-01-01", "train_end": "2024-03-31", ...},
    {"fold_id": 1, "train_start": "2024-02-01", "train_end": "2024-04-30", ...},
    ...
  ]
}
```

#### `GET /folds/summary`
Overview of preprocessing status:

```bash
curl http://localhost:8100/folds/summary
```

Response:
```json
{
  "total_symbols_with_data": 8,
  "total_symbols_with_folds": 3,
  "symbols_needing_preprocessing": ["NVDA", "QQQ", "RDDT", "VIXY", "XOM"],
  "symbols_ready_for_training": ["AAPL", "GOOGL", "MSFT"]
}
```

#### `POST /streaming/walk_forward` (Updated)
Generate folds - **run this ONCE per symbol**:

```json
{
  "symbols": ["AAPL", "GOOGL"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "train_months": 3,
  "test_months": 1,
  "step_months": 1,
  "windows": [20, 50, 200],
  "output_base_path": "/app/data/walk_forward_folds"
}
```

This creates:
```
/app/data/walk_forward_folds/
├── AAPL/
│   ├── fold_000/
│   │   ├── train/
│   │   │   └── *.parquet  (with features calculated on Jan-Mar)
│   │   └── test/
│   │       └── *.parquet
│   ├── fold_001/
│   │   ├── train/  (features calculated on Feb-Apr)
│   │   └── test/
│   └── ...
└── GOOGL/
    └── ...
```

#### `POST /train/walk_forward` (Updated with Validation)
Now validates folds exist **BEFORE** starting training:

```python
# Automatically checks:
# 1. Do folds exist for all symbols?
# 2. Are features pre-calculated?
# 3. Raises HTTPException if not ready
```

## Workflow

### Step 1: Generate Folds (One-Time Preprocessing)

```bash
# Generate folds for your symbols
curl -X POST http://localhost:8100/streaming/walk_forward \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "train_months": 3,
    "test_months": 1,
    "step_months": 1,
    "windows": [20, 50, 200]
  }'
```

**What this does:**
- Loads raw 1-minute bars from `/app/data/parquet/`
- Creates rolling windows (Fold 1: Jan-Mar→Apr, Fold 2: Feb-Apr→May, etc.)
- For each fold:
  - Calculates MA, RSI, Bollinger Bands, ATR on **training window only**
  - Applies same parameters to test window
  - Saves to `/app/data/walk_forward_folds/{symbol}/fold_XXX/`

### Step 2: Verify Folds Exist

```bash
# Check what's ready
curl http://localhost:8100/folds/summary

# Check specific symbol
curl http://localhost:8100/folds/list/AAPL
```

### Step 3: Train Models

```bash
curl -X POST http://localhost:8100/train/walk_forward \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "algorithm": "xgboost",
    "param_space": {
      "max_depth": [3, 5, 7],
      "learning_rate": [0.01, 0.05, 0.1]
    },
    "num_samples": 50
  }'
```

**What this does:**
- ✅ **Validates** folds exist (fails fast if not)
- ✅ **Loads** pre-calculated features via `load_fold_from_disk()`
- ✅ **Trains** each hyperparameter config across ALL folds
- ✅ **Reports** average metrics (prevents overfitting to one time period)

## Code Examples

### Training Script (Correct)

```python
from ray_orchestrator.data import load_fold_from_disk, get_available_folds
from sklearn.ensemble import RandomForestRegressor
import numpy as np

symbol = "AAPL"
fold_ids = get_available_folds(symbol)

fold_metrics = []
for fold_id in fold_ids:
    # Load pre-processed fold
    train_df, test_df = load_fold_from_disk(fold_id, symbol)
    
    # Features are already calculated with proper isolation
    X_train = train_df[['ma_20', 'ma_50', 'rsi_14', 'bb_upper', 'bb_lower']]
    y_train = train_df['target']
    
    X_test = test_df[['ma_20', 'ma_50', 'rsi_14', 'bb_upper', 'bb_lower']]
    y_test = test_df['target']
    
    # Train
    model = RandomForestRegressor(max_depth=5)
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    fold_metrics.append(score)

# Average across folds
avg_score = np.mean(fold_metrics)
print(f"Average R² across {len(fold_ids)} folds: {avg_score:.4f}")
```

### What NOT to Do

```python
# ❌ DON'T LOAD RAW DATA DIRECTLY FOR TRAINING
from ray_orchestrator.data import load_symbol_data_pandas

df = load_symbol_data_pandas("AAPL")  # ← Shows deprecation warning
df['ma_50'] = df['close'].rolling(50).mean()  # ← Look-ahead bias!
```

## Benefits

1. **No Look-Ahead Bias**: Features calculated independently per fold
2. **Realistic Backtests**: Model performance reflects production behavior
3. **Time-Series Aware**: Respects temporal ordering
4. **Reproducible**: Same folds = same features = same results
5. **Efficient**: Pre-calculate once, train many times
6. **Cacheable**: Reuse folds across experiments

## Directory Structure

```
/app/data/
├── parquet/                      # Raw OHLCV bars (source)
│   ├── AAPL/
│   │   ├── dt=2024-01-01/
│   │   └── dt=2024-01-02/
│   └── GOOGL/
│
├── walk_forward_folds/           # Pre-processed folds (training)
│   ├── AAPL/
│   │   ├── fold_000/
│   │   │   ├── train/
│   │   │   │   └── part-0.parquet  # With ma_20, ma_50, rsi_14, etc.
│   │   │   ├── test/
│   │   │   └── metadata.json       # Fold config
│   │   ├── fold_001/
│   │   └── fold_002/
│   └── GOOGL/
│
├── models/ray/                   # Trained models
└── ray_checkpoints/              # Ray Tune checkpoints
```

## Migration Guide

### For Existing Code

**Before:**
```python
df = load_symbol_data_pandas("AAPL", "2024-01-01", "2024-12-31")
df['ma_50'] = calculate_ma(df, 50)
train_test_split(df, ...)
```

**After:**
```python
# 1. One-time: Generate folds
# POST /streaming/walk_forward

# 2. Training: Use folds
fold_ids = get_available_folds("AAPL")
for fold_id in fold_ids:
    train_df, test_df = load_fold_from_disk(fold_id, "AAPL")
    # Features already in train_df and test_df
```

## FAQs

**Q: Do I need to regenerate folds for each experiment?**
A: No! Generate once, reuse for all hyperparameter searches.

**Q: How often should I regenerate folds?**
A: When you:
- Add new data (extend date range)
- Change feature engineering logic
- Change window sizes (20 vs 50 periods)
- Change train/test split strategy

**Q: Can I use raw parquet for anything?**
A: Yes, for:
- Data exploration/visualization
- Generating new folds
- Non-ML analytics

**Q: What if folds don't exist?**
A: Training endpoints now **validate** and return clear error:
```json
{
  "detail": "No pre-processed folds found for AAPL. Run preprocessing first: POST /streaming/walk_forward with symbols=['AAPL']"
}
```

**Q: How do I check fold quality?**
A: Use `GET /folds/list/AAPL` to see metadata including:
- Date ranges
- Row counts
- Feature columns
- Fold configuration

## Summary

✅ **DO**:
- Generate folds: `POST /streaming/walk_forward`
- Load folds: `load_fold_from_disk()`
- Verify folds: `GET /folds/summary`
- Train on folds: `POST /train/walk_forward`

❌ **DON'T**:
- Load raw parquet for training
- Calculate features on full dataset then split
- Skip fold validation
- Assume features exist without checking
