# Training Service Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Data Pipeline](#data-pipeline)
3. [Data Leakage Protection](#data-leakage-protection)
4. [Validation Methodology](#validation-methodology)
5. [Feature Engineering & Selection](#feature-engineering--selection)
6. [Model Training Pipeline](#model-training-pipeline)
7. [Metrics & Evaluation](#metrics--evaluation)
8. [Grid Search & Hyperparameter Tuning](#grid-search--hyperparameter-tuning)
9. [Model Relationships](#model-relationships)
10. [Configuration Options](#configuration-options)

---

## Architecture Overview

The Training Service is a sophisticated ML pipeline designed for time-series forecasting of stock prices using engineered features from DuckDB/Parquet storage. It emphasizes data integrity, prevents leakage, and provides comprehensive model evaluation.

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Service                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │   data.py   │──▶│  trainer.py  │──▶│  PostgreSQL Models  │  │
│  │  (Loader)   │   │  (Training)  │   │     (Metadata)      │  │
│  └─────────────┘   └──────────────┘   └─────────────────────┘  │
│         │                  │                      │              │
│         ▼                  ▼                      ▼              │
│  DuckDB/Parquet      Joblib Models         Feature Analysis     │
│   (Features)         (Artifacts)           (SHAP, Importance)   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Files:**
- [training_service/data.py](training_service/data.py) - Data loading and preprocessing
- [training_service/trainer.py](training_service/trainer.py) - Model training and evaluation
- [training_service/main.py](training_service/main.py) - FastAPI endpoints
- [training_service/db.py](training_service/db.py) - PostgreSQL metadata storage
- [training_service/config.py](training_service/config.py) - Configuration settings

---

## Data Pipeline

### 1. **Multi-Ticker Loading**

The service supports training on **multi-ticker datasets** where context symbols (e.g., QQQ, VIX, MSFT) provide additional features for predicting a primary symbol (e.g., AAPL).

**Example:** Training AAPL with context from QQQ and VIX:
```python
# Primary symbol: AAPL
# Context symbols: QQQ, VIX
# Result: Features like close_QQQ, macd_line_VIX, return_z_score_20_QQQ
```

**Implementation:** [data.py#L150-L180](training_service/data.py)
```python
def load_training_data(symbol: str, target_col: str = "close", ...):
    # Parse primary and context symbols
    symbols = symbol.split(",")
    primary_symbol = symbols[0]
    context_symbols = symbols[1:] if len(symbols) > 1 else []
    
    # Load primary data
    df = _load_single(primary_symbol)
    
    # Merge context symbols via INNER JOIN on 'ts' (timestamp)
    for ctx_sym in context_symbols:
        ctx_df = _load_single(ctx_sym)
        ctx_df = ctx_df.rename(columns={c: f"{c}_{ctx_sym}" for c in ctx_df.columns if c != "ts"})
        df = pd.merge(df, ctx_df, on="ts", how="inner")
```

**Key Protections:**
- **Inner Join Alignment:** Only rows with matching timestamps across ALL tickers are kept
- **Column Renaming:** Context features are suffixed (e.g., `close_QQQ`) to avoid collisions
- **Missing Data Handling:** Context symbols with no data are skipped with warnings

### 2. **DuckDB Parquet Scanning**

Features are stored in partitioned Parquet files (`/app/data/features_parquet/{symbol}/dt={date}/`). The loader uses DuckDB's `read_parquet` with glob patterns for efficient scanning.

**Options Filtering:** The `data_options` parameter filters feature sets (e.g., "options_v2_full_vol", "options_v2_ma_only") to control which indicators are loaded.

**Legacy Handling:** Supports older data with NULL or empty options columns via fallback logic.

### 3. **Time-Based Resampling**

Data is resampled to the target timeframe (e.g., 1m → 5m → 1h → 1d) using custom aggregation rules:

| Column         | Aggregation | Rationale |
|----------------|-------------|-----------|
| `open`         | first       | Opening price at period start |
| `high`         | max         | Highest price in period |
| `low`          | min         | Lowest price in period |
| `close`        | last        | Closing price at period end |
| `volume`       | sum         | Total volume traded |
| `data_split`   | aggressive_test_label | Anti-leakage (see below) |
| indicators     | last        | Final indicator value |

**Implementation:** [data.py#L200-L240](training_service/data.py)

---

## Data Leakage Protection

### Critical Mechanisms

#### 1. **Train/Test Boundary Protection**

When using a `data_split` column (e.g., "train"/"test" labels), the service implements **aggressive boundary handling**:

**Problem:** At row T (train), the target uses row T+1. If T+1 is "test", we leak test data into training labels.

**Solution:**
```python
# During resampling, if ANY record in a bucket is 'test', mark entire bucket as 'test'
def aggressive_test_label(series):
    vals = set(series.astype(str).unique())
    if "test" in vals: return "test"
    return "train"

# After resampling, drop rows where current=train but future=test
future_split = df[split_col].shift(-lookforward)
is_leak = (df[split_col] == 'train') & (future_split == 'test')
df = df[~is_leak]  # Drop boundary rows
```

**Implementation:** [data.py#L220-L270](training_service/data.py)

#### 2. **Raw Price Column Removal**

Raw price columns (`open`, `high`, `low`, `close`, `vwap`) from ALL tickers are dropped from features to prevent overfitting and ensure models learn from **derived stationary features** (returns, technical indicators):

```python
# Drop primary raw prices: close, open, high, low, vwap
# Drop context raw prices: close_QQQ, open_MSFT, vwap_VIX
raw_price_cols = ["open", "high", "low", "close", "vwap"]
cols_to_drop = []
for col in df.columns:
    if col in raw_price_cols or any(col.startswith(f"{raw}_") for raw in raw_price_cols):
        cols_to_drop.append(col)
```

**Exception:** The primary `target_col` (e.g., "close") is kept for **reference only** (used to calculate direction for classifiers and reconstruct prices for metrics). It is explicitly dropped from X in trainer.py.

**Implementation:** [data.py#L270-L310](training_service/data.py)

#### 3. **Target Column Exclusion**

In trainer.py, the target column and metadata are explicitly excluded from feature matrix X:

```python
drop_cols = ["target", "ts", "symbol", "date", "source", "options", "target_col_shifted", target_col]
if split_col_name:
    drop_cols.append(split_col_name)

feature_cols = [c for c in df_numeric.columns if c not in drop_cols]
X = df[feature_cols]
y = df["target"]
```

**Implementation:** [trainer.py#L360-L380](training_service/trainer.py)

#### 4. **Lookforward Target Creation**

The target is created by shifting the target column **backwards** (or equivalently, future values forward):

```python
# Predict future_val which is N steps ahead
future_val = df[target_col].shift(-lookforward)

# Transform based on user selection
if target_transform == "log_return":
    df["target"] = np.log((future_val + 1e-9) / (df[target_col] + 1e-9))
elif target_transform == "pct_change":
    df["target"] = (future_val - df[target_col]) / (df[target_col] + 1e-9)
else:
    df["target"] = future_val  # Raw price (non-stationary, not recommended)
```

**Stationarity:** Using `log_return` or `pct_change` ensures the target is stationary (mean-reverting), which is critical for time-series ML.

**Implementation:** [data.py#L180-L200](training_service/data.py)

---

## Validation Methodology

### 1. **Walk-Forward Cross-Validation**

The service detects **multiple train/test blocks** in the `data_split` column and performs walk-forward CV:

**Detection Logic:**
```python
# Identify contiguous train/test chunks
block_ids = (split_vals != split_vals.shift()).cumsum()
blocks = split_vals.groupby(block_ids)

# Find Train -> Test pairs
folds = []
for bid in unique_blocks:
    idx = blocks.groups[bid]
    val = split_vals.loc[idx[0]]
    if val == 'train':
        current_train_idx = idx
    elif val == 'test' and current_train_idx is not None:
        folds.append((current_train_idx, idx))
```

**If multiple folds are found:**
- Train on each fold independently
- Evaluate on corresponding test set
- Aggregate metrics (mean MSE, accuracy) across folds
- Final model is trained on **all train data** (including orphan blocks)

**Example Timeline:**
```
[Train1] [Test1] [Train2] [Test2] [Train3] [Test3]
   │        │       │        │       │        │
   └────────┘       └────────┘       └────────┘
    Fold 1           Fold 2           Fold 3
```

**Implementation:** [trainer.py#L540-L610](training_service/trainer.py)

### 2. **Time-Based Split (Fallback)**

If no `data_split` column exists, the service performs an **80/20 time-based split**:

```python
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

**Critical:** This preserves temporal ordering (train comes before test), preventing look-ahead bias.

**Implementation:** [trainer.py#L600-L610](training_service/trainer.py)

---

## Feature Engineering & Selection

### 1. **Top-Down Pruning (P-Value Filtering)**

Before training, features are pruned based on **univariate F-test** significance:

**Process:**
1. Impute missing values with mean
2. Standardize features with StandardScaler
3. Calculate F-statistic and p-values using `f_regression`
4. Drop features with `p > threshold` (default: 0.05)
5. Log standardized beta coefficients (feature ranking)

**Safeguard:** If all features are pruned, revert to the single best feature (lowest p-value).

**Implementation:** [trainer.py#L420-L470](training_service/trainer.py)

### 2. **Parent Feature Inheritance**

When `parent_model_id` is provided (for multi-generational evolution), the service loads features from the parent model and **intersects** them with current features:

```python
if parent_model_id:
    parent_model = db.get_model(parent_model_id)
    parent_features = json.loads(parent_model['feature_cols'])
    feature_cols = [c for c in feature_cols if c in parent_features]
    log.info(f"Applied parent feature mask: {original_count} -> {len(feature_cols)} features")
```

**Use Case:** Grid search over regularization parameters while using a fixed feature set from a champion parent model.

**Implementation:** [trainer.py#L300-L320](training_service/trainer.py)

### 3. **Heterogeneous Scaling Pipeline**

Features are grouped by **statistical properties** and scaled appropriately:

| Feature Group       | Scaler          | Columns                                    |
|---------------------|-----------------|--------------------------------------------|
| **Regimes**         | OneHotEncoder (Linear) / Passthrough (Trees) | `regime_vix`, `regime_gmm` |
| **Robust**          | RobustScaler    | `volume`, `trade_count`, `pro_vol_*`       |
| **Bounded**         | Passthrough     | `rsi`, `ibs`, `aroon`, `stoch`, `bop`, `mfi` |
| **Standard**        | StandardScaler  | Returns, z-scores, `ma_dist`, momentum     |

**Rationale:**
- **Regimes:** OneHot for linear models (dummy variables), passthrough for trees (ordinal)
- **Robust:** Volume/counts have heavy outliers, RobustScaler resists them
- **Bounded:** Oscillators (RSI, IBS) are already scaled 0-1 or 0-100, no transformation needed
- **Standard:** Returns and z-scores are normally distributed, StandardScaler is optimal

**Implementation:** [trainer.py#L640-L720](training_service/trainer.py)

---

## Model Training Pipeline

### 1. **Supported Algorithms**

| Algorithm                | Type           | Library     | Key Use Case |
|--------------------------|----------------|-------------|--------------|
| `linear_regression`      | Regression     | sklearn     | Baseline, interpretable |
| `elasticnet_regression`  | Regression     | sklearn     | L1+L2 regularization, feature selection |
| `random_forest_regressor`| Regression     | sklearn     | Non-linear, robust to outliers |
| `xgboost_regressor`      | Regression     | XGBoost     | Gradient boosting, high performance |
| `lightgbm_regressor`     | Regression     | LightGBM    | Fast, efficient, handles large datasets |
| `logistic_classification`| Classification | sklearn     | Binary/multi-class classification |
| `random_forest_classifier`| Classification| sklearn     | Non-linear classification |
| `xgboost_classifier`     | Classification | XGBoost     | Gradient boosting classification |
| `lightgbm_classifier`    | Classification | LightGBM    | Efficient classification |

**Implementation:** [trainer.py#L84-L100](training_service/trainer.py)

### 2. **Training Workflow**

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Load Data (DuckDB Parquet) → Multi-ticker merge            │
│ 2. Resample to target timeframe (1m → 5m → 1h)                │
│ 3. Create target (log_return, pct_change, raw)                │
│ 4. Anti-leakage: Drop boundary rows, remove raw prices        │
│ 5. Feature pruning (p-value threshold)                        │
│ 6. Train/test split (walk-forward CV or time-based 80/20)     │
│ 7. Impute missing values (mean strategy)                      │
│ 8. Build preprocessing pipeline (heterogeneous scaling)        │
│ 9. Grid search (if grids provided) or single model training   │
│ 10. Evaluate on test set (comprehensive metrics)              │
│ 11. Feature importance (coefficients, tree, permutation, SHAP)│
│ 12. Save model artifact (.joblib) and metadata (PostgreSQL)   │
└────────────────────────────────────────────────────────────────┘
```

### 3. **Classification Direction Logic**

For classification algorithms, the target is converted to binary direction:

```python
if "classifier" in algorithm or "classification" in algorithm:
    # Binary: 1 = Future >= Current (Up/Same), 0 = Future < Current (Down)
    y = (df["target"] >= df[target_col]).astype(int)
```

**Use Case:** Predict if price will go up (1) or down (0) in the next N steps.

**Implementation:** [trainer.py#L480-L495](training_service/trainer.py)

---

## Metrics & Evaluation

### Regression Metrics

| Metric       | Formula | Interpretation |
|--------------|---------|----------------|
| **R²**       | 1 - (SS_res / SS_tot) | Variance explained (0 = no fit, 1 = perfect) |
| **MSE**      | mean((y_true - y_pred)²) | Mean Squared Error (lower is better) |
| **MAE**      | mean(\|y_true - y_pred\|) | Mean Absolute Error (units of target) |
| **RMSE**     | sqrt(MSE) | Root Mean Squared Error (penalizes large errors) |
| **RMSE_price** | RMSE in $ (reconstructed from log_return) | Interpretable price error |

**Price Reconstruction:**
For `log_return` targets, predictions are transformed back to price for interpretability:

```python
if target_transform == "log_return":
    rec_preds = base_prices * np.exp(preds)  # Pred Price = Base * e^(log_return)
    rec_true = base_prices * np.exp(y_test)
    rec_rmse = sqrt(mean((rec_true - rec_preds)²))
    metrics["rmse_price"] = rec_rmse  # e.g., $0.52 error per minute
```

**Implementation:** [trainer.py#L1250-L1280](training_service/trainer.py)

### Classification Metrics

| Metric              | Formula | Interpretation |
|---------------------|---------|----------------|
| **Accuracy**        | correct / total | Overall correctness |
| **Precision**       | TP / (TP + FP) | How many predicted positives are correct |
| **Recall**          | TP / (TP + FN) | How many actual positives were caught |
| **F1 Score**        | 2 * (precision * recall) / (precision + recall) | Harmonic mean of precision/recall |
| **Confusion Matrix**| [[TN, FP], [FN, TP]] | Error breakdown |

**Implementation:** [trainer.py#L1295-L1330](training_service/trainer.py)

### Feature Importance Analysis

The service computes **four types** of feature importance:

#### 1. **Linear Coefficients** (Linear Models)
- Direct coefficients from Linear/ElasticNet/Logistic models
- Interpretable: coefficient magnitude = feature impact
- Mapped back from OneHotEncoded features to original columns

#### 2. **Tree Importance** (Tree-Based Models)
- `feature_importances_` from RandomForest/XGBoost/LightGBM
- Based on Gini impurity reduction (RF) or gain (XGB/LGBM)
- Fast to compute, built into model

#### 3. **Permutation Importance** (Model-Agnostic)
- Measures performance drop when feature is shuffled
- Uses 5 repeats on 500 test samples for speed
- Scoring: `neg_mean_squared_error` (regression) or `accuracy` (classification)

#### 4. **SHAP Values** (Shapley Additive Explanations)
- **TreeExplainer:** Fast, exact for tree-based models
- **LinearExplainer:** For linear models
- **KernelExplainer:** Fallback for XGBoost errors (uses model.predict)
- Computes mean absolute SHAP value per feature
- **Regime-Conditional Importance:** SHAP values grouped by regime (VIX, GMM)

**Implementation:** [trainer.py#L1335-L1460](training_service/trainer.py)

**Example Output:**
```json
{
  "feature_details": {
    "macd_line_QQQ": {
      "tree_importance": 0.1245,
      "permutation_mean": 0.0032,
      "shap_mean_abs": 0.0089
    },
    "return_z_score_20_MSFT": {
      "coefficient": -0.0234,
      "permutation_mean": 0.0021,
      "shap_mean_abs": 0.0067
    }
  },
  "regime_importance": {
    "regime_vix": {
      "0": {"macd_line_QQQ": 0.0056, ...},  # Low VIX
      "1": {"macd_line_QQQ": 0.0123, ...}   # High VIX
    }
  }
}
```

---

## Grid Search & Hyperparameter Tuning

### 1. **Duplicate Detection**

Before training, the service computes a **fingerprint** for each hyperparameter combination and checks if it already exists:

**Fingerprint Components:**
```python
fingerprint = SHA256(
    symbol + algorithm + target_col + timeframe + 
    target_transform + data_options + json(params)
)
# NOTE: parent_model_id is NOT included (siblings can share parameters)
```

**Pre-Check Before GridSearchCV:**
```python
all_combinations = []
for alpha in alphas:
    for l1_ratio in l1_ratios:
        param_set = {'alpha': alpha, 'l1_ratio': l1_ratio}
        fp, _ = compute_fingerprint(...)
        if db.get_model_by_fingerprint(fp):
            existing_count += 1

if existing_count == len(all_combinations):
    log.info("All grid combinations already exist! Skipping grid search.")
```

**Double-Check at Save Time:** Even after GridSearchCV, each model is checked again before saving (race condition protection).

**Implementation:** [trainer.py#L860-L890](training_service/trainer.py)

### 2. **ElasticNet Grid Search**

**Default Grid:**
```python
alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]  # L2 penalty
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]  # L1 ratio
# Grid size: 8 × 7 = 56 combinations
```

**Adaptive Grid for Weak Signals:**
If max correlation with target < 0.05, the grid is automatically reduced to prevent over-regularization:

```python
if max_corr < 0.05:
    alphas = [0.0001, 0.001, 0.01, 0.05, 0.1]  # Use only low alphas
```

**OLS Fallback:** If ElasticNet produces all-zero coefficients, fallback to LinearRegression (OLS, no regularization).

**Implementation:** [trainer.py#L850-L930](training_service/trainer.py)

### 3. **XGBoost Grid Search**

**Default Grid:**
```python
max_depth = [3, 4, 5, 6, 7, 8, 9]
min_child_weight = [1, 3, 5, 10, 15, 20, 30]
reg_alpha = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]  # L1
reg_lambda = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]  # L2
learning_rate = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
# Grid size: 7 × 7 × 8 × 7 × 7 = 19,208 combinations (!)
```

**Parallel Training:** `n_jobs=-1` uses all CPU cores.

**Implementation:** [trainer.py#L932-L1010](training_service/trainer.py)

### 4. **LightGBM Grid Search**

**Default Grid:**
```python
num_leaves = [7, 15, 31, 63, 95, 127, 191]
min_data_in_leaf = [5, 10, 20, 40, 60, 80, 100]
lambda_l1 = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]  # L1
lambda_l2 = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]  # L2
learning_rate = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
# Grid size: 7 × 7 × 8 × 7 × 7 = 19,208 combinations
```

**Implementation:** [trainer.py#L1012-L1090](training_service/trainer.py)

### 5. **RandomForest Grid Search**

**Default Grid:**
```python
max_depth = [5, 10, 15, 20, 30, 50, None]
min_samples_split = [2, 5, 10, 20, 30, 50, 100]
min_samples_leaf = [1, 2, 4, 8, 12, 16, 20]
n_estimators = [25, 50, 75, 100, 150, 200, 300]
max_features = ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9, 1.0]  # Feature sampling
# Grid size: 7 × 7 × 7 × 7 × 7 = 16,807 combinations
```

**Implementation:** [trainer.py#L1092-L1170](training_service/trainer.py)

### 6. **Saving All Grid Models**

When `save_all_grid_models=True`, EVERY hyperparameter combination is saved as a separate model record with:
- Full metrics (R², RMSE, MAE, feature importance, SHAP)
- Shared `cohort_id` (grid search siblings)
- Unique `fingerprint` (includes specific params)
- Grid search rank (sorted by CV score)

**Use Case:** Analyze regularization path, compare performance across lambda/alpha, multi-generational evolution.

**Implementation:** [trainer.py#L110-L250](training_service/trainer.py)

---

## Model Relationships

### 1. **Cohort System**

**Cohort ID:** Shared identifier for models trained in the same grid search run.

**Purpose:**
- Group siblings (same algorithm, same data, different hyperparameters)
- Enable bulk operations (delete cohort, compare cohort performance)
- UI: Display cohort as clickable badge, modal shows all siblings

**Example:**
```
Cohort ID: abc123
├─ Model 1: alpha=0.001, l1_ratio=0.1, R²=0.72
├─ Model 2: alpha=0.001, l1_ratio=0.5, R²=0.78 ← Best
├─ Model 3: alpha=0.01, l1_ratio=0.1, R²=0.65
└─ Model 4: alpha=0.01, l1_ratio=0.5, R²=0.70
```

**Implementation:** [trainer.py#L110-L250](training_service/trainer.py)

### 2. **Parent-Child Evolution**

**Parent Model ID:** Identifies the model whose features were inherited.

**Multi-Generational Pruning:**
1. **Generation 0:** Train with all features, prune low p-value features → save as Model A
2. **Generation 1:** Load Model A's features, grid search over alpha/l1_ratio → save as Models B1-B50 (children of A)
3. **Generation 2:** Load best Model B's features, further prune or grid search → save as Models C1-C30 (grandchildren of A)

**Purpose:**
- Iterative feature selection (prune → validate → prune again)
- Regularization path exploration (fixed features, vary penalties)
- Reproducibility (trace feature evolution lineage)

**UI:** Parent model ID is displayed as a purple badge with click-to-highlight (scrolls to parent row, flashes background).

**Implementation:** [trainer.py#L300-L320](training_service/trainer.py)

---

## Configuration Options

### Environment Variables

| Variable                | Default                            | Description |
|-------------------------|------------------------------------|-------------|
| `FEATURES_PARQUET_DIR`  | `/app/data/features_parquet`       | Parquet files with engineered features |
| `MODELS_DIR`            | `/app/data/models`                 | Model artifacts (.joblib) |
| `METADATA_DB_PATH`      | `/app/data/duckdb/models.db`       | PostgreSQL connection for model metadata |
| `LOG_LEVEL`             | `INFO`                             | Logging verbosity |

**Implementation:** [config.py](training_service/config.py)

### Training Parameters

| Parameter              | Type   | Default | Description |
|------------------------|--------|---------|-------------|
| `symbol`               | str    | -       | Primary ticker (e.g., "AAPL,QQQ,VIX" for multi-ticker) |
| `algorithm`            | str    | -       | Algorithm name (e.g., "elasticnet_regression") |
| `target_col`           | str    | "close" | Price column to predict |
| `timeframe`            | str    | "1m"    | Resample to this timeframe (1m, 5m, 15m, 1h, 1d) |
| `target_transform`     | str    | "none"  | Target transformation: "log_return", "pct_change", "log", "none" |
| `data_options`         | str    | None    | Filter feature sets (JSON: `{"reference_symbols": ["QQQ"]}`) |
| `parent_model_id`      | str    | None    | Inherit features from this model |
| `feature_whitelist`    | list   | None    | Override parent features with explicit list |
| `p_value_threshold`    | float  | 0.05    | Prune features with p > threshold |
| `lookforward`          | int    | 1       | Predict N steps ahead |
| `save_all_grid_models` | bool   | False   | Save every grid combination (not just best) |

### Grid Search Parameters

**ElasticNet:**
- `alpha_grid`: List of L2 penalties (default: [0.0001, ..., 0.5])
- `l1_ratio_grid`: List of L1 ratios (default: [0.1, ..., 0.99])

**XGBoost:**
- `max_depth_grid`, `min_child_weight_grid`, `reg_alpha_grid`, `reg_lambda_grid`, `learning_rate_grid`

**LightGBM:**
- `num_leaves_grid`, `min_data_in_leaf_grid`, `lambda_l1_grid`, `lambda_l2_grid`, `lgbm_learning_rate_grid`

**RandomForest:**
- `rf_max_depth_grid`, `min_samples_split_grid`, `min_samples_leaf_grid`, `n_estimators_grid`, `max_features_grid`

---

## Best Practices

### 1. **Prevent Data Leakage**
- ✅ Always use `log_return` or `pct_change` for targets (stationary)
- ✅ Verify `data_split` column exists for walk-forward CV
- ✅ Check for raw price columns in feature importance (should not appear)
- ✅ Inspect `rmse_price` metric to ensure realistic price errors

### 2. **Feature Engineering**
- ✅ Use multi-ticker context (e.g., VIX for volatility, QQQ for market)
- ✅ Enable p-value pruning (default 0.05) to remove noise
- ✅ Review SHAP values to understand feature impact
- ✅ Use regime-conditional importance to detect market-dependent features

### 3. **Hyperparameter Tuning**
- ✅ Start with `save_all_grid_models=False` to find best model quickly
- ✅ Use `save_all_grid_models=True` for regularization path analysis
- ✅ Enable multi-generational pruning for iterative feature selection
- ✅ Monitor duplicate detection logs (avoid redundant training)

### 4. **Model Evaluation**
- ✅ Use walk-forward CV (multiple folds) for time-series robustness
- ✅ Compare R² across cohort siblings to find optimal regularization
- ✅ Check `rmse_price` for interpretability (e.g., $0.52 error = acceptable?)
- ✅ Inspect confusion matrix for classification (balanced precision/recall?)

### 5. **Production Deployment**
- ✅ Use parent_model_id to track feature evolution lineage
- ✅ Save comprehensive metrics for reproducibility
- ✅ Monitor logs for OLS fallback (indicates weak signal or over-regularization)
- ✅ Verify SHAP values computed successfully (fallback to KernelExplainer if needed)

---

## Troubleshooting

### Problem: ElasticNet produces all-zero coefficients
**Cause:** Over-regularization (alpha too high) or weak signal (features uncorrelated with target).

**Solutions:**
1. Check `max_corr` in logs (should be > 0.05)
2. Reduce `alpha_grid` to lower values: [0.0001, 0.001, 0.01]
3. Use `l1_ratio` < 0.5 (more L2, less L1 sparsity)
4. Service auto-fallback to OLS (LinearRegression) if detected

**Implementation:** [trainer.py#L870-L930](training_service/trainer.py)

### Problem: SHAP fails with XGBoost
**Error:** `"The passed model is not callable"`

**Solution:** Service automatically falls back to KernelExplainer:
```python
try:
    explainer = shap.TreeExplainer(estimator)
except:
    explainer = shap.KernelExplainer(estimator.predict, X_bg)
```

**Implementation:** [trainer.py#L1400-L1420](training_service/trainer.py)

### Problem: Data leakage suspected (MSE too low)
**Diagnosis:**
1. Check feature importance: Raw prices (`close`, `close_QQQ`) should NOT appear
2. Verify `data_split` boundary handling in logs: "Dropping X rows at Train->Test boundary"
3. Inspect `rmse_price`: Realistic? (e.g., $0.50 is OK, $0.01 is suspicious)
4. Review target creation: Future values should use `.shift(-lookforward)`

**Implementation:** [data.py#L250-L310](training_service/data.py), [trainer.py#L360-L380](training_service/trainer.py)

### Problem: GridSearchCV times out
**Cause:** Grid too large (e.g., 19,000+ combinations for XGBoost).

**Solutions:**
1. Reduce grid dimensions: Use fewer values per parameter
2. Enable duplicate detection (skips existing models)
3. Use `save_all_grid_models=False` (only save best model)
4. Monitor logs: "X/Y combinations already exist. Training remaining Z models."

**Implementation:** [trainer.py#L860-L890](trainer.py)

---

## Future Enhancements

### Planned Features
1. **Time Series Split:** Replace simple 80/20 with sklearn's TimeSeriesSplit for robust CV
2. **Feature Clustering:** Group correlated features to reduce multicollinearity
3. **Ensemble Models:** Combine predictions from multiple algorithms (voting/stacking)
4. **Online Learning:** Incremental model updates as new data arrives
5. **Hyperparameter Optimization:** Bayesian optimization (Optuna) instead of grid search
6. **Model Monitoring:** Drift detection, performance degradation alerts
7. **Explainability Dashboard:** Interactive SHAP waterfall plots, force plots

### Research Directions
- Test alternative targets: Volatility, bid-ask spread, order flow
- Incorporate NLP sentiment (news, Twitter) as features
- Deep learning: LSTM, Transformer for sequence modeling
- Reinforcement learning: Direct policy optimization for trading

---

## References

### Code Files
- [training_service/trainer.py](training_service/trainer.py) - Main training logic
- [training_service/data.py](training_service/data.py) - Data loading and preprocessing
- [training_service/main.py](training_service/main.py) - FastAPI endpoints
- [training_service/config.py](training_service/config.py) - Configuration
- [feature_service/pipeline.py](feature_service/pipeline.py) - Feature engineering

### Libraries Used
- **scikit-learn:** ML algorithms, preprocessing, metrics
- **XGBoost:** Gradient boosting (regression/classification)
- **LightGBM:** Efficient gradient boosting
- **SHAP:** Shapley values for model explainability
- **joblib:** Model serialization
- **DuckDB:** SQL analytics on Parquet files
- **PostgreSQL:** Model metadata storage

### Academic References
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions" (SHAP)
- Zou & Hastie (2005): "Regularization and variable selection via the elastic net"
- Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"
- Breiman (2001): "Random Forests"

---

**Last Updated:** 2025-01-XX  
**Maintained By:** Training Service Team  
**Questions?** See [README.md](README.md) for project overview or open an issue.
