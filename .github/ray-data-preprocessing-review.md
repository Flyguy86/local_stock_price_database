# Ray Data Preprocessing Review - Context Symbols & Feature Tracking

**Date**: 2026-01-19  
**Reviewer**: GitHub Copilot  
**Focus Areas**: Context symbol handling, TS alignment, feature importance logging, dropped columns tracking

---

## Executive Summary

### ✅ What's Working
1. **TS alignment architecture** is correctly designed with `pd.merge(..., on='ts', how='left')` 
2. **Context symbols are loaded** in the pipeline via `load_fold_data()`
3. **Feature importance** is already extracted and logged for grid search models
4. **Dropped columns** are tracked at preprocessing phase

### ❌ Critical Issues Found

1. **Context features are NOT being merged** - The `_join_context_features()` method is a stub that returns primary data unchanged
2. **Context data is loaded but discarded** - Context symbols are fetched but never actually joined
3. **No verification logging** for context symbol alignment
4. **Feature importance not logged for final/best models** (only for grid search members)
5. **Dropped columns logging lacks detail** about *why* columns were dropped

---

## Detailed Findings

### 1. Context Symbol Merge Issue (CRITICAL)

**Location**: [ray_orchestrator/streaming.py](ray_orchestrator/streaming.py#L375-L433)

**Current Code**:
```python
def _join_context_features(
    self,
    primary_ds: Dataset,
    context_ds: Dataset,
    primary_symbol: str
) -> Dataset:
    """Join primary symbol data with context symbols (QQQ, VIX)..."""
    
    def join_and_calculate_context(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # ... code to prepare batch ...
        
        # TODO: Implement proper Ray Data join operation
        # Current limitation: Ray Data's join is complex for time-series
        return batch  # ⚠️ RETURNS UNCHANGED - NO CONTEXT DATA ADDED
    
    log.info(f"Context feature joining for {primary_symbol} (stub - not yet implemented)")
    return primary_ds  # ⚠️ RETURNS PRIMARY DS WITHOUT CONTEXT
```

**Impact**:
- Context symbols (QQQ, VIX) are **loaded but never merged** into training data
- Models are trained **without market context features**
- Beta, relative strength, VIX regime features are **not available** to models
- This defeats the purpose of specifying `context_symbols` in requests

**Evidence**:
- [ray_orchestrator/streaming.py:L321-L322](ray_orchestrator/streaming.py#L321-L322) calls `_join_context_features()` but receives unchanged dataset
- [ray_orchestrator/streaming.py:L433](ray_orchestrator/streaming.py#L433) confirms: `return primary_ds` (no merge)

### 2. Alternative Context Merge (Partially Working)

**Location**: [training_service/data.py](training_service/data.py#L87-L165)

**Current Code**:
```python
def load_training_data(...):
    symbols = [s.strip() for s in symbol.split(",")]
    primary_symbol = symbols[0]
    context_symbols = symbols[1:]  # ✅ Extracts context from comma-separated string
    
    # ... load primary ...
    
    # 2. Load and Merge Context Tickers
    for ctx_sym in context_symbols:
        log.info(f"Loading context ticker: {ctx_sym}")
        ctx_df = _load_single(ctx_sym)
        
        # Merge via Inner Join to ensure time alignment ✅
        df = pd.merge(df, ctx_df, on="ts", how="inner")  # ✅ CORRECT
        log.info(f"Merged {ctx_sym}. Resulting rows: {len(df)}")
```

**Status**: ✅ This works, but only for **non-Ray training paths** (legacy training_service)

**Limitation**:
- Ray Data preprocessing (used for walk-forward) **does NOT use this code**
- Context symbols must be passed as comma-separated string (e.g., `"AAPL,QQQ,VIX"`)
- Not the same as the dedicated `context_symbols` parameter in Ray API

### 3. TS Alignment Verification

**Current State**:
- `pd.merge(..., on='ts', how='inner')` ensures exact timestamp alignment ✅
- Inner join prevents future data leakage ✅
- **Missing**: No logging to verify alignment worked (row count changes, NaN handling)

**Recommendation**: Add verification logging:
```python
pre_merge_rows = len(primary_df)
merged = pd.merge(primary_df, context_df, on='ts', how='inner')
post_merge_rows = len(merged)
dropped_rows = pre_merge_rows - post_merge_rows

log.info(f"Context merge [{context_symbol}]: {pre_merge_rows} → {post_merge_rows} rows "
         f"(dropped {dropped_rows} due to timestamp mismatch)")

# Verify no future leakage
assert merged['ts'].is_monotonic_increasing, "Timestamps out of order after merge!"
```

---

## Feature Importance & Dropped Columns Analysis

### 4. Feature Importance Logging

**Current Implementation**: [training_service/trainer.py:L180-L201](training_service/trainer.py#L180-L201)

**What's Working**:
```python
# Grid search models: ✅ Feature importance extracted
if hasattr(estimator, "feature_importances_"):
    feature_importance = dict(zip(feature_cols_used, [float(x) for x in imps]))
    metrics["feature_importance"] = dict(sorted(feature_importance.items(), 
                                                key=lambda x: abs(x[1]), reverse=True))
```

**What's Missing**:
1. **Final/Best model**: Feature importance not separately logged (only embedded in metrics JSON)
2. **Context feature highlighting**: No indication which features came from context symbols
3. **Top-N summary**: No console log of top 10 most important features

**Recommendation**:
```python
# After extracting feature importance
top_features = list(metrics["feature_importance"].items())[:10]
log.info(f"Top 10 Most Important Features:")
for rank, (feat, importance) in enumerate(top_features, 1):
    context_marker = " [CONTEXT]" if any(sym in feat for sym in context_symbols) else ""
    log.info(f"  {rank:2d}. {feat:30s} = {importance:8.5f}{context_marker}")

# Count context vs primary features
context_feats = [f for f in feature_cols_used if any(sym in f for sym in context_symbols)]
log.info(f"Context features in model: {len(context_feats)}/{len(feature_cols_used)} "
         f"({100*len(context_feats)/len(feature_cols_used):.1f}%)")
```

### 5. Dropped Columns Tracking

**Current Implementation**: [training_service/trainer.py:L421-L426](training_service/trainer.py#L421-L426)

**What's Working**:
```python
dropped_feature_cols = sorted(list(set(initial_feature_cols) - set(feature_cols_used)))
log.info(f"Column counts after preprocessing: {columns_remaining}/{columns_initial} "
         f"features (dropped {len(dropped_feature_cols)})")
```

**What's Missing**:
1. **No list of dropped columns** (only count)
2. **No reason tracking** (dropped for NaN? non-numeric? pruning?)
3. **Context columns not highlighted**

**Current Drop Reasons**:
- Line 411: All-NaN columns
- Line 441: Non-numeric columns
- Line 505: P-value pruning

**Recommendation**:
```python
# Track drops with reasons
drop_reasons = {
    'all_nan': [],
    'non_numeric': [],
    'p_value_pruning': []
}

# Phase 1: All-NaN
X_before = X.copy()
X = X.dropna(axis=1, how='all')
drop_reasons['all_nan'] = sorted(list(set(X_before.columns) - set(X.columns)))

# Phase 2: Non-numeric
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
drop_reasons['non_numeric'] = sorted(list(set(X.columns) - set(numeric_cols)))

# Phase 3: P-value pruning
if p_val_thresh < 1.0:
    removed_feats = sorted(list(set(feature_cols_used) - set(kept_feats)))
    drop_reasons['p_value_pruning'] = removed_feats

# Log comprehensive summary
log.info(f"=== Feature Dropping Summary ===")
log.info(f"Initial features: {columns_initial}")
for reason, cols in drop_reasons.items():
    if cols:
        context_count = sum(1 for c in cols if any(sym in c for sym in context_symbols))
        log.info(f"  Dropped ({reason}): {len(cols)} features "
                 f"({context_count} from context symbols)")
        if len(cols) <= 20:  # Only list if manageable
            log.info(f"    Columns: {cols}")
log.info(f"Final features: {columns_remaining}")
log.info(f"Retention rate: {100*columns_remaining/columns_initial:.1f}%")
```

---

## Recommendations Summary

### High Priority Fixes

1. **[CRITICAL] Implement context symbol merging in Ray Data**
   - Fix `_join_context_features()` to actually merge datasets
   - Use Ray Data's join operations or convert to pandas for merge then back to Dataset
   - Add verification logging

2. **Add context symbol alignment verification**
   - Log row count changes during merge
   - Verify timestamps remain sorted
   - Check for unexpected NaN introduction

3. **Enhanced feature importance logging**
   - Highlight context features in output
   - Show top-N summary after each training
   - Track context feature usage percentage

4. **Detailed dropped columns logging**
   - List dropped columns by category (NaN/non-numeric/pruned)
   - Separate context vs primary column drops
   - Include retention rate statistics

### Implementation Approach

**Option A: Fix Ray Data Join (Recommended)**
```python
def _join_context_features(self, primary_ds: Dataset, context_ds: Dataset, 
                          primary_symbol: str) -> Dataset:
    # Convert to pandas for complex join operations
    primary_pdf = primary_ds.to_pandas()
    context_pdf = context_ds.to_pandas()
    
    # Calculate context indicators
    context_with_indicators = self.calculate_indicators_gpu(
        batch=context_pdf,
        windows=[50, 200],
        drop_warmup=False
    )
    
    # Suffix context columns
    context_symbols_in_df = context_pdf['symbol'].unique()
    for ctx_sym in context_symbols_in_df:
        ctx_data = context_with_indicators[context_with_indicators['symbol'] == ctx_sym].copy()
        ctx_cols = [c for c in ctx_data.columns if c != 'ts']
        ctx_data = ctx_data.rename(columns={c: f"{c}_{ctx_sym}" for c in ctx_cols})
        
        # Merge on timestamp
        pre_rows = len(primary_pdf)
        primary_pdf = pd.merge(primary_pdf, ctx_data[['ts'] + list(ctx_data.columns)], 
                               on='ts', how='left')
        post_rows = len(primary_pdf)
        
        log.info(f"Merged context {ctx_sym}: {pre_rows} → {post_rows} rows")
    
    # Convert back to Ray Dataset
    return ray.data.from_pandas(primary_pdf)
```

**Option B: Pre-merge before creating Dataset**
- Load and merge in `load_fold_data()` before creating Ray Datasets
- Simpler but less scalable for very large datasets

---

## Testing Checklist

After implementing fixes:

- [ ] Verify context columns appear in saved parquet files
- [ ] Check training logs show context feature importance
- [ ] Confirm row counts match expected merges (no unexpected drops)
- [ ] Validate timestamps remain aligned (no future leakage)
- [ ] Test with multiple context symbols (QQQ + VIX)
- [ ] Ensure dropped columns report includes context features
- [ ] Compare model performance with/without context features

---

## Code Files to Modify

1. **[ray_orchestrator/streaming.py](ray_orchestrator/streaming.py)**
   - Lines 375-433: Fix `_join_context_features()` implementation
   - Lines 460-505: Add merge verification in `_calculate_context_features()`

2. **[training_service/trainer.py](training_service/trainer.py)**
   - Lines 180-201: Enhance feature importance logging
   - Lines 411-450: Add detailed dropped columns tracking with reasons

3. **[training_service/data.py](training_service/data.py)**
   - Lines 147-165: Add verification logging to existing merge logic

---

## Next Steps

1. Implement Option A (fix Ray Data join) in `streaming.py`
2. Add comprehensive logging as outlined above
3. Run test with AAPL + QQQ + VIX context symbols
4. Verify context features appear in:
   - Saved parquet files
   - Training logs (feature importance)
   - Model metrics
5. Document expected context feature counts in orchestrator-agent-plan.md
