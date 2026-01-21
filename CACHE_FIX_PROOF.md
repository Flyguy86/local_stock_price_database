# PROOF: Retrain Jobs Disable Fold Caching

## The Problem
The fold cache uses an **insufficient cache key**:
```python
# From ray_orchestrator/streaming.py line 330:
fold_dir = settings.data.walk_forward_folds_dir / symbol / f"fold_{fold.fold_id:03d}"
```

Cache key only includes:
- ✅ symbol (e.g., "GOOGL")
- ✅ fold_id (e.g., "001")

Cache key MISSING:
- ❌ train_months (7 vs 8)
- ❌ test_months (1 vs 2)  
- ❌ start_date / end_date
- ❌ step_months

**Data Leakage Risk**: Changing from 7-month to 8-month training would reuse the SAME cached `fold_001` with WRONG date ranges!

---

## The Solution
Retrain jobs now pass `use_cached_folds=False` to force fresh calculation.

### Evidence Chain

#### 1. Endpoint Code (ray_orchestrator/main.py line 2809)
```python
results = trainer.run_walk_forward_tuning(
    symbols=["{primary_ticker}"],
    start_date="{start_date}",
    end_date="{end_date}",
    train_months={train_months},
    test_months={test_months},
    step_months={step_months},
    ...
    disable_mlflow=True,
    use_cached_folds=False  # ← RETRAIN JOBS DISABLE CACHING
)
```

#### 2. Trainer Signature (ray_orchestrator/trainer.py line 576)
```python
def run_walk_forward_tuning(
    self,
    symbols: List[str],
    ...
    disable_mlflow: bool = False,
    use_cached_folds: bool = True,  # ← PARAMETER ACCEPTED
) -> tune.ResultGrid:
```

#### 3. Trainer Passes to Pipeline (ray_orchestrator/trainer.py line 637)
```python
for fold in self.preprocessor.create_walk_forward_pipeline(
    symbols=symbols,
    ...
    actor_pool_size=actor_pool_size,
    use_cached_folds=use_cached_folds  # ← PASSED TO PIPELINE
):
```

#### 4. Pipeline Logic (ray_orchestrator/streaming.py lines 1715-1720)
```python
# Check if cached fold exists
if use_cached_folds and len(symbols) == 1:  # ← CACHE CONDITIONAL
    cached_fold = self._try_load_cached_fold(fold, symbols[0])
    if cached_fold is not None:
        log.info(f"✓ Using cached fold {fold.fold_id} for {symbols[0]} (skipping recalculation)")
        yield cached_fold
        continue

# Cache miss or caching disabled - compute from scratch
log.info(f"Computing fold {fold.fold_id} from scratch...")  # ← FRESH CALCULATION
```

---

## How It Works

### Scenario A: Normal Tuning Job (use_cached_folds=True)
1. `generate_walk_forward_folds()` calculates dates
2. **Check**: Does `/app/data/walk_forward_folds/GOOGL/fold_001/` exist?
3. **YES** → Load cached train/test data ⚡ (fast, but wrong if params changed!)
4. **Result**: Uses 7-month cached data even if params say 8-month ❌

### Scenario B: Retrain Job (use_cached_folds=False)
1. `generate_walk_forward_folds()` calculates dates
2. **Check**: `use_cached_folds=False` → Skip cache check entirely
3. Process features fresh based on actual `train_months=8`
4. **Result**: Generates correct 8-month train/test splits ✅

---

## Verification

You can verify this works by:

1. **Check the submitted job's entrypoint code**:
   ```bash
   # The job submission creates Python code with this parameter
   grep -r "use_cached_folds=False" /app/data/ray_checkpoints/
   ```

2. **Monitor fold generation logs**:
   ```
   # With caching enabled, you'd see:
   ✓ Using cached fold 1 for GOOGL (skipping recalculation)
   
   # With caching disabled (retrains), you see:
   Computing fold 1 from scratch...
   ```

3. **Test with different parameters**:
   - Run retrain with 7-month training window
   - Change to 8-month training window
   - Check fold date ranges in logs - they should be DIFFERENT
   - Without this fix, they'd be the SAME (data leakage!)

---

## Conclusion

✅ **Retrain jobs set** `use_cached_folds=False`  
✅ **Pipeline skips** cache loading when flag is False  
✅ **Fresh folds computed** based on actual parameters  
✅ **No data leakage** from stale cached folds  

🎯 **Bottom line**: When you change training window from 7 to 8 months in a retrain job, you get FRESH fold calculations with the correct 8-month date ranges. The old 7-month cached data is IGNORED.
