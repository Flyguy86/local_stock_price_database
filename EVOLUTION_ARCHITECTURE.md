# Evolution Architecture: Two-Phase Approach

## Overview
The evolution engine now uses a two-phase approach to separate model training from simulation testing.

## Phase 1: Training & Pruning
**Goal**: Build all model variants through progressive feature pruning

1. **Initialize**: Train or load initial model with all features
2. **For each generation** (up to `max_generations`):
   - Get feature importance from current model
   - Prune bottom X% of features (configurable via `prune_fraction`)
   - Check fingerprint cache for existing model
   - Train child model with reduced features (or reuse cached)
   - Record model in `trained_models` list
   - Advance to next generation

**Output**: List of all trained model IDs with metadata (generation, feature count)

## Phase 2: Simulation Grid Search
**Goal**: Test all models across full simulation parameter grid

1. **Queue simulations** for each model:
   - Thresholds (signal strength cutoffs)
   - Z-score thresholds (volatility filtering)
   - Regime configs (market conditions)
   - Simulation tickers (generalization testing)

2. **Wait for completion**: Poll until all simulations finish
   - Updates progress: `simulations_completed / simulations_total`
   - Timeout: 2 hours for all simulations

3. **Evaluate results**: Find best model + config combination
   - Check Holy Grail criteria (SQN, profit factor, trade count)
   - Record promotions if criteria met
   - Update run status and best results

## Benefits

### 1. Better Resource Utilization
- **Training**: Sequential, CPU-intensive, completes quickly
- **Simulations**: Parallel, can saturate worker capacity
- No idle time waiting for simulations while training is possible

### 2. Clearer Progress Tracking
- **Phase 1**: `models_trained / models_total`
- **Phase 2**: `simulations_completed / simulations_total`
- UI shows clear separation of work phases

### 3. Faster Execution
- Training doesn't block on simulation results
- Simulations run in massive parallel batches
- Overall wall-clock time reduced significantly

### 4. Simplified Logic
- No interleaved training/simulation coordination
- Easier to debug and maintain
- Clear separation of concerns

## Configuration

### Training Grid
- **ElasticNet**: `alpha_grid × l1_ratio_grid` combinations per model
- **Other algorithms**: 1 model per generation

### Simulation Grid
Per model: `tickers × thresholds × z_scores × regimes`

Example: `1 ticker × 4 thresholds × 5 z-scores × 7 regimes = 140 simulations/model`

### Total Work Estimate
- **Models**: `max_generations × grid_size_per_model`
- **Simulations**: `models_trained × simulations_per_model`

## Database Schema

### Evolution Runs
- `models_trained`, `models_total`: Training phase progress
- `simulations_completed`, `simulations_total`: Simulation phase progress
- `step_status`: Current phase and progress description

### Simulation Jobs
- Tagged with `generation` number for per-model grouping
- Priority queue ensures important models simulated first
- Fingerprint deduplication prevents redundant work

## Example Flow

```
Starting evolution run for AAPL
Max generations: 4
Work estimate: 168 model configs, 560 simulations

=============================================================
PHASE 1: TRAINING ALL MODELS
=============================================================
=== Generation 0 ===
Current model: seed-model-123
Current features count: 46
Step A: Got importance for 46 features
Step B: Pruning - 12 removed, 34 remaining
Step C: Training child model (34 features)
Step D: Training complete - child-model-456
Evolution lineage recorded: Gen 1

=== Generation 1 ===
Current model: child-model-456
Current features count: 34
...

=============================================================
PHASE 2: SIMULATING ALL MODELS (4 models)
=============================================================
Queueing simulations for model 1/4: seed-model-123 (Gen 0)
  Queued 140 simulations
Queueing simulations for model 2/4: child-model-456 (Gen 1)
  Queued 140 simulations
...
Total simulations queued: 560

Waiting for all simulations to complete...
Phase 2: Simulations 234/560 (42%) - 120s
...
All simulations complete (560 total)

Evaluating all simulation results...
Model 1/4 (Gen 0): Best SQN = 2.34
Model 2/4 (Gen 1): Best SQN = 3.87
...
Overall best result: Model child-model-789, SQN = 4.23

🎯 Model child-model-789 PROMOTED!
Evolution run finished: COMPLETED (reason: promoted)
```

## Migration Notes

### Breaking Changes
- Simulations no longer happen per-generation
- All simulations queued at once after training completes
- Progress tracking now two-phase instead of interleaved

### Backward Compatibility
- Existing database schema unchanged
- Simulation fingerprinting still works
- Promotion criteria unchanged

### Performance Impact
- Expect 30-50% reduction in total run time
- Worker utilization significantly improved
- Database query load reduced (batch polling vs per-generation)
