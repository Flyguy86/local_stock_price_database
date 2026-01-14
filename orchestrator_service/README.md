# Orchestrator Service - Recursive Strategy Factory

## Overview

The Orchestrator Service automates the **Train → Prune → Simulate** evolution loop to find optimal trading models without manual intervention. It implements a recursive strategy factory that:

1. **Trains** models on engineered features (with internal grid search)
2. **Simulates** each model across all regime/threshold/z-score combinations
3. **Evaluates** results against Holy Grail criteria (auto-promote if met)
4. **Prunes** ineffective features based on importance
5. **Evolves** to next generation with reduced feature set
6. **Repeats** until features cannot be pruned further or max generations reached

### Key Insight: Multi-Dimensional Grid Search

The evolution process explores THREE orthogonal search spaces simultaneously:

1. **Feature Space (Evolution Loop)**: Progressive feature reduction across generations
2. **Hyperparameter Space (Training Service)**: Grid search over regularization (alpha, l1_ratio) and regime configs
3. **Strategy Space (Simulation Service)**: Grid search over thresholds, z-scores, and regime filters

Each evolution generation trains ~294 models internally and simulates each across ~140 strategy configurations, ensuring comprehensive optimization before moving to the next feature set.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR SERVICE (Port 8400)                                 │
│ - Evolution Engine (automated loop coordination)                 │
│ - Model Fingerprinting (deduplication via SHA-256)               │
│ - Priority Queue (parent SQN-based job ordering)                 │
│ - Holy Grail Criteria (configurable success thresholds)          │
└────────────────────────┬─────────────────────────────────────────┘
                         │ HTTP calls
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Training     │  │ Simulation   │  │ Feature      │
│ Service:8200 │  │ Service:8300 │  │ Service:8100 │
└──────────────┘  └──────────────┘  └──────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ PostgreSQL (Shared State)            │
│ - evolution_runs (top-level tracking)│
│ - evolution_log (lineage DAG)        │
│ - priority_jobs (simulation queue)   │
│ - model_fingerprints (deduplication) │
│ - promoted_models (Holy Grail wins)  │
│ - workers (heartbeat tracking)       │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ Priority Workers (Distributed)       │
│ - Claim highest-priority jobs        │
│ - Run simulations via HTTP           │
│ - Report results to PostgreSQL       │
│ - Scale horizontally across nodes    │
└──────────────────────────────────────┘
```

## Key Features

### 1. Model Fingerprinting (Deduplication)
- **Purpose**: Prevent retraining identical model configurations
- **Method**: SHA-256 hash of complete training configuration:
  ```python
  fingerprint = hash(
      features,           # Feature list
      hyperparams,        # Base algorithm config
      target_transform,   # log_return, pct_change, etc.
      symbol,             # AAPL, SPY, etc.
      alpha_grid,         # L2 regularization grid search values
      l1_ratio_grid,      # L1/L2 mixing ratio grid search values
      regime_configs      # Market regime filter configurations
  )
  ```
- **Benefit**: Skip expensive training when pruning leads back to a known configuration
- **Smart Detection**: Different grid search parameters produce different fingerprints, ensuring accurate deduplication
- **Example**: Gen 3 with features `[A, B, C, D]` matches Gen 1 → reuses existing model instead of retraining 294 models

### 2. Priority Queue
- **Ordering**: Jobs from higher-performing parents (higher SQN) run first
- **Rationale**: Good parents are more likely to produce good children
- **Implementation**: PostgreSQL `ORDER BY parent_sqn DESC, created_at ASC`
- **Atomicity**: `FOR UPDATE SKIP LOCKED` prevents race conditions

### 3. Holy Grail Criteria
Configurable success thresholds for model promotion:

| Metric | Default Range | Purpose |
|--------|---------------|---------|
| **SQN** | 3.0 - 5.0 | System Quality Number (statistical robustness) |
| **Profit Factor** | 2.0 - 4.0 | Gross Profit / Gross Loss |
| **Trade Count** | 200 - 10,000 | Ensure sufficient statistical sample |
| **Weekly Consistency** | < 0.5 | StdDev/Mean ratio (even trade distribution) |

### 4. Evolution Lineage Tracking
- **Full Ancestry**: Recursive queries trace model genealogy back to seed
- **Pruning History**: Record which features were removed and why
- **Reproducibility**: Link promoted models to exact feature sets and hyperparameters

## Quick Start

### Using Docker Compose

```bash
# 1. Build all services (first time only)
./build.sh

# 2. Start orchestrator + dependencies
docker-compose up -d postgres orchestrator priority_worker

# 3. Verify services are running
docker-compose ps

# 4. Access dashboard
open http://localhost:8400
```

### Manual Start (Development)

```bash
# 1. Install dependencies
pip install -r orchestrator_service/requirements.txt

# 2. Start PostgreSQL (or use docker-compose for just postgres)
docker-compose up -d postgres

# 3. Set environment variables
export POSTGRES_URL="postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory"
export TRAINING_URL="http://localhost:8200"
export SIMULATION_URL="http://localhost:8300"
export FEATURE_URL="http://localhost:8100"

# 4. Start orchestrator API
uvicorn orchestrator_service.main:app --host 0.0.0.0 --port 8400

# 5. (In another terminal) Start priority worker
python -m orchestrator_service.priority_worker
```

## API Endpoints

### Evolution Management

#### POST `/evolve`
Start a new evolution run.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "algorithm": "RandomForest",
  "target_col": "close",
  "target_transform": "log_return",
  "max_generations": 4,
  "timeframe": "1m",
  "data_options": "{\"train_window\": 30, \"test_window\": 5}",
  "seed_features": ["sma_20", "rsi_14", "macd_line"],
  "reference_symbols": ["SPY", "QQQ"],
  "thresholds": [0.0001, 0.0003, 0.0005],
  "regime_configs": [
    {"regime_gmm": [1]},
    {"regime_vix": [2, 3]}
  ],
  "sqn_min": 3.0,
  "sqn_max": 5.0,
  "profit_factor_min": 2.0,
  "profit_factor_max": 4.0,
  "trade_count_min": 200,
  "trade_count_max": 10000
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Evolution run started for AAPL",
  "max_generations": 4
}
```

#### GET `/runs`
List evolution runs (with optional status filter).

**Query Params:**
- `status`: Filter by status (PENDING, RUNNING, COMPLETED, FAILED, PROMOTED)
- `limit`: Max results (default 50)

#### GET `/runs/{run_id}`
Get evolution run details with full lineage.

#### GET `/runs/{run_id}/generations`
Get per-generation summary with best results for each epoch.

### Promoted Models

#### GET `/promoted`
List models that met Holy Grail criteria (sorted by SQN descending).

#### GET `/promoted/{promoted_id}`
Get full details of a promoted model including:
- Model configuration (features, hyperparameters)
- Simulation parameters (threshold, regime filter)
- Full lineage (ancestry back to seed)
- Complete simulation results

### Worker Queue

#### POST `/jobs/claim`
Claim the highest-priority pending job (used by workers).

#### POST `/jobs/{job_id}/complete`
Mark a job as completed with results (used by workers).

### Utilities

#### GET `/health`
Health check endpoint.

#### GET `/api/stats`
System statistics (runs, jobs, promoted models).

#### GET `/api/features/symbols`
Proxy to feature service to list available symbols.

#### GET `/api/features/columns?symbol={symbol}`
Get feature columns for a symbol from feature service.

## Evolution Loop Workflow - Complete End-to-End Process

### High-Level Flow

Each evolution run explores three orthogonal optimization dimensions:
1. **Feature Space**: Progressive feature reduction (evolution loop)
2. **Hyperparameter Space**: Grid search over alpha/l1_ratio/regimes (training)
3. **Strategy Space**: Grid search over thresholds/z-scores/regimes (simulation)

### Detailed Step-by-Step Process

#### **Initialization Phase**
1. User submits evolution request via dashboard (`POST /evolve`)
2. Orchestrator creates `evolution_run` record (status = PENDING)
3. If `seed_model_id` provided → load features from existing model
4. If `seed_features` provided → train initial model with these features
5. **Generation 0 begins**

---

#### **Generation N Loop (Repeats until stopping condition)**

**STEP A: Get Feature Importance from Current Model** 📊  
*Extract importance scores from the trained model's coefficients/SHAP values*

- Call training service: `GET /api/model/{model_id}/importance`
- Returns dict of `{feature_name: importance_value}`
- Importance calculation methods:
  - **ElasticNet/Lasso/Ridge**: `abs(coefficient)` from linear model weights
  - **RandomForest/XGBoost**: SHAP values (Shapley Additive Explanations)
  - **Linear models**: Standardized coefficients (scaled by feature variance)

**STEP B: Prune Low-Importance Features** ✂️

- Sort features by **absolute importance** (ascending order: lowest first)
- Calculate `num_to_prune = floor(num_features × prune_fraction)`
  - Default `prune_fraction = 0.25` → remove bottom 25%
- Remove bottom `num_to_prune` features
- **Stopping Checks** (if ANY condition true, loop exits):
  - ⛔ **All features have equal importance** → can't determine which to prune → **STOP**
  - ⛔ **Pruning would go below `min_features`** (default 5) → **STOP**
  - ⛔ **No features remaining after pruning** → **STOP**

**STEP C: Compute Fingerprint** 🔍

- Generate SHA-256 hash of **complete training configuration**:
  ```python
  hash_input = canonicalize({
      "features": sorted(remaining_features),
      "hyperparameters": hyperparameters,
      "target_transform": target_transform,
      "symbol": symbol,
      "target_col": target_col,
      "alpha_grid": sorted(alpha_grid),
      "l1_ratio_grid": sorted(l1_ratio_grid),
      "regime_configs": sorted_normalized(regime_configs)
  })
  fingerprint = sha256(hash_input).hexdigest()
  ```
- **If match found**: Reuse existing model_id (skip training) → go to Step E
- **Else**: Continue to Step D

**STEP D: Train Child Model with Pruned Features** 🏗️

- Call training service: `POST /api/train`
- Payload: pruned features, symbol, hyperparameters, alpha/l1_ratio grids
- **Training Service Internal Grid Search**:
  - Creates 7 × 7 × 6 = **294 model variations**
  - Uses `GridSearchCV(n_jobs=-1, cv=5)` → parallel CPU training
  - Picks best model by validation score
  - Returns `child_model_id`

**STEP E: Simulate Child Model** ⚙️  
*Test the pruned model with full strategy grid search*

- Queue 140 simulation jobs (4 thresholds × 5 z-scores × 7 regimes)
- Priority workers execute backtests in parallel
- Wait for completion, retrieve best result (highest SQN)
- **Holy Grail Check**:
  - If criteria met → **PROMOTE child & STOP** 🎯
  - Else continue to Step F

**STEP F: Record Lineage & Advance Generation** 📝

- Insert evolution lineage (parent→child, pruned features, SQN)
- Update evolution run state (generation++, best_sqn, best_model_id)
- **Child becomes current** for next iteration
- **Loop back to STEP A** with child as new current model

---

#### **Stopping Conditions**

The loop exits when ANY of the following occur:

| Condition | Reason | Results Saved? |
|-----------|--------|----------------|
| **Holy Grail Met** | Child model achieved target SQN, profit factor, trade count | ✅ Yes (promoted to `promoted_models` table) |
| **Max Generations** | Reached `max_generations` limit (e.g., 4) | ✅ Yes (best model recorded) |
| **No Pruning Possible** | All features have equal importance | ⚠️ Partial (current model not simulated) |
| **Min Features Reached** | Pruning would violate `min_features` constraint | ⚠️ Partial (current model not simulated) |
| **Training Error** | Training service failed to create model | ⚠️ Partial (previous generations saved) |
| **Simulation Error** | All simulation jobs failed | ⚠️ Partial (model trained but not evaluated) |

**Note**: With the new flow (importance → prune → train → simulate), if pruning fails, the current model has NOT been simulated. Only the child models (after successful pruning and training) get simulated.

---

### Example Evolution Run

**Request Configuration:**
- Symbol: `AAPL`
- Algorithm: `ElasticNet`
- Seed Features: `['sma_20', 'rsi_14', 'macd_line', 'atr_14', 'volume_sma_20']` (5 features)
- Max Generations: 3
- Prune Fraction: 0.25 (remove bottom 25%)
- Min Features: 2
- Holy Grail: SQN ≥ 3.0, Profit Factor ≥ 2.0, Trades ≥ 200

**Evolution Timeline:**

| Generation | Features | Training Grid | Simulations | Best SQN | Action |
|------------|----------|---------------|-------------|----------|--------|
| **0** | 5 | 294 models | 140 configs | 2.1 | Not promoted → prune |
| **1** | 4 (pruned `volume_sma_20`) | 294 models | 140 configs | 2.5 | Not promoted → prune |
| **2** | 3 (pruned `atr_14`) | 294 models | 140 configs | 3.2 | **PROMOTED** 🎯 |

**Total Work:**
- **882 models trained** (3 generations × 294 models)
- **420 simulations run** (3 generations × 140 configs)
- **Result**: Found optimal model with 3 features achieving SQN = 3.2

**Fingerprint Reuse Example:**
If Generation 3 tried to prune back to 4 features, the orchestrator would detect this fingerprint already exists (from Gen 1) and skip training, reusing the cached model.

---

### Grid Search Dimensions Explained

#### 1. Feature Space (Evolution Loop)
- **Dimension**: Number and selection of input features
- **Method**: Progressive pruning based on importance
- **Scope**: Across generations (4-8 iterations typical)

#### 2. Hyperparameter Space (Training Service)
- **Dimension**: Regularization strength and mixing ratio
- **Method**: GridSearchCV over alpha/l1_ratio
- **Scope**: Within each generation (294 models per generation)
- **Parameters**:
  - `alpha`: Controls total regularization strength (7 values)
  - `l1_ratio`: Mixes L1 (Lasso) vs L2 (Ridge) penalties (6 values)
  - `regime_configs`: Train on specific market conditions (7 configs)

#### 3. Strategy Space (Simulation Service)
- **Dimension**: Trading rules and filters
- **Method**: Exhaustive grid search over strategy parameters
- **Scope**: Per model (140 simulations per model)
- **Parameters**:
  - `threshold`: Minimum prediction confidence to trade (4 values)
  - `z_score_cutoff`: Outlier filter for predictions (5 values, including 0 = no filter)
  - `regime_filter`: Only trade in specific market conditions (7 configs)

**Total Search Space per Evolution Run:**
- Feature combinations: ~C(N, k) where N = seed features, k = min_features
- Hyperparameter combinations: 294 per generation
- Strategy combinations: 140 per model
- **Example**: 4 generations × 294 models × 140 strategies = ~165,000 total evaluations

---

### Parallelization Analysis & Optimization Opportunities

#### Current Parallel Execution

**✅ What's Already Parallel:**

1. **Training Grid Search (Within Step E)**
   - 294 models (7 regimes × 7 alphas × 6 l1_ratios)
   - Training service uses `GridSearchCV(n_jobs=-1)` → all CPU cores
   - **Execution**: 294 models in parallel threads
   - **Bottleneck**: CPU-bound (sklearn training)

2. **Simulation Grid Search (Within Step A)**
   - 140 simulations (4 thresholds × 5 z-scores × 7 regimes)
   - Queued to PostgreSQL `priority_jobs` table
   - Priority workers (`priority_worker_1`, `priority_worker_2`, ...) pull and process
   - **Execution**: Distributed across N workers
   - **Bottleneck**: I/O-bound (DuckDB reads, backtest computation)
   - **Scaling**: Add more worker containers → linear speedup

**⛔ What's Serial:**

1. **Generation Loop (Steps A-F)**
   - Each generation depends on the previous generation's results
   - Gen 1 needs Gen 0's feature importance to know what to prune
   - Gen 2 needs Gen 1's pruned model, etc.
   - **Cannot parallelize** due to data dependency chain

2. **Single Evolution Run at a Time**
   - Dashboard only starts one evolution run per user action
   - Multiple symbols could evolve simultaneously but independently
   - **Current**: No batch "evolve all symbols" feature

---

#### Optimization Opportunities

##### 🚀 **Immediate Wins** (Low Effort, High Impact)

**1. Add More Priority Workers**
```yaml
# docker-compose.yml
priority_worker_2:
  build:
    context: .
    dockerfile: Dockerfile.optimize
  container_name: priority_worker_2
  restart: unless-stopped
  command: python -m orchestrator_service.priority_worker
  environment:
    WORKER_ID: priority_worker_2
    POSTGRES_URL: postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory
    SIMULATION_URL: http://simulation_service:8300
  depends_on:
    - postgres
    - simulation_service
```

- **Benefit**: 2 workers = 2× simulation throughput
- **Linear scaling**: 4 workers = 4× throughput (up to CPU limits)
- **Cost**: Each worker needs 1 CPU core + 2GB RAM
- **Recommended**: N workers = number of CPU cores / 2

**2. Batch Evolution Across Symbols**
```python
# New endpoint: POST /evolve/batch
symbols = ["AAPL", "GOOGL", "MSFT", "NVDA"]
for symbol in symbols:
    background_tasks.add_task(engine.run_evolution, config_for_symbol)
```

- **Benefit**: 4 symbols evolving simultaneously (independent)
- **Current limitation**: Each run creates separate workers pool contention
- **Recommended**: Limit to 2-3 concurrent evolution runs to avoid resource starvation

##### ⚡ **Medium-Effort Optimizations**

**3. Pipeline Overlapping (Step A + Step E)**

Currently:
```
Gen 0: [Train] → [Simulate 140 jobs] → [Wait] → [Prune]
Gen 1:           [Wait]                 → [Train] → [Simulate 140 jobs]
```

Optimized:
```
Gen 0: [Train] → [Simulate 140 jobs concurrently with Gen 1 training]
Gen 1:                                  [Train] → [Simulate]
```

- **Implementation**: Don't block on simulations completing before training next model
- **Risk**: Gen 1 might not need to be trained if Gen 0 gets promoted
- **Benefit**: ~30% faster (overlaps waiting with computation)

**4. Simulation Result Streaming**

Currently:
```python
# Wait for ALL 140 simulations, then pick best
await wait_for_all_jobs()
best = max(results, key=lambda r: r['sqn'])
```

Optimized:
```python
# Check results as they arrive
while pending > 0:
    completed = await get_latest_completed()
    if any(meets_holy_grail(r) for r in completed):
        PROMOTE_IMMEDIATELY()
        cancel_remaining_jobs()
        break
```

- **Benefit**: Early termination on Holy Grail hit (saves ~70% of simulations)
- **Risk**: Might miss a slightly better configuration
- **Tradeoff**: Speed vs exhaustive search

**5. Fingerprint Cache Pre-check Before Training**

Currently:
```
[Get Importance] → [Prune] → [Compute Fingerprint] → [Check Cache] → Maybe Skip Training
```

Optimized:
```
[Get Importance] → [Prune] → [Compute Fingerprint] → [Check Cache]
   ↓ (if cache miss)
[Predict likely fingerprints for next 2 generations] → [Warm cache]
```

- **Benefit**: Detect cycles earlier (e.g., Gen 3 = Gen 1)
- **Implementation**: Simulate pruning trajectories
- **Savings**: Skip 1-2 unnecessary training rounds per run

##### 🔬 **Advanced Optimizations** (High Effort, Research Required)

**6. Adaptive Grid Search**

Instead of exhaustive 7×7×6 = 294 models:
```python
# Bayesian optimization or Hyperband
- Start with 10 random configs
- Identify promising regions
- Refine top 5 with tighter grids
- Total: ~50 models instead of 294
```

- **Benefit**: 6× faster training per generation
- **Risk**: Might miss global optimum
- **Research**: sklearn has `HalvingGridSearchCV` for this

**7. Multi-Generation Look-Ahead**

Currently: Prune 25% → train → evaluate → repeat

Optimized:
```python
# Predict multiple pruning paths in parallel
Gen 1a: Remove features [A, B]    → Train → Simulate
Gen 1b: Remove features [A, C]    → Train → Simulate
Gen 1c: Remove features [B, C]    → Train → Simulate
Pick best path based on simulation results
```

- **Benefit**: Explore feature space faster
- **Cost**: 3× training/simulation cost per generation
- **Tradeoff**: Breadth vs depth (genetic algorithm vs hill climbing)

**8. Simulation Approximation via Surrogate Model**

Instead of running full backtest on DuckDB:
```python
# Train a lightweight "simulation predictor"
surrogate = RandomForest(X=strategy_params, y=historical_sqn)
predicted_sqn = surrogate.predict(new_strategy_config)
# Only run actual simulation for top 20 predicted configs
```

- **Benefit**: 7× faster (20 sims instead of 140)
- **Risk**: Surrogate might be inaccurate
- **Implementation**: Requires 1000+ historical simulations to train surrogate

---

#### Bottleneck Analysis

**Current Timing Breakdown** (for 1 evolution run, 4 generations):

| Step | Time | Parallelized? | Bottleneck |
|------|------|---------------|------------|
| **Step E: Train Model** | 120s | ✅ Yes (n_jobs=-1) | CPU (294 models × 0.4s each) |
| **Step A: Queue Simulations** | 5s | ⛔ No | PostgreSQL inserts |
| **Step A: Run Simulations** | 600s | ✅ Yes (N workers) | DuckDB I/O + backtest logic |
| **Step A: Wait & Evaluate** | 10s | ⛔ No | Single-threaded result aggregation |
| **Step B: Get Importance** | 15s | ⛔ No | HTTP request + SHAP calculation |
| **Step C: Prune Features** | 1s | ⛔ No | In-memory sorting |
| **Step D: Fingerprint** | 0.1s | ⛔ No | SHA-256 hash |
| **Step F: Record Lineage** | 2s | ⛔ No | PostgreSQL insert |
| **Total per generation** | ~753s (12.5 min) | - | - |
| **4 generations** | ~50 minutes | - | - |

**Critical Path**: Step A (simulation grid search) is 80% of total time.

**Optimization Priority:**
1. 🥇 Add more priority workers (immediate 2-4× speedup)
2. 🥈 Early termination on Holy Grail hit (saves 70% when successful)
3. 🥉 Reduce simulation grid (e.g., 70 configs instead of 140 = 50% faster)

---

#### Scaling Recommendations

**Small Setup** (1-2 symbols, rapid iteration):
- 1 orchestrator
- 1 training service
- 1 simulation service
- 2 priority workers
- **Throughput**: 1 evolution run every ~30 minutes

**Medium Setup** (5-10 symbols, production):
- 1 orchestrator
- 2 training services (load balanced)
- 2 simulation services (load balanced)
- 8 priority workers (4 per simulation service)
- **Throughput**: 2-3 concurrent evolution runs

**Large Setup** (50+ symbols, research cluster):
- 1 orchestrator
- 4 training services (load balanced)
- 4 simulation services (load balanced)
- 32 priority workers (8 per simulation service)
- **Throughput**: 10+ concurrent evolution runs
- **Cost**: 16-core CPU, 64GB RAM, NVMe SSD for DuckDB

## Database Schema

### Tables

#### `model_fingerprints`
Deduplication lookup table. Stores complete training configuration for accurate cache hits.

```sql
fingerprint VARCHAR(64) PRIMARY KEY,  -- SHA-256 hash of full config
model_id VARCHAR(64) NOT NULL,         -- Training service model UUID
features_json JSONB NOT NULL,          -- Sorted feature list
hyperparameters_json JSONB NOT NULL,   -- Includes alpha_grid, l1_ratio_grid, regime_configs
target_transform VARCHAR(32),
symbol VARCHAR(16) NOT NULL
```

**Fingerprint Components:**
- Features: `["sma_20", "rsi_14", "macd_line"]`
- Base hyperparams: `{"algorithm": "ElasticNet"}`
- Grid search params:
  - `alpha_grid`: `[0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]`
  - `l1_ratio_grid`: `[0.1, 0.3, 0.5, 0.7, 0.9, 0.95]`
  - `regime_configs`: `[{"regime_vix": [0, 1, 2, 3]}, ...]`

**Cache Hit Behavior:**
- Same features + same grids → Reuse model (skip training) ✅
- Same features + different grids → Train new model ✅
- Normalization ensures order doesn't matter: `[10, 1, 0.1] == [0.1, 1, 10]`

#### `evolution_runs`
Top-level tracking for evolution jobs.

```sql
id VARCHAR(64) PRIMARY KEY,
seed_model_id VARCHAR(64),
symbol VARCHAR(16) NOT NULL,
max_generations INTEGER NOT NULL,
current_generation INTEGER NOT NULL,
status VARCHAR(16) NOT NULL,           -- PENDING/RUNNING/COMPLETED/STOPPED/FAILED
config JSONB NOT NULL,
best_sqn DOUBLE PRECISION,
best_model_id VARCHAR(64),
promoted BOOLEAN DEFAULT FALSE
```

#### `evolution_log`
Lineage DAG (parent → child relationships).

```sql
id VARCHAR(64) PRIMARY KEY,
run_id VARCHAR(64),
parent_model_id VARCHAR(64),
child_model_id VARCHAR(64),
generation INTEGER,
parent_sqn DOUBLE PRECISION,
pruned_features JSONB,                 -- Features removed this generation
remaining_features JSONB,              -- Features kept after pruning
pruning_reason VARCHAR(64)
```

#### `priority_jobs`
Simulation job queue with priority ordering.

```sql
id VARCHAR(64) PRIMARY KEY,
batch_id VARCHAR(64),
run_id VARCHAR(64),
model_id VARCHAR(64),
generation INTEGER,
parent_sqn DOUBLE PRECISION,           -- Higher = higher priority
status VARCHAR(16),                    -- PENDING/RUNNING/COMPLETED/FAILED
params JSONB,                          -- Simulation parameters
result JSONB,                          -- Simulation results
worker_id VARCHAR(64)
```

#### `promoted_models`
Holy Grail success records.

```sql
id VARCHAR(64) PRIMARY KEY,
model_id VARCHAR(64),
run_id VARCHAR(64),
generation INTEGER,
sqn DOUBLE PRECISION,
profit_factor DOUBLE PRECISION,
trade_count INTEGER,
weekly_consistency DOUBLE PRECISION,
ticker VARCHAR(16),
regime_config JSONB,
threshold DOUBLE PRECISION,
full_result JSONB
```

## Worker Architecture

### Priority Worker Process

The `priority_worker.py` module implements a distributed simulation executor:

1. **Poll**: Query PostgreSQL for highest-priority pending job
2. **Claim**: Atomically update job status to RUNNING (prevents duplicates)
3. **Execute**: Call simulation service HTTP API with job parameters
4. **Report**: Write results back to PostgreSQL
5. **Repeat**: Loop until stopped

### Scaling Workers

**Vertical Scaling** (single machine):
```bash
# Start 4 workers on the same host
for i in {1..4}; do
  WORKER_ID="worker_$i" python -m orchestrator_service.priority_worker &
done
```

**Horizontal Scaling** (distributed):
```bash
# On machine A
docker-compose up -d priority_worker

# On machine B (point to shared postgres)
export POSTGRES_URL="postgresql://orchestrator:secret@machine-a:5432/strategy_factory"
export SIMULATION_URL="http://simulation-cluster:8300"
python -m orchestrator_service.priority_worker
```

**Kubernetes** (future):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: priority-workers
spec:
  replicas: 10  # Scale to 10 parallel workers
  template:
    spec:
      containers:
      - name: worker
        image: stock_base:latest
        command: ["python", "-m", "orchestrator_service.priority_worker"]
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_URL` | `postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory` | PostgreSQL connection string |
| `TRAINING_URL` | `http://training:8200` | Training service endpoint |
| `SIMULATION_URL` | `http://simulation:8300` | Simulation service endpoint |
| `FEATURE_URL` | `http://feature_service:8100` | Feature service endpoint |
| `DEFAULT_MAX_GENERATIONS` | `4` | Default evolution depth |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `WORKER_ID` | `worker_{pid}` | Unique worker identifier |
| `POLL_INTERVAL` | `5` | Seconds between job polls |

## Monitoring

### Dashboard (Web UI)

Access at `http://localhost:8400`:

- **System Status**: Real-time metrics (runs, pending jobs, promoted models, active workers)
- **Start Evolution**: Web form to configure and launch evolution runs
- **Evolution Runs**: List with status, generation progress, best SQN
- **Generation Details**: Per-epoch breakdown with top results
- **Promoted Models**: Hall of fame with full lineage
- **Job Queue**: Pending/running/completed simulation jobs

### Logs

**Orchestrator Service:**
```bash
docker-compose logs -f orchestrator
```

**Priority Workers:**
```bash
docker-compose logs -f priority_worker
```

**PostgreSQL:**
```bash
docker-compose exec postgres psql -U orchestrator -d strategy_factory
```

## Troubleshooting

### "No jobs available"
- Check if evolution run is queued: `GET /runs`
- Verify simulation service is running: `docker-compose ps simulation_service`
- Check pending job count: `GET /jobs/pending`

### "Fingerprint collision" (unexpected reuse)
- Review fingerprint logic in `orchestrator_service/fingerprint.py`
- Ensure hyperparameters are being normalized consistently
- Check if symbol/transform are being included in hash

### "Model training timeout"
- Training service may be overloaded
- Check training logs: `docker-compose logs training_service`
- Increase HTTP client timeout in `evolution.py` (default 120s)

### "PostgreSQL connection refused"
- Ensure PostgreSQL container is healthy: `docker-compose ps postgres`
- Wait for postgres healthcheck: `docker-compose up -d postgres && docker-compose logs -f postgres`
- Verify connection string matches docker-compose settings

## Advanced Usage

### Custom Pruning Logic

To implement permutation-based pruning instead of importance-based:

1. Modify `evolution.py::_get_feature_importance()` to call a different training service endpoint
2. Update `evolution.py::_prune_features()` to filter on permutation importance
3. Adjust `pruning_reason` in lineage log

### Multi-Symbol Batch Evolution

To evolve multiple symbols in parallel:

```python
import asyncio
import httpx

async def batch_evolve(symbols):
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                "http://localhost:8400/evolve",
                json={
                    "symbol": sym,
                    "algorithm": "RandomForest",
                    "max_generations": 4
                }
            )
            for sym in symbols
        ]
        return await asyncio.gather(*tasks)

symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
asyncio.run(batch_evolve(symbols))
```

### Dynamic Holy Grail Criteria

To adjust thresholds per-symbol based on volatility:

```python
import httpx

def get_symbol_volatility(symbol):
    # Fetch ATR or realized vol from feature service
    pass

def evolve_with_adaptive_criteria(symbol):
    vol = get_symbol_volatility(symbol)
    
    # Higher volatility = relax SQN requirement
    sqn_min = 3.0 if vol < 0.02 else 2.5
    
    httpx.post(
        "http://localhost:8400/evolve",
        json={
            "symbol": symbol,
            "sqn_min": sqn_min,
            "profit_factor_min": 2.0,
            "trade_count_min": 200
        }
    )
```

## Related Documentation

- [Training Service](../training_service/README.md) - Model training workflows
- [Simulation Service](../simulation_service/README.md) - Backtesting engine
- [Feature Service](../feature_service/README.md) - Technical indicator pipeline
- [Architecture Plan](../.github/orchestrator-agent-plan.md) - Design document

## Contributing

When modifying the orchestrator service:

1. **New Database Tables**: Update `init.sql` AND add migration logic
2. **New API Endpoints**: Add to `main.py` AND update this README
3. **Worker Protocol Changes**: Maintain backward compatibility with existing workers
4. **Fingerprint Changes**: Bump fingerprint version to avoid collisions

**Testing Checklist:**
- [ ] Evolution run completes without errors
- [ ] Fingerprint deduplication prevents retraining
- [ ] Priority ordering works (high SQN parents first)
- [ ] Holy Grail criteria correctly filters models
- [ ] Lineage tracking shows full ancestry
- [ ] Workers claim and complete jobs atomically
