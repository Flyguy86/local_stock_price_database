# Orchestrator Service - Recursive Strategy Factory

## Overview

The Orchestrator Service automates the **Train → Prune → Simulate** evolution loop to find optimal trading models without manual intervention. It implements a recursive strategy factory that:

1. **Trains** models on engineered features
2. **Prunes** ineffective features (importance ≤ 0)
3. **Simulates** strategies with regime filtering and thresholds
4. **Evolves** by repeating with pruned feature sets
5. **Promotes** models that meet "Holy Grail" success criteria

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
- **Method**: SHA-256 hash of `(features, hyperparams, target_transform, symbol)`
- **Benefit**: Skip expensive training when pruning leads back to a known configuration

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

## Evolution Loop Workflow

### Generation 0 (Seed)
1. Fetch seed features from feature service (or use provided list)
2. Train initial model
3. Queue simulations across all threshold/regime combinations
4. Wait for results, find best SQN
5. Evaluate against Holy Grail criteria → **If promoted, stop; else continue**

### Generation 1+ (Recursive)
1. Get feature importance from previous generation's model
2. Prune features with `importance <= 0`
3. Compute fingerprint from remaining features
4. Check if fingerprint exists in database:
   - **If yes**: Reuse existing model (skip training)
   - **If no**: Train new model, record fingerprint
5. Record evolution lineage (parent → child, pruned features)
6. Queue simulations with **priority = parent_sqn** (good parents → higher priority)
7. Wait for results, find best SQN
8. Evaluate Holy Grail criteria → **If promoted, stop; else continue**

### Stopping Conditions
- ✅ **Promoted**: Model meets Holy Grail criteria
- ✅ **Max Generations**: Reached `max_generations` limit
- ✅ **No Pruning**: All features have positive importance
- ✅ **All Pruned**: No features remaining after pruning

## Database Schema

### Tables

#### `model_fingerprints`
Deduplication lookup table.

```sql
fingerprint VARCHAR(64) PRIMARY KEY,  -- SHA-256 hash
model_id VARCHAR(64) NOT NULL,         -- Training service model UUID
features_json JSONB NOT NULL,          -- Sorted feature list
hyperparameters_json JSONB NOT NULL,
target_transform VARCHAR(32),
symbol VARCHAR(16) NOT NULL
```

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
