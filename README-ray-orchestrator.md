# Ray Orchestrator

**Distributed ML Training & Deployment for Trading Bots**

Ray Orchestrator leverages the [Ray](https://ray.io) ecosystem to provide industry-standard distributed training, hyperparameter tuning, and production model deployment. This replaces manual grid search and training orchestration with battle-tested tools used by OpenAI, Uber, and other companies at scale.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Ray Ecosystem Components](#ray-ecosystem-components)
5. [API Reference](#api-reference)
6. [Search Strategies](#search-strategies)
7. [Deduplication (Avoiding Double Work)](#deduplication-avoiding-double-work)
8. [Ensemble Deployment](#ensemble-deployment)
9. [Configuration](#configuration)
10. [Dashboard](#dashboard)
11. [Migration from Legacy Training](#migration-from-legacy-training)

---

## Overview

### Why Ray?

The previous training infrastructure required manual orchestration of:
- Grid search loops across ticker/parameter combinations
- Worker process management
- Checkpoint saving and fault recovery
- Model ensemble deployment

**Ray solves all of these problems:**

| Problem | Manual Approach | Ray Solution |
|---------|-----------------|--------------|
| Parallel training | ProcessPoolExecutor with manual scheduling | Ray Core: Automatic task distribution |
| Hyperparameter search | Nested loops, manual tracking | Ray Tune: Built-in schedulers (PBT, ASHA) |
| Early stopping | Manual metric checks | Ray Tune: Automatic trial termination |
| Checkpointing | joblib.dump with manual paths | Ray Train: Automatic checkpoints |
| Fault tolerance | Try/except, manual restart | Ray: Automatic task recovery |
| Ensemble inference | Sequential model calls | Ray Serve: Parallel async inference |
| Dashboard | Custom FastAPI endpoints | Ray Dashboard: Built-in monitoring |

### Key Features

- **Population-Based Training (PBT)**: Multi-generational evolution of hyperparameters
- **ASHA Scheduler**: Aggressive early stopping of poor trials
- **Bayesian Optimization**: Smart search with `skip_duplicate=True` fingerprinting
- **Multi-Ticker Optimization**: Find hyperparameters that work across ALL tickers
- **Voting Ensembles**: Deploy PBT survivors as production consensus models
- **Zero-Downtime Model Updates**: Swap models without taking the service offline
- **Deduplication**: SQLite fingerprint database prevents re-training identical configs
- **Experiment Resuming**: Continue crashed experiments without re-training completed trials
- **Real-Time Dashboard**: Monitor experiments via Ray Dashboard + custom UI

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Ray Orchestrator                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  FastAPI     │    │  Ray Tune    │    │    Ray Serve             │  │
│  │  Dashboard   │───▶│  (Training)  │───▶│    (Deployment)          │  │
│  │  :8100       │    │              │    │    :8000                  │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│         │                   │                        │                   │
│         ▼                   ▼                        ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Ray Dashboard│    │  Checkpoints │    │  Model Ensembles         │  │
│  │  :8265       │    │  /data/ray_* │    │  Voting (Hard/Soft)      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Data Layer                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  DuckDB (features.db)  │  Parquet (features_parquet)  │  PostgreSQL    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components

| File | Purpose |
|------|---------|
| [main.py](ray_orchestrator/main.py) | FastAPI app with REST endpoints |
| [tuner.py](ray_orchestrator/tuner.py) | Ray Tune orchestrator (Grid, PBT, ASHA, Bayesian) |
| [objectives.py](ray_orchestrator/objectives.py) | Training objective functions |
| [ensemble.py](ray_orchestrator/ensemble.py) | Ray Serve ensemble deployment |
| [data.py](ray_orchestrator/data.py) | Data loading with Ray Data |
| [fingerprint.py](ray_orchestrator/fingerprint.py) | SQLite fingerprint database for deduplication |
| [config.py](ray_orchestrator/config.py) | Configuration settings |

---

## Quick Start

### 1. Start Ray Orchestrator

```bash
# Start only the Ray stack (without other services)
docker-compose --profile ray up ray_orchestrator

# Or start everything
docker-compose --profile ray up
```

### 2. Access the Dashboard

- **Ray Orchestrator UI**: http://localhost:8100
- **Ray Dashboard**: http://localhost:8265

### 3. Run Your First PBT Search

```bash
# Via cURL
curl -X POST http://localhost:8100/search/pbt \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "elasticnet",
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "population_size": 20,
    "num_generations": 10,
    "target_transform": "log_return",
    "timeframe": "5m"
  }'
```

### 4. Deploy PBT Survivors as Ensemble

```bash
curl -X POST http://localhost:8100/ensemble/deploy-pbt-survivors \
  -H "Content-Type: application/json" \
  -d '{
    "ensemble_name": "trading_bot_v1",
    "experiment_name": "pbt_elasticnet_20260115_143022",
    "top_n": 5,
    "voting": "soft",
    "threshold": 0.7
  }'
```

### 5. Get Trading Signal

```bash
curl -X POST http://localhost:8100/ensemble/trading_bot_v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "macd_line_QQQ": 0.5,
      "return_z_score_20": -1.2,
      "rsi_14": 35.0,
      "volatility_20": 0.02
    }
  }'

# Response:
{
  "signal": "BUY",
  "confidence": 0.82,
  "num_models": 5,
  "voting_method": "soft"
}
```

---

## Ray Ecosystem Components

### Ray Core

Distributed task execution. Turns any Python function into a remote task:

```python
import ray

@ray.remote
def train_model(config):
    # Runs on any available worker
    model = fit_model(config)
    return model.score()

# Launch 100 parallel training jobs
futures = [train_model.remote(cfg) for cfg in configs]
results = ray.get(futures)  # Collects all results
```

### Ray Tune

Hyperparameter tuning with schedulers:

```python
from ray import tune

# Define search space
search_space = {
    "ticker": tune.grid_search(["AAPL", "MSFT", "GOOGL"]),
    "alpha": tune.loguniform(1e-5, 1.0),
    "l1_ratio": tune.uniform(0.0, 1.0),
}

# Run with PBT scheduler (multi-generational)
tuner = tune.Tuner(
    train_trading_model,
    param_space=search_space,
    tune_config=TuneConfig(
        scheduler=PopulationBasedTraining(
            perturbation_interval=5,
            hyperparam_mutations={"alpha": tune.loguniform(1e-5, 1.0)}
        ),
        num_samples=20
    )
)
results = tuner.fit()
```

### Ray Data

Streaming data without memory overflow:

```python
import ray.data

# Stream gigabytes of parquet files
ds = ray.data.read_parquet("/app/data/features_parquet/AAPL/*/*.parquet")

# Process in parallel
ds = ds.map(lambda row: preprocess(row))
ds = ds.filter(lambda row: row["volume"] > 1000)

# Feed to training (batched, doesn't load all into RAM)
for batch in ds.iter_batches(batch_size=1024):
    model.partial_fit(batch)
```

### Ray Serve

Production model deployment:

```python
from ray import serve

@serve.deployment
class TradingBot:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
    
    async def __call__(self, request):
        features = await request.json()
        return {"signal": self.model.predict(features)}

# Deploy with autoscaling
bot = TradingBot.bind("model.joblib")
serve.run(bot, route_prefix="/predict")
```

---

## API Reference

### Search Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search/grid` | Exhaustive grid search |
| POST | `/search/pbt` | Population-Based Training (supports `resume`) |
| POST | `/search/asha` | ASHA early stopping (supports `resume`, `bayesopt`) |
| POST | `/search/bayesian` | Bayesian Optimization with `skip_duplicate=True` |
| POST | `/search/multi-ticker-pbt` | Multi-ticker PBT (universal hyperparams) |

### Experiment Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/experiments` | List all experiments |
| GET | `/experiments/{name}` | Get experiment details |
| GET | `/experiments/{name}/top/{n}` | Get top N models |

### Ensemble Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ensemble/deploy` | Deploy ensemble from model paths |
| POST | `/ensemble/deploy-pbt-survivors` | Deploy top N from experiment |
| GET | `/ensemble` | List deployed ensembles |
| GET | `/ensemble/{name}` | Get ensemble status |
| POST | `/ensemble/{name}/predict` | Get prediction from ensemble |
| DELETE | `/ensemble/{name}` | Delete ensemble |

### System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | System status (Ray, resources) |
| GET | `/symbols` | Available ticker symbols |
| GET | `/algorithms` | Available algorithms |

### Fingerprint/Deduplication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/fingerprint/stats` | Get database statistics (total cached configs) |
| GET | `/fingerprint/{experiment}` | Get fingerprints for a specific experiment |
| DELETE | `/fingerprint/{experiment}` | Clear fingerprints for an experiment |
| DELETE | `/fingerprint` | Clear ALL fingerprints (use with caution) |

---

## Search Strategies

### Grid Search

**Use When:** You have a small, specific set of hyperparameters to test exhaustively.

```json
POST /search/grid
{
  "algorithm": "elasticnet",
  "tickers": ["AAPL"],
  "param_grid": {
    "alpha": [0.001, 0.01, 0.1],
    "l1_ratio": [0.3, 0.5, 0.7]
  }
}
```

**Total Trials:** 3 × 3 = 9 combinations

### Population-Based Training (PBT)

**Use When:** You want to find hyperparameters that adapt to changing market conditions. This is the "multi-generational evolution" approach.

```json
POST /search/pbt
{
  "algorithm": "xgboost_regressor",
  "tickers": ["AAPL"],
  "population_size": 20,
  "num_generations": 10
}
```

**How It Works:**
1. Start 20 random configurations (population)
2. After each generation, the bottom 25% are killed
3. They're replaced with mutated clones of the top 25%
4. Repeat for 10 generations
5. Survivors have hyperparameters that generalize well

### ASHA (Asynchronous Successive Halving)

**Use When:** You want to explore a large search space efficiently by stopping bad trials early.

```json
POST /search/asha
{
  "algorithm": "lightgbm_regressor",
  "tickers": ["AAPL", "MSFT"],
  "num_samples": 100,
  "search_alg": "optuna"
}
```

**How It Works:**
1. Start many trials with random hyperparameters
2. After a grace period, stop the worst performers
3. Let promising trials continue
4. Repeat until only the best remain

### Multi-Ticker PBT

**Use When:** You want hyperparameters that work across ALL tickers, not just optimized for one.

```json
POST /search/multi-ticker-pbt
{
  "algorithm": "elasticnet",
  "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
  "population_size": 30,
  "num_generations": 15
}
```

**How It Works:**
- Each trial trains on ALL tickers with the same hyperparameters
- Optimizes for **average Sharpe + minimum Sharpe** across tickers
- Prevents overfitting to a single ticker's patterns

### Bayesian Optimization

**Use When:** You want efficient, intelligent exploration with automatic deduplication.

```json
POST /search/bayesian
{
  "algorithm": "elasticnet",
  "tickers": ["AAPL", "MSFT"],
  "num_samples": 50,
  "resume": false
}
```

**How It Works:**
1. Uses a Gaussian Process surrogate model to predict promising regions
2. Automatically enables `skip_duplicate=True` to hash configurations
3. If a config has already been tested, it skips and suggests a different one
4. More sample-efficient than random search

---

## Deduplication (Avoiding Double Work)

One of the biggest advantages of using Ray is built-in deduplication. This ensures you never waste compute re-training identical configurations.

### 1. The `skip_duplicate` Flag (Ray Level)

When using Bayesian Optimization, Ray hashes the config dictionary:

```python
from ray.tune.search.bayesopt import BayesOptSearch

search_alg = BayesOptSearch(
    metric="sharpe_ratio", 
    mode="max", 
    skip_duplicate=True  # <-- This is your "fingerprint" toggle
)
```

**How the fingerprint works:**
- Nested dicts are flattened: `{"params": {"lr": 0.1}}` → `params/lr: 0.1`
- Floats are rounded to ~5 decimal places
- If two configs are identical up to this precision, Ray skips the duplicate

### 2. Fingerprint Database (Application Level)

For complex multi-generational setups (swapping datasets, tickers, timeframes), we maintain a SQLite fingerprint database:

```python
from ray_orchestrator.fingerprint import FingerprintDB

fp = FingerprintDB()

# Before training
if fp.exists(config):
    cached = fp.get_cached_result(config)
    if cached:
        tune.report(**cached)  # Report cached result, skip training
        return

# After training
fp.record(config, result)  # Store for future lookups
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/fingerprint/stats` | Get fingerprint database stats |
| GET | `/fingerprint/{experiment}` | Get fingerprints for experiment |
| DELETE | `/fingerprint/{experiment}` | Clear fingerprints for experiment |
| DELETE | `/fingerprint` | Clear ALL fingerprints |

### 3. Experiment Resuming

If a node crashes or you stop the search, set `resume=true` to continue:

```json
POST /search/pbt
{
  "algorithm": "elasticnet",
  "tickers": ["AAPL"],
  "name": "my_experiment_v1",
  "resume": true
}
```

**How it works:**
- Ray saves a JSON file of every result to the storage path
- When you restart with `resume=true`, Ray restores from that checkpoint
- Only unfinished/errored trials are re-run
- Completed trials report their cached results

### Why This Matters for Trading Bots

By ensuring no duplicate work, you can spend your **Compute Budget** on diversity:

| Without Deduplication | With Deduplication |
|-----------------------|--------------------|
| Train same XGBoost model twice | Each config is unique |
| Wasted compute on duplicates | More hyperparameter exploration |
| Slower convergence | Faster discovery of optimal configs |
| Less balanced ensemble | Well-balanced "committee" of diverse models |

---

## Ensemble Deployment

### Voting Strategies

**Hard Voting (Majority Rule):**
```
Model 1: BUY  │
Model 2: SELL │ → 3 BUY, 2 SELL → Signal: BUY (60% confidence)
Model 3: BUY  │
Model 4: BUY  │
Model 5: SELL │
```

**Soft Voting (Confidence Weighted):**
```
Model 1: +0.8 (0.9 confidence)  │
Model 2: -0.3 (0.6 confidence)  │ → Weighted avg: +0.42
Model 3: +0.5 (0.8 confidence)  │ → Signal: BUY if > threshold
Model 4: +0.2 (0.7 confidence)  │
Model 5: -0.1 (0.5 confidence)  │
```

### Deploy PBT Survivors

After a PBT experiment completes, deploy the survivors as an ensemble:

```bash
# 1. Check experiment results
curl http://localhost:8100/experiments/pbt_elasticnet_20260115/top/5

# 2. Deploy top 5 as ensemble
curl -X POST http://localhost:8100/ensemble/deploy-pbt-survivors \
  -d '{"ensemble_name": "bot_v1", "experiment_name": "pbt_elasticnet_20260115", "top_n": 5}'

# 3. Use the ensemble
curl -X POST http://localhost:8100/ensemble/bot_v1/predict \
  -d '{"features": {...}}'
```

### Dynamic Model Swapping

Update models without downtime:

```python
# The ensemble can be updated while serving
ensemble_manager.add_model_to_ensemble(
    ensemble_name="bot_v1",
    model_path="/app/data/models/ray/new_champion.joblib",
    model_id="champion_v2"
)

# Redeploy to include new model
ensemble_manager.deploy_ensemble(
    ensemble_name="bot_v1",
    model_paths=[...existing..., "/new_champion.joblib"]
)
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAY_ADDRESS` | `auto` | Ray cluster address |
| `RAY_DASHBOARD_PORT` | `8265` | Ray Dashboard port |
| `RAY_NAMESPACE` | `trading_bot` | Ray namespace |
| `RAY_NUM_CPUS_PER_TRIAL` | `1.0` | CPUs per training trial |
| `RAY_NUM_GPUS_PER_TRIAL` | `0.0` | GPUs per trial |
| `RAY_MAX_CONCURRENT_TRIALS` | `8` | Max parallel trials |
| `TUNE_METRIC` | `sharpe_ratio` | Optimization metric |
| `TUNE_MODE` | `max` | Optimization direction |
| `TUNE_PBT_POPULATION_SIZE` | `20` | PBT population |
| `TUNE_ASHA_GRACE_PERIOD` | `10` | ASHA min iterations |
| `TUNE_SKIP_DUPLICATE` | `true` | Enable fingerprint deduplication |
| `TUNE_FLOAT_PRECISION` | `5` | Decimal places for float hashing |
| `TUNE_USE_FINGERPRINT_DB` | `true` | Use SQLite fingerprint database |
| `TUNE_RESUME_ERRORED` | `true` | Resume only errored trials |
| `TUNE_RESUME_UNFINISHED` | `true` | Continue unfinished experiments |
| `SERVE_VOTING` | `soft` | Ensemble voting method |
| `SERVE_THRESHOLD` | `0.7` | Soft voting threshold |

### Resource Allocation

Tell Ray how much each algorithm needs:

```python
# In tuner.py
@ray.remote(num_cpus=1, num_gpus=0)
def train_elasticnet(config): ...

@ray.remote(num_cpus=2, num_gpus=0.5)
def train_xgboost(config): ...

@ray.remote(num_cpus=4, num_gpus=1)
def train_lstm(config): ...
```

Ray will pack as many trials as possible onto your hardware while respecting resource constraints.

---

## Dashboard

### Ray Orchestrator Dashboard (http://localhost:8100)

Custom dashboard for:
- Starting new searches (Grid, PBT, ASHA)
- Viewing experiment results
- Deploying ensembles
- Testing predictions

### Ray Dashboard (http://localhost:8265)

Built-in Ray dashboard for:
- Real-time trial progress
- Resource utilization
- Log streaming
- Actor debugging
- Memory profiling

---

## Migration from Legacy Training

### What Changes

| Legacy (training_service) | Ray Orchestrator |
|---------------------------|------------------|
| `train_model_task()` | `train_trading_model()` (Ray remote) |
| `GridSearchCV` loops | `ray.tune.Tuner` with schedulers |
| Manual checkpointing | Automatic `ray.train.Checkpoint` |
| `ProcessPoolExecutor` | Ray Core distributed tasks |
| Custom ensemble code | Ray Serve `VotingEnsemble` |
| PostgreSQL model registry | Ray experiment tracking + checkpoints |

### Gradual Migration

1. **Phase 1:** Run Ray Orchestrator alongside legacy services (different ports)
2. **Phase 2:** Route new experiments to Ray, legacy for production
3. **Phase 3:** Deploy Ray Serve ensembles for production inference
4. **Phase 4:** Deprecate legacy training_service

### Data Compatibility

Ray Orchestrator uses the same:
- Feature parquet files (`/app/data/features_parquet`)
- Model artifacts (`.joblib` format)
- PostgreSQL metadata (optional, for tracking)

---

## Example Workflow

### Complete Trading Bot Development

```bash
# 1. Start Ray Orchestrator
docker-compose --profile ray up -d ray_orchestrator

# 2. Run multi-ticker PBT to find universal hyperparameters
curl -X POST http://localhost:8100/search/multi-ticker-pbt \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "xgboost_regressor",
    "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA", "QQQ"],
    "population_size": 50,
    "num_generations": 20,
    "target_transform": "log_return",
    "timeframe": "5m"
  }'

# 3. Monitor progress on Ray Dashboard
open http://localhost:8265

# 4. Once complete, check top models
curl http://localhost:8100/experiments/multi_pbt_xgboost_20260115/top/10

# 5. Deploy top 5 as voting ensemble
curl -X POST http://localhost:8100/ensemble/deploy-pbt-survivors \
  -d '{
    "ensemble_name": "universal_bot",
    "experiment_name": "multi_pbt_xgboost_20260115",
    "top_n": 5,
    "voting": "soft",
    "threshold": 0.65
  }'

# 6. Integrate with trading system
# Endpoint: http://localhost:8000/universal_bot
```

---

## Troubleshooting

### Ray Not Connecting

```bash
# Check if Ray is running
docker logs ray_orchestrator | grep "Ray initialized"

# Verify cluster resources
curl http://localhost:8100/status | jq '.cluster_resources'
```

### Out of Memory

```python
# Use Ray Data for streaming (doesn't load all data)
ds = ray.data.read_parquet("*.parquet")
for batch in ds.iter_batches(batch_size=1024):
    # Process batch by batch
```

### Trial Failures

```bash
# Check Ray Dashboard for trial logs
open http://localhost:8265

# Or via API
curl http://localhost:8100/experiments/my_experiment | jq '.all_trials[] | select(.status == "failed")'
```

### Ensemble Not Predicting

```bash
# Verify models are loaded
curl http://localhost:8100/ensemble/my_ensemble

# Check model paths exist
ls -la /app/data/ray_checkpoints/*/checkpoint*/model.joblib
```

---

## References

- [Ray Documentation](https://docs.ray.io/)
- [Ray Tune User Guide](https://docs.ray.io/en/latest/tune/index.html)
- [Ray Serve User Guide](https://docs.ray.io/en/latest/serve/index.html)
- [Population-Based Training Paper](https://arxiv.org/abs/1711.09846)
- [ASHA Scheduler Paper](https://arxiv.org/abs/1810.05934)

---

**Last Updated:** January 2026  
**Maintained By:** Trading Bot Team
