# Training Service Migration to Async PostgreSQL + ProcessPoolExecutor

## Overview

The training service has been migrated from synchronous DuckDB to asynchronous PostgreSQL with multi-process parallel execution. This enables:

✅ **True CPU parallelism** - Bypass Python's GIL with ProcessPoolExecutor  
✅ **Multi-worker scaling** - Run multiple training jobs simultaneously across all CPU cores  
✅ **Database concurrency** - PostgreSQL handles concurrent access without file locks  
✅ **Future cluster support** - Process-based architecture can be extended to network cluster (Celery/Ray)

---

## Architecture Changes

### Before (Thread-based + DuckDB)
```
FastAPI (async) 
  ↓
ThreadPoolExecutor (GIL-limited)
  ↓
train_model_task (sync)
  ↓
DuckDB (file locks, single writer)
```

**Problems:**
- Python GIL prevents true parallelism
- DuckDB file locks block concurrent writes
- "PID 0" lock conflicts across services
- Limited to sequential training execution

### After (Process-based + PostgreSQL)
```
FastAPI (async)
  ↓
submit_training_task (async wrapper)
  ↓
ProcessPoolExecutor (no GIL!)
  ↓ (per-process)
train_model_task (sync)
  ↓ (per-process)
SyncDBWrapper (creates own asyncpg pool)
  ↓ (per-process)
PostgreSQL (concurrent reads/writes)
```

**Benefits:**
- Each worker runs in separate process (CPU cores)
- Each process has its own PostgreSQL connection pool
- No shared state between workers (process-safe)
- Scales to `CPU_COUNT` parallel training jobs

---

## Key Components

### 1. PostgreSQL Schema (`training_service/pg_db.py`)

```python
async def ensure_tables():
    # models table with comprehensive fingerprinting
    # features_log table for feature importance
    # simulation_history table for backtest results
```

**Fingerprint fields** (SHA-256 hash for deduplication):
- features, hyperparameters, target_transform
- symbol, target_col, timeframe
- train_window, test_window, context_symbols
- cv_folds, cv_strategy
- alpha_grid, l1_ratio_grid, regime_configs

**Connection pooling**: asyncpg (min=2, max=10 per service)

### 2. Process-Safe DB Wrapper (`training_service/sync_db_wrapper.py`)

```python
class SyncDBWrapper:
    def _get_pool(self):
        """Each process creates its own asyncpg connection pool."""
        # Uses per-process event loop
        # Reads POSTGRES_URL from environment
        
    def _execute_async(self, coro):
        """Executes async operation in process event loop."""
        loop = asyncio.get_event_loop() or asyncio.new_event_loop()
        return loop.run_until_complete(coro)
```

**Key insight**: Each worker process instantiates `SyncDBWrapper()` which creates its own PostgreSQL pool. No shared state.

### 3. Process Pool Executor (`training_service/main.py`)

```python
CPU_COUNT = os.cpu_count() or 4
_process_pool = ProcessPoolExecutor(
    max_workers=CPU_COUNT,
    max_tasks_per_child=10  # Prevent memory leaks
)

async def submit_training_task(...):
    """Async wrapper to submit to process pool."""
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(_process_pool, train_model_task, ...)
    except Exception as e:
        # Ensure failed status set in DB
        await db.update_model_status(training_id, status="failed", error=str(e))
```

**Lifecycle management**:
```python
@asynccontextmanager
async def lifespan(app):
    await ensure_tables()  # PostgreSQL
    yield
    _process_pool.shutdown(wait=True, cancel_futures=True)
    await close_pool()
```

### 4. Updated Endpoints

All training submission points now use `submit_training_task()`:

- **`/train`** - Single model training
- **`/train/batch`** - 4-model batch (open/close/high/low)
- **`/retrain/{model_id}`** - Retrain existing model
- **`/api/train_with_parent`** - Evolution chain training

```python
# OLD (thread-based):
background_tasks.add_task(train_model_task, ...)

# NEW (process-based):
asyncio.create_task(submit_training_task(...))
```

---

## Migration Checklist

### ✅ Completed

1. **PostgreSQL schema** designed with fingerprint fields
2. **Enhanced fingerprinting** with 12+ configuration parameters
3. **Migration script** created (`scripts/migrate_to_postgres.py`)
4. **training_service** migrated to async PostgreSQL
5. **All endpoints** converted to async
6. **SyncDBWrapper** rewritten for multi-process architecture
7. **ProcessPoolExecutor** added with CPU_COUNT workers
8. **submit_training_task()** wrapper for process pool submission
9. **All training endpoints** updated to use process pool

### 🔄 Next Steps

1. **Test deployment**:
   ```bash
   docker compose up training_service -d --build
   docker compose logs -f training_service
   ```

2. **Verify parallel execution**:
   - Submit multiple `/train` requests
   - Check logs for different PIDs per task
   - Monitor CPU utilization (should use all cores)

3. **Run test script**:
   ```bash
   python test_process_pool.py
   ```
   - Should see 8 tasks running across multiple processes
   - Different PIDs confirm parallel execution

4. **Migrate simulation_service**:
   - Update to use PostgreSQL for simulation_history
   - Similar async pattern as training_service

5. **Run migration script**:
   ```bash
   python scripts/migrate_to_postgres.py
   ```
   - Transfers existing DuckDB data to PostgreSQL
   - Handles ON CONFLICT for idempotent runs

---

## Performance Expectations

### Before
- **Parallelism**: 1 training job at a time (GIL-limited threads)
- **Concurrency**: DuckDB locks prevent concurrent writes
- **Scaling**: Limited to single-threaded performance

### After
- **Parallelism**: Up to `CPU_COUNT` simultaneous jobs (8-16+ on typical machines)
- **Concurrency**: PostgreSQL handles unlimited concurrent connections
- **Scaling**: Linear improvement with CPU cores

### Example Workload

**8-core machine, 16 training jobs:**

| Architecture | Execution Time | CPU Utilization |
|--------------|----------------|-----------------|
| Thread-based | 16 × single_job_time | 12-25% (GIL-limited) |
| Process-based | 2 × single_job_time | 95%+ (all cores) |

**Speed improvement**: ~8× faster for CPU-bound training workloads

---

## Configuration

### Environment Variables

```bash
# PostgreSQL connection
POSTGRES_URL=postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory

# Optional: Override CPU count
# CPU_COUNT=16  # Defaults to os.cpu_count()
```

### Docker Compose

```yaml
training_service:
  environment:
    POSTGRES_URL: postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory
  depends_on:
    postgres:
      condition: service_healthy
```

---

## Future Enhancements

### Distributed Cluster (Network Scaling)

Current process-based architecture prepares for:

**Option 1: Celery**
```python
# Replace ProcessPoolExecutor with Celery
@celery_app.task
def train_model_task(...):
    # Same code, runs on remote worker
    
# Submit to distributed queue
train_model_task.delay(training_id, symbol, ...)
```

**Option 2: Ray**
```python
# Use Ray for distributed execution
@ray.remote
def train_model_task(...):
    # Same code, runs on cluster node
    
# Submit to Ray cluster
ray.get([train_model_task.remote(...) for _ in range(100)])
```

**Requirements**:
- Redis/RabbitMQ for task queue
- Multiple worker machines
- Shared filesystem or S3 for model storage
- Load balancer for FastAPI endpoints

---

## Testing

### Unit Tests

```python
# Test process pool submission
async def test_submit_training_task():
    result = await submit_training_task(...)
    assert result["status"] == "completed"

# Test per-process connection pools
def test_sync_db_wrapper():
    db = SyncDBWrapper()
    db.update_model_status(...)
    # Each process should have own pool
```

### Integration Tests

```bash
# 1. Start services
docker compose up -d

# 2. Submit multiple training requests
for i in {1..8}; do
  curl -X POST http://localhost:8200/train \
    -H "Content-Type: application/json" \
    -d '{"symbol":"RDDT","algorithm":"RandomForest","target_col":"close"}' &
done

# 3. Check logs for parallel execution
docker compose logs training_service | grep "starting in process"
# Should see different PIDs (e.g., 45, 46, 47, 48...)

# 4. Monitor CPU usage
docker stats training_service
# Should show high CPU% (800%+ on 8-core machine = all cores used)
```

### Load Testing

```python
import asyncio
import aiohttp

async def submit_training(session, i):
    async with session.post(
        "http://localhost:8200/train",
        json={"symbol": "RDDT", "algorithm": "RandomForest"}
    ) as resp:
        return await resp.json()

async def load_test(num_requests=50):
    async with aiohttp.ClientSession() as session:
        tasks = [submit_training(session, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        print(f"Submitted {len(results)} training jobs")

# Expected: 50 jobs queued, processed in parallel batches of CPU_COUNT
asyncio.run(load_test())
```

---

## Troubleshooting

### Issue: "No asyncpg pool available"

**Cause**: Worker process failed to create connection pool

**Fix**: Check POSTGRES_URL environment variable
```bash
docker compose exec training_service env | grep POSTGRES
```

### Issue: All tasks run in same process

**Cause**: ProcessPoolExecutor not being used

**Fix**: Verify endpoints use `submit_training_task()` not `train_model_task()`
```bash
grep -n "submit_training_task" training_service/main.py
```

### Issue: High memory usage

**Cause**: Workers not recycling after max_tasks_per_child

**Fix**: Adjust max_tasks_per_child parameter
```python
ProcessPoolExecutor(max_workers=CPU_COUNT, max_tasks_per_child=5)
```

### Issue: Database connection errors

**Cause**: Connection pool exhausted

**Fix**: Increase PostgreSQL max_connections or reduce connection pool size
```python
# In pg_db.py
pool = await asyncpg.create_pool(min_size=2, max_size=5)  # Reduce from 10
```

---

## Monitoring

### Key Metrics

1. **Process count**: Should see CPU_COUNT worker processes
   ```bash
   docker compose exec training_service ps aux | grep train_model_task
   ```

2. **CPU utilization**: Should approach 100% per core
   ```bash
   docker stats training_service
   ```

3. **Database connections**: Should scale with active workers
   ```sql
   SELECT count(*) FROM pg_stat_activity WHERE datname='strategy_factory';
   ```

4. **Training throughput**: Jobs per minute
   ```sql
   SELECT 
     date_trunc('minute', created_at) as minute,
     count(*) as jobs_started
   FROM models
   GROUP BY minute
   ORDER BY minute DESC
   LIMIT 10;
   ```

### Logs to Watch

```bash
# Process pool creation
docker compose logs training_service | grep "Creating ProcessPoolExecutor"

# Task submissions
docker compose logs training_service | grep "starting in process"

# Completions
docker compose logs training_service | grep "completed in process"

# Errors
docker compose logs training_service | grep -i "error\|failed"
```

---

## Summary

The training service now uses **ProcessPoolExecutor with per-process PostgreSQL connection pools** for true CPU parallelism. This architecture:

- ✅ Bypasses Python's GIL for multi-core training
- ✅ Eliminates DuckDB lock conflicts  
- ✅ Scales to all available CPU cores
- ✅ Prepares for future distributed cluster (Celery/Ray)

**Next**: Test parallel execution, migrate simulation_service, run data migration script.
