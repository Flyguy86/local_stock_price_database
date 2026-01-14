# PostgreSQL Migration - Deployment Guide

## Quick Start

### 1. Build and Start Services

```bash
# Build base image with dependencies
docker compose build stock_base

# Start PostgreSQL
docker compose up postgres -d

# Wait for PostgreSQL to be ready
docker compose logs postgres | grep "ready to accept connections"

# Start training and simulation services
docker compose up training_service simulation_service -d
```

### 2. Verify Services

```bash
# Check all services are running
docker compose ps

# Check logs
docker compose logs -f training_service simulation_service
```

### 3. Run Tests

```bash
# Validate migration
python validate_migration.py

# Test end-to-end workflow
python test_end_to_end.py

# Test process pool
python test_process_pool.py
```

### 4. Migrate Existing Data (Optional)

If you have existing models in DuckDB:

```bash
python scripts/migrate_to_postgres.py
```

---

## Service Endpoints

### Training Service (port 8200)

- `GET /health` - Health check
- `GET /models` - List all models
- `GET /models/{id}` - Get model details
- `POST /train` - Submit training job
- `POST /train/batch` - Submit batch training (4 models)
- `POST /retrain/{model_id}` - Retrain existing model
- `DELETE /models/{id}` - Delete model
- `DELETE /models/all` - Delete all models

### Simulation Service (port 8300)

- `GET /health` - Health check
- `GET /api/config` - Get available models and tickers
- `GET /api/history?limit=50` - Get simulation history
- `GET /history/top?limit=15&offset=0` - Get top strategies by SQN
- `POST /api/simulate` - Run single simulation
- `POST /api/batch_simulate` - Run batch simulations
- `DELETE /history/all` - Delete all simulation history

---

## Testing Parallel Training

Submit multiple training requests to test CPU parallelism:

```bash
# Submit 8 training jobs in parallel
for i in {1..8}; do
  curl -X POST http://localhost:8200/train \
    -H "Content-Type: application/json" \
    -d '{
      "symbol": "RDDT",
      "algorithm": "RandomForest",
      "target_col": "close",
      "timeframe": "1m",
      "hyperparameters": {"n_estimators": 100},
      "target_transform": "log_return"
    }' &
done
wait

# Check logs for parallel execution (different PIDs)
docker compose logs training_service | grep "starting in process"

# Monitor CPU usage (should use all cores)
docker stats training_service
```

Expected output:
- Multiple "starting in process {PID}" logs with different PIDs
- CPU usage 400-800%+ (4-8+ cores utilized)

---

## Monitoring

### PostgreSQL Connection Pool

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U orchestrator -d strategy_factory

# Check active connections
SELECT count(*), state FROM pg_stat_activity 
WHERE datname='strategy_factory' 
GROUP BY state;

# Check table sizes
SELECT 
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

# Check model count
SELECT count(*) FROM models;

# Check simulation history count
SELECT count(*) FROM simulation_history;
```

### Training Service Metrics

```bash
# Check number of training jobs
docker compose exec postgres psql -U orchestrator -d strategy_factory -c \
  "SELECT status, count(*) FROM models GROUP BY status;"

# Check recent training jobs
docker compose exec postgres psql -U orchestrator -d strategy_factory -c \
  "SELECT id, symbol, algorithm, status, created_at 
   FROM models 
   ORDER BY created_at DESC 
   LIMIT 10;"
```

### Process Pool Status

```bash
# Check worker processes inside container
docker compose exec training_service ps aux | grep python

# Should see multiple Python processes if jobs are running
```

---

## Troubleshooting

### Issue: "Connection refused" to PostgreSQL

**Solution**: Ensure PostgreSQL is running and healthy

```bash
docker compose up postgres -d
docker compose logs postgres | tail -20
```

Wait for: `database system is ready to accept connections`

### Issue: "asyncpg.exceptions.UndefinedTableError"

**Solution**: Tables not created yet. Check startup logs:

```bash
docker compose logs training_service | grep "Tables ensured"
docker compose logs simulation_service | grep "Tables ensured"
```

If missing, restart services:

```bash
docker compose restart training_service simulation_service
```

### Issue: High memory usage

**Solution**: Adjust ProcessPoolExecutor max_tasks_per_child

In [training_service/main.py](training_service/main.py):

```python
_process_pool = ProcessPoolExecutor(
    max_workers=CPU_COUNT,
    max_tasks_per_child=5  # Reduce from 10
)
```

### Issue: Training jobs stuck in "preprocessing" status

**Solution**: Check worker process logs

```bash
docker compose logs training_service | grep -A 5 "error\|exception\|failed"

# Check if worker processes crashed
docker compose exec training_service ps aux | grep python
```

Restart if needed:

```bash
docker compose restart training_service
```

### Issue: Cannot find model files

**Solution**: Ensure data volume mounted correctly

```bash
# Check volume mount
docker compose exec training_service ls -la /app/data/models

# Check environment variables
docker compose exec training_service env | grep MODELS
```

---

## Performance Tuning

### Optimize PostgreSQL Connection Pool

In services' pg_db.py:

```python
# Reduce max connections if hitting limits
pool = await asyncpg.create_pool(
    POSTGRES_URL,
    min_size=2,
    max_size=5,  # Reduce from 10
    command_timeout=60
)
```

### Optimize Process Pool Workers

In training_service/main.py:

```python
# Use fewer workers if memory-constrained
CPU_COUNT = min(os.cpu_count() or 4, 4)  # Cap at 4 workers

# Recycle workers more aggressively
_process_pool = ProcessPoolExecutor(
    max_workers=CPU_COUNT,
    max_tasks_per_child=5  # Reduce from 10
)
```

### Optimize PostgreSQL Settings

In docker-compose.yml postgres service:

```yaml
environment:
  POSTGRES_MAX_CONNECTIONS: "50"  # Increase if needed
  POSTGRES_SHARED_BUFFERS: "256MB"
  POSTGRES_EFFECTIVE_CACHE_SIZE: "1GB"
```

---

## Migration Rollback

If you need to revert to DuckDB:

1. **Stop services**:
   ```bash
   docker compose down
   ```

2. **Restore old code** (git checkout)

3. **Keep existing data**:
   - DuckDB files: `data/duckdb/*.db`
   - Model files: `data/models/*.joblib`
   - Parquet data: `data/parquet/`, `data/features_parquet/`

4. **Restart with old setup**:
   ```bash
   docker compose up -d
   ```

---

## Success Criteria

✅ All services start without errors  
✅ PostgreSQL connection pool created  
✅ Tables created in strategy_factory database  
✅ Training jobs submit successfully  
✅ Multiple training jobs run in parallel (different PIDs)  
✅ CPU utilization increases with parallel jobs  
✅ Simulation history saves to PostgreSQL  
✅ Top strategies query returns results  
✅ Fingerprint deduplication works  

---

## Next Steps

1. **Load Testing**: Submit 50+ training jobs to test concurrent execution
2. **Monitoring**: Set up Grafana + Prometheus for metrics
3. **Backup**: Configure PostgreSQL backups (pg_dump)
4. **Scaling**: Add more training_service instances behind load balancer
5. **Distributed Training**: Migrate to Celery/Ray for network cluster
