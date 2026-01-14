import asyncpg
import uuid
import json
import logging
import os
from datetime import datetime

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory")
log = logging.getLogger("optimization.db")

# Global connection pool
_pool = None

async def get_pool():
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(POSTGRES_URL, min_size=2, max_size=10)
    return _pool

async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None

async def ensure_tables():
    """Create optimization tables if they don't exist."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS optimization_jobs (
                id VARCHAR PRIMARY KEY,
                batch_id VARCHAR,
                status VARCHAR,
                params JSONB,
                result JSONB,
                worker_id VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                progress DOUBLE PRECISION
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS optimization_workers (
                id VARCHAR PRIMARY KEY,
                last_heartbeat TIMESTAMP,
                current_job_id VARCHAR,
                status VARCHAR
            )
        """)
        
        # Create indexes for performance
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_opt_jobs_status ON optimization_jobs(status)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_opt_jobs_batch ON optimization_jobs(batch_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_opt_jobs_created ON optimization_jobs(created_at)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_opt_jobs_created ON optimization_jobs(created_at)
        """)

async def create_jobs(batch_params_list):
    """Create a batch of optimization jobs."""
    await ensure_tables()
    batch_id = str(uuid.uuid4())[:8]
    now = datetime.now()
    
    if not batch_params_list:
        return batch_id

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            for params in batch_params_list:
                job_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO optimization_jobs 
                    (id, batch_id, status, params, result, worker_id, created_at, updated_at, progress)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, job_id, batch_id, "PENDING", json.dumps(params), None, None, now, now, 0.0)
        
    return batch_id

async def claim_job(worker_id):
    """
    Atomically claim a pending job using SELECT FOR UPDATE.
    """
    await ensure_tables()
    updated_at = datetime.now()
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Find and lock one pending job
            row = await conn.fetchrow("""
                SELECT id, params FROM optimization_jobs 
                WHERE status = 'PENDING' 
                ORDER BY created_at ASC 
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            """)
            
            if row:
                job_id = row['id']
                params_json = row['params']
                
                # Update to RUNNING
                await conn.execute("""
                    UPDATE optimization_jobs 
                    SET status = 'RUNNING', worker_id = $1, updated_at = $2
                    WHERE id = $3
                """, worker_id, updated_at, job_id)
                
                # Parse JSON string if needed
                if isinstance(params_json, str):
                    params = json.loads(params_json)
                else:
                    params = params_json
                
                return {"id": job_id, "params": params}
            
    return None

async def update_job_progress(job_id, progress):
    """Update job progress (0.0 to 1.0)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE optimization_jobs SET progress = $1, updated_at = $2 WHERE id = $3
        """, progress, datetime.now(), job_id)

async def complete_job(job_id, result_dict, status="COMPLETED"):
    """Mark job as completed with result."""
    updated_at = datetime.now()
    result_json = json.dumps(result_dict) if result_dict else None
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE optimization_jobs 
            SET status = $1, result = $2, updated_at = $3, progress = 1.0
            WHERE id = $4
        """, status, result_json, updated_at, job_id)

async def worker_heartbeat(worker_id, current_job_id=None):
    """Record worker heartbeat."""
    await ensure_tables()
    now = datetime.now()
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Upsert worker
        await conn.execute("""
            INSERT INTO optimization_workers (id, last_heartbeat, current_job_id, status)
            VALUES ($1, $2, $3, 'ACTIVE')
            ON CONFLICT (id) DO UPDATE SET
                last_heartbeat = $2,
                current_job_id = $3,
                status = 'ACTIVE'
        """, worker_id, now, current_job_id)

async def get_active_workers():
    """Get list of active workers with their current jobs."""
    await ensure_tables()
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch("""
                SELECT w.id, w.last_heartbeat, w.current_job_id, j.params, j.progress
                FROM optimization_workers w
                LEFT JOIN optimization_jobs j ON w.current_job_id = j.id
                WHERE w.last_heartbeat > NOW() - INTERVAL '30 seconds'
            """)
            
            workers = []
            for r in rows:
                params = r['params']
                if isinstance(params, str):
                    params = json.loads(params)
                    
                workers.append({
                    "id": r['id'],
                    "last_heartbeat": str(r['last_heartbeat']),
                    "current_job_id": r['current_job_id'],
                    "job_params": params if params else None,
                    "progress": r['progress'] if r['progress'] else 0.0
                })
            return workers
        except Exception as e:
            log.warning(f"Error reading workers: {e}")
            return []

async def get_dashboard_stats():
    """Get optimization dashboard statistics."""
    await ensure_tables()
    
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from simulation_service.core import get_available_models
        models_list = get_available_models()
        model_name_lookup = {m['id']: m['name'] for m in models_list}
    except Exception as e:
        log.warning(f"Could not load model names: {e}")
        model_name_lookup = {}
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            # Status counts
            status_rows = await conn.fetch("""
                SELECT status, COUNT(*) as cnt FROM optimization_jobs GROUP BY status
            """)
            status_counts = {r['status']: r['cnt'] for r in status_rows}
            
            # Recent completed
            recent_rows = await conn.fetch("""
                SELECT id, batch_id, params, result, updated_at 
                FROM optimization_jobs 
                WHERE status = 'COMPLETED' 
                ORDER BY updated_at DESC 
                LIMIT 20
            """)
            
            recent_completed = []
            for r in recent_rows:
                params = r['params'] if not isinstance(r['params'], str) else json.loads(r['params'])
                result = r['result'] if not isinstance(r['result'], str) else json.loads(r['result'])
                recent_completed.append({
                    "id": r['id'],
                    "batch": r['batch_id'],
                    "params": params,
                    "result": result,
                    "ts": r['updated_at']
                })
            
            # Leaderboard
            leaderboard = []
            try:
                all_completed = await conn.fetch("""
                    SELECT params, result FROM optimization_jobs WHERE status = 'COMPLETED'
                """)
                
                for row in all_completed:
                    p = row['params'] if not isinstance(row['params'], str) else json.loads(row['params'])
                    r = row['result'] if not isinstance(row['result'], str) else json.loads(row['result'])
                    
                    if not r or "error" in r:
                        continue
                        
                    metric = r.get("strategy_return_pct", -999)
                    hit_rate = r.get("hit_rate_pct", 0.0)
                    sqn = r.get("sqn", 0.0)
                    
                    # Build simulation methods summary
                    sim_methods = []
                    sim_methods.append(f"Initial Capital: ${p.get('initial_cash', 10000):,.2f}")
                    sim_methods.append(f"Prediction Threshold: {p.get('min_prediction_threshold', 0.0):.4f}")
                    
                    if r.get("slippage_enabled", True):
                        bars = r.get("slippage_bars", 4)
                        sim_methods.append(f"SLIPPAGE: {bars}-bar execution delay")
                    else:
                        sim_methods.append("SLIPPAGE: DISABLED")
                    
                    total_fees = r.get("total_fees", 0.0)
                    avg_fee = r.get("avg_fee_per_trade", 0.02)
                    sim_methods.append(f"TRANSACTION COSTS: ${avg_fee:.2f}/trade, Total: ${total_fees:.2f}")
                    
                    if p.get("use_bot", False):
                        sim_methods.append("TRADING BOT: Enabled")
                    
                    regime_col = p.get("regime_col")
                    if regime_col:
                        allowed = p.get("allowed_regimes", [])
                        sim_methods.append(f"REGIME: {regime_col} in {allowed}")
                    
                    if p.get("enable_z_score_check", False):
                        sim_methods.append("Z-SCORE: Enabled")
                    
                    if p.get("volatility_normalization", False):
                        sim_methods.append("VOL NORM: Enabled")
                    
                    model_id = p.get("model_id", "")
                    model_display_name = model_name_lookup.get(model_id, model_id)
                    
                    leaderboard.append({
                        "ticker": p.get("ticker"),
                        "model": model_display_name,
                        "model_id": model_id,
                        "return": metric,
                        "hit_rate": hit_rate,
                        "trades": r.get("total_trades"),
                        "sqn": sqn,
                        "expectancy": r.get("expectancy", 0.0),
                        "profit_factor": r.get("profit_factor", 0.0),
                        "threshold": p.get("min_prediction_threshold", 0.0),
                        "z_score": p.get("enable_z_score_check", False),
                        "vol_norm": p.get("volatility_normalization", False),
                        "use_bot": p.get("use_bot", False),
                        "regime_col": p.get("regime_col", "None"),
                        "allowed_regimes": ",".join(map(str, p.get("allowed_regimes", []))) if p.get("allowed_regimes") else "All",
                        "initial_cash": p.get("initial_cash", 10000),
                        "full_params": p,
                        "sim_methods": "\n".join(sim_methods)
                    })
                    
                leaderboard.sort(key=lambda x: (x["sqn"], x["return"]), reverse=True)
                
            except Exception as e:
                log.error(f"Leaderboard error: {e}")
                leaderboard = []

            workers = await get_active_workers()

            return {
                "counts": status_counts,
                "recent": recent_completed,
                "leaderboard": leaderboard,
                "workers": workers
            }
        except Exception as e:
            log.error(f"Error getting dashboard stats: {e}")
            return {"counts": {}, "recent": [], "leaderboard": [], "workers": []}
