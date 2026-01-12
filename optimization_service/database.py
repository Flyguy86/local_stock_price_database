import duckdb
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime

DB_PATH = Path("/app/data/duckdb/optimization.db")
log = logging.getLogger("optimization.db")

def ensure_tables():
    if not DB_PATH.parent.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with duckdb.connect(str(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id VARCHAR PRIMARY KEY,
                batch_id VARCHAR,
                status VARCHAR,
                params JSON,
                result JSON,
                worker_id VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                progress DOUBLE
            )
        """)
        
        # Add progress column if missing (migration)
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN progress DOUBLE")
        except:
            pass
            
        # Worker tracking table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workers (
                id VARCHAR PRIMARY KEY,
                last_heartbeat TIMESTAMP,
                current_job_id VARCHAR,
                status VARCHAR
            )
        """)

def create_jobs(batch_params_list):
    ensure_tables()
    batch_id = str(uuid.uuid4())[:8]
    now = datetime.now()
    
    data = []
    for params in batch_params_list:
        job_id = str(uuid.uuid4())
        data.append((
            job_id, 
            batch_id, 
            "PENDING", 
            json.dumps(params), 
            None, 
            None, 
            now, 
            now,
            0.0
        ))
    
    # Insert efficiently
    if not data:
        return batch_id

    with duckdb.connect(str(DB_PATH)) as conn:
        conn.executemany("""
            INSERT INTO jobs (id, batch_id, status, params, result, worker_id, created_at, updated_at, progress)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
    return batch_id

def claim_job(worker_id):
    """
    Atomically claim a pending job (simplistic lock via status update).
    DuckDB concurrency is limited for writes, but this is a low-frequency operation.
    """
    ensure_tables()
    updated_at = datetime.now()
    
    with duckdb.connect(str(DB_PATH)) as conn:
        # Find one pending
        job = conn.execute("""
            SELECT id, params FROM jobs 
            WHERE status = 'PENDING' 
            ORDER BY created_at ASC 
            LIMIT 1
        """).fetchone()
        
        if job:
            job_id, params_json = job
            # Update to RUNNING
            conn.execute("""
                UPDATE jobs 
                SET status = 'RUNNING', worker_id = ?, updated_at = ?
                WHERE id = ?
            """, [worker_id, updated_at, job_id])
            
            return {"id": job_id, "params": json.loads(params_json)}
            
    return None

def update_job_progress(job_id, progress):
    """Update job progress (0.0 to 1.0)."""
    with duckdb.connect(str(DB_PATH)) as conn:
        conn.execute("""
            UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?
        """, [progress, datetime.now(), job_id])

def complete_job(job_id, result_dict, status="COMPLETED"):
    updated_at = datetime.now()
    result_json = json.dumps(result_dict) if result_dict else None
    
    with duckdb.connect(str(DB_PATH)) as conn:
        conn.execute("""
            UPDATE jobs 
            SET status = ?, result = ?, updated_at = ?, progress = 1.0
            WHERE id = ?
        """, [status, result_json, updated_at, job_id])

def worker_heartbeat(worker_id, current_job_id=None):
    """Record worker heartbeat."""
    ensure_tables()
    now = datetime.now()
    
    with duckdb.connect(str(DB_PATH)) as conn:
        # Upsert worker
        conn.execute("""
            INSERT INTO workers (id, last_heartbeat, current_job_id, status)
            VALUES (?, ?, ?, 'ACTIVE')
            ON CONFLICT (id) DO UPDATE SET
                last_heartbeat = EXCLUDED.last_heartbeat,
                current_job_id = EXCLUDED.current_job_id,
                status = 'ACTIVE'
        """, [worker_id, now, current_job_id])

def _get_active_workers_inner(conn):
    """Internal helper that reuses an existing connection. REMOVED ensure_tables() call."""
    try:
        rows = conn.execute("""
            SELECT w.id, w.last_heartbeat, w.current_job_id, j.params, j.progress
            FROM workers w
            LEFT JOIN jobs j ON w.current_job_id = j.id
            WHERE w.last_heartbeat > NOW() - INTERVAL 30 SECONDS
        """).fetchall()
        
        workers = []
        for r in rows:
            workers.append({
                "id": r[0],
                "last_heartbeat": str(r[1]),
                "current_job_id": r[2],
                "job_params": json.loads(r[3]) if r[3] else None,
                "progress": r[4] if r[4] else 0.0
            })
        return workers
    except Exception as e:
        log.warning(f"Error reading workers (tables might not exist yet): {e}")
        return []

def get_dashboard_stats():
    ensure_tables()
    
    with duckdb.connect(str(DB_PATH)) as conn:
        try:
            status_counts = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
            recent_completed = conn.execute("""
                SELECT id, batch_id, params, result, updated_at 
                FROM jobs 
                WHERE status = 'COMPLETED' 
                ORDER BY updated_at DESC 
                LIMIT 20
            """).fetchall()
            
            leaderboard = []
            try:
                all_completed = conn.execute("SELECT params, result FROM jobs WHERE status = 'COMPLETED' LIMIT 200").fetchall()
                for p_raw, r_raw in all_completed:
                    if r_raw:
                        r = json.loads(r_raw)
                        p = json.loads(p_raw)
                        
                        if "error" in r:
                            continue
                            
                        metric = r.get("strategy_return_pct", -999)
                        hit_rate = r.get("hit_rate_pct", 0.0)
                        sqn = r.get("sqn", 0.0)
                        
                        # Build simulation methods summary
                        sim_methods = []
                        sim_methods.append(f"Initial Capital: ${p.get('initial_cash', 10000):,.2f}")
                        sim_methods.append(f"Prediction Threshold: {p.get('min_prediction_threshold', 0.0):.4f}")
                        
                        # Slippage
                        if r.get("slippage_enabled", True):
                            bars = r.get("slippage_bars", 4)
                            sim_methods.append(f"SLIPPAGE: {bars}-bar execution delay with midpoint pricing")
                            sim_methods.append(f"  → Orders fill at mean(open, close) of bar T+{bars}")
                        else:
                            sim_methods.append("SLIPPAGE: DISABLED (instant fills - UNREALISTIC)")
                        
                        # Transaction costs
                        total_fees = r.get("total_fees", 0.0)
                        avg_fee = r.get("avg_fee_per_trade", 0.02)
                        sim_methods.append(f"TRANSACTION COSTS: ${avg_fee:.2f} per trade")
                        sim_methods.append(f"  → Total fees paid: ${total_fees:.2f}")
                        
                        # Trading bot
                        if p.get("use_bot", False):
                            sim_methods.append("TRADING BOT: Enabled (ML-based signal confirmation)")
                        else:
                            sim_methods.append("TRADING BOT: Disabled (raw model predictions)")
                        
                        # Regime filter
                        regime_col = p.get("regime_col")
                        if regime_col:
                            allowed = p.get("allowed_regimes", [])
                            sim_methods.append(f"REGIME FILTER: {regime_col} in {allowed}")
                        else:
                            sim_methods.append("REGIME FILTER: Disabled (all market conditions)")
                        
                        # Z-score check
                        if p.get("enable_z_score_check", False):
                            sim_methods.append("Z-SCORE CHECK: Enabled (outlier removal >4σ)")
                        
                        # Volatility normalization
                        if p.get("volatility_normalization", False):
                            sim_methods.append("VOLATILITY NORM: Enabled (StandardScaler)")
                        
                        leaderboard.append({
                            "ticker": p.get("ticker"),
                            "model": p.get("model_id", "")[:12] + "...",
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
                            "sim_methods": "\n".join(sim_methods)  # NEW: Simulation methods text
                        })
                        
                leaderboard.sort(key=lambda x: (x["sqn"], x["return"]), reverse=True)
                leaderboard = leaderboard[:15]
            except Exception as e:
                log.error(f"Leaderboard error: {e}")
                leaderboard = []

            return {
                "counts": {s: c for s, c in status_counts},
                "recent": [
                    {"id": r[0], "batch": r[1], "params": json.loads(r[2]), "result": json.loads(r[3]), "ts": r[4]} 
                    for r in recent_completed
                ],
                "leaderboard": leaderboard,
                "workers": _get_active_workers_inner(conn)
            }
        except duckdb.CatalogException:
            return {"counts": {}, "recent": [], "leaderboard": [], "workers": []}
