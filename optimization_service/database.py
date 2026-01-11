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
                status VARCHAR, -- PENDING, RUNNING, COMPLETED, FAILED
                params JSON,
                result JSON,
                worker_id VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

def create_jobs(batch_params_list):
    """
    Bulk insert jobs.
    batch_params_list: list of dicts containing simulation params
    """
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
            now
        ))
    
    # Insert efficiently
    if not data:
        return batch_id

    with duckdb.connect(str(DB_PATH)) as conn:
        conn.executemany("""
            INSERT INTO jobs (id, batch_id, status, params, result, worker_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

def complete_job(job_id, result_dict, status="COMPLETED"):
    updated_at = datetime.now()
    result_json = json.dumps(result_dict) if result_dict else None
    
    with duckdb.connect(str(DB_PATH)) as conn:
        conn.execute("""
            UPDATE jobs 
            SET status = ?, result = ?, updated_at = ?
            WHERE id = ?
        """, [status, result_json, updated_at, job_id])

def get_dashboard_stats():
    ensure_tables()
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        # Check if table exists
        try:
            status_counts = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
            recent_completed = conn.execute("""
                SELECT id, batch_id, params, result, updated_at 
                FROM jobs 
                WHERE status = 'COMPLETED' 
                ORDER BY updated_at DESC 
                LIMIT 20
            """).fetchall()
            
            # Simple leaderboard
            # We need to extract return_pct from result JSON safely
            leaderboard = []
            try:
                # DuckDB JSON extraction syntax depends on version, using Python fallback for simplicity
                all_completed = conn.execute("SELECT params, result FROM jobs WHERE status = 'COMPLETED' LIMIT 100").fetchall()
                for p_raw, r_raw in all_completed:
                    if r_raw:
                        r = json.loads(r_raw)
                        p = json.loads(p_raw)
                        metric = r.get("return_pct", -999)
                        leaderboard.append({
                            "ticker": p.get("ticker"),
                            "model": p.get("model_id"),
                            "return": metric,
                            "trades": r.get("total_trades"),
                            "params": p
                        })
                leaderboard.sort(key=lambda x: x["return"], reverse=True)
                leaderboard = leaderboard[:10]
            except Exception:
                leaderboard = []

            return {
                "counts": {s: c for s, c in status_counts},
                "recent": [
                    {"id": r[0], "batch": r[1], "params": json.loads(r[2]), "result": json.loads(r[3]), "ts": r[4]} 
                    for r in recent_completed
                ],
                "leaderboard": leaderboard
            }
        except duckdb.CatalogException:
            return {"counts": {}, "recent": [], "leaderboard": []}
