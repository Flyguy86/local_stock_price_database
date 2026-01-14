from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import uvicorn
import logging
import sys
from pathlib import Path
import threading

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from optimization_service import database
from optimization_service.database import (
    get_dashboard_stats, ensure_tables, create_jobs, 
    claim_job, complete_job, worker_heartbeat, get_pool, close_pool
)
from simulation_service.core import get_available_models, get_available_tickers

log = logging.getLogger("optimization.server")

# Import worker logic
from optimization_service.worker import run_worker

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await database.ensure_tables()
    log.info("Optimization C2 startup complete")
    
    # Start internal worker thread
    try:
        log.info("Attempting to start internal worker thread...")
        worker_thread = threading.Thread(target=run_worker, daemon=True, name="InternalWorker")
        worker_thread.start()
        log.info(f"✓ Started internal worker thread (daemon={worker_thread.daemon}, alive={worker_thread.is_alive()})")
        
        import time
        time.sleep(1)
        
        if worker_thread.is_alive():
            log.info("✓ Worker thread is running and healthy")
        else:
            log.error("✗ Worker thread died immediately after start")
            
    except Exception as e:
        log.error(f"✗ Failed to start worker thread: {e}", exc_info=True)
    
    # Start stuck job monitor
    async def monitor_stuck_jobs():
        """Background task to auto-fail jobs stuck for >10 minutes."""
        import asyncio
        from datetime import datetime, timedelta
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                pool = await get_pool()
                async with pool.acquire() as conn:
                    # Find jobs claimed >10 minutes ago still in RUNNING state
                    cutoff = datetime.now() - timedelta(minutes=10)
                    stuck = await conn.fetch("""
                        SELECT id, updated_at FROM optimization_jobs 
                        WHERE status = 'RUNNING' 
                        AND updated_at < $1
                    """, cutoff)
                    
                    if stuck:
                        log.warning(f"Found {len(stuck)} stuck jobs (>10min). Auto-failing...")
                        for row in stuck:
                            job_id = row['id']
                            updated_at = row['updated_at']
                            log.warning(f"Auto-failing stuck job {job_id} (last updated at {updated_at})")
                            await conn.execute("""
                                UPDATE optimization_jobs 
                                SET status = 'FAILED', 
                                    result = $1,
                                    updated_at = $2
                                WHERE id = $3
                            """, '{"error": "Job stuck >10 minutes, auto-failed"}', datetime.now(), job_id)
            except Exception as e:
                log.error(f"Stuck job monitor error: {e}")
                await asyncio.sleep(60)
    
    yield
    log.info("Optimization C2 shutdown")

app = FastAPI(title="Optimization Service C2", lifespan=lifespan)

# Setup Jinja2 templates
TEMPLATES_DIR = Path(__file__).parent / "templates"
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main dashboard using Jinja2 template."""
    try:
        models = get_available_models()
        tickers = get_available_tickers()
    except Exception as e:
        log.error(f"Failed to load models/tickers: {e}")
        models = []
        tickers = []
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "models": models,
        "tickers": tickers
    })

@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics including leaderboard."""
    try:
        return await get_dashboard_stats()
    except Exception as e:
        log.error(f"Failed to get stats: {e}")
        return {"counts": {}, "recent": [], "leaderboard": [], "workers": [], "error": str(e)}

@app.post("/api/create_batch")
async def create_batch_endpoint(config: dict):
    """Create a batch of optimization jobs."""
    try:
        jobs = []
        models = config.get("models", [])
        tickers = config.get("tickers", [])
        thresholds = config.get("thresholds", [0.0])
        z_scores = config.get("z_score", [False])
        use_bots = config.get("use_bot", [False])
        vol_norms = config.get("volatility_normalization", [False])
        regime_configs = config.get("regime_configs", [None])
        
        if not models:
            return {"error": "Please select at least one model", "batch_id": None, "jobs_created": 0}
        
        if not tickers:
            return {"error": "Please select at least one ticker", "batch_id": None, "jobs_created": 0}
        
        log.info(f"Building grid: {len(models)} models x {len(tickers)} tickers x {len(thresholds)} thresholds x {len(regime_configs)} regimes")
        
        for m in models:
            for t in tickers:
                for thresh in thresholds:
                    for z in z_scores:
                        for bot in use_bots:
                            for vol in vol_norms:
                                for regime_cfg in regime_configs:
                                    job_params = {
                                        "model_id": m,
                                        "ticker": t,
                                        "initial_cash": 10000,
                                        "use_bot": bot,
                                        "min_prediction_threshold": thresh,
                                        "enable_z_score_check": z,
                                        "volatility_normalization": vol
                                    }
                                    
                                    if regime_cfg and regime_cfg.get("col") and regime_cfg.get("allowed"):
                                        job_params["regime_col"] = regime_cfg["col"]
                                        job_params["allowed_regimes"] = regime_cfg["allowed"]
                                    
                                    jobs.append(job_params)
        
        log.info(f"Created {len(jobs)} job configurations")
        batch_id = await create_jobs(jobs)
        log.info(f"✓ Batch {batch_id} created successfully with {len(jobs)} jobs")
        
        return {"batch_id": batch_id, "jobs_created": len(jobs)}
        
    except Exception as e:
        log.error(f"Failed to create batch: {e}", exc_info=True)
        return {"error": str(e), "batch_id": None, "jobs_created": 0}

@app.post("/api/worker/heartbeat")
async def worker_heartbeat_endpoint(payload: dict):
    worker_id = payload.get("worker_id")
    job_id = payload.get("job_id")
    await worker_heartbeat(worker_id, job_id)
    return {"status": "ok"}

@app.post("/api/worker/claim")
async def claim_job_endpoint(payload: dict):
    worker_id = payload.get("worker_id", "unknown")
    job = await claim_job(worker_id)
    return job

@app.post("/api/worker/complete")
async def complete_job_endpoint(payload: dict):
    job_id = payload.get("job_id")
    result = payload.get("result")
    status = payload.get("status", "COMPLETED")
    await complete_job(job_id, result, status)
    return {"status": "ok"}

@app.get("/history/top")
def get_top_history_endpoint(limit: int = 15, offset: int = 0):
    """Returns paginated top strategies sorted by SQN."""
    try:
        stats = get_dashboard_stats()
        leaderboard = stats.get("leaderboard", [])
        
        total = len(leaderboard)
        paginated = leaderboard[offset:offset + limit]
        
        items = []
        for entry in paginated:
            items.append({
                "id": entry.get("model", ""),
                "timestamp": "",
                "model_id": entry.get("model", ""),
                "ticker": entry.get("ticker", ""),
                "return_pct": entry.get("return", 0.0),
                "trades_count": entry.get("trades", 0),
                "hit_rate_pct": entry.get("hit_rate", 0.0),
                "sqn": entry.get("sqn", 0.0),
                "params": entry.get("full_params", {})
            })
        
        return {"items": items, "total": total}
    except Exception as e:
        log.error(f"Failed to get history: {e}")
        return {"items": [], "total": 0, "error": str(e)}

@app.delete("/history/all")
def delete_all_history_endpoint():
    """Deletes all optimization job history."""
    try:
        import duckdb
        
        ensure_tables()
        with duckdb.connect(str(DB_PATH)) as conn:
            conn.execute("DELETE FROM jobs")
        
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import logging
    
    # Suppress noisy logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
