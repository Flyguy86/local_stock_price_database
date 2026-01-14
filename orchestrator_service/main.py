"""
Orchestrator Service - FastAPI Application
Recursive Strategy Factory for automated model evolution.
"""
import os
import logging
import threading
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from collections import deque

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

from .db import db
from .evolution import engine, EvolutionConfig
from .criteria import HolyGrailCriteria

# Log Buffer for Live Logs
log_buffer = deque(maxlen=10000)
log_lock = threading.Lock()

class BufferHandler(logging.Handler):
    """Custom handler to capture logs in memory buffer."""
    def emit(self, record):
        try:
            msg = self.format(record)
            # Filter out noisy GET requests
            if any(x in msg for x in ["GET /runs", "GET /promoted", "GET /jobs/pending", "GET /health"]):
                return
            with log_lock:
                log_buffer.append(msg)
        except Exception:
            self.handleError(record)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True
)

# Add buffer handler to root logger to capture ALL logs
# Only add to root logger - child loggers will propagate up automatically
_handler = BufferHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
root_logger = logging.getLogger()
root_logger.addHandler(_handler)

log = logging.getLogger("orchestrator.api")

# Paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    log.info("Orchestrator Service starting...")
    log.info(f"Log buffer initialized with capacity: {log_buffer.maxlen}")
    await db.connect()
    await engine.start()
    
    # Update service URLs from environment
    engine.training_url = os.getenv("TRAINING_URL", "http://training:8200")
    engine.simulation_url = os.getenv("SIMULATION_URL", "http://simulation:8300")
    
    log.info(f"Training URL: {engine.training_url}")
    log.info(f"Simulation URL: {engine.simulation_url}")
    
    # Recovery: Mark any RUNNING/PENDING runs as STOPPED (container restarted)
    try:
        async with db.acquire() as conn:
            # Find runs that were interrupted
            interrupted_runs = await conn.fetch(
                "SELECT id, symbol FROM evolution_runs WHERE status IN ('RUNNING', 'PENDING')"
            )
            
            if interrupted_runs:
                log.warning(f"Found {len(interrupted_runs)} interrupted evolution runs - cleaning up")
                
                for row in interrupted_runs:
                    run_id = row['id']
                    symbol = row['symbol']
                    
                    # Cancel all pending/running simulation jobs for this run
                    cancelled = await conn.execute(
                        """
                        UPDATE priority_jobs 
                        SET status = 'CANCELLED'
                        WHERE run_id = $1 AND status IN ('PENDING', 'RUNNING')
                        """,
                        run_id
                    )
                    log.info(f"Cancelled pending simulations for run {run_id} (symbol: {symbol})")
                
                # Mark runs as stopped
                await conn.execute(
                    """
                    UPDATE evolution_runs 
                    SET status = 'STOPPED', 
                        step_status = 'Interrupted by container restart'
                    WHERE status IN ('RUNNING', 'PENDING')
                    """
                )
                
                for row in interrupted_runs:
                    log.warning(f"Marked run {row['id']} (symbol: {row['symbol']}) as STOPPED")
            else:
                log.info("No interrupted evolution runs to recover")
    except Exception as e:
        log.error(f"Failed to recover interrupted runs: {e}")
    
    yield
    
    # Shutdown
    log.info("Orchestrator Service shutting down...")
    await engine.stop()
    await db.disconnect()


app = FastAPI(
    title="Orchestrator Service",
    description="Recursive Strategy Factory - Automated Train → Prune → Simulate Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================
# Health & Status
# ============================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "orchestrator"}


@app.get("/logs")
async def get_logs():
    """Get recent logs from buffer."""
    with log_lock:
        logs_list = list(log_buffer)
        # Add debug info if buffer is empty
        if not logs_list:
            return ["[No logs in buffer yet - check if handler is capturing logs]"]
        return logs_list


@app.get("/training/logs")
async def get_training_logs():
    """Proxy training service logs to avoid CORS issues."""
    training_url = os.getenv("TRAINING_URL", "http://training_service:8200")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{training_url}/logs")
            return resp.json()
    except Exception as e:
        log.warning(f"Failed to fetch training logs: {e}")
        return []


@app.get("/simulation/logs")
async def get_simulation_logs():
    """Proxy simulation service logs to avoid CORS issues."""
    simulation_url = os.getenv("SIMULATION_URL", "http://simulation_service:8300")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{simulation_url}/logs")
            return resp.json()
    except Exception as e:
        log.warning(f"Failed to fetch simulation logs: {e}")
        return []


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the orchestrator dashboard UI."""
    # Use the clean HTML that references external CSS/JS
    template_path = TEMPLATES_DIR / "dashboard_new.html"
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text())
    # Fallback if template not found
    return HTMLResponse(content="""
        <html><body>
        <h1>Orchestrator Service</h1>
        <p>Dashboard template not found. API endpoints available at <a href="/docs">/docs</a></p>
        </body></html>
    """)


@app.get("/models", response_class=HTMLResponse)
async def models_browser():
    """Serve the model browser and simulation launcher UI."""
    template_path = TEMPLATES_DIR / "models.html"
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text())
    return HTMLResponse(content="<html><body><h1>Model Browser</h1><p>Template not found</p></body></html>")


@app.get("/api/info")
async def api_info():
    """API information endpoint (JSON)."""
    return {
        "service": "Orchestrator - Recursive Strategy Factory",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/evolve": "Start new evolution run",
            "GET /api/runs": "List evolution runs",
            "GET /api/runs/{run_id}": "Get run details with lineage",
            "GET /api/promoted": "List promoted models",
            "GET /api/stats": "Get system statistics",
            "GET /api/features/options": "List available data folds/options from feature service",
            "GET /api/features/symbols": "List available symbols from feature service (optionally filtered by options)",
            "GET /api/features/columns": "Get feature columns for a symbol"
        }
    }


@app.get("/api/stats")
async def api_stats():
    """Get system statistics for dashboard."""
    runs = await db.list_evolution_runs(limit=1000)
    promoted = await db.list_promoted_models(limit=1000)
    jobs = await db.list_jobs(limit=10000)  # Get all jobs
    
    active_runs = [r for r in runs if r.get("status") == "running"]
    completed_runs = [r for r in runs if r.get("status") == "completed"]
    pending_jobs = [j for j in jobs if j.get("status") == "PENDING"]
    running_jobs = [j for j in jobs if j.get("status") == "RUNNING"]
    
    return {
        "total_runs": len(runs),
        "active_runs": len(active_runs),
        "completed_runs": len(completed_runs),
        "pending_jobs": len(pending_jobs),
        "running_jobs": len(running_jobs),
        "promoted_models": len(promoted),
        "total_jobs": len(jobs)
    }


# ============================================
# Direct Data Access (reads mounted /app/data)
# ============================================
FEATURES_PARQUET_DIR = Path("/app/data/features_parquet")


def _get_options_from_parquet() -> list:
    """Read options directly from mounted parquet files."""
    try:
        if not FEATURES_PARQUET_DIR.exists():
            log.warning(f"Features parquet dir not found: {FEATURES_PARQUET_DIR}")
            return []
        
        # Get first symbol directory
        symbol_dirs = [d for d in FEATURES_PARQUET_DIR.iterdir() if d.is_dir()]
        if not symbol_dirs:
            log.warning("No symbol directories in parquet")
            return []
        
        # Find first parquet file
        first_symbol = symbol_dirs[0]
        date_dirs = [d for d in first_symbol.iterdir() if d.is_dir()]
        if not date_dirs:
            return ["Legacy / No Config"]
        
        parquet_files = list(date_dirs[0].glob("*.parquet"))
        if not parquet_files:
            return ["Legacy / No Config"]
        
        # Read with pyarrow
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_files[0]))
        
        if 'options' not in table.column_names:
            return ["Legacy / No Config"]
        
        # Get unique options
        options_column = table.column('options').to_pylist()
        unique_options = list(set(opt for opt in options_column if opt and opt != '{}'))
        
        if not unique_options:
            return ["Legacy / No Config"]
        
        log.info(f"Found options from parquet: {unique_options}")
        return unique_options
        
    except Exception as e:
        log.exception(f"Failed to read options from parquet: {e}")
        return ["Legacy / No Config"]


def _get_symbols_from_parquet() -> list:
    """List symbols directly from mounted parquet directory."""
    try:
        if not FEATURES_PARQUET_DIR.exists():
            return []
        
        symbols = sorted([d.name for d in FEATURES_PARQUET_DIR.iterdir() if d.is_dir()])
        log.info(f"Found symbols from parquet: {symbols}")
        return symbols
        
    except Exception as e:
        log.exception(f"Failed to list symbols from parquet: {e}")
        return []


@app.get("/api/features/options")
async def get_feature_options():
    """List available data folds/options from mounted parquet files."""
    return _get_options_from_parquet()


@app.get("/api/features/symbols")
async def get_feature_symbols(options: Optional[str] = None):
    """List available symbols from mounted parquet directory."""
    symbols = _get_symbols_from_parquet()
    return {"symbols": symbols}


def _get_feature_columns_from_parquet(symbol: str) -> list:
    """Read feature columns directly from parquet files for a symbol."""
    try:
        symbol_dir = FEATURES_PARQUET_DIR / symbol
        if not symbol_dir.exists():
            log.warning(f"Symbol directory not found: {symbol_dir}")
            return []
        
        # Find first parquet file
        date_dirs = [d for d in symbol_dir.iterdir() if d.is_dir()]
        if not date_dirs:
            log.warning(f"No date directories for {symbol}")
            return []
        
        parquet_files = list(date_dirs[0].glob("*.parquet"))
        if not parquet_files:
            log.warning(f"No parquet files for {symbol}")
            return []
        
        # Read with pyarrow
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_files[0]))
        all_cols = table.column_names
        
        # Filter out OHLCV and metadata columns
        excluded = {'ts', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 
                    'options', 'dt', 'data_split'}
        features = [col for col in all_cols if col not in excluded]
        
        log.info(f"Found {len(features)} feature columns for {symbol}")
        return features
        
    except Exception as e:
        log.exception(f"Failed to read columns for {symbol}: {e}")
        return []


def _get_multi_ticker_features(target_symbol: str, reference_symbols: Optional[List[str]] = None) -> list:
    """
    Get feature columns for multi-ticker TS-aligned training.
    
    Returns features from target symbol plus all reference symbols with prefixes.
    For example:
    - Target AAPL: ['rsi_14', 'sma_20', ...]
    - Reference SPY: ['SPY_rsi_14', 'SPY_sma_20', ...]
    - Reference QQQ: ['QQQ_rsi_14', 'QQQ_sma_20', ...]
    
    Total: ~30 features/symbol × (1 target + N references) = 30×(N+1) features
    """
    all_features = []
    
    # Get target symbol features (no prefix)
    target_features = _get_feature_columns_from_parquet(target_symbol)
    if target_features:
        all_features.extend(target_features)
        log.info(f"Target {target_symbol}: {len(target_features)} features")
    else:
        log.warning(f"No features found for target symbol {target_symbol}")
    
    # Get reference symbol features (with prefix)
    if reference_symbols:
        for ref_symbol in reference_symbols:
            ref_features = _get_feature_columns_from_parquet(ref_symbol)
            if ref_features:
                # Prefix each feature with the reference symbol
                prefixed_features = [f"{ref_symbol}_{feat}" for feat in ref_features]
                all_features.extend(prefixed_features)
                log.info(f"Reference {ref_symbol}: {len(ref_features)} features (prefixed)")
            else:
                log.warning(f"No features found for reference symbol {ref_symbol}")
    
    log.info(f"Total multi-ticker features: {len(all_features)} ({target_symbol} + {len(reference_symbols or [])} references)")
    return all_features


@app.get("/api/features/columns")
async def get_feature_columns(symbol: str, limit: int = 1):
    """Get feature columns directly from mounted parquet files."""
    features = _get_feature_columns_from_parquet(symbol)
    
    if features:
        return {
            "symbol": symbol,
            "feature_columns": features,
            "sample_count": 1
        }
    return {"error": f"No parquet data for {symbol}", "symbol": symbol, "feature_columns": []}


# ============================================
# Evolution Endpoints
# ============================================

class EvolveRequest(BaseModel):
    """Request to start an evolution run."""
    seed_model_id: Optional[str] = None
    seed_features: Optional[List[str]] = None
    symbol: str
    reference_symbols: Optional[List[str]] = None  # Additional tickers for relational features
    simulation_tickers: Optional[List[str]] = None  # Tickers to run simulations on
    algorithm: str = "RandomForest"
    target_col: str = "close"
    hyperparameters: dict = {}
    target_transform: str = "log_return"
    max_generations: int = 4
    prune_fraction: float = 0.25  # Prune bottom X% each generation
    min_features: int = 5         # Stop when reaching this many features
    data_options: Optional[str] = None
    timeframe: str = "1m"
    thresholds: List[float] = [0.0001, 0.0003, 0.0005, 0.0007]
    z_score_thresholds: List[float] = [0, 2.0, 2.5, 3.0, 3.5]  # Z-score cutoffs (0 = no filter)
    regime_configs: List[dict] = [
        {"regime_gmm": [0]},
        {"regime_gmm": [1]},
        {"regime_vix": [0, 1]}
    ]
    # Grid search for regularization (ElasticNet, Ridge, Lasso)
    alpha_grid: Optional[List[float]] = None    # L2 penalty: [0.001, 0.01, 0.1, 1, 10, 50, 100]
    l1_ratio_grid: Optional[List[float]] = None # L1/L2 mix: [0.1, 0.3, 0.5, 0.7, 0.9]
    # XGBoost hyperparameter grids
    max_depth_grid: Optional[List[int]] = None
    min_child_weight_grid: Optional[List[int]] = None
    reg_lambda_grid: Optional[List[float]] = None
    learning_rate_grid: Optional[List[float]] = None
    # LightGBM hyperparameter grids
    num_leaves_grid: Optional[List[int]] = None
    min_data_in_leaf_grid: Optional[List[int]] = None
    lambda_l2_grid: Optional[List[float]] = None
    lgbm_learning_rate_grid: Optional[List[float]] = None
    # RandomForest hyperparameter grids
    rf_max_depth_grid: Optional[List[Any]] = None  # Can include None
    min_samples_split_grid: Optional[List[int]] = None
    min_samples_leaf_grid: Optional[List[int]] = None
    n_estimators_grid: Optional[List[int]] = None
    # Holy Grail thresholds
    sqn_min: float = 3.0
    sqn_max: float = 5.0
    profit_factor_min: float = 2.0
    profit_factor_max: float = 4.0
    trade_count_min: int = 200
    trade_count_max: int = 10000


@app.post("/evolve")
async def start_evolution(req: EvolveRequest, background_tasks: BackgroundTasks):
    """
    Start a new evolution run.
    
    The evolution loop will:
    1. Fetch seed features from parquet if not provided
    2. Prune features with importance <= 0
    3. Check for existing model with same fingerprint
    4. Train new model or reuse existing (with TS-aligned multi-ticker data)
    5. Queue simulations with priority based on parent SQN
    6. Evaluate results against Holy Grail criteria
    7. Repeat until max_generations or promotion
    """
    seed_features = req.seed_features
    
    # Auto-fetch seed features from mounted parquet if not provided
    # Includes features from target symbol AND all reference symbols (TS-aligned)
    if not req.seed_model_id and not seed_features:
        log.info(f"Fetching multi-ticker seed features: {req.symbol} + {req.reference_symbols or []}")
        seed_features = _get_multi_ticker_features(req.symbol, req.reference_symbols)
        
        if seed_features:
            log.info(f"Auto-fetched {len(seed_features)} features ({req.symbol} + {len(req.reference_symbols or [])} references)")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No feature data found for symbol {req.symbol} in parquet directory"
            )
    
    if not req.seed_model_id and not seed_features:
        raise HTTPException(
            status_code=400,
            detail="Must provide either seed_model_id or seed_features, or have feature data in parquet"
        )
    
    # Build data_options to include reference symbols for TS-aligned multi-ticker training
    data_options = req.data_options
    if req.reference_symbols and len(req.reference_symbols) > 0:
        import json
        opts = json.loads(data_options) if data_options else {}
        opts['reference_symbols'] = req.reference_symbols
        data_options = json.dumps(opts)
        log.info(f"Training with reference symbols: {req.reference_symbols}")
    
    config = EvolutionConfig(
        seed_model_id=req.seed_model_id,
        seed_features=seed_features,
        symbol=req.symbol,
        simulation_tickers=req.simulation_tickers,
        algorithm=req.algorithm,
        target_col=req.target_col,
        hyperparameters=req.hyperparameters,
        target_transform=req.target_transform,
        max_generations=req.max_generations,
        prune_fraction=req.prune_fraction,
        min_features=req.min_features,
        data_options=data_options,
        timeframe=req.timeframe,
        thresholds=req.thresholds,
        z_score_thresholds=req.z_score_thresholds,
        regime_configs=req.regime_configs,
        alpha_grid=req.alpha_grid,
        l1_ratio_grid=req.l1_ratio_grid,
        max_depth_grid=req.max_depth_grid,
        min_child_weight_grid=req.min_child_weight_grid,
        reg_lambda_grid=req.reg_lambda_grid,
        learning_rate_grid=req.learning_rate_grid,
        num_leaves_grid=req.num_leaves_grid,
        min_data_in_leaf_grid=req.min_data_in_leaf_grid,
        lambda_l2_grid=req.lambda_l2_grid,
        lgbm_learning_rate_grid=req.lgbm_learning_rate_grid,
        rf_max_depth_grid=req.rf_max_depth_grid,
        min_samples_split_grid=req.min_samples_split_grid,
        min_samples_leaf_grid=req.min_samples_leaf_grid,
        n_estimators_grid=req.n_estimators_grid,
        sqn_min=req.sqn_min,
        sqn_max=req.sqn_max,
        profit_factor_min=req.profit_factor_min,
        profit_factor_max=req.profit_factor_max,
        trade_count_min=req.trade_count_min,
        trade_count_max=req.trade_count_max
    )
    
    # Run evolution in background
    async def run_in_background():
        try:
            result = await engine.run_evolution(config)
            log.info(f"Evolution completed: {result}")
        except Exception as e:
            log.error(f"Evolution failed: {e}")
    
    background_tasks.add_task(run_in_background)
    
    return {
        "status": "started",
        "message": f"Evolution run started for {req.symbol}",
        "max_generations": req.max_generations
    }


@app.get("/runs")
async def list_runs(status: Optional[str] = None, limit: int = 50):
    """List evolution runs."""
    runs = await db.list_evolution_runs(status=status, limit=limit)
    return {"runs": runs, "count": len(runs)}


@app.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Cancel/fail a stale or running evolution run."""
    run = await db.get_evolution_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run["status"] not in ("RUNNING", "PENDING"):
        raise HTTPException(status_code=400, detail=f"Run is already {run['status']}")
    
    await db.update_evolution_run(run_id, status="CANCELLED", step_status="Cancelled by user")
    log.info(f"Cancelled evolution run {run_id}")
    return {"status": "cancelled", "run_id": run_id}


@app.post("/runs/{run_id}/stop")
async def stop_run(run_id: str):
    """Stop a RUNNING/PENDING evolution run."""
    run = await db.get_evolution_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run["status"] not in ("RUNNING", "PENDING"):
        raise HTTPException(status_code=400, detail=f"Run is not running (status: {run['status']})")
    
    await db.update_evolution_run(run_id, status="STOPPED", step_status="Stopped by user")
    log.info(f"Stopped evolution run {run_id}")
    return {"status": "stopped", "run_id": run_id}


@app.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, background_tasks: BackgroundTasks):
    """Resume a STOPPED evolution run (e.g., after container restart)."""
    run = await db.get_evolution_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run["status"] not in ("STOPPED", "FAILED", "CANCELLED"):
        raise HTTPException(
            status_code=400, 
            detail=f"Can only resume STOPPED/FAILED/CANCELLED runs, current status: {run['status']}"
        )
    
    # Check if there are pending/running simulations for this run
    pending_count = await db.get_pending_job_count(run_id)
    
    if pending_count > 0:
        # Just mark as RUNNING again - simulations are still in queue
        await db.update_evolution_run(
            run_id, 
            status="RUNNING", 
            step_status=f"Resumed: {pending_count} simulations pending"
        )
        log.info(f"Resumed run {run_id}: {pending_count} simulations still in queue")
        return {
            "status": "resumed",
            "run_id": run_id,
            "message": f"Resumed with {pending_count} pending simulations"
        }
    
    # No pending jobs - check if we have completed jobs to evaluate
    completed_count = await db.get_completed_job_count(run_id, run["current_generation"])
    
    if completed_count > 0:
        # Evolution loop is not running - cannot truly resume mid-generation
        # User should start a new evolution using best model as seed
        await db.update_evolution_run(
            run_id,
            status="STOPPED",
            step_status=f"Cannot resume: {completed_count} sims completed but evolution loop not running. Start new run with best model as seed."
        )
        log.warning(f"Run {run_id}: Cannot resume - evolution engine not running. User should start new evolution.")
        
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume this run - the evolution engine is not running. "
                   f"{completed_count} simulations completed in generation {run['current_generation']}. "
                   f"To continue: Start a new evolution run using the best model from this run as 'seed_model_id'."
        )
    
    # No pending or completed jobs - mark as failed
    await db.update_evolution_run(
        run_id,
        status="FAILED",
        step_status="No active simulations found"
    )
    
    raise HTTPException(
        status_code=400,
        detail="No pending or completed simulations found - run cannot be resumed"
    )


@app.post("/runs/cleanup-stale")
async def cleanup_stale_runs(stale_minutes: int = 10):
    """Mark runs as FAILED if they haven't been updated in stale_minutes."""
    from datetime import datetime, timedelta, timezone
    
    runs = await db.list_evolution_runs(status="RUNNING", limit=100)
    cleaned = []
    
    now = datetime.now(timezone.utc)
    for run in runs:
        updated_at = run.get("updated_at")
        if updated_at:
            # Handle both aware and naive datetimes
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            age_minutes = (now - updated_at).total_seconds() / 60
            if age_minutes > stale_minutes:
                await db.update_evolution_run(
                    run["id"], 
                    status="FAILED", 
                    step_status=f"Stale - no update for {int(age_minutes)} minutes"
                )
                cleaned.append(run["id"])
                log.info(f"Cleaned up stale run {run['id']} (last updated {int(age_minutes)} min ago)")
    
    return {"cleaned": len(cleaned), "run_ids": cleaned}


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get evolution run details with full lineage."""
    run = await db.get_evolution_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    lineage = await db.get_lineage(run_id)
    completed_jobs = await db.get_completed_jobs(run_id)
    
    return {
        "run": run,
        "lineage": lineage,
        "completed_jobs": len(completed_jobs),
        "results_sample": completed_jobs[:10]
    }


@app.get("/runs/{run_id}/generations")
async def get_run_generations(run_id: str):
    """Get per-generation summary with best results for each epoch."""
    import json
    run = await db.get_evolution_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    lineage = await db.get_lineage(run_id)
    
    # Group jobs by generation
    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT generation, 
                   COUNT(*) as total_jobs,
                   COUNT(*) FILTER (WHERE status = 'COMPLETED') as completed,
                   COUNT(*) FILTER (WHERE status = 'PENDING') as pending,
                   COUNT(*) FILTER (WHERE status = 'RUNNING') as running,
                   MAX((result->>'sqn')::float) as best_sqn,
                   MAX((result->>'profit_factor')::float) as best_pf
            FROM priority_jobs 
            WHERE run_id = $1 
            GROUP BY generation 
            ORDER BY generation
            """,
            run_id
        )
        
        generations = []
        for row in rows:
            gen = dict(row)
            gen_num = gen["generation"]
            
            # Find lineage entry for this generation
            lin = next((l for l in lineage if l.get("generation") == gen_num), None)
            if lin:
                gen["parent_model_id"] = lin.get("parent_model_id")
                gen["child_model_id"] = lin.get("child_model_id")
                gen["pruned_features"] = json.loads(lin.get("pruned_features", "[]")) if isinstance(lin.get("pruned_features"), str) else lin.get("pruned_features", [])
                gen["remaining_features"] = json.loads(lin.get("remaining_features", "[]")) if isinstance(lin.get("remaining_features"), str) else lin.get("remaining_features", [])
                gen["pruning_reason"] = lin.get("pruning_reason")
            
            # Get top 3 results for this generation
            top_results = await conn.fetch(
                """
                SELECT model_id, result 
                FROM priority_jobs 
                WHERE run_id = $1 AND generation = $2 AND status = 'COMPLETED'
                ORDER BY (result->>'sqn')::float DESC NULLS LAST
                LIMIT 3
                """,
                run_id, gen_num
            )
            gen["top_results"] = []
            for r in top_results:
                res = json.loads(r["result"]) if isinstance(r.get("result"), str) else r.get("result", {})
                gen["top_results"].append({
                    "model_id": r["model_id"],
                    "sqn": res.get("sqn"),
                    "profit_factor": res.get("profit_factor"),
                    "trade_count": res.get("trade_count") or res.get("trades_count"),
                    "params": res.get("params", {})
                })
            
            generations.append(gen)
        
        return {
            "run_id": run_id,
            "symbol": run.get("symbol"),
            "status": run.get("status"),
            "max_generations": run.get("max_generations"),
            "generations": generations
        }


# ============================================
# Promoted Models
# ============================================

@app.get("/promoted")
async def list_promoted(limit: int = 50):
    """List models that met Holy Grail criteria."""
    promoted = await db.list_promoted_models(limit=limit)
    return {"promoted": promoted, "count": len(promoted)}


@app.get("/promoted/{promoted_id}")
async def get_promoted_detail(promoted_id: str):
    """Get full details of a promoted model including config, features, and lineage."""
    async with db.acquire() as conn:
        # Get the promoted model record
        row = await conn.fetchrow(
            "SELECT * FROM promoted_models WHERE id = $1",
            promoted_id
        )
        if not row:
            raise HTTPException(status_code=404, detail="Promoted model not found")
        
        promoted = dict(row)
        
        # Parse JSON fields
        import json
        if isinstance(promoted.get("regime_config"), str):
            promoted["regime_config"] = json.loads(promoted["regime_config"])
        if isinstance(promoted.get("full_result"), str):
            promoted["full_result"] = json.loads(promoted["full_result"])
        
        # Get the fingerprint entry for this model (contains features, hyperparams)
        fingerprint = await conn.fetchrow(
            "SELECT * FROM model_fingerprints WHERE model_id = $1",
            promoted["model_id"]
        )
        if fingerprint:
            fp_dict = dict(fingerprint)
            if isinstance(fp_dict.get("features"), str):
                fp_dict["features"] = json.loads(fp_dict["features"])
            if isinstance(fp_dict.get("hyperparams"), str):
                fp_dict["hyperparams"] = json.loads(fp_dict["hyperparams"])
            promoted["model_config"] = fp_dict
        
        # Get full lineage (ancestry)
        lineage_rows = await conn.fetch(
            """
            WITH RECURSIVE ancestry AS (
                SELECT * FROM evolution_log WHERE child_model_id = $1
                UNION ALL
                SELECT e.* FROM evolution_log e
                JOIN ancestry a ON e.child_model_id = a.parent_model_id
            )
            SELECT * FROM ancestry ORDER BY generation ASC
            """,
            promoted["model_id"]
        )
        lineage = []
        for l in lineage_rows:
            ld = dict(l)
            if isinstance(ld.get("pruned_features"), str):
                ld["pruned_features"] = json.loads(ld["pruned_features"])
            if isinstance(ld.get("remaining_features"), str):
                ld["remaining_features"] = json.loads(ld["remaining_features"])
            lineage.append(ld)
        
        promoted["lineage"] = lineage
        
        return promoted


@app.get("/promoted/{model_id}/lineage")
async def get_model_lineage(model_id: str):
    """Get full ancestry of a model."""
    # Find all evolution logs where this model appears
    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH RECURSIVE ancestry AS (
                SELECT * FROM evolution_log WHERE child_model_id = $1
                UNION ALL
                SELECT e.* FROM evolution_log e
                JOIN ancestry a ON e.child_model_id = a.parent_model_id
            )
            SELECT * FROM ancestry ORDER BY generation ASC
            """,
            model_id
        )
        return {"lineage": [dict(r) for r in rows]}


# ============================================
# Worker Job Queue
# ============================================

class ClaimRequest(BaseModel):
    worker_id: str


class CompleteRequest(BaseModel):
    worker_id: str
    result: dict
    success: bool = True


@app.post("/jobs/claim")
async def claim_job(req: ClaimRequest):
    """
    Claim the highest priority pending job.
    
    Priority is based on parent_sqn (higher SQN parents get their
    children processed first).
    """
    await db.register_worker(req.worker_id)
    
    job = await db.claim_job(req.worker_id)
    if not job:
        return {"status": "no_jobs", "job": None}
    
    await db.update_worker_status(req.worker_id, "BUSY", job["id"])
    
    return {
        "status": "claimed",
        "job": job
    }


@app.post("/jobs/{job_id}/complete")
async def complete_job(job_id: str, req: CompleteRequest):
    """Mark a job as completed with results."""
    await db.complete_job(job_id, req.result, req.success)
    await db.update_worker_status(req.worker_id, "IDLE", None)
    
    return {"status": "completed", "job_id": job_id}


@app.get("/jobs/pending")
async def get_pending_jobs(run_id: Optional[str] = None):
    """Get count of pending jobs."""
    if run_id:
        count = await db.get_pending_job_count(run_id)
    else:
        async with db.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as cnt FROM priority_jobs WHERE status = 'PENDING'"
            )
            count = row["cnt"]
    
    return {"pending": count}


# ============================================
# Train-Only Mode (Evolution without Simulations)
# ============================================

class TrainOnlyRequest(BaseModel):
    """Request to train and evolve models without running simulations."""
    seed_model_id: Optional[str] = None
    seed_features: Optional[List[str]] = None
    symbol: str
    reference_symbols: Optional[List[str]] = None
    algorithm: str = "elasticnet_regression"
    target_col: str = "close"
    hyperparameters: Dict[str, Any] = {}
    target_transform: str = "log_return"
    max_generations: int = 4
    prune_fraction: float = 0.25
    min_features: int = 5
    data_options: Optional[str] = None
    timeframe: str = "1m"
    # ElasticNet/Ridge/Lasso grids
    alpha_grid: Optional[List[float]] = None
    l1_ratio_grid: Optional[List[float]] = None
    # XGBoost grids
    max_depth_grid: Optional[List[int]] = None
    min_child_weight_grid: Optional[List[int]] = None
    reg_lambda_grid: Optional[List[float]] = None
    learning_rate_grid: Optional[List[float]] = None
    # LightGBM grids
    num_leaves_grid: Optional[List[int]] = None
    min_data_in_leaf_grid: Optional[List[int]] = None
    lambda_l2_grid: Optional[List[float]] = None
    lgbm_learning_rate_grid: Optional[List[float]] = None
    # RandomForest grids
    rf_max_depth_grid: Optional[List[Any]] = None  # Can include None
    min_samples_split_grid: Optional[List[int]] = None
    min_samples_leaf_grid: Optional[List[int]] = None
    n_estimators_grid: Optional[List[int]] = None


@app.post("/train-only")
async def train_only(req: TrainOnlyRequest, background_tasks: BackgroundTasks):
    """
    Train and evolve models for X generations WITHOUT running simulations.
    
    Use this when you want to explore different model configurations
    and manually select which ones to run grid search simulations on later.
    """
    seed_features = req.seed_features
    
    # Auto-fetch features if not provided
    # Includes features from target symbol AND all reference symbols (TS-aligned)
    if not req.seed_model_id and not seed_features:
        log.info(f"Fetching multi-ticker seed features: {req.symbol} + {req.reference_symbols or []}")
        seed_features = _get_multi_ticker_features(req.symbol, req.reference_symbols)
        
        if seed_features:
            log.info(f"Auto-fetched {len(seed_features)} features ({req.symbol} + {len(req.reference_symbols or [])} references)")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No feature data found for symbol {req.symbol}"
            )
    
    # Build data_options
    data_options = req.data_options
    if req.reference_symbols and len(req.reference_symbols) > 0:
        import json
        opts = json.loads(data_options) if data_options else {}
        opts['reference_symbols'] = req.reference_symbols
        data_options = json.dumps(opts)
    
    config = EvolutionConfig(
        seed_model_id=req.seed_model_id,
        seed_features=seed_features,
        symbol=req.symbol,
        simulation_tickers=[],  # Empty - no simulations
        algorithm=req.algorithm,
        target_col=req.target_col,
        hyperparameters=req.hyperparameters,
        target_transform=req.target_transform,
        max_generations=req.max_generations,
        prune_fraction=req.prune_fraction,
        min_features=req.min_features,
        data_options=data_options,
        timeframe=req.timeframe,
        thresholds=[],  # No simulation grid
        z_score_thresholds=[],
        regime_configs=[],
        alpha_grid=req.alpha_grid,
        l1_ratio_grid=req.l1_ratio_grid,
        max_depth_grid=req.max_depth_grid,
        min_child_weight_grid=req.min_child_weight_grid,
        reg_lambda_grid=req.reg_lambda_grid,
        learning_rate_grid=req.learning_rate_grid,
        num_leaves_grid=req.num_leaves_grid,
        min_data_in_leaf_grid=req.min_data_in_leaf_grid,
        lambda_l2_grid=req.lambda_l2_grid,
        lgbm_learning_rate_grid=req.lgbm_learning_rate_grid,
        rf_max_depth_grid=req.rf_max_depth_grid,
        min_samples_split_grid=req.min_samples_split_grid,
        min_samples_leaf_grid=req.min_samples_leaf_grid,
        n_estimators_grid=req.n_estimators_grid
    )
    
    # Run in background
    async def run_in_background():
        try:
            result = await engine.run_evolution(config)
            log.info(f"Train-only evolution completed: {result}")
        except Exception as e:
            log.error(f"Train-only evolution failed: {e}")
    
    background_tasks.add_task(run_in_background)
    
    return {
        "status": "started",
        "mode": "train-only",
        "message": f"Training {req.max_generations} generations for {req.symbol} (no simulations)",
        "max_generations": req.max_generations
    }


# ============================================
# Grid Search Endpoints (Algorithm-Specific)
# ============================================

class GridSearchElasticNetRequest(BaseModel):
    """Grid search for ElasticNet regression - no feature pruning, just hyperparameter exploration."""
    seed_features: Optional[List[str]] = None
    symbol: str
    reference_symbols: Optional[List[str]] = None
    target_col: str = "close"
    target_transform: str = "log_return"
    data_options: Optional[str] = None
    timeframe: str = "1m"
    # ElasticNet-specific grids
    alpha_grid: List[float] = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    l1_ratio_grid: List[float] = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]


class GridSearchXGBoostRequest(BaseModel):
    """Grid search for XGBoost - explore tree depth, regularization, and learning rates."""
    seed_features: Optional[List[str]] = None
    symbol: str
    reference_symbols: Optional[List[str]] = None
    target_col: str = "close"
    target_transform: str = "log_return"
    data_options: Optional[str] = None
    timeframe: str = "1m"
    regressor: bool = True  # True for XGBRegressor, False for XGBClassifier
    # XGBoost-specific grids
    max_depth_grid: List[int] = [3, 4, 5, 6, 7, 8, 9]
    min_child_weight_grid: List[int] = [1, 3, 5, 10, 15, 20, 30]
    reg_alpha_grid: List[float] = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]  # L1 regularization (Lasso) - drives coefficients to zero
    reg_lambda_grid: List[float] = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]  # L2 regularization (Ridge) - shrinks coefficients
    learning_rate_grid: List[float] = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]


class GridSearchLightGBMRequest(BaseModel):
    """Grid search for LightGBM - explore leaf complexity and regularization."""
    seed_features: Optional[List[str]] = None
    symbol: str
    reference_symbols: Optional[List[str]] = None
    target_col: str = "close"
    target_transform: str = "log_return"
    data_options: Optional[str] = None
    timeframe: str = "1m"
    regressor: bool = True  # True for LGBMRegressor, False for LGBMClassifier
    # LightGBM-specific grids
    num_leaves_grid: List[int] = [7, 15, 31, 63, 95, 127, 191]
    min_data_in_leaf_grid: List[int] = [5, 10, 20, 40, 60, 80, 100]
    lambda_l1_grid: List[float] = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]  # L1 regularization (Lasso) - feature selection
    lambda_l2_grid: List[float] = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]  # L2 regularization (Ridge) - coefficient shrinkage
    learning_rate_grid: List[float] = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]


class GridSearchRandomForestRequest(BaseModel):
    """Grid search for RandomForest - explore tree depth, splits, and ensemble size."""
    seed_features: Optional[List[str]] = None
    symbol: str
    reference_symbols: Optional[List[str]] = None
    target_col: str = "close"
    target_transform: str = "log_return"
    data_options: Optional[str] = None
    timeframe: str = "1m"
    regressor: bool = True  # True for RandomForestRegressor, False for Classifier
    # RandomForest-specific grids
    max_depth_grid: List[Any] = [5, 10, 15, 20, 30, 50, None]  # None = no limit
    min_samples_split_grid: List[int] = [2, 5, 10, 20, 30, 50, 100]
    min_samples_leaf_grid: List[int] = [1, 2, 4, 8, 12, 16, 20]
    n_estimators_grid: List[int] = [25, 50, 75, 100, 150, 200, 300]
    max_features_grid: List[Any] = ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9, 1.0]  # Feature sampling: sqrt/log2 for regularization, 0.5/1.0 for % of features


@app.post("/grid-search/elasticnet")
async def grid_search_elasticnet(req: GridSearchElasticNetRequest, background_tasks: BackgroundTasks):
    """
    Train ALL ElasticNet hyperparameter combinations (alpha × l1_ratio).
    
    Example: 5 alphas × 3 l1_ratios = 15 models saved.
    No feature pruning - all models use the same feature set.
    Returns immediately and runs training in background.
    """
    log.info(f"=== ElasticNet grid search request received for {req.symbol} ===")
    
    # Auto-fetch features if not provided
    seed_features = req.seed_features
    if not seed_features:
        log.info(f"Fetching multi-ticker seed features: {req.symbol} + {req.reference_symbols or []}")
        seed_features = _get_multi_ticker_features(req.symbol, req.reference_symbols)
        
        if not seed_features:
            log.error(f"No feature data found for symbol {req.symbol}")
            raise HTTPException(
                status_code=404,
                detail=f"No feature data found for symbol {req.symbol}"
            )
    
    log.info(f"Using {len(seed_features)} features for training")
    
    # Build data_options with standard flags + reference symbols
    data_options = req.data_options
    if req.reference_symbols and len(req.reference_symbols) > 0:
        import json
        # Start with defaults if data_options is empty
        opts = json.loads(data_options) if data_options else {
            "use_rsi": True,
            "use_macd": True,
            "use_bb": True,
            "use_sma": True,
            "use_vol": True,
            "use_atr": True,
            "use_time": True,
            "enable_segmentation": True,
            "train_window": 20000,
            "test_window": 1000
        }
        opts['reference_symbols'] = req.reference_symbols
        data_options = json.dumps(opts)
    elif not data_options:
        # Provide defaults even if no reference symbols
        import json
        data_options = json.dumps({
            "use_rsi": True,
            "use_macd": True,
            "use_bb": True,
            "use_sma": True,
            "use_vol": True,
            "use_atr": True,
            "use_time": True,
            "enable_segmentation": True,
            "train_window": 20000,
            "test_window": 1000
        })

    
    grid_size = len(req.alpha_grid) * len(req.l1_ratio_grid)
    log.info(f"Grid size: {grid_size} models ({len(req.alpha_grid)} alphas × {len(req.l1_ratio_grid)} l1_ratios)")
    
    # Create a run record to track progress
    run_id = str(uuid.uuid4())
    async with db.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO evolution_runs 
            (id, symbol, status, max_generations, config, algorithm, target_col, target_transform, timeframe, seed_features)
            VALUES ($1, $2, 'RUNNING', $3, '{}'::jsonb, 'elasticnet_regression', $4, $5, $6, $7::jsonb)
            """,
            run_id, req.symbol, grid_size, req.target_col, req.target_transform, req.timeframe, json.dumps(seed_features)
        )
    
    log.info(f"Created grid search run {run_id}: ElasticNet {grid_size} models for {req.symbol}")
    
    # Use asyncio.create_task instead of background_tasks for async functions
    async def run_grid_search():
        log.info(f"[{run_id}] ===== STARTING BACKGROUND GRID SEARCH TASK =====")
        try:
            payload = {
                "symbol": req.symbol,
                "algorithm": "elasticnet_regression",
                "target_col": req.target_col,
                "hyperparameters": {},
                "target_transform": req.target_transform,
                "data_options": data_options,
                "timeframe": req.timeframe,
                "feature_whitelist": seed_features,
                "alpha_grid": req.alpha_grid,
                "l1_ratio_grid": req.l1_ratio_grid,
                "save_all_grid_models": True
            }
            
            log.info(f"[{run_id}] Payload prepared: {grid_size} models, {len(seed_features)} features")
            log.info(f"[{run_id}] Sending POST to http://training_service:8200/train ...")
            async with httpx.AsyncClient(timeout=1800.0) as client:
                resp = await client.post(
                    "http://training_service:8200/train",
                    json=payload
                )
                
                log.info(f"[{run_id}] Training service responded with status {resp.status_code}")
                log.info(f"[{run_id}] Response body: {resp.text[:500]}")
                
                if resp.status_code == 200:
                    async with db.acquire() as conn:
                        await conn.execute(
                            "UPDATE evolution_runs SET status = 'COMPLETED' WHERE id = $1",
                            run_id
                        )
                    log.info(f"[{run_id}] Grid search completed successfully")
                else:
                    async with db.acquire() as conn:
                        await conn.execute(
                            "UPDATE evolution_runs SET status = 'FAILED' WHERE id = $1",
                            run_id
                        )
                    log.error(f"[{run_id}] Grid search failed with status {resp.status_code}: {resp.text}")
        except Exception as e:
            log.error(f"[{run_id}] Grid search EXCEPTION: {type(e).__name__}: {e}", exc_info=True)
            async with db.acquire() as conn:
                await conn.execute(
                    "UPDATE evolution_runs SET status = 'FAILED' WHERE id = $1",
                    run_id
                )
    
    # Start the background task
    log.info(f"[{run_id}] About to create asyncio background task...")
    import asyncio
    task = asyncio.create_task(run_grid_search())
    log.info(f"[{run_id}] Background task created: {task}")
    
    return {
        "status": "started",
        "run_id": run_id,
        "algorithm": "elasticnet_regression",
        "grid_size": grid_size,
        "symbol": req.symbol,
        "message": f"Grid search started: training {grid_size} ElasticNet models in background"
    }


@app.post("/grid-search/xgboost")
async def grid_search_xgboost(req: GridSearchXGBoostRequest, background_tasks: BackgroundTasks):
    """
    Train ALL XGBoost hyperparameter combinations.
    Returns immediately and runs training in background.
    """
    seed_features = req.seed_features
    if not seed_features:
        log.info(f"Fetching multi-ticker seed features: {req.symbol} + {req.reference_symbols or []}")
        seed_features = _get_multi_ticker_features(req.symbol, req.reference_symbols)
        
        if not seed_features:
            raise HTTPException(status_code=404, detail=f"No feature data found for {req.symbol}")
    
    data_options = req.data_options
    if req.reference_symbols:
        import json
        opts = json.loads(data_options) if data_options else {
            "use_rsi": True,
            "use_macd": True,
            "use_bb": True,
            "use_sma": True,
            "use_vol": True,
            "use_atr": True,
            "use_time": True,
            "enable_segmentation": True,
            "train_window": 20000,
            "test_window": 1000
        }
        opts['reference_symbols'] = req.reference_symbols
        data_options = json.dumps(opts)
    elif not data_options:
        import json
        data_options = json.dumps({
            "use_rsi": True,
            "use_macd": True,
            "use_bb": True,
            "use_sma": True,
            "use_vol": True,
            "use_atr": True,
            "use_time": True,
            "enable_segmentation": True,
            "train_window": 20000,
            "test_window": 1000
        })
    
    algorithm = "xgboost_regressor" if req.regressor else "xgboost_classifier"
    grid_size = len(req.max_depth_grid) * len(req.min_child_weight_grid) * \
                len(req.reg_alpha_grid) * len(req.reg_lambda_grid) * len(req.learning_rate_grid)
    
    # Create run record
    run_id = str(uuid.uuid4())
    async with db.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO evolution_runs 
            (id, symbol, status, max_generations, config, algorithm, target_col, target_transform, timeframe, seed_features)
            VALUES ($1, $2, 'RUNNING', $3, '{}'::jsonb, $4, $5, $6, $7, $8::jsonb)
            """,
            run_id, req.symbol, grid_size, algorithm, req.target_col, req.target_transform, req.timeframe, json.dumps(seed_features)
        )
    
    log.info(f"Created grid search run {run_id}: XGBoost {grid_size} models for {req.symbol}")
    
    # Run in background
    async def run_grid_search():
        try:
            payload = {
                "symbol": req.symbol,
                "algorithm": algorithm,
                "target_col": req.target_col,
                "hyperparameters": {},
                "target_transform": req.target_transform,
                "data_options": data_options,
                "timeframe": req.timeframe,
                "feature_whitelist": seed_features,
                "max_depth_grid": req.max_depth_grid,
                "min_child_weight_grid": req.min_child_weight_grid,
                "reg_alpha_grid": req.reg_alpha_grid,
                "reg_lambda_grid": req.reg_lambda_grid,
                "learning_rate_grid": req.learning_rate_grid,
                "save_all_grid_models": True
            }
            
            async with httpx.AsyncClient(timeout=1800.0) as client:
                resp = await client.post("http://training_service:8200/train", json=payload)
                
                if resp.status_code == 200:
                    async with db.acquire() as conn:
                        await conn.execute("UPDATE evolution_runs SET status = 'COMPLETED' WHERE id = $1", run_id)
                    log.info(f"Grid search {run_id} completed")
                else:
                    async with db.acquire() as conn:
                        await conn.execute("UPDATE evolution_runs SET status = 'FAILED' WHERE id = $1", run_id)
                    log.error(f"Grid search {run_id} failed: {resp.text}")
        except Exception as e:
            log.error(f"Grid search {run_id} error: {e}")
            async with db.acquire() as conn:
                await conn.execute("UPDATE evolution_runs SET status = 'FAILED' WHERE id = $1", run_id)
    
    background_tasks.add_task(run_grid_search)
    
    return {
        "status": "started",
        "run_id": run_id,
        "algorithm": algorithm,
        "grid_size": grid_size,
        "symbol": req.symbol,
        "message": f"Grid search started: training {grid_size} XGBoost models in background"
    }


@app.post("/grid-search/lightgbm")
async def grid_search_lightgbm(req: GridSearchLightGBMRequest, background_tasks: BackgroundTasks):
    """
    Train ALL LightGBM hyperparameter combinations.
    Returns immediately and runs training in background.
    """
    seed_features = req.seed_features
    if not seed_features:
        log.info(f"Fetching multi-ticker seed features: {req.symbol} + {req.reference_symbols or []}")
        seed_features = _get_multi_ticker_features(req.symbol, req.reference_symbols)
        
        if not seed_features:
            raise HTTPException(status_code=404, detail=f"No feature data found for {req.symbol}")
    
    data_options = req.data_options
    if req.reference_symbols:
        import json
        opts = json.loads(data_options) if data_options else {
            "use_rsi": True,
            "use_macd": True,
            "use_bb": True,
            "use_sma": True,
            "use_vol": True,
            "use_atr": True,
            "use_time": True,
            "enable_segmentation": True,
            "train_window": 20000,
            "test_window": 1000
        }
        opts['reference_symbols'] = req.reference_symbols
        data_options = json.dumps(opts)
    elif not data_options:
        import json
        data_options = json.dumps({
            "use_rsi": True,
            "use_macd": True,
            "use_bb": True,
            "use_sma": True,
            "use_vol": True,
            "use_atr": True,
            "use_time": True,
            "enable_segmentation": True,
            "train_window": 20000,
            "test_window": 1000
        })
    
    algorithm = "lightgbm_regressor" if req.regressor else "lightgbm_classifier"
    grid_size = len(req.num_leaves_grid) * len(req.min_data_in_leaf_grid) * \
                len(req.lambda_l1_grid) * len(req.lambda_l2_grid) * len(req.learning_rate_grid)
    
    # Create run record
    run_id = str(uuid.uuid4())
    async with db.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO evolution_runs 
            (id, symbol, status, max_generations, config, algorithm, target_col, target_transform, timeframe, seed_features)
            VALUES ($1, $2, 'RUNNING', $3, '{}'::jsonb, $4, $5, $6, $7, $8::jsonb)
            """,
            run_id, req.symbol, grid_size, algorithm, req.target_col, req.target_transform, req.timeframe, json.dumps(seed_features)
        )
    
    log.info(f"Created grid search run {run_id}: LightGBM {grid_size} models for {req.symbol}")
    
    # Run in background
    async def run_grid_search():
        try:
            payload = {
                "symbol": req.symbol,
                "algorithm": algorithm,
                "target_col": req.target_col,
                "hyperparameters": {},
                "target_transform": req.target_transform,
                "data_options": data_options,
                "timeframe": req.timeframe,
                "feature_whitelist": seed_features,
                "num_leaves_grid": req.num_leaves_grid,
                "min_data_in_leaf_grid": req.min_data_in_leaf_grid,
                "lambda_l1_grid": req.lambda_l1_grid,
                "lambda_l2_grid": req.lambda_l2_grid,
                "lgbm_learning_rate_grid": req.learning_rate_grid,
                "save_all_grid_models": True
            }
            
            async with httpx.AsyncClient(timeout=1800.0) as client:
                resp = await client.post("http://training_service:8200/train", json=payload)
                
                if resp.status_code == 200:
                    async with db.acquire() as conn:
                        await conn.execute("UPDATE evolution_runs SET status = 'COMPLETED' WHERE id = $1", run_id)
                    log.info(f"Grid search {run_id} completed")
                else:
                    async with db.acquire() as conn:
                        await conn.execute("UPDATE evolution_runs SET status = 'FAILED' WHERE id = $1", run_id)
                    log.error(f"Grid search {run_id} failed: {resp.text}")
        except Exception as e:
            log.error(f"Grid search {run_id} error: {e}")
            async with db.acquire() as conn:
                await conn.execute("UPDATE evolution_runs SET status = 'FAILED' WHERE id = $1", run_id)
    
    background_tasks.add_task(run_grid_search)
    
    return {
        "status": "started",
        "run_id": run_id,
        "algorithm": algorithm,
        "grid_size": grid_size,
        "symbol": req.symbol,
        "message": f"Grid search started: training {grid_size} LightGBM models in background"
    }


@app.post("/grid-search/randomforest")
async def grid_search_randomforest(req: GridSearchRandomForestRequest, background_tasks: BackgroundTasks):
    """
    Train ALL RandomForest hyperparameter combinations.
    Returns immediately and runs training in background.
    """
    seed_features = req.seed_features
    if not seed_features:
        log.info(f"Fetching multi-ticker seed features: {req.symbol} + {req.reference_symbols or []}")
        seed_features = _get_multi_ticker_features(req.symbol, req.reference_symbols)
        
        if not seed_features:
            raise HTTPException(status_code=404, detail=f"No feature data found for {req.symbol}")
    
    data_options = req.data_options
    if req.reference_symbols:
        import json
        opts = json.loads(data_options) if data_options else {
            "use_rsi": True,
            "use_macd": True,
            "use_bb": True,
            "use_sma": True,
            "use_vol": True,
            "use_atr": True,
            "use_time": True,
            "enable_segmentation": True,
            "train_window": 20000,
            "test_window": 1000
        }
        opts['reference_symbols'] = req.reference_symbols
        data_options = json.dumps(opts)
    elif not data_options:
        import json
        data_options = json.dumps({
            "use_rsi": True,
            "use_macd": True,
            "use_bb": True,
            "use_sma": True,
            "use_vol": True,
            "use_atr": True,
            "use_time": True,
            "enable_segmentation": True,
            "train_window": 20000,
            "test_window": 1000
        })
    
    algorithm = "random_forest_regressor" if req.regressor else "random_forest_classifier"
    grid_size = len(req.max_depth_grid) * len(req.min_samples_split_grid) * \
                len(req.min_samples_leaf_grid) * len(req.n_estimators_grid) * len(req.max_features_grid)
    
    # Create run record
    run_id = str(uuid.uuid4())
    async with db.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO evolution_runs 
            (id, symbol, status, max_generations, config, algorithm, target_col, target_transform, timeframe, seed_features)
            VALUES ($1, $2, 'RUNNING', $3, '{}'::jsonb, $4, $5, $6, $7, $8::jsonb)
            """,
            run_id, req.symbol, grid_size, algorithm, req.target_col, req.target_transform, req.timeframe, json.dumps(seed_features)
        )
    
    log.info(f"Created grid search run {run_id}: RandomForest {grid_size} models for {req.symbol}")
    
    # Run in background
    async def run_grid_search():
        try:
            payload = {
                "symbol": req.symbol,
                "algorithm": algorithm,
                "target_col": req.target_col,
                "hyperparameters": {},
                "target_transform": req.target_transform,
                "data_options": data_options,
                "timeframe": req.timeframe,
                "feature_whitelist": seed_features,
                "rf_max_depth_grid": req.max_depth_grid,
                "min_samples_split_grid": req.min_samples_split_grid,
                "min_samples_leaf_grid": req.min_samples_leaf_grid,
                "n_estimators_grid": req.n_estimators_grid,
                "max_features_grid": req.max_features_grid,
                "save_all_grid_models": True
            }
            
            async with httpx.AsyncClient(timeout=1800.0) as client:
                resp = await client.post("http://training_service:8200/train", json=payload)
                
                if resp.status_code == 200:
                    async with db.acquire() as conn:
                        await conn.execute("UPDATE evolution_runs SET status = 'COMPLETED' WHERE id = $1", run_id)
                    log.info(f"Grid search {run_id} completed")
                else:
                    async with db.acquire() as conn:
                        await conn.execute("UPDATE evolution_runs SET status = 'FAILED' WHERE id = $1", run_id)
                    log.error(f"Grid search {run_id} failed: {resp.text}")
        except Exception as e:
            log.error(f"Grid search {run_id} error: {e}")
            async with db.acquire() as conn:
                await conn.execute("UPDATE evolution_runs SET status = 'FAILED' WHERE id = $1", run_id)
    
    background_tasks.add_task(run_grid_search)
    
    return {
        "status": "started",
        "run_id": run_id,
        "algorithm": algorithm,
        "grid_size": grid_size,
        "symbol": req.symbol,
        "message": f"Grid search started: training {grid_size} RandomForest models in background"
    }


# ============================================
# Model Browser with Fingerprints & Metrics
# ============================================

@app.get("/models/browse")
async def browse_models(
    symbol: Optional[str] = None,
    algorithm: Optional[str] = None,
    status: Optional[str] = None,
    min_accuracy: Optional[float] = None,
    limit: int = 100
):
    """
    Browse all models with full details: fingerprint, metrics, feature importance.
    
    Use this to explore trained models and select which ones to run simulations with.
    """
    import httpx
    
    # Get models from training service
    params = {}
    if symbol:
        params["symbol"] = symbol
    if algorithm:
        params["algorithm"] = algorithm
    if status:
        params["status"] = status
    params["limit"] = limit
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            "http://training_service:8200/models",
            params=params
        )
        resp.raise_for_status()
        models = resp.json()
    
    # Filter by accuracy if requested
    if min_accuracy is not None:
        models = [
            m for m in models 
            if m.get("metrics", {}).get("accuracy", 0) >= min_accuracy
        ]
    
    # Enrich with feature importance for each model
    enriched_models = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in models:
            try:
                # Get feature importance
                imp_resp = await client.get(
                    f"http://training_service:8200/api/model/{model['id']}/importance"
                )
                if imp_resp.status_code == 200:
                    model["importance"] = imp_resp.json()
            except:
                model["importance"] = None
            
            enriched_models.append(model)
    
    return {
        "models": enriched_models,
        "count": len(enriched_models),
        "filtered_by": {
            "symbol": symbol,
            "algorithm": algorithm,
            "status": status,
            "min_accuracy": min_accuracy
        }
    }


# ============================================
# Manual Simulation Launcher
# ============================================

class ManualSimulationRequest(BaseModel):
    """Request to run grid search simulations for specific models."""
    model_ids: List[str]
    simulation_tickers: List[str]
    thresholds: List[float] = [0.0001, 0.0003, 0.0005, 0.0007]
    z_score_thresholds: List[float] = [0, 2.0, 2.5, 3.0, 3.5]
    regime_configs: List[Dict[str, Any]] = [
        {"regime_vix": [3]},  # Bull Quiet (Best)
        {"regime_gmm": [0]},  # Low volatility
    ]
    sqn_min: float = 1.9
    profit_factor_min: float = 1.3
    trade_count_min: int = 20


@app.post("/simulations/manual")
async def run_manual_simulations(req: ManualSimulationRequest):
    """
    Run grid search simulations for manually selected models.
    
    After training and reviewing model metrics/fingerprints,
    use this to run full simulation grid searches on the best candidates.
    """
    if not req.model_ids:
        raise HTTPException(status_code=400, detail="No model IDs provided")
    
    if not req.simulation_tickers:
        raise HTTPException(status_code=400, detail="No simulation tickers provided")
    
    # Calculate total simulations
    total_per_model = (
        len(req.simulation_tickers) *
        len(req.thresholds) *
        len(req.z_score_thresholds) *
        len(req.regime_configs)
    )
    total_sims = total_per_model * len(req.model_ids)
    
    log.info(f"Manual simulation request: {len(req.model_ids)} models × {total_per_model} sims/model = {total_sims} total")
    
    # Create simulation jobs for each model
    job_ids = []
    for model_id in req.model_ids:
        for ticker in req.simulation_tickers:
            for threshold in req.thresholds:
                for z_score in req.z_score_thresholds:
                    for regime_cfg in req.regime_configs:
                        job_id = await db.create_priority_job(
                            job_type="SIMULATION",
                            model_id=model_id,
                            ticker=ticker,
                            config={
                                "threshold": threshold,
                                "z_score_threshold": z_score,
                                "regime_config": regime_cfg,
                                "sqn_min": req.sqn_min,
                                "profit_factor_min": req.profit_factor_min,
                                "trade_count_min": req.trade_count_min
                            },
                            priority=5  # Normal priority
                        )
                        job_ids.append(job_id)
    
    return {
        "status": "queued",
        "model_count": len(req.model_ids),
        "ticker_count": len(req.simulation_tickers),
        "simulations_per_model": total_per_model,
        "total_simulations": total_sims,
        "job_ids": job_ids[:10],  # Return first 10 job IDs
        "total_jobs": len(job_ids)
    }


# ============================================
# Fingerprint Lookup
# ============================================

class FingerprintCheckRequest(BaseModel):
    features: List[str]
    hyperparameters: dict
    target_transform: str
    symbol: str
    target_col: str = "close"
    alpha_grid: Optional[List[float]] = None
    l1_ratio_grid: Optional[List[float]] = None
    regime_configs: Optional[List[Dict[str, Any]]] = None


@app.post("/fingerprint/check")
async def check_fingerprint(req: FingerprintCheckRequest):
    """Check if a model configuration already exists."""
    from .fingerprint import compute_fingerprint
    
    fp = compute_fingerprint(
        features=req.features,
        hyperparams=req.hyperparameters,
        target_transform=req.target_transform,
        symbol=req.symbol,
        target_col=req.target_col,
        alpha_grid=req.alpha_grid,
        l1_ratio_grid=req.l1_ratio_grid,
        regime_configs=req.regime_configs
    )
    
    existing_model_id = await db.get_model_by_fingerprint(fp)
    
    return {
        "fingerprint": fp,
        "exists": existing_model_id is not None,
        "model_id": existing_model_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8400)
