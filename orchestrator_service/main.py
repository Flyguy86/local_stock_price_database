"""
Orchestrator Service - FastAPI Application
Recursive Strategy Factory for automated model evolution.
"""
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import db
from .evolution import engine, EvolutionConfig
from .criteria import HolyGrailCriteria

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
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
    await db.connect()
    await engine.start()
    
    # Update service URLs from environment
    engine.training_url = os.getenv("TRAINING_URL", "http://training:8200")
    engine.simulation_url = os.getenv("SIMULATION_URL", "http://simulation:8300")
    
    log.info(f"Training URL: {engine.training_url}")
    log.info(f"Simulation URL: {engine.simulation_url}")
    
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
    algorithm: str = "RandomForest"
    target_col: str = "close"
    hyperparameters: dict = {}
    target_transform: str = "log_return"
    max_generations: int = 4
    data_options: Optional[str] = None
    timeframe: str = "1m"
    thresholds: List[float] = [0.0001, 0.0003, 0.0005, 0.0007]
    regime_configs: List[dict] = [
        {"regime_gmm": [0]},
        {"regime_gmm": [1]},
        {"regime_vix": [0, 1]}
    ]
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
    if not req.seed_model_id and not seed_features:
        log.info(f"Fetching seed features for {req.symbol} from parquet")
        seed_features = _get_feature_columns_from_parquet(req.symbol)
        
        if seed_features:
            log.info(f"Auto-fetched {len(seed_features)} features for {req.symbol}")
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
        algorithm=req.algorithm,
        target_col=req.target_col,
        hyperparameters=req.hyperparameters,
        target_transform=req.target_transform,
        max_generations=req.max_generations,
        data_options=data_options,
        timeframe=req.timeframe,
        thresholds=req.thresholds,
        regime_configs=req.regime_configs,
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
# Fingerprint Lookup
# ============================================

class FingerprintCheckRequest(BaseModel):
    features: List[str]
    hyperparameters: dict
    target_transform: str
    symbol: str
    target_col: str = "close"


@app.post("/fingerprint/check")
async def check_fingerprint(req: FingerprintCheckRequest):
    """Check if a model configuration already exists."""
    from .fingerprint import compute_fingerprint
    
    fp = compute_fingerprint(
        features=req.features,
        hyperparams=req.hyperparameters,
        target_transform=req.target_transform,
        symbol=req.symbol,
        target_col=req.target_col
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
