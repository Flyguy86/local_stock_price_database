from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import uuid
import threading
import asyncio
import os
from pathlib import Path
from collections import deque
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor

from .config import settings
from .pg_db import TrainingDB, ensure_tables, get_pool, close_pool
from .trainer import train_model_task, ALGORITHMS
from .data import get_data_options, get_feature_map
from training_service.models.hybrid import HybridRegressor

settings.ensure_paths()

# Process pool for parallel training (scales with CPU count)
CPU_COUNT = os.cpu_count() or 4
_process_pool = ProcessPoolExecutor(
    max_workers=CPU_COUNT,
    max_tasks_per_child=10  # Restart workers periodically to prevent memory leaks
)


async def submit_training_task(
    training_id: str,
    symbol: str,
    algorithm: str,
    target_col: str,
    params: dict,
    data_options: Optional[str] = None,
    timeframe: str = "1m",
    parent_model_id: Optional[str] = None,
    feature_whitelist: Optional[list] = None,
    group_id: Optional[str] = None,
    target_transform: str = "none",
    alpha_grid: Optional[list] = None,
    l1_ratio_grid: Optional[list] = None
):
    """Submit training task to process pool and monitor it."""
    loop = asyncio.get_event_loop()
    
    try:
        # Submit to process pool (runs in separate process for true parallelism)
        await loop.run_in_executor(
            _process_pool,
            train_model_task,
            training_id, symbol, algorithm, target_col, params,
            data_options, timeframe, parent_model_id, feature_whitelist,
            group_id, target_transform, alpha_grid, l1_ratio_grid
        )
        log.info(f"Training {training_id} completed successfully")
    except Exception as e:
        log.error(f"Training {training_id} failed: {e}")
        # Update status to failed (task may have already done this, but ensure it)
        try:
            await db.update_model_status(training_id, status="failed", error=str(e))
        except:
            pass

async def async_start_training(
    symbol: str,
    algorithm: str,
    target_col: str = "close",
    params: dict = None,
    data_options: str = None,
    timeframe: str = "1m",
    parent_model_id: str = None,
    group_id: str = None,
    target_transform: str = "none"
) -> str:
    """Create a new training job record in the database."""
    import json
    from datetime import datetime
    
    if params is None:
        params = {}
    
    training_id = str(uuid.uuid4())
    
    # Create initial model record
    await db.create_model_record({
        "id": training_id,
        "name": f"{symbol}-{algorithm}-{target_col}-{timeframe}-{datetime.now().strftime('%Y%m%d%H%M')}",
        "algorithm": algorithm,
        "symbol": symbol,
        "target_col": target_col,
        "feature_cols": json.dumps([]),
        "hyperparameters": json.dumps(params),
        "status": "pending",
        "metrics": json.dumps({}),
        "data_options": data_options,
        "parent_model_id": parent_model_id,
        "group_id": group_id,
        "timeframe": timeframe,
        "target_transform": target_transform
    })
    
    return training_id

# --- Custom Log Handler ---
log_buffer = deque(maxlen=200)
log_lock = threading.Lock()

class BufferHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            with log_lock:
                log_buffer.append(msg)
        except Exception:
            self.handleError(record)

# Setup Logging
logging.basicConfig(level=settings.log_level)
# Add buffer handler to root logger so we catch everything (api, trainer, data)
_handler = BufferHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.getLogger().addHandler(_handler)

log = logging.getLogger("training.api")

# Ensure static/js directory exists
(Path(__file__).parent / "static" / "js").mkdir(parents=True, exist_ok=True)
# Ensure templates directory exists
(Path(__file__).parent / "templates").mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    log.info("Training Service starting...")
    log.info(f"Process pool configured with {CPU_COUNT} workers")
    log.info("Initializing PostgreSQL connection pool...")
    await ensure_tables()
    log.info("PostgreSQL tables ready")
    
    yield
    
    # Shutdown
    log.info("Shutting down process pool...")
    _process_pool.shutdown(wait=True, cancel_futures=True)
    log.info("Closing PostgreSQL connection pool...")
    await close_pool()
    log.info("Training Service shutdown complete")

app = FastAPI(title="Training Service", lifespan=lifespan)

# Add CORS middleware to allow requests from orchestrator frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a global db instance
db = TrainingDB()

# Mount Static Files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/logs")
def get_logs():
    """Get recent training logs from buffer."""
    with log_lock:
        logs_list = list(log_buffer)
        if not logs_list:
            return ["[No training logs yet]"]
        return logs_list

@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(Path(__file__).parent / "templates" / "dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/data/options")
def list_global_options():
    return get_data_options()

@app.get("/data/options/{symbol}")
def list_options(symbol: str):
    return get_data_options(symbol)

@app.get("/data/map")
def get_map():
    log.info("Request received: GET /data/map (Scanning Parquet features)")
    return get_feature_map()

class TrainRequest(BaseModel):
    symbol: str
    algorithm: str
    target_col: str = "close"
    hyperparameters: Optional[Dict[str, Any]] = None
    data_options: Optional[str] = None
    timeframe: str = "1m"
    p_value_threshold: float = 0.05
    parent_model_id: Optional[str] = None
    feature_whitelist: Optional[list[str]] = None
    group_id: Optional[str] = None
    target_transform: str = "none" # none, log_return, pct_change
    # Grid search parameters for regularization
    alpha_grid: Optional[list[float]] = None  # L2 penalty values (Ridge/ElasticNet alpha)
    l1_ratio_grid: Optional[list[float]] = None  # L1/L2 mix for ElasticNet (0=Ridge, 1=Lasso)

# Validating batch request schema
class TrainBatchRequest(BaseModel):
    symbol: str
    algorithm: str
    hyperparameters: Optional[Dict[str, Any]] = None
    data_options: Optional[str] = None
    p_value_threshold: float = 0.05
    parent_model_id: Optional[str] = None
    feature_whitelist: Optional[list[str]] = None
    timeframe_oc: str = "1m"
    timeframe_hl: str = "1d"
    target_transform: str = "log_return"
    target_transform: str = "log_return" # Default to 'log_return' for batch to encourage stationarity

@app.get("/algorithms")
def list_algorithms():
    try:
        keys = list(ALGORITHMS.keys())
        log.info(f"Serving algorithms list: {keys}")
        return keys
    except Exception as e:
        log.error(f"Failed to list algorithms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    return await db.list_models()

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    model = await db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@app.delete("/models/all")
async def delete_all_models_endpoint():
    # 1. Delete all files in models dir
    try:
        # Check if dir exists first
        if settings.models_dir.exists():
            for f in settings.models_dir.glob("*.joblib"):
                try:
                    f.unlink()
                except Exception as e:
                    log.warning(f"Could not delete {f}: {e}")
    except Exception as e:
        log.error(f"Failed to clear models dir: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete files: {e}")

    # 2. Clear DB
    await db.delete_all_models()
    return {"status": "all deleted"}

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    # Delete from DB
    await db.delete_model(model_id)
    
    # Delete file
    try:
        model_path = settings.models_dir / f"{model_id}.joblib"
        if model_path.exists():
            model_path.unlink()
    except Exception as e:
        log.error(f"Failed to delete model file {model_id}: {e}")
        
    return {"status": "deleted", "id": model_id}

@app.post("/retrain/{model_id}")
async def retrain_model(model_id: str, background_tasks: BackgroundTasks):
    import json
    
    # Fetch original parameters
    model = await db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Original model not found")
    
    symbol = model.get('symbol')
    algo = model.get('algorithm')
    target = model.get('target_col')
    params_json = model.get('hyperparameters')
    d_opt = model.get('data_options')
    tf = model.get('timeframe')
    transform = model.get('target_transform') or "none"
    
    # Parse params
    params = {}
    if params_json:
        try:
            if isinstance(params_json, str):
                params = json.loads(params_json)
            else:
                params = params_json
        except Exception:
            log.warning(f"Could not parse hyperparameters for {model_id}, using empty dict")
    
    try:
        # Start new training job
        training_id = await async_start_training(
            symbol, algo, target, params, d_opt, tf, 
            parent_model_id=model_id, target_transform=transform
        )
    
        # Submit to process pool for parallel execution
        asyncio.create_task(submit_training_task(
            training_id, 
            symbol, 
            algo, 
            target, 
            params,
            d_opt,
            tf,
            model_id,  # parent is now the original model
            None,  # features
            None,  # group
            transform
        ))
        return {"id": training_id, "status": "started", "retrained_from": model_id}
            
    except Exception as e:
        log.error(f"Retrain failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/batch")
async def train_batch(req: TrainBatchRequest, background_tasks: BackgroundTasks):
    if req.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Algorithm must be one of {list(ALGORITHMS.keys())}")

    group_id = str(uuid.uuid4())
    
    # Define the 4 models configuration
    configs = [
        {"target": "open", "tf": req.timeframe_oc},
        {"target": "close", "tf": req.timeframe_oc},
        {"target": "high", "tf": req.timeframe_hl},
        {"target": "low", "tf": req.timeframe_hl}
    ]
    
    started_ids = []
    
    params = req.hyperparameters or {}
    params["p_value_threshold"] = req.p_value_threshold

    for cfg in configs:
        tid = await async_start_training(
            symbol=req.symbol,
            algorithm=req.algorithm,
            target_col=cfg["target"],
            params=params,
            data_options=req.data_options,
            timeframe=cfg["tf"],
            parent_model_id=req.parent_model_id,
            group_id=group_id,
            target_transform=req.target_transform
        )
        started_ids.append(tid)
        
        # Submit to process pool for parallel execution
        asyncio.create_task(submit_training_task(
            tid,
            req.symbol,
            req.algorithm,
            cfg["target"],
            params,
            req.data_options,
            cfg["tf"],
            req.parent_model_id,
            req.feature_whitelist,
            group_id,  # Pass group_id
            req.target_transform
        ))

    return {"group_id": group_id, "ids": started_ids, "status": "started batch"}

@app.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Algorithm must be one of {list(ALGORITHMS.keys())}")
    
    # Inject p_value_threshold into parameters for persistence and task usage
    params = req.hyperparameters or {}
    params["p_value_threshold"] = req.p_value_threshold
    
    training_id = await async_start_training(
        req.symbol, 
        req.algorithm, 
        req.target_col, 
        params, 
        req.data_options, 
        req.timeframe, 
        req.parent_model_id, 
        req.group_id,
        target_transform=req.target_transform
    )
    
    # Submit to process pool for parallel execution
    asyncio.create_task(submit_training_task(
        training_id, 
        req.symbol, 
        req.algorithm, 
        req.target_col, 
        params,
        req.data_options, 
        req.timeframe,
        req.parent_model_id, 
        req.feature_whitelist,
        req.group_id,
        req.target_transform,
        req.alpha_grid,
        req.l1_ratio_grid
    ))
    
    return {"id": training_id, "status": "started"}


# ============================================
# NEW ENDPOINTS FOR ORCHESTRATOR INTEGRATION
# ============================================

@app.get("/api/model/{model_id}/importance")
async def get_model_importance(model_id: str):
    """
    Get feature importance scores for a trained model.
    Used by orchestrator to determine which features to prune.
    
    Returns:
        {
            "model_id": "abc-123",
            "importance": {"feature_name": importance_value, ...},
            "importance_type": "tree_importance" | "permutation_mean" | "coefficient"
        }
    """
    import json
    try:
        model = await db.get_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Parse metrics
        metrics_json = model.get('metrics')
        if metrics_json:
            try:
                metrics = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json
            except:
                metrics = {}
        else:
            metrics = {}
        
        # Extract importance from metrics
        importance = {}
        importance_type = None
        
        # Try feature_importance (legacy/simple format)
        if "feature_importance" in metrics:
            importance = metrics["feature_importance"]
            importance_type = "feature_importance"
        
        # Try feature_details for more detailed importance
        if "feature_details" in metrics:
            feature_details = metrics["feature_details"]
            for feat_name, details in feature_details.items():
                # Prefer permutation_mean, then tree_importance, then coefficient
                if "permutation_mean" in details:
                    importance[feat_name] = details["permutation_mean"]
                    importance_type = importance_type or "permutation_mean"
                elif "tree_importance" in details:
                    importance[feat_name] = details["tree_importance"]
                    importance_type = importance_type or "tree_importance"
                elif "coefficient" in details:
                    importance[feat_name] = abs(details["coefficient"])
                    importance_type = importance_type or "coefficient"
        
        return {
            "model_id": model_id,
            "importance": importance,
            "importance_type": importance_type,
            "feature_count": len(importance)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting importance for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/{model_id}/config")
async def get_model_config(model_id: str):
    """
    Get the full configuration used to train a model.
    Used by orchestrator for fingerprint computation.
    
    Returns:
        {
            "model_id": "abc-123",
            "features": ["sma_20", "rsi_14", ...],
            "hyperparameters": {...},
            "target_transform": "log_return",
            "symbol": "RDDT",
            "target_col": "close",
            "algorithm": "RandomForest"
        }
    """
    import json
    try:
        model = await db.get_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Parse JSON fields
        features = model.get('feature_cols', [])
        if isinstance(features, str):
            try:
                features = json.loads(features)
            except:
                features = []
        
        hyperparams = model.get('hyperparameters', {})
        if isinstance(hyperparams, str):
            try:
                hyperparams = json.loads(hyperparams)
            except:
                hyperparams = {}
        
        return {
            "model_id": model_id,
            "features": features,
            "hyperparameters": hyperparams,
            "target_transform": model.get('target_transform') or "none",
            "symbol": model.get('symbol'),
            "target_col": model.get('target_col'),
            "algorithm": model.get('algorithm'),
            "data_options": model.get('data_options'),
            "timeframe": model.get('timeframe')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting config for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TrainWithParentRequest(BaseModel):
    """Request for training with explicit parent lineage."""
    symbol: str
    algorithm: str
    target_col: str = "close"
    hyperparameters: Optional[Dict[str, Any]] = None
    data_options: Optional[str] = None
    timeframe: str = "1m"
    p_value_threshold: float = 0.05
    parent_model_id: str  # Required for lineage
    feature_whitelist: list[str]  # Required - the pruned features
    target_transform: str = "log_return"


@app.post("/api/train_with_parent")
async def train_with_parent(req: TrainWithParentRequest, background_tasks: BackgroundTasks):
    """
    Train a new model with explicit parent lineage.
    Used by orchestrator for evolution chain.
    
    This is similar to /train but:
    - parent_model_id is required
    - feature_whitelist is required (the pruned feature set)
    """
    if req.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Algorithm must be one of {list(ALGORITHMS.keys())}")
    
    if not req.feature_whitelist:
        raise HTTPException(status_code=400, detail="feature_whitelist is required for lineage training")
    
    params = req.hyperparameters or {}
    params["p_value_threshold"] = req.p_value_threshold
    
    training_id = await async_start_training(
        req.symbol,
        req.algorithm,
        req.target_col,
        params,
        req.data_options,
        req.timeframe,
        req.parent_model_id,
        None,  # group_id
        target_transform=req.target_transform
    )
    
    # Submit to process pool for parallel execution
    asyncio.create_task(submit_training_task(
        training_id,
        req.symbol,
        req.algorithm,
        req.target_col,
        params,
        req.data_options,
        req.timeframe,
        req.parent_model_id,
        req.feature_whitelist,
        None,  # group_id
        req.target_transform
    ))
    
    log.info(f"Started training {training_id} with parent {req.parent_model_id}, {len(req.feature_whitelist)} features")
    
    return {
        "id": training_id,
        "status": "started",
        "parent_model_id": req.parent_model_id,
        "feature_count": len(req.feature_whitelist)
    }
