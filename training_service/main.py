from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import uuid
import threading
from pathlib import Path
from collections import deque

from .config import settings
from .db import db
from .trainer import start_training, train_model_task, ALGORITHMS
from .data import get_data_options, get_feature_map
from training_service.models.hybrid import HybridRegressor

settings.ensure_paths()

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

app = FastAPI(title="Training Service")

# Mount Static Files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/logs")
def get_logs():
    with log_lock:
        return list(log_buffer)

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
def list_models():
    return db.list_models()

@app.get("/models/{model_id}")
def get_model(model_id: str):
    model = db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    # Convert result tuples/rows to dict if necessary (DuckDB fetchone returns tuple)
    return {"id": model_id, "data": str(model)}

@app.delete("/models/all")
def delete_all_models_endpoint():
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
    db.delete_all_models()
    return {"status": "all deleted"}

@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    # Delete from DB
    db.delete_model(model_id)
    
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
    conn = db.get_connection()
    try:
        # Fetch original parameters
        row = conn.execute("SELECT symbol, algorithm, target_col, hyperparameters, data_options, timeframe, target_transform FROM models WHERE id = ?", [model_id]).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Original model not found")
        
        symbol, algo, target, params_json, d_opt, tf, transform = row
        # Support legacy rows where transform might be null
        transform = transform or "none"
        
        # Parse params
        params = {}
        if params_json:
            try:
                params = json.loads(params_json)
            except Exception:
                log.warning(f"Could not parse hyperparameters for {model_id}, using empty dict")
                
        # Start new training job
        training_id = start_training(symbol, algo, target, params, d_opt, tf, parent_model_id=model_id, target_transform=transform)
    
        background_tasks.add_task(
            train_model_task, 
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
        )
        return {"id": training_id, "status": "started", "retrained_from": model_id}
            
    except Exception as e:
        log.error(f"Retrain failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

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
        tid = start_training(
            symbol=req.symbol,
            algorithm=req.algorithm,
            target_col=cfg["target"],
            params=params,
            data_options=req.data_options,
            timeframe=cfg["tf"],
            parent_model_id=req.parent_model_id,
            group_id=group_id
        )
        started_ids.append(tid)
        
        background_tasks.add_task(
            train_model_task,
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
        )

    return {"group_id": group_id, "ids": started_ids, "status": "started batch"}

@app.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Algorithm must be one of {list(ALGORITHMS.keys())}")
    
    # Inject p_value_threshold into parameters for persistence and task usage
    params = req.hyperparameters or {}
    params["p_value_threshold"] = req.p_value_threshold
    
    training_id = start_training(
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
    
    background_tasks.add_task(
        train_model_task, 
        training_id, 
        req.symbol, 
        req.algorithm, 
        req.target_col, 
        params,
        req.data_options, 
        req.timeframe,
        req.parent_model_id, 
        req.feature_whitelist,
        req.group_id,  # Pass group_id
        req.target_transform,
        req.alpha_grid,  # Grid search: L2 penalty values
        req.l1_ratio_grid  # Grid search: L1/L2 mix values
    )
    
    return {"id": training_id, "status": "started"}


# ============================================
# NEW ENDPOINTS FOR ORCHESTRATOR INTEGRATION
# ============================================

@app.get("/api/model/{model_id}/importance")
def get_model_importance(model_id: str):
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
    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT metrics, feature_cols FROM models WHERE id = ?", 
            [model_id]
        ).fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")
        
        metrics_json, feature_cols_json = row
        
        # Parse metrics
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
    finally:
        conn.close()


@app.get("/api/model/{model_id}/config")
def get_model_config(model_id: str):
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
    conn = db.get_connection()
    try:
        row = conn.execute(
            """SELECT symbol, algorithm, target_col, feature_cols, 
                      hyperparameters, target_transform, data_options, timeframe
               FROM models WHERE id = ?""",
            [model_id]
        ).fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")
        
        symbol, algorithm, target_col, feature_cols_json, hyperparams_json, transform, data_opts, tf = row
        
        # Parse JSON fields
        features = []
        if feature_cols_json:
            try:
                features = json.loads(feature_cols_json) if isinstance(feature_cols_json, str) else feature_cols_json
            except:
                features = []
        
        hyperparams = {}
        if hyperparams_json:
            try:
                hyperparams = json.loads(hyperparams_json) if isinstance(hyperparams_json, str) else hyperparams_json
            except:
                hyperparams = {}
        
        return {
            "model_id": model_id,
            "features": features,
            "hyperparameters": hyperparams,
            "target_transform": transform or "none",
            "symbol": symbol,
            "target_col": target_col,
            "algorithm": algorithm,
            "data_options": data_opts,
            "timeframe": tf
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting config for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


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
    
    training_id = start_training(
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
    
    background_tasks.add_task(
        train_model_task,
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
    )
    
    log.info(f"Started training {training_id} with parent {req.parent_model_id}, {len(req.feature_whitelist)} features")
    
    return {
        "id": training_id,
        "status": "started",
        "parent_model_id": req.parent_model_id,
        "feature_count": len(req.feature_whitelist)
    }
