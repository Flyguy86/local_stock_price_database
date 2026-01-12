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
        req.target_transform
    )
    
    return {"id": training_id, "status": "started"}
