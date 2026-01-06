from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from .config import settings
from .db import db
from .trainer import start_training, train_model_task, ALGORITHMS

settings.ensure_paths()
logging.basicConfig(level=settings.log_level)
log = logging.getLogger("training.api")

app = FastAPI(title="Training Service")

class TrainRequest(BaseModel):
    symbol: str
    algorithm: str
    target_col: str = "close"
    hyperparameters: Optional[Dict[str, Any]] = None

@app.get("/algorithms")
def list_algorithms():
    return list(ALGORITHMS.keys())

@app.get("/models")
def list_models():
    return db.list_models()

@app.get("/models/{model_id}")
def get_model(model_id: str):
    model = db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    # Convert result tuples/rows to dict if necessary (DuckDB fetchone returns tuple)
    # The get_model in db.py returns a tuple, let's map it. 
    # Actually, simpler to return as part of list or fix db.py to return dict.
    # For now, let's rely on list_models for UI.
    return {"id": model_id, "data": str(model)}

@app.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Algorithm must be one of {list(ALGORITHMS.keys())}")
    
    training_id = start_training(req.symbol, req.algorithm, req.target_col, req.hyperparameters)
    
    background_tasks.add_task(
        train_model_task, 
        training_id, 
        req.symbol, 
        req.algorithm, 
        req.target_col, 
        req.hyperparameters or {}
    )
    
    return {"id": training_id, "status": "started"}
