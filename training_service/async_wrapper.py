"""
Async wrappers for training operations.
Bridges sync training code with async PostgreSQL database.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
import json

from .trainer import train_model_task
from .pg_db import TrainingDB

log = logging.getLogger("training.async_wrapper")

# Thread pool for CPU-bound training tasks
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trainer")


async def async_train_model_task(
    db: TrainingDB,
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
    """
    Async wrapper for train_model_task that handles database operations.
    
    The training itself runs in a thread pool (CPU-bound),
    but database updates are async.
    """
    from .config import settings
    import joblib
    from pathlib import Path
    import uuid as uuid_module
    from datetime import datetime
    
    model_path = str(settings.models_dir / f"{training_id}.joblib")
    
    try:
        log.info(f"Starting async training {training_id} for {symbol} using {algorithm}")
        
        # Update status to preprocessing
        await db.update_model_status(training_id, status="preprocessing")
        
        # Load parent features if needed
        parent_features = None
        if feature_whitelist:
            parent_features = feature_whitelist
            log.info(f"Using {len(feature_whitelist)} whitelisted features")
        elif parent_model_id:
            try:
                parent_model = await db.get_model(parent_model_id)
                if parent_model and parent_model.get('feature_cols'):
                    parent_features = parent_model['feature_cols']
                    if isinstance(parent_features, str):
                        parent_features = json.loads(parent_features)
                    log.info(f"Loaded {len(parent_features)} features from parent {parent_model_id}")
            except Exception as e:
                log.warning(f"Failed to load parent features: {e}")
        
        # Run the actual training in thread pool (it's CPU-bound)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _executor,
            train_model_task,
            training_id, symbol, algorithm, target_col, params,
            data_options, timeframe, parent_model_id, parent_features,
            group_id, target_transform, alpha_grid, l1_ratio_grid
        )
        
        log.info(f"Training {training_id} completed successfully")
        
    except Exception as e:
        log.exception(f"Training {training_id} failed: {e}")
        await db.update_model_status(
            training_id,
            status="failed",
            error=str(e)
        )
        raise
