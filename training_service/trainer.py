import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import json
import logging
from datetime import datetime
import uuid

from .config import settings
from .db import db
from .data import load_training_data

log = logging.getLogger("training.core")

ALGORITHMS = {
    "linear_regression": LinearRegression,
    "random_forest_regressor": RandomForestRegressor,
    "logistic_classification": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier
}

def train_model_task(training_id: str, symbol: str, algorithm: str, target_col: str, params: dict):
    model_path = str(settings.models_dir / f"{training_id}.joblib")
    
    try:
        log.info(f"Starting training {training_id} for {symbol} using {algorithm}")
        
        # 1. Load Data
        df = load_training_data(symbol, target_col=target_col)
        
        # 2. Prepare Features
        # Drop metadata columns and target
        drop_cols = ["target", "ts", "symbol", "date", "source"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        X = df[feature_cols]
        y = df["target"]
        
        # Simple time-based split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 3. Initialize Model
        ModelClass = ALGORITHMS.get(algorithm)
        if not ModelClass:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        model = ModelClass(**params)
        
        # 4. Train
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        preds = model.predict(X_test)
        
        metrics = {}
        if "regressor" in algorithm or "regression" in algorithm:
            mse = mean_squared_error(y_test, preds)
            metrics["mse"] = mse
            metrics["rmse"] = float(mse ** 0.5)
        else:
            acc = accuracy_score(y_test, preds)
            metrics["accuracy"] = acc
            
        # 6. Save
        joblib.dump(model, model_path)
        
        # 7. Update DB
        db.update_model_status(
            training_id, 
            status="completed", 
            metrics=json.dumps(metrics), 
            artifact_path=model_path
        )
        log.info(f"Training {training_id} completed. Metrics: {metrics}")
        
    except Exception as e:
        log.exception(f"Training {training_id} failed")
        db.update_model_status(
            training_id, 
            status="failed", 
            error=str(e)
        )

def start_training(symbol: str, algorithm: str, target_col: str = "close", params: dict = None):
    if params is None:
        params = {}
        
    training_id = str(uuid.uuid4())
    
    # Init Record
    db.create_model_record({
        "id": training_id,
        "name": f"{symbol}-{algorithm}-{datetime.now().strftime('%Y%m%d%H%M')}",
        "algorithm": algorithm,
        "symbol": symbol,
        "target_col": target_col,
        "feature_cols": json.dumps([]), # Filled later?
        "hyperparameters": json.dumps(params),
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "metrics": json.dumps({})
    })
    
    # Run synchronously for now (or convert to background task if using FastAPI BackgroundTasks)
    # We will call this via BackgroundTasks in main.py
    return training_id
