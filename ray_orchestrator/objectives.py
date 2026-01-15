"""
Training Objective Functions for Ray Tune.

Each objective function defines a complete training pipeline
that receives hyperparameters and reports metrics back to Tune.

Deduplication:
    - Uses fingerprint database to check for already-tested configs
    - Records results after successful training
    - Allows early exit if duplicate detected
"""

import logging
from typing import Any, Optional
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = None
    XGBClassifier = None

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:
    LGBMRegressor = None
    LGBMClassifier = None

import ray
from ray import tune
from ray.train import Checkpoint

from .data import load_symbol_data_pandas, prepare_training_data
from .config import settings
from .fingerprint import check_and_skip_duplicate, record_trial_result

log = logging.getLogger("ray_orchestrator.objectives")


# Algorithm registry
ALGORITHMS = {
    "linear_regression": LinearRegression,
    "elasticnet": ElasticNet,
    "logistic": LogisticRegression,
    "random_forest_regressor": RandomForestRegressor,
    "random_forest_classifier": RandomForestClassifier,
}

if XGBRegressor:
    ALGORITHMS["xgboost_regressor"] = XGBRegressor
    ALGORITHMS["xgboost_classifier"] = XGBClassifier

if LGBMRegressor:
    ALGORITHMS["lightgbm_regressor"] = LGBMRegressor
    ALGORITHMS["lightgbm_classifier"] = LGBMClassifier


def get_algorithm_class(name: str):
    """Get algorithm class by name."""
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[name]


def is_regression(algorithm: str) -> bool:
    """Check if algorithm is regression (vs classification)."""
    return "regressor" in algorithm or "regression" in algorithm or algorithm == "elasticnet"


def build_preprocessing_pipeline(X: pd.DataFrame, algorithm: str) -> ColumnTransformer:
    """
    Build heterogeneous scaling pipeline based on feature types.
    
    Groups:
    - Robust: volume, counts (heavy outliers)
    - Passthrough: bounded oscillators (RSI, IBS, etc.)
    - Standard: returns, z-scores, everything else
    """
    cols_robust = []
    cols_passthrough = []
    cols_standard = []
    
    for col in X.columns:
        cl = col.lower()
        if "volume" in cl or "count" in cl or "pro_vol" in cl:
            cols_robust.append(col)
        elif any(x in cl for x in ["rsi", "ibs", "aroon", "stoch", "bop", "mfi", "willr"]):
            cols_passthrough.append(col)
        else:
            cols_standard.append(col)
    
    transformers = []
    if cols_robust:
        transformers.append(("robust", RobustScaler(), cols_robust))
    if cols_standard:
        transformers.append(("standard", StandardScaler(), cols_standard))
    if cols_passthrough:
        transformers.append(("bounded", "passthrough", cols_passthrough))
    
    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough"
    )


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 390  # 1-minute bars
) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252*390 for 1m bars)
        
    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    # Annualize
    excess_returns = np.mean(returns) - (risk_free_rate / periods_per_year)
    annualized_return = excess_returns * periods_per_year
    annualized_volatility = np.std(returns) * np.sqrt(periods_per_year)
    
    if annualized_volatility == 0:
        return 0.0
    
    return annualized_return / annualized_volatility


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    if len(cumulative_returns) == 0:
        return 0.0
    
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (running_max - cumulative_returns) / np.maximum(running_max, 1e-9)
    
    return float(np.max(drawdowns))


def train_trading_model(config: dict) -> None:
    """
    Training objective function for Ray Tune.
    
    This is the core function that trains a single model with given hyperparameters
    and reports metrics back to Tune.
    
    Deduplication:
        Before any heavy training, we check the fingerprint database to see
        if this exact config has already been tested. If so, we report the
        cached result and return early, saving compute.
    
    Config Parameters:
        - ticker: Symbol to train on
        - algorithm: Algorithm name
        - target_col: Column to predict
        - target_transform: "log_return", "pct_change", "raw"
        - timeframe: "1m", "5m", "15m", "1h", "1d"
        - start_date: Training start date
        - end_date: Training end date
        - Algorithm-specific hyperparameters (alpha, l1_ratio, max_depth, etc.)
    """
    # ===== DEDUPLICATION CHECK =====
    # Check if this config has already been tested (fingerprint lookup)
    # This is the "Database Method" for complex multi-generational setups
    cached_result = check_and_skip_duplicate(config)
    if cached_result:
        # Report cached result and return early - no heavy training needed!
        log.info(f"Skipping duplicate: returning cached result for {config.get('ticker', '?')}")
        tune.report(**cached_result)
        return
    
    # ===== EXTRACT CONFIG =====
    ticker = config.get("ticker", "AAPL")
    algorithm = config.get("algorithm", "elasticnet")
    target_col = config.get("target_col", "close")
    target_transform = config.get("target_transform", "log_return")
    timeframe = config.get("timeframe", "1m")
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    lookforward = config.get("lookforward", 1)
    
    log.info(f"Training {algorithm} on {ticker} ({timeframe})")
    
    # Load data
    df = load_symbol_data_pandas(ticker, start_date, end_date, timeframe)
    
    if df.empty or len(df) < 100:
        # Report failure metrics
        tune.report(
            sharpe_ratio=-999,
            r2=-999,
            rmse=999,
            status="failed",
            error="Insufficient data"
        )
        return
    
    # Prepare training data
    X_train, X_test, y_train, y_test = prepare_training_data(
        df, target_col, target_transform, lookforward
    )
    
    if X_train.empty or X_test.empty:
        tune.report(
            sharpe_ratio=-999,
            r2=-999,
            rmse=999,
            status="failed",
            error="Empty train/test split"
        )
        return
    
    # Build model with hyperparameters
    try:
        AlgoClass = get_algorithm_class(algorithm)
        
        # Extract algorithm-specific hyperparameters
        algo_params = {}
        
        if algorithm == "elasticnet":
            algo_params["alpha"] = config.get("alpha", 0.01)
            algo_params["l1_ratio"] = config.get("l1_ratio", 0.5)
            algo_params["max_iter"] = config.get("max_iter", 2000)
            
        elif algorithm in ["xgboost_regressor", "xgboost_classifier"]:
            algo_params["max_depth"] = config.get("max_depth", 6)
            algo_params["learning_rate"] = config.get("learning_rate", 0.1)
            algo_params["n_estimators"] = config.get("n_estimators", 100)
            algo_params["min_child_weight"] = config.get("min_child_weight", 1)
            algo_params["reg_alpha"] = config.get("reg_alpha", 0.0)
            algo_params["reg_lambda"] = config.get("reg_lambda", 1.0)
            algo_params["subsample"] = config.get("subsample", 1.0)
            algo_params["n_jobs"] = -1
            
        elif algorithm in ["lightgbm_regressor", "lightgbm_classifier"]:
            algo_params["num_leaves"] = config.get("num_leaves", 31)
            algo_params["max_depth"] = config.get("max_depth", -1)
            algo_params["learning_rate"] = config.get("learning_rate", 0.1)
            algo_params["n_estimators"] = config.get("n_estimators", 100)
            algo_params["min_data_in_leaf"] = config.get("min_data_in_leaf", 20)
            algo_params["lambda_l1"] = config.get("lambda_l1", 0.0)
            algo_params["lambda_l2"] = config.get("lambda_l2", 0.0)
            algo_params["n_jobs"] = -1
            algo_params["verbosity"] = -1
            
        elif algorithm in ["random_forest_regressor", "random_forest_classifier"]:
            algo_params["max_depth"] = config.get("max_depth")
            algo_params["n_estimators"] = config.get("n_estimators", 100)
            algo_params["min_samples_split"] = config.get("min_samples_split", 2)
            algo_params["min_samples_leaf"] = config.get("min_samples_leaf", 1)
            algo_params["max_features"] = config.get("max_features", "sqrt")
            algo_params["n_jobs"] = -1
        
        # Build pipeline
        preprocessor = build_preprocessing_pipeline(X_train, algorithm)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("preprocessor", preprocessor),
            ("model", AlgoClass(**algo_params))
        ])
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        tune.report(
            sharpe_ratio=-999,
            r2=-999,
            rmse=999,
            status="failed",
            error=str(e)
        )
        return
    
    # Calculate metrics
    metrics = {"status": "completed"}
    
    if is_regression(algorithm):
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, preds)
        
        metrics["mse"] = float(mse)
        metrics["mae"] = float(mae)
        metrics["rmse"] = rmse
        metrics["r2"] = float(r2)
        
        # Trading-specific metrics
        # Simulate: if prediction > 0, go long; else flat
        # Returns = actual return * signal
        signals = (preds > 0).astype(float)
        strategy_returns = np.array(y_test) * signals
        
        sharpe = calculate_sharpe_ratio(strategy_returns)
        cumulative = np.cumprod(1 + strategy_returns)
        max_dd = calculate_max_drawdown(cumulative)
        
        metrics["sharpe_ratio"] = sharpe
        metrics["max_drawdown"] = max_dd
        metrics["win_rate"] = float(np.mean((strategy_returns > 0)))
        metrics["total_return"] = float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0
        
    else:
        # Classification metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        
        metrics["accuracy"] = float(acc)
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1_score"] = float(f1)
        
        # For classification, use accuracy as proxy for sharpe
        metrics["sharpe_ratio"] = float(acc - 0.5) * 2  # Scale to [-1, 1]
        metrics["rmse"] = 1.0 - acc
        metrics["r2"] = float(acc)
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        joblib.dump(model, model_path)
        
        checkpoint = Checkpoint.from_directory(tmpdir)
        
        # ===== RECORD TO FINGERPRINT DATABASE =====
        # Store the result so future runs with the same config can skip training
        record_trial_result(config, metrics)
        
        # Report metrics to Tune
        tune.report(
            **metrics,
            checkpoint=checkpoint
        )
    
    log.info(f"Training completed: {ticker}/{algorithm} - Sharpe: {metrics.get('sharpe_ratio', 0):.4f}")


def multi_ticker_objective(config: dict) -> None:
    """
    Multi-ticker training objective.
    
    Trains on multiple tickers with shared hyperparameters and reports
    aggregate metrics.
    """
    tickers = config.get("tickers", ["AAPL"])
    if isinstance(tickers, str):
        tickers = [tickers]
    
    all_metrics = []
    
    for ticker in tickers:
        ticker_config = {**config, "ticker": ticker}
        
        # Load and train for this ticker
        df = load_symbol_data_pandas(
            ticker,
            config.get("start_date"),
            config.get("end_date"),
            config.get("timeframe", "1m")
        )
        
        if df.empty or len(df) < 100:
            continue
        
        X_train, X_test, y_train, y_test = prepare_training_data(
            df,
            config.get("target_col", "close"),
            config.get("target_transform", "log_return"),
            config.get("lookforward", 1)
        )
        
        if X_train.empty or X_test.empty:
            continue
        
        try:
            algorithm = config.get("algorithm", "elasticnet")
            AlgoClass = get_algorithm_class(algorithm)
            
            # Build model (simplified for brevity)
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", AlgoClass(
                    alpha=config.get("alpha", 0.01),
                    l1_ratio=config.get("l1_ratio", 0.5)
                ) if algorithm == "elasticnet" else AlgoClass())
            ])
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            # Calculate metrics
            signals = (preds > 0).astype(float)
            strategy_returns = np.array(y_test) * signals
            sharpe = calculate_sharpe_ratio(strategy_returns)
            
            all_metrics.append({
                "ticker": ticker,
                "sharpe": sharpe,
                "r2": r2_score(y_test, preds) if is_regression(algorithm) else 0
            })
            
        except Exception as e:
            log.warning(f"Failed on {ticker}: {e}")
            continue
    
    if not all_metrics:
        tune.report(
            sharpe_ratio=-999,
            avg_sharpe=-999,
            status="failed"
        )
        return
    
    # Aggregate metrics across tickers
    avg_sharpe = np.mean([m["sharpe"] for m in all_metrics])
    min_sharpe = np.min([m["sharpe"] for m in all_metrics])
    avg_r2 = np.mean([m["r2"] for m in all_metrics])
    
    tune.report(
        sharpe_ratio=avg_sharpe,  # Primary metric
        avg_sharpe=avg_sharpe,
        min_sharpe=min_sharpe,
        avg_r2=avg_r2,
        num_tickers=len(all_metrics),
        ticker_metrics=all_metrics,
        status="completed"
    )
