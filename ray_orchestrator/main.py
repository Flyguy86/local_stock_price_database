"""
FastAPI Application for Ray Orchestrator.

Provides HTTP endpoints for:
- Starting hyperparameter searches (Grid, PBT, ASHA)
- Deploying model ensembles
- Monitoring experiments and deployments
- Dashboard UI
"""

import logging
from typing import Optional
from datetime import datetime
from pathlib import Path
import json
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import ray
from ray.job_submission import JobSubmissionClient

from .config import settings
from .data import get_available_symbols, get_symbol_date_range
from .tuner import orchestrator, SEARCH_SPACES
from .ensemble import ensemble_manager
from .fingerprint import fingerprint_db
from .streaming import create_preprocessing_pipeline, BarDataLoader
from .trainer import create_walk_forward_trainer
from .backtester import create_backtester
from .mlflow_integration import MLflowTracker

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("ray_orchestrator.api")

# Global fold generation status tracking
fold_generation_status = {}

# Create FastAPI app
app = FastAPI(
    title="Ray Orchestrator",
    description="Distributed ML Training & Deployment for Trading Bots",
    version="0.1.0"
)

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


# ============================================================================
# Request/Response Models
# ============================================================================

class GridSearchRequest(BaseModel):
    """Request model for grid search."""
    algorithm: str = "elasticnet"
    tickers: list[str] = ["AAPL"]
    param_grid: dict = {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.3, 0.5, 0.7]}
    target_col: str = "close"
    target_transform: str = "log_return"
    timeframe: str = "1m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    name: Optional[str] = None


class PBTSearchRequest(BaseModel):
    """Request model for PBT search."""
    algorithm: str = "elasticnet"
    tickers: list[str] = ["AAPL"]
    population_size: int = 20
    num_generations: int = 10
    target_col: str = "close"
    target_transform: str = "log_return"
    timeframe: str = "1m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    name: Optional[str] = None
    resume: bool = False  # Resume crashed/stopped experiment


class ASHASearchRequest(BaseModel):
    """Request model for ASHA search."""
    algorithm: str = "elasticnet"
    tickers: list[str] = ["AAPL"]
    num_samples: int = 50
    search_alg: str = "random"  # random, bayesopt, optuna, hyperopt
    target_col: str = "close"
    target_transform: str = "log_return"
    timeframe: str = "1m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    name: Optional[str] = None
    resume: bool = False  # Resume crashed/stopped experiment


class BayesianSearchRequest(BaseModel):
    """Request model for Bayesian Optimization search."""
    algorithm: str = "elasticnet"
    tickers: list[str] = ["AAPL"]


class StreamingPreprocessRequest(BaseModel):
    """Request model for streaming preprocessing."""
    symbols: Optional[list[str]] = None  # None = all symbols
    output_path: str = "/app/data/preprocessed_parquet"
    market_hours_only: bool = True
    rolling_windows: list[int] = [5, 10, 20]
    partition_by: Optional[list[str]] = None


class GenerateFoldsRequest(BaseModel):
    """Request model for generating fold data only (no training)."""
    symbol: str  # Primary symbol (e.g., "AAPL")
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    train_months: int = 12
    test_months: int = 3
    step_months: int = 3
    output_base_path: str = "/app/data/walk_forward_folds"


class WalkForwardPreprocessRequest(BaseModel):
    """Request model for walk-forward preprocessing with fold isolation."""
    symbols: list[str]  # Primary symbols (e.g., ["AAPL"])
    context_symbols: Optional[list[str]] = None  # Context symbols (e.g., ["QQQ", "VIX"])
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    train_months: int = 3
    test_months: int = 1
    step_months: int = 1
    windows: list[int] = [50, 200]  # SMA windows
    resampling_timeframes: Optional[list[str]] = None  # e.g., ["5min", "15min"]
    output_base_path: str = "/app/data/walk_forward_folds"
    num_gpus: float = 0.0  # Set to 1.0+ for GPU acceleration
    actor_pool_size: Optional[int] = None  # None = auto-detect all CPUs, or specify manually


class WalkForwardTrainRequest(BaseModel):
    """Request model for walk-forward training with hyperparameter tuning."""
    symbols: list[str]  # Primary symbols
    context_symbols: Optional[list[str]] = None
    start_date: str
    end_date: str
    train_months: int = 3
    test_months: int = 1
    step_months: int = 1
    algorithm: str = "elasticnet"  # elasticnet, ridge, lasso, randomforest
    param_space: Optional[dict] = None  # Custom search space
    num_samples: int = 50  # Number of trials
    windows: list[int] = [50, 200]
    resampling_timeframes: Optional[list[str]] = None
    num_gpus: float = 0.0  # GPU acceleration for preprocessing
    actor_pool_size: Optional[int] = None  # None = auto-detect all CPUs
    skip_empty_folds: bool = False  # If True, skip empty folds; if False, fail on empty folds
    target_col: str = "close"
    target_transform: str = "log_return"
    timeframe: str = "1m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    name: Optional[str] = None
    resume: bool = False  # Resume crashed/stopped experiment


class MultiTickerPBTRequest(BaseModel):
    """Request model for multi-ticker PBT."""
    algorithm: str = "elasticnet"
    tickers: list[str] = ["AAPL", "MSFT", "GOOGL"]
    population_size: int = 20
    num_generations: int = 10
    target_col: str = "close"
    target_transform: str = "log_return"
    timeframe: str = "1m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    name: Optional[str] = None
    resume: bool = False  # Resume crashed/stopped experiment


class BacktestRequest(BaseModel):
    """Request model for backtesting a trained model."""
    experiment_name: str
    symbols: Optional[list[str]] = None  # Symbols to backtest, None = all available
    train_months: int = 3
    test_months: int = 1
    step_months: int = 1
    start_date: Optional[str] = None  # Override fold start date
    end_date: Optional[str] = None    # Override fold end date
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%


class DeployEnsembleRequest(BaseModel):
    """Request model for deploying an ensemble."""
    ensemble_name: str
    model_paths: list[str]
    model_ids: Optional[list[str]] = None
    voting: str = "soft"  # hard or soft
    threshold: float = 0.7


class DeployPBTSurvivorsRequest(BaseModel):
    """Request model for deploying PBT survivors."""
    ensemble_name: str
    experiment_name: str
    top_n: int = 5
    voting: str = "soft"
    threshold: float = 0.7


class PredictRequest(BaseModel):
    """Request model for ensemble prediction."""
    features: dict


# ============================================================================
# Streaming Preprocessing Endpoints
# ============================================================================

@app.post("/streaming/preprocess")
async def run_streaming_preprocess(request: StreamingPreprocessRequest, background_tasks: BackgroundTasks):
    """
    Run streaming preprocessing on bar data using Ray Data.
    
    Transforms raw 1-minute bars into ML-ready features using Ray's
    streaming engine for efficient parallel processing.
    """
    log.info(f"Starting streaming preprocessing for symbols: {request.symbols or 'ALL'}")
    
    def run_preprocessing():
        try:
            # Create pipeline
            preprocessor = create_preprocessing_pipeline(
                parquet_dir=str(settings.data.parquet_dir)
            )
            
            # Run preprocessing
            ds = preprocessor.create_training_pipeline(
                symbols=request.symbols,
                market_hours_only=request.market_hours_only,
                rolling_windows=request.rolling_windows
            )
            
            # Save results
            preprocessor.save_processed_data(
                ds,
                output_path=request.output_path,
                partition_by=request.partition_by or ["symbol"]
            )
            
            log.info(f"Streaming preprocessing completed: {request.output_path}")
        except Exception as e:
            log.error(f"Streaming preprocessing failed: {e}", exc_info=True)
    
    background_tasks.add_task(run_preprocessing)
    
    return {
        "status": "started",
        "task": "streaming_preprocessing",
        "symbols": request.symbols or "all",
        "output_path": request.output_path
    }


@app.get("/folds/list/{symbol}")
async def list_available_folds(symbol: str):
    """
    List available pre-processed walk-forward folds for a symbol.
    
    Returns fold metadata including date ranges and row counts.
    Models MUST use these folds for training.
    """
    from .data import get_available_folds
    
    fold_ids = get_available_folds(symbol)
    
    if not fold_ids:
        raise HTTPException(
            status_code=404,
            detail=f"No walk-forward folds found for {symbol}. "
                   f"Run preprocessing first: POST /streaming/walk_forward"
        )
    
    folds_info = []
    fold_base_dir = settings.data.walk_forward_folds_dir
    
    for fold_id in fold_ids:
        fold_dir = fold_base_dir / symbol / f"fold_{fold_id:03d}"
        
        # Get metadata if available
        metadata_file = fold_dir / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {"fold_id": fold_id}
        
        folds_info.append(metadata)
    
    return {
        "symbol": symbol,
        "num_folds": len(fold_ids),
        "folds": folds_info,
        "base_dir": str(fold_base_dir)
    }


@app.get("/folds/summary")
async def get_folds_summary():
    """
    Get summary of all available walk-forward folds across all symbols.
    
    Use this to verify preprocessing has been run before training.
    """
    fold_base_dir = settings.data.walk_forward_folds_dir
    
    if not fold_base_dir.exists():
        return {
            "status": "no_folds",
            "message": "No walk-forward folds directory found. Run preprocessing first.",
            "required_action": "POST /streaming/walk_forward"
        }
    
    from .data import get_available_symbols
    
    # Check which symbols have raw data
    symbols_with_data = get_available_symbols()
    
    # Check which symbols have processed folds
    symbols_with_folds = []
    for symbol_dir in fold_base_dir.iterdir():
        if symbol_dir.is_dir():
            symbols_with_folds.append(symbol_dir.name)
    
    summary = {
        "total_symbols_with_data": len(symbols_with_data),
        "total_symbols_with_folds": len(symbols_with_folds),
        "symbols_needing_preprocessing": list(set(symbols_with_data) - set(symbols_with_folds)),
        "symbols_ready_for_training": symbols_with_folds,
    }
    
    return summary


@app.get("/folds/{symbol}/{fold_id}/columns")
async def get_fold_columns(symbol: str, fold_id: int):
    """
    Get column names and sample data for a specific fold.
    
    Useful for inspecting what features are available in pre-processed folds.
    """
    fold_base_dir = settings.data.walk_forward_folds_dir
    fold_dir = fold_base_dir / symbol / f"fold_{fold_id:03d}"
    
    if not fold_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Fold {fold_id} not found for {symbol}"
        )
    
    train_path = fold_dir / "train"
    
    if not train_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Train data not found for fold {fold_id}"
        )
    
    try:
        # Read first file to get schema
        import duckdb
        parquet_files = list(train_path.glob("*.parquet"))
        if not parquet_files:
            raise HTTPException(status_code=404, detail="No parquet files found")
        
        # Get schema and sample data
        sample_query = f"""
            SELECT * FROM read_parquet('{train_path}/*.parquet')
            LIMIT 5
        """
        sample_df = duckdb.query(sample_query).df()
        
        # Get column stats
        columns_info = []
        for col in sample_df.columns:
            col_info = {
                "name": col,
                "dtype": str(sample_df[col].dtype),
                "null_count": int(sample_df[col].isnull().sum()),
                "sample_values": sample_df[col].head(3).tolist()
            }
            
            # Add type-specific info
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                col_info["min"] = float(sample_df[col].min()) if not sample_df[col].isnull().all() else None
                col_info["max"] = float(sample_df[col].max()) if not sample_df[col].isnull().all() else None
                col_info["mean"] = float(sample_df[col].mean()) if not sample_df[col].isnull().all() else None
            
            columns_info.append(col_info)
        
        # Categorize columns
        context_cols = [c for c in sample_df.columns if any(ctx in c for ctx in ['_QQQ', '_VIX', '_MSFT', '_SPY'])]
        indicator_cols = [c for c in sample_df.columns if any(ind in c for ind in ['sma_', 'ema_', 'rsi_', 'macd_', 'bb_', 'atr_'])]
        price_cols = [c for c in sample_df.columns if any(p in c for p in ['open', 'high', 'low', 'close', 'volume'])]
        metadata_cols = [c for c in sample_df.columns if c in ['symbol', 'ts', 'date', 'dt']]
        
        return {
            "symbol": symbol,
            "fold_id": fold_id,
            "total_columns": len(sample_df.columns),
            "row_count_sample": len(sample_df),
            "columns": columns_info,
            "column_categories": {
                "context_features": context_cols,
                "indicators": indicator_cols,
                "price_features": price_cols,
                "metadata": metadata_cols,
                "other": list(set(sample_df.columns) - set(context_cols) - set(indicator_cols) - set(price_cols) - set(metadata_cols))
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading fold data: {str(e)}")


@app.get("/folds/summary")
    }
    
    return summary


@app.get("/streaming/status")
async def get_streaming_status():
    """Get status of Ray Data streaming jobs with symbol and date information."""
    if not ray.is_initialized():
        return {"error": "Ray not initialized"}
    
    # Get Ray Data job stats
    try:
        stats = {
            "ray_initialized": True,
            "timestamp": datetime.utcnow().isoformat(),
            "available_resources": ray.available_resources(),
        }
        
        # List available data sources
        parquet_dir = Path("/app/data/parquet")
        log.info(f"Checking parquet directory: {parquet_dir.absolute()} (exists={parquet_dir.exists()})")
        
        # Debug: list what's actually in /app/data
        data_dir = Path("/app/data")
        if data_dir.exists():
            log.info(f"Contents of /app/data: {list(data_dir.iterdir())}")
        
        loader = BarDataLoader(parquet_dir=str(parquet_dir))
        files = loader._discover_parquet_files()
        
        log.info(f"BarDataLoader found {len(files)} parquet files")
        
        # Extract symbols and date ranges from files
        symbols_info = {}
        for file_path in files:
            # Parse /app/data/parquet/AAPL/2024-01-15.parquet
            try:
                parts = Path(file_path).parts
                if 'parquet' in parts:
                    idx = parts.index('parquet')
                    if len(parts) > idx + 2:
                        symbol = parts[idx + 1]
                        date_str = parts[idx + 2].replace('.parquet', '')
                        
                        if symbol not in symbols_info:
                            symbols_info[symbol] = {
                                'symbol': symbol,
                                'file_count': 0,
                                'date_range': {'start': date_str, 'end': date_str}
                            }
                        
                        symbols_info[symbol]['file_count'] += 1
                        
                        # Update date range
                        if date_str < symbols_info[symbol]['date_range']['start']:
                            symbols_info[symbol]['date_range']['start'] = date_str
                        if date_str > symbols_info[symbol]['date_range']['end']:
                            symbols_info[symbol]['date_range']['end'] = date_str
            except Exception as e:
                log.debug(f"Error parsing file path {file_path}: {e}")
                continue
        
        # Cap max date to last day of previous month (validation requirement)
        from datetime import date
        today = date.today()
        if today.month == 1:
            max_allowed_year = today.year - 1
            max_allowed_month = 12
        else:
            max_allowed_year = today.year
            max_allowed_month = today.month - 1
        
        # Last day of previous month
        import calendar
        last_day = calendar.monthrange(max_allowed_year, max_allowed_month)[1]
        max_allowed_date = date(max_allowed_year, max_allowed_month, last_day).isoformat()
        
        # Cap all symbol date ranges
        for symbol_info in symbols_info.values():
            if symbol_info['date_range']['end'] > max_allowed_date:
                symbol_info['date_range']['end'] = max_allowed_date
                symbol_info['date_range']['capped'] = True  # Mark as capped for UI
        
        stats["data_sources"] = {
            "total_parquet_files": len(files),
            "sample_files": files[:10] if files else [],
            "symbols": list(symbols_info.values()),
            "max_allowed_date": max_allowed_date  # Tell UI the boundary
        }
        
        return stats
    except Exception as e:
        log.error(f"Error getting streaming status: {e}")
        return {"error": str(e)}


@app.post("/streaming/preview")
async def preview_streaming_data(
    symbols: Optional[list[str]] = None,
    limit: int = 100
):
    """
    Preview raw bar data before preprocessing.
    
    Useful for checking data quality and schema.
    """
    try:
        loader = BarDataLoader(parquet_dir=str(settings.data.parquet_dir))
        ds = loader.load_all_bars(symbols=symbols)
        
        # Get sample rows
        sample = ds.take(limit)
        
        return {
            "status": "success",
            "row_count": len(sample),
            "total_estimated": ds.count(),
            "sample_data": [dict(row) for row in sample[:10]],  # Show first 10
            "schema": str(ds.schema()) if hasattr(ds, 'schema') else None
        }
    except Exception as e:
        log.error(f"Error previewing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/streaming/generate-folds")
async def generate_folds_only(request: GenerateFoldsRequest, background_tasks: BackgroundTasks):
    """
    Generate walk-forward fold data WITHOUT training models.
    
    Use this when you already have trained models but need to generate
    fold data for backtesting. This will:
    
    1. Load raw OHLCV data for the symbol
    2. Generate walk-forward folds with proper date splits
    3. Save train.parquet and test.parquet for each fold
    
    Much faster than full training since it skips:
    - Feature engineering
    - Hyperparameter tuning
    - Model training
    
    Output: /app/data/walk_forward_folds/{symbol}/fold_{id}/
    """
    log.info(f"Generating folds for {request.symbol}: {request.start_date} to {request.end_date}")
    
    def run_fold_generation():
        try:
            from pathlib import Path
            import os
            import pandas as pd
            import numpy as np
            
            # Auto-detect CPU count
            try:
                cpu_count = len(os.sched_getaffinity(0))
            except AttributeError:
                cpu_count = os.cpu_count() or 4
            
            log.info(f"Using {cpu_count} parallel actors for fold generation")
            
            # Create preprocessing pipeline (reuses training code!)
            preprocessor = create_preprocessing_pipeline(
                parquet_dir=str(settings.data.parquet_dir)
            )
            
            # Use the SAME walk-forward pipeline as training
            fold_count = 0
            for fold in preprocessor.create_walk_forward_pipeline(
                symbols=[request.symbol],
                start_date=request.start_date,
                end_date=request.end_date,
                train_months=request.train_months,
                test_months=request.test_months,
                step_months=request.step_months,
                context_symbols=None,  # No context symbols for simple backtesting
                windows=[5, 10, 20, 50, 200],  # Standard windows
                resampling_timeframes=None,
                num_gpus=0.0,  # CPU only for fold generation
                actor_pool_size=cpu_count  # Use all available CPUs
            ):
                fold_count += 1
                log.info(f"Processing {fold}")
                
                # Create output directory with zero-padded fold ID
                symbol_dir = Path(request.output_base_path) / request.symbol
                fold_dir = symbol_dir / f"fold_{fold.fold_id:03d}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                
                def transform_to_log_returns(batch: pd.DataFrame) -> pd.DataFrame:
                    """Transform OHLC to log returns and add target."""
                    batch = batch.copy()
                    
                    # Keep raw close for VectorBT simulation
                    batch['close_raw'] = batch['close']
                    
                    # Transform OHLC to log returns
                    for col in ['open', 'high', 'low', 'close']:
                        if col in batch.columns:
                            batch[f'{col}_log_return'] = np.log((batch[col] + 1e-9) / (batch[col].shift(1) + 1e-9))
                            batch.drop(columns=[col], inplace=True)
                    
                    # Add target (future close log return)
                    future_close = batch['close_raw'].shift(-1)
                    batch['target'] = np.log((future_close + 1e-9) / (batch['close_raw'] + 1e-9))
                    
                    return batch
                
                # Save train data with target
                if fold.train_ds:
                    train_path = str(fold_dir / "train")
                    fold.train_ds.map_batches(transform_to_log_returns, batch_format="pandas").write_parquet(train_path, try_create_dir=True)
                    train_count = fold.train_ds.count()
                    log.info(f"✓ Saved train data: {train_count:,} rows → {train_path}/")
                
                # Save test data with target
                if fold.test_ds:
                    test_path = str(fold_dir / "test")
                    fold.test_ds.map_batches(transform_to_log_returns, batch_format="pandas").write_parquet(test_path, try_create_dir=True)
                    test_count = fold.test_ds.count()
                    log.info(f"✓ Saved test data: {test_count:,} rows → {test_path}/")
            
            log.info(f"✅ Fold generation completed: {fold_count} folds with features saved to {symbol_dir}")
            
        except Exception as e:
            log.error(f"Fold generation failed: {e}", exc_info=True)
    
    background_tasks.add_task(run_fold_generation)
    
    return {
        "status": "started",
        "task": "generate_folds",
        "symbol": request.symbol,
        "date_range": f"{request.start_date} to {request.end_date}",
        "fold_config": {
            "train_months": request.train_months,
            "test_months": request.test_months,
            "step_months": request.step_months
        },
        "output_path": f"{request.output_base_path}/{request.symbol}",
        "note": "Generating fold data only (no training). Check logs for progress."
    }


@app.post("/streaming/walk_forward")
async def run_walk_forward_preprocess(request: WalkForwardPreprocessRequest, background_tasks: BackgroundTasks):
    """
    Run walk-forward preprocessing with fold isolation for balanced backtesting.
    
    This is the RECOMMENDED approach for ML trading bots. Each fold calculates
    indicators independently, ensuring no look-ahead bias across train/test splits.
    
    Example:
        - Fold 1: Train Jan-Mar (SMA calculated on Jan-Mar only), Test Apr
        - Fold 2: Train Feb-Apr (SMA calculated on Feb-Apr only), Test May
        
    The SMA at the start of each Test period does NOT know what happened in
    previous folds, eliminating look-ahead bias.
    """
    # Auto-detect CPU count if not specified
    import os
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 4
    
    actor_pool_size = request.actor_pool_size if request.actor_pool_size is not None else cpu_count
    
    log.info(f"Starting walk-forward preprocessing: {request.symbols} from {request.start_date} to {request.end_date}")
    log.info(f"Using {actor_pool_size} parallel actors (CPUs available: {cpu_count})")
    
    def run_preprocessing():
        try:
            from pathlib import Path
            import pandas as pd
            import numpy as np
            
            # Create pipeline
            preprocessor = create_preprocessing_pipeline(
                parquet_dir=str(settings.data.parquet_dir)
            )
            
            # Process folds
            fold_count = 0
            for fold in preprocessor.create_walk_forward_pipeline(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                train_months=request.train_months,
                test_months=request.test_months,
                step_months=request.step_months,
                context_symbols=request.context_symbols,
                windows=request.windows,
                resampling_timeframes=request.resampling_timeframes,
                num_gpus=request.num_gpus,
                actor_pool_size=actor_pool_size
            ):
                fold_count += 1
                log.info(f"Processing {fold}")
                
                # Transform OHLC to log returns and add target
                def transform_to_log_returns(batch: pd.DataFrame) -> pd.DataFrame:
                    """
                    Transform OHLC to log returns to prevent absolute price leakage.
                    Keep close_raw for VectorBT simulation.
                    """
                    batch = batch.copy()
                    
                    # Preserve raw close for VectorBT
                    batch['close_raw'] = batch['close']
                    
                    # Transform OHLC to log returns
                    for col in ['open', 'high', 'low', 'close']:
                        if col in batch.columns:
                            batch[f'{col}_log_return'] = np.log((batch[col] + 1e-9) / (batch[col].shift(1) + 1e-9))
                            batch.drop(columns=[col], inplace=True)
                    
                    # Add target as future close log return
                    future_close = batch['close_raw'].shift(-1)
                    batch['target'] = np.log((future_close + 1e-9) / (batch['close_raw'] + 1e-9))
                    
                    return batch
                
                # Save train data with log returns and target
                if fold.train_ds:
                    # Validate before saving
                    train_sample = fold.train_ds.take(1000)
                    if train_sample:
                        train_df_check = pd.DataFrame(train_sample)
                        nan_pct = train_df_check.isna().sum().sum() / (train_df_check.shape[0] * train_df_check.shape[1])
                        if nan_pct > 0.05:
                            log.error(f"Train data has {nan_pct*100:.2f}% NaN values before saving (fold {fold.fold_id})")
                        else:
                            log.info(f"Train data validation passed: {nan_pct*100:.2f}% NaN (fold {fold.fold_id})")
                    
                    train_path = f"{request.output_base_path}/fold_{fold.fold_id}/train"
                    fold.train_ds.map_batches(transform_to_log_returns, batch_format="pandas").write_parquet(train_path, try_create_dir=True)
                    log.info(f"Saved train data to {train_path}")
                
                # Save test data with log returns and target
                if fold.test_ds:
                    # Validate before saving
                    test_sample = fold.test_ds.take(1000)
                    if test_sample:
                        test_df_check = pd.DataFrame(test_sample)
                        nan_pct = test_df_check.isna().sum().sum() / (test_df_check.shape[0] * test_df_check.shape[1])
                        if nan_pct > 0.05:
                            log.error(f"Test data has {nan_pct*100:.2f}% NaN values before saving (fold {fold.fold_id})")
                        else:
                            log.info(f"Test data validation passed: {nan_pct*100:.2f}% NaN (fold {fold.fold_id})")
                    
                    test_path = f"{request.output_base_path}/fold_{fold.fold_id}/test"
                    fold.test_ds.map_batches(transform_to_log_returns, batch_format="pandas").write_parquet(test_path, try_create_dir=True)
                    log.info(f"Saved test data to {test_path}")
            
            log.info(f"Walk-forward preprocessing completed: {fold_count} folds processed")
        except Exception as e:
            log.error(f"Walk-forward preprocessing failed: {e}", exc_info=True)
    
    background_tasks.add_task(run_preprocessing)
    
    return {
        "status": "started",
        "task": "walk_forward_preprocessing",
        "symbols": request.symbols,
        "context_symbols": request.context_symbols,
        "date_range": f"{request.start_date} to {request.end_date}",
        "fold_config": {
            "train_months": request.train_months,
            "test_months": request.test_months,
            "step_months": request.step_months
        },
        "output_base_path": request.output_base_path,
        "note": "Each fold calculates indicators independently to prevent look-ahead bias"
    }


@app.post("/train/walk_forward")
async def train_walk_forward(request: WalkForwardTrainRequest, background_tasks: BackgroundTasks):
    """
    Train models using walk-forward validation with hyperparameter tuning.
    
    This is the COMPLETE end-to-end training pipeline:
    1. Generates walk-forward folds with proper date splits
    2. Preprocesses each fold independently (no look-ahead)
    3. Runs hyperparameter tuning across all folds
    4. Reports average metrics across folds
    
    Each hyperparameter configuration is tested on ALL folds, ensuring
    the model generalizes across different time periods.
    
    Jobs are submitted via Ray Job Submission API for better visibility
    in the Ray Dashboard with full log access.
    """
    log.info(f"Submitting training job: {request.symbols} with {request.algorithm}")
    
    try:
        # Create job submission client
        client = JobSubmissionClient("http://127.0.0.1:8265")
        
        # Prepare Python entrypoint
        entrypoint_code = f'''
import ray
from ray_orchestrator.trainer import create_walk_forward_trainer

ray.init(address="auto", ignore_reinit_error=True)

trainer = create_walk_forward_trainer(parquet_dir="/app/data/parquet")

results = trainer.run_walk_forward_tuning(
    symbols={request.symbols!r},
    start_date="{request.start_date}",
    end_date="{request.end_date}",
    train_months={request.train_months},
    test_months={request.test_months},
    step_months={request.step_months},
    algorithm="{request.algorithm}",
    param_space={request.param_space!r},
    num_samples={request.num_samples},
    context_symbols={request.context_symbols!r},
    windows={request.windows!r},
    resampling_timeframes={request.resampling_timeframes!r},
    num_gpus={request.num_gpus},
    actor_pool_size={request.actor_pool_size!r},
    skip_empty_folds={request.skip_empty_folds}
)

best = results.get_best_result()
print(f"=== TRAINING COMPLETE ===")
print(f"Best config: {{best.config}}")
print(f"Best test RMSE: {{best.metrics['test_rmse']:.6f}}")
print(f"Best test R2: {{best.metrics['test_r2']:.4f}}")
'''
        
        # Submit job
        # No working_dir needed - code is already in the Docker container
        # Just set PYTHONPATH to ensure imports work correctly
        runtime_env = {
            "env_vars": {
                "PYTHONPATH": "/app"
            }
        }
        
        metadata = {
            "symbols": str(request.symbols),
            "algorithm": request.algorithm,
            "date_range": f"{request.start_date} to {request.end_date}"
        }
        
        # Write Python code to a temp file and execute it to avoid shell escaping issues
        import tempfile
        import base64
        
        # Base64 encode the Python code to avoid any escaping issues
        encoded_code = base64.b64encode(entrypoint_code.encode()).decode()
        
        job_id = client.submit_job(
            entrypoint=f'python -c "import base64; exec(base64.b64decode(\'{encoded_code}\').decode())"',
            runtime_env=runtime_env,
            metadata=metadata
        )
        
        log.info(f"Training job submitted: {job_id}")
        
        return {
            "status": "submitted",
            "job_id": job_id,
            "task": "walk_forward_training",
            "symbols": request.symbols,
            "algorithm": request.algorithm,
            "num_samples": request.num_samples,
            "date_range": f"{request.start_date} to {request.end_date}",
            "fold_config": {
                "train_months": request.train_months,
                "test_months": request.test_months,
                "step_months": request.step_months
            },
            "note": "Job submitted via Ray Job Submission API. View logs at Ray Dashboard."
        }
        
    except Exception as e:
        log.error(f"Failed to submit training job: {{e}}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Job submission failed: {{str(e)}}")


@app.get("/train/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a submitted training job."""
    try:
        client = JobSubmissionClient("http://127.0.0.1:8265")
        status = client.get_job_status(job_id)
        logs = client.get_job_logs(job_id)
        
        return {
            "job_id": job_id,
            "status": str(status),
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {str(e)}")


@app.get("/train/jobs")
async def list_training_jobs():
    """List all submitted training jobs."""
    try:
        client = JobSubmissionClient("http://127.0.0.1:8265")
        jobs = client.list_jobs()
        return {"jobs": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Backtesting Endpoints (VectorBT)
# ============================================================================

@app.get("/backtest/fold-status/{symbol}")
async def get_fold_generation_status(symbol: str):
    """Get the status of fold generation for a symbol."""
    status = fold_generation_status.get(symbol, {
        "status": "not_started",
        "symbol": symbol,
        "message": "No fold generation in progress"
    })
    return status

@app.post("/backtest/validate")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run VectorBT backtesting on a trained model.
    
    Loads per-fold trained models from Ray checkpoints and runs realistic
    backtesting with transaction costs. Uses the exact models and features
    that were used during training for each fold.
    
    Returns aggregated metrics: Sharpe ratio, max drawdown, consistency scores.
    """
    try:
        # Use symbols from request, not from model name
        if not request.symbols or len(request.symbols) == 0:
            raise ValueError("Please select at least one symbol in 'Symbols to Backtest'")
        
        if len(request.symbols) > 1:
            raise ValueError("Please select only ONE symbol at a time for backtesting")
        
        symbol = request.symbols[0]
        
        # Handle new format: experiment_name/trial_name or old format: experiment_name
        experiment_name = request.experiment_name
        checkpoint_path = None
        base_experiment = experiment_name
        
        if '/' in experiment_name:
            # New format with trial path - extract experiment and get checkpoint
            parts = experiment_name.split('/')
            base_experiment = parts[0]
            trial_name = parts[1]
            
            # Find the checkpoint path for this specific trial
            from ray_orchestrator.list_models import list_trained_models
            models = list_trained_models(str(settings.data.checkpoints_dir))
            
            matching_model = None
            for model in models:
                if model['experiment_name'] == base_experiment and model['trial_name'] == trial_name:
                    matching_model = model
                    break
            
            if not matching_model:
                raise ValueError(f"Could not find checkpoint for {experiment_name}")
            
            checkpoint_path = matching_model['checkpoint_path']
            best_config = matching_model['params']
            algorithm = matching_model['algorithm']
            
            log.info(f"Using specific checkpoint: {checkpoint_path}")
            log.info(f"Algorithm: {algorithm}, Config: {best_config}")
            
            # Use base experiment name for backtester
            experiment_name = base_experiment
            
        else:
            # Old format - extract algorithm from experiment name
            parts = experiment_name.split('_')
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid experiment name format: {experiment_name}. "
                    "Expected: walk_forward_{{algorithm}}_{{symbol}}"
                )
            algorithm = parts[-2]
            best_config = None  # Will be loaded by backtester
        
        log.info(f"Starting backtest for {request.experiment_name} on symbol: {symbol}, algorithm: {algorithm}")
        
        # Create backtester with base experiment name and checkpoint path
        from .backtester import ModelBacktester
        backtester = ModelBacktester(
            experiment_name=experiment_name,
            checkpoint_path=checkpoint_path  # Will be None for old format, which is fine
        )
        
        # Load best hyperparameters if not already set
        if best_config is None:
            model_info = backtester.load_best_model()
            best_config = model_info.get('config', {})
            
            if not best_config:
                raise ValueError(
                    f"No hyperparameters found for {experiment_name}. "
                    "The model may not have completed training successfully."
                )
        else:
            # Create model_info from our loaded data
            # For specific trial, mark that best_trial is loaded
            backtester.best_trial = {'checkpoint_dir': checkpoint_path}
            backtester.best_config = best_config
            
            model_info = {
                'config': best_config,
                'checkpoint_path': checkpoint_path
            }
        
        # Get available folds for the SELECTED symbol (not model symbol)
        from .data import get_available_folds, load_fold_from_disk
        folds = get_available_folds(symbol)
        
        if not folds:
            # Kick off fold generation in background and return immediately
            log.info(f"No folds found for {symbol}, starting background fold generation...")
            
            def generate_folds_background():
                try:
                    import os
                    import pandas as pd
                    import numpy as np
                    
                    # Initialize status
                    fold_generation_status[symbol] = {
                        "status": "generating",
                        "symbol": symbol,
                        "current_fold": 0,
                        "total_folds": "calculating...",
                        "message": "Initializing fold generation...",
                        "started_at": datetime.utcnow().isoformat()
                    }
                    
                    try:
                        cpu_count = len(os.sched_getaffinity(0))
                    except AttributeError:
                        cpu_count = os.cpu_count() or 4
                    
                    log.info(f"Background: Generating folds with {cpu_count} parallel actors")
                    
                    fold_generation_status[symbol]["message"] = f"Creating preprocessing pipeline with {cpu_count} actors..."
                    
                    preprocessor = create_preprocessing_pipeline(
                        parquet_dir=str(settings.data.parquet_dir)
                    )
                    
                    start_date = request.start_date or "2020-01-01"
                    end_date = request.end_date or "2024-12-31"
                    
                    log.info(f"Background: {symbol} from {start_date} to {end_date}")
                    
                    fold_generation_status[symbol]["message"] = f"Generating walk-forward folds from {start_date} to {end_date}..."
                    
                    fold_count = 0
                    for fold in preprocessor.create_walk_forward_pipeline(
                        symbols=[symbol],
                        start_date=start_date,
                        end_date=end_date,
                        train_months=request.train_months,
                        test_months=request.test_months,
                        step_months=request.step_months,
                        context_symbols=None,
                        windows=[5, 10, 20, 50, 200],
                        resampling_timeframes=None,
                        num_gpus=0.0,
                        actor_pool_size=cpu_count
                    ):
                        fold_count += 1
                        
                        # Update status
                        fold_generation_status[symbol].update({
                            "current_fold": fold_count,
                            "message": f"Processing fold {fold_count} (fold_id: {fold.fold_id})..."
                        })
                        
                        symbol_dir = Path(settings.data.walk_forward_folds_dir) / symbol
                        fold_dir = symbol_dir / f"fold_{fold.fold_id:03d}"
                        fold_dir.mkdir(parents=True, exist_ok=True)
                        
                        def transform_to_log_returns(batch: pd.DataFrame) -> pd.DataFrame:
                            """Transform OHLC to log returns and add target."""
                            batch = batch.copy()
                            
                            # Keep raw close for VectorBT simulation (needed for portfolio tracking)
                            batch['close_raw'] = batch['close']
                            
                            # Transform OHLC to log returns (prevents absolute price leakage)
                            for col in ['open', 'high', 'low', 'close']:
                                if col in batch.columns:
                                    batch[f'{col}_log_return'] = np.log((batch[col] + 1e-9) / (batch[col].shift(1) + 1e-9))
                                    # Drop raw price column
                                    batch.drop(columns=[col], inplace=True)
                            
                            # Add target (future close log return)
                            future_close = batch['close_raw'].shift(-1)
                            batch['target'] = np.log((future_close + 1e-9) / (batch['close_raw'] + 1e-9))
                            
                            return batch
                        
                        if fold.train_ds:
                            train_path = str(fold_dir / "train")
                            fold.train_ds.map_batches(transform_to_log_returns, batch_format="pandas").write_parquet(train_path, try_create_dir=True)
                        
                        if fold.test_ds:
                            test_path = str(fold_dir / "test")
                            fold.test_ds.map_batches(transform_to_log_returns, batch_format="pandas").write_parquet(test_path, try_create_dir=True)
                        
                        log.info(f"Background: Saved fold {fold_count} (fold_id: {fold.fold_id})")
                    
                    # Mark as complete
                    fold_generation_status[symbol] = {
                        "status": "completed",
                        "symbol": symbol,
                        "total_folds": fold_count,
                        "message": f"✅ Successfully generated {fold_count} folds",
                        "completed_at": datetime.utcnow().isoformat()
                    }
                    
                    log.info(f"Background: ✅ Generated {fold_count} folds for {symbol}")
                    
                except Exception as e:
                    log.error(f"Background fold generation failed: {e}", exc_info=True)
                    fold_generation_status[symbol] = {
                        "status": "failed",
                        "symbol": symbol,
                        "message": f"❌ Fold generation failed: {str(e)}",
                        "failed_at": datetime.utcnow().isoformat()
                    }
            
            background_tasks.add_task(generate_folds_background)
            
            return {
                "status": "folds_required",
                "message": f"No walk-forward folds exist for {symbol}. Please generate folds first using the Ray Data preprocessing pipeline.",
                "symbol": symbol,
                "experiment_name": request.experiment_name,
                "instructions": {
                    "step_1": "Go to the Training Dashboard tab",
                    "step_2": f"Set 'Primary Symbol' to {symbol}",
                    "step_3": "Click 'Generate Fold Data Only' button (under Advanced: Manual Fold Generation)",
                    "step_4": "Wait 2-5 minutes for fold generation with full feature engineering",
                    "step_5": "Return here and run backtest again",
                    "alternative": f"Or use API: POST /streaming/walk_forward with symbols=['{symbol}']"
                },
                "background_task": "Fold generation started in background (watch docker logs)",
                "note": "Folds are being generated in background, but for best results use the Training Dashboard to ensure all features match the trained model."
            }
        
        # Folds exist - proceed with backtest
        log.info(f"Found {len(folds)} folds to backtest")
        
        # Backtest each fold
        all_results = []
        
        # Let backtester load per-fold models from checkpoints
        # This uses the exact model that was trained on each fold
        log.info(f"Backtesting {len(folds)} folds (loading models from checkpoint)")
        
        for fold_id in folds:
            try:
                # backtester.backtest_fold() loads fold-specific model and features from checkpoint
                fold_result = backtester.backtest_fold(
                    fold_id=fold_id,
                    symbol=symbol,
                    model=None,  # Load from checkpoint
                    fees=request.commission,
                    slippage=request.slippage
                )
                
                all_results.append(fold_result)
                log.info(f"✓ Fold {fold_id}: Return={fold_result.get('total_return', 0):.2%}")
                
            except Exception as e:
                log.error(f"Failed to backtest fold {fold_id}: {str(e)}", exc_info=True)
                continue
        
        if not all_results:
            raise ValueError(
                "No folds successfully backtested. "
                "Check logs for details. "
                "Most common issue: Folds have no features (only OHLCV). "
                "Solution: Delete /app/data/walk_forward_folds/{symbol}/ and run full training with POST /streaming/walk_forward"
            )
        
        # Aggregate results
        import pandas as pd
        results_df = pd.DataFrame(all_results)
        summary = backtester.aggregate_results(results_df)
        
        # Convert numpy types to native Python for JSON serialization
        def convert_to_native(obj):
            """Convert numpy/pandas types to native Python types."""
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            return obj
        
        # Cache results for dashboard
        # Sanitize experiment_name for filename (replace / with _)
        safe_experiment_name = request.experiment_name.replace('/', '_')
        results_dir = Path(settings.data.checkpoints_dir) / "backtest_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{safe_experiment_name}_{symbol}_results.json"
        
        dashboard_results = {
            "symbol": symbol,
            "model": request.experiment_name,
            "model_info": convert_to_native(model_info),
            "backtest_config": {
                "initial_capital": request.initial_capital,
                "commission": request.commission,
                "slippage": request.slippage,
                "train_months": request.train_months,
                "test_months": request.test_months,
                "step_months": request.step_months
            },
            "aggregate_metrics": convert_to_native(summary),
            "fold_results": convert_to_native(all_results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(dashboard_results, f, indent=2)
        
        log.info(f"✅ Backtest complete: {len(all_results)} folds, results saved to {results_file}")
        
        # Convert return value to native types for FastAPI JSON response
        return convert_to_native({
            "status": "completed",
            "experiment_name": request.experiment_name,
            "symbol": symbol,
            "algorithm": algorithm,
            "num_folds": len(all_results),
            "model_info": model_info,
            "results": summary,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except Exception as e:
        log.error(f"Backtest error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backtesting failed: {str(e)}")


@app.get("/backtest/results/list")
async def list_backtest_results():
    """
    List all available backtest results files.
    
    Returns metadata about each completed backtest.
    """
    try:
        results_dir = settings.data.checkpoints_dir / "backtest_results"
        log.info(f"Looking for backtest results in: {results_dir}")
        
        if not results_dir.exists():
            log.warning(f"Backtest results directory does not exist: {results_dir}")
            return []
        
        results = []
        for filepath in results_dir.glob("*_results.json"):
            # Parse filename: {experiment_name}_{symbol}_results.json
            filename = filepath.stem  # Remove .json
            parts = filename.rsplit("_", 2)  # Split from right: [..., symbol, "results"]
            
            if len(parts) >= 2:
                symbol = parts[-2]
                model_name = "_".join(parts[:-2])
                
                # Get file timestamp
                timestamp = datetime.fromtimestamp(filepath.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                
                results.append({
                    "filepath": str(filepath),
                    "model": model_name,
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "filename": filepath.name
                })
        
        # Sort by timestamp descending
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        log.info(f"Found {len(results)} backtest result files")
        return results
        
    except Exception as e:
        log.error(f"Error listing backtest results: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/results/load")
async def load_backtest_result(filepath: str):
    """
    Load a specific backtest result file.
    
    Returns the complete results including aggregate metrics and fold-by-fold data.
    """
    try:
        import json
        result_path = Path(filepath)
        
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Result file not found")
        
        # Verify it's in the backtest_results directory for security
        results_dir = settings.data.checkpoints_dir / "backtest_results"
        if not result_path.is_relative_to(results_dir):
            raise HTTPException(status_code=403, detail="Access denied")
        
        with open(result_path, 'r') as f:
            data = json.load(f)
        
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error loading backtest result: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/results/{experiment_name}")
async def get_backtest_results(experiment_name: str):
    """
    Get detailed backtest results for a specific experiment.
    
    Returns fold-by-fold performance metrics and aggregated assessment.
    """
    try:
        checkpoint_dir = f"/app/data/ray_checkpoints/{experiment_name}"
        fold_dir = "/app/data/walk_forward_folds"
        
        # Check if experiment exists
        if not os.path.exists(checkpoint_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{experiment_name}' not found"
            )
        
        backtester = create_backtester(
            experiment_name=experiment_name
        )
        
        # Load model info
        model_info = backtester.load_best_model()
        
        return {
            "experiment_name": experiment_name,
            "model_info": model_info,
            "checkpoint_dir": checkpoint_dir,
            "fold_dir": fold_dir,
            "status": "ready"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/experiments")
async def list_backtest_experiments():
    """
    List all available trained models with their metadata for backtesting.
    
    Returns individual model checkpoints with performance metrics and config.
    """
    try:
        from ray_orchestrator.list_models import list_trained_models
        
        # Get all trained models with their metadata
        models = list_trained_models(str(settings.data.checkpoints_dir))
        
        # Transform to match expected format for UI
        experiments = []
        for model in models:
            experiments.append({
                "name": f"{model['experiment_name']}/{model['trial_name']}",
                "experiment_name": model['experiment_name'],
                "trial_name": model['trial_name'],
                "checkpoint_path": model['checkpoint_path'],
                "model_info": {
                    "algorithm": model['algorithm'],
                    "ticker": model['ticker'],
                    "feature_version": model['feature_version'],
                    "test_rmse": model['metrics']['avg_test_rmse'],
                    "test_r2": model['metrics']['avg_test_r2'],
                    "test_mae": model['metrics']['avg_test_mae'],
                    "train_rmse": model['metrics']['avg_train_rmse'],
                    "num_folds": model['metrics']['num_folds'],
                    "date_range": model['date_range'],
                    "params": model['params']
                },
                "path": model['checkpoint_path']
            })
        
        return {
            "experiments": experiments,
            "total": len(experiments)
        }
        
    except Exception as e:
        log.error(f"Failed to list backtest experiments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dashboard UI
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the training dashboard UI."""
    return templates.TemplateResponse("training_dashboard.html", {"request": request})


@app.get("/registry", response_class=HTMLResponse)
async def model_registry(request: Request):
    """Serve the Model Registry UI."""
    return templates.TemplateResponse("model_registry.html", {"request": request})


@app.get("/backtest/dashboard", response_class=HTMLResponse)
async def vectorbt_dashboard(request: Request):
    """Serve the VectorBT backtest analysis dashboard."""
    return templates.TemplateResponse("vectorbt_dashboard.html", {"request": request})


@app.get("/backtest/dashboard", response_class=HTMLResponse)
async def vectorbt_dashboard(request: Request):
    """Serve the VectorBT backtest analysis dashboard."""
    return templates.TemplateResponse("vectorbt_dashboard.html", {"request": request})


@app.get("/backtest/results_viewer", response_class=HTMLResponse)
async def backtest_results_viewer(request: Request):
    """Serve the Backtest Results Viewer page."""
    return templates.TemplateResponse("backtest_results.html", {"request": request})


@app.get("/backtest/results")
async def get_dashboard_results(model: str, symbol: str):
    """
    Get backtest results for the VectorBT dashboard.
    
    Loads cached results from previous backtest runs.
    """
    try:
        results_dir = Path(settings.data.checkpoints_dir) / "backtest_results"
        results_file = results_dir / f"{model}_{symbol}_results.json"
        
        if not results_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No backtest results found for {model} on {symbol}. Run backtest first."
            )
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return results
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Results not found")
    except Exception as e:
        log.error(f"Failed to load results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_trained_models")
async def list_trained_models():
    """
    List all available trained models for backtesting.
    
    Scans Ray checkpoints directory for completed experiments.
    """
    try:
        checkpoints_dir = Path(settings.data.checkpoints_dir)
        models = []
        
        if not checkpoints_dir.exists():
            return []
        
        for exp_dir in checkpoints_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "backtest_results":
                continue
            
            # Parse experiment name (e.g., "walk_forward_xgboost_GOOGL")
            parts = exp_dir.name.split('_')
            if len(parts) >= 3:
                algorithm = parts[-2] if len(parts) >= 3 else "unknown"
                symbol = parts[-1]
                
                # Check if has valid trials
                has_trials = any(
                    (trial_dir / "result.json").exists()
                    for trial_dir in exp_dir.iterdir()
                    if trial_dir.is_dir()
                )
                
                if has_trials:
                    models.append({
                        "name": exp_dir.name,
                        "symbol": symbol,
                        "algorithm": algorithm,
                        "date": datetime.fromtimestamp(exp_dir.stat().st_mtime).strftime("%Y-%m-%d")
                    })
        
        # Sort by date descending
        models.sort(key=lambda x: x['date'], reverse=True)
        
        return models
        
    except Exception as e:
        log.error(f"Failed to list models: {e}", exc_info=True)
        return []


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ray_initialized": ray.is_initialized()
    }


@app.get("/status")
async def get_status():
    """Get overall system status."""
    status = {
        "ray_initialized": ray.is_initialized(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if ray.is_initialized():
        status["cluster_resources"] = ray.cluster_resources()
        status["available_resources"] = ray.available_resources()
    
    status["active_experiments"] = list(orchestrator.results.keys())
    status["deployed_ensembles"] = ensemble_manager.list_ensembles()
    status["available_symbols"] = get_available_symbols()[:20]  # First 20
    
    return status


@app.get("/symbols")
async def list_symbols():
    """List available ticker symbols with date ranges."""
    symbols = get_available_symbols()
    
    symbol_info = []
    for sym in symbols:
        start, end = get_symbol_date_range(sym)
        symbol_info.append({
            "symbol": sym,
            "start_date": start,
            "end_date": end
        })
    
    return {"symbols": symbol_info, "count": len(symbols)}


# ============================================================================
# Model Discovery - List trained models from Ray checkpoints
# ============================================================================

@app.get("/models/list")
async def list_trained_models_endpoint():
    """List all trained models from Ray checkpoints."""
    try:
        from ray_orchestrator.list_models import list_trained_models
        
        models = list_trained_models(str(settings.data.checkpoints_dir))
        return {"models": models, "count": len(models)}
    except Exception as e:
        log.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MLflow Model Registry Endpoints
# ============================================================================

@app.get("/mlflow/models")
async def list_mlflow_models():
    """List all registered models in MLflow."""
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        log.info(f"📡 /mlflow/models called - tracking_uri: {tracking_uri}")
        
        mlflow_tracker = MLflowTracker(tracking_uri=tracking_uri)
        log.info(f"📡 MLflowTracker created")
        
        models = mlflow_tracker.get_registered_models()
        log.info(f"📡 get_registered_models() returned {len(models)} models")
        
        if models:
            log.info(f"📡 First model: {models[0]}")
        else:
            log.warning(f"📡 No models returned from MLflow!")
            
        return {"models": models, "count": len(models)}
    except Exception as e:
        log.error(f"❌ Failed to fetch MLflow models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/model/{model_name}/{version}")
async def get_model_details(model_name: str, version: str):
    """Get detailed information about a specific model version."""
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        
        client = mlflow.tracking.MlflowClient()
        
        # Get model version details
        mv = client.get_model_version(model_name, version)
        
        # Get run details
        run = client.get_run(mv.run_id)
        
        # Get permutation importance if logged
        permutation_importance = None
        try:
            artifacts = client.list_artifacts(mv.run_id)
            for artifact in artifacts:
                if "permutation_importance" in artifact.path:
                    # Download and parse
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        local_path = client.download_artifacts(mv.run_id, artifact.path, tmpdir)
                        with open(local_path, 'r') as f:
                            import pandas as pd
                            permutation_importance = pd.read_json(f).to_dict(orient='records')
                    break
        except Exception as e:
            log.warning(f"Could not load permutation importance: {e}")
        
        return {
            "model_name": model_name,
            "version": version,
            "stage": mv.current_stage,
            "run_id": mv.run_id,
            "description": mv.description or "",
            "created_at": mv.creation_timestamp,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "permutation_importance": permutation_importance
        }
        
    except Exception as e:
        log.error(f"Failed to get model details: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))


class ModelStageTransition(BaseModel):
    stage: str  # None, Staging, Production, Archived


@app.post("/mlflow/model/{model_name}/{version}/transition")
async def transition_model_stage(model_name: str, version: str, transition: ModelStageTransition):
    """Transition a model to a new stage."""
    try:
        mlflow_tracker = MLflowTracker(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow_tracker.transition_model_stage(model_name, version, transition.stage)
        return {"status": "success", "model": model_name, "version": version, "new_stage": transition.stage}
    except Exception as e:
        log.error(f"Failed to transition model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class BacktestRequest(BaseModel):
    model_name: str
    model_version: str
    ticker: str
    start_date: str
    end_date: str
    slippage_pct: float = 0.0001  # 0.01% default
    commission_per_share: float = 0.001  # $0.001 per share
    initial_capital: float = 100000.0
    position_size_pct: float = 0.1  # 10% of capital per trade


@app.post("/backtest/simulate")
async def run_backtest_simulation(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run a backtest simulation with a trained model.
    
    This loads the model from MLflow and runs it on historical data
    with realistic transaction costs and slippage.
    """
    try:
        # Validate model exists
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = mlflow.tracking.MlflowClient()
        
        try:
            mv = client.get_model_version(request.model_name, request.model_version)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} v{request.model_version} not found")
        
        # Create backtest job
        job_id = f"backtest_{request.ticker}_{request.model_name}_v{request.model_version}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Queue backtest (run in background)
        background_tasks.add_task(
            execute_backtest,
            job_id=job_id,
            model_name=request.model_name,
            model_version=request.model_version,
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            slippage_pct=request.slippage_pct,
            commission_per_share=request.commission_per_share,
            initial_capital=request.initial_capital,
            position_size_pct=request.position_size_pct
        )
        
        return {
            "status": "queued",
            "job_id": job_id,
            "message": "Backtest simulation queued. Check /backtest/status/{job_id} for progress."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to queue backtest: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def execute_backtest(
    job_id: str,
    model_name: str,
    model_version: str,
    ticker: str,
    start_date: str,
    end_date: str,
    slippage_pct: float,
    commission_per_share: float,
    initial_capital: float,
    position_size_pct: float
):
    """Execute backtest simulation (runs in background)."""
    import subprocess
    
    results_dir = Path(settings.data.checkpoints_dir) / "backtest_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    status_file = results_dir / f"{job_id}_status.json"
    results_file = results_dir / f"{job_id}_results.json"
    
    # Write initial status
    with open(status_file, 'w') as f:
        json.dump({"status": "running", "started_at": datetime.utcnow().isoformat()}, f)
    
    try:
        # Run backtest_model.py script
        cmd = [
            "python", "/app/backtest_model.py",
            "--checkpoint", f"models:/{model_name}/{model_version}",  # MLflow URI
            "--ticker", ticker,
            "--start-date", start_date,
            "--end-date", end_date,
            "--output-dir", str(results_dir),
            "--mlflow-mode"  # Special flag to load from MLflow instead of Ray checkpoint
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            # Success - update status
            with open(status_file, 'w') as f:
                json.dump({
                    "status": "completed",
                    "started_at": datetime.utcnow().isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                    "results_file": str(results_file)
                }, f)
        else:
            # Failure
            with open(status_file, 'w') as f:
                json.dump({
                    "status": "failed",
                    "error": result.stderr,
                    "started_at": datetime.utcnow().isoformat(),
                    "completed_at": datetime.utcnow().isoformat()
                }, f)
                
    except Exception as e:
        log.error(f"Backtest execution failed: {e}", exc_info=True)
        with open(status_file, 'w') as f:
            json.dump({
                "status": "failed",
                "error": str(e),
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat()
            }, f)


@app.get("/backtest/status/{job_id}")
async def get_backtest_status(job_id: str):
    """Get status of a backtest job."""
    results_dir = Path(settings.data.checkpoints_dir) / "backtest_results"
    status_file = results_dir / f"{job_id}_status.json"
    
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    with open(status_file, 'r') as f:
        status = json.load(f)
    
    return status


@app.get("/backtest/results/{job_id}")
async def get_backtest_results(job_id: str):
    """Get results of a completed backtest."""
    results_dir = Path(settings.data.checkpoints_dir) / "backtest_results"
    results_file = results_dir / f"{job_id}_results.json"
    
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


@app.get("/algorithms")
async def list_algorithms():
    """List available algorithms with their search spaces."""
    return {
        "algorithms": list(SEARCH_SPACES.keys()),
        "search_spaces": SEARCH_SPACES
    }


# ============================================================================
# Hyperparameter Search Endpoints
# ============================================================================

@app.post("/search/grid")
async def start_grid_search(request: GridSearchRequest, background_tasks: BackgroundTasks):
    """
    Start a grid search experiment.
    
    Runs exhaustive search over all parameter combinations.
    """
    log.info(f"Starting grid search: {request.algorithm} on {request.tickers}")
    
    def run_search():
        try:
            result = orchestrator.run_grid_search(
                algorithm=request.algorithm,
                tickers=request.tickers,
                param_grid=request.param_grid,
                name=request.name,
                target_col=request.target_col,
                target_transform=request.target_transform,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )
            log.info(f"Grid search completed: {result.get('experiment_name')}")
        except Exception as e:
            log.error(f"Grid search failed: {e}")
    
    background_tasks.add_task(run_search)
    
    return {
        "status": "started",
        "experiment_type": "grid_search",
        "algorithm": request.algorithm,
        "tickers": request.tickers,
        "estimated_trials": sum(len(v) if isinstance(v, list) else 1 for v in request.param_grid.values()) * len(request.tickers)
    }


@app.post("/search/pbt")
async def start_pbt_search(request: PBTSearchRequest, background_tasks: BackgroundTasks):
    """
    Start a Population-Based Training (PBT) experiment.
    
    Evolves a population of models over generations, replacing poor
    performers with mutated versions of the best models.
    
    Deduplication: Set resume=true to continue a crashed/stopped experiment
    without re-training completed trials.
    """
    log.info(f"Starting PBT search: {request.algorithm} on {request.tickers} (resume={request.resume})")
    
    def run_search():
        try:
            result = orchestrator.run_pbt_search(
                algorithm=request.algorithm,
                tickers=request.tickers,
                population_size=request.population_size,
                num_generations=request.num_generations,
                name=request.name,
                resume=request.resume,
                target_col=request.target_col,
                target_transform=request.target_transform,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )
            log.info(f"PBT search completed: {result.get('experiment_name')}")
        except Exception as e:
            log.error(f"PBT search failed: {e}")
    
    background_tasks.add_task(run_search)
    
    return {
        "status": "started",
        "experiment_type": "pbt",
        "algorithm": request.algorithm,
        "tickers": request.tickers,
        "population_size": request.population_size,
        "num_generations": request.num_generations,
        "resume": request.resume
    }


@app.post("/search/asha")
async def start_asha_search(request: ASHASearchRequest, background_tasks: BackgroundTasks):
    """
    Start an ASHA (Asynchronous Successive Halving) experiment.
    
    Aggressively stops underperforming trials early to focus
    resources on the most promising configurations.
    
    Deduplication:
        - Uses skip_duplicate=True for Bayesian/Optuna/HyperOpt search algorithms
        - Set resume=true to continue a crashed/stopped experiment
    """
    log.info(f"Starting ASHA search: {request.algorithm} on {request.tickers} (resume={request.resume})")
    
    def run_search():
        try:
            result = orchestrator.run_asha_search(
                algorithm=request.algorithm,
                tickers=request.tickers,
                num_samples=request.num_samples,
                search_alg=request.search_alg,
                name=request.name,
                resume=request.resume,
                target_col=request.target_col,
                target_transform=request.target_transform,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )
            log.info(f"ASHA search completed: {result.get('experiment_name')}")
        except Exception as e:
            log.error(f"ASHA search failed: {e}")
    
    background_tasks.add_task(run_search)
    
    return {
        "status": "started",
        "experiment_type": "asha",
        "algorithm": request.algorithm,
        "tickers": request.tickers,
        "num_samples": request.num_samples,
        "search_algorithm": request.search_alg,
        "resume": request.resume
    }


@app.post("/search/multi-ticker-pbt")
async def start_multi_ticker_pbt(request: MultiTickerPBTRequest, background_tasks: BackgroundTasks):
    """
    Start a multi-ticker PBT experiment.
    
    Finds hyperparameters that work well across ALL tickers,
    not just optimized for individual symbols.
    
    Deduplication: Set resume=true to continue a crashed/stopped experiment.
    """
    log.info(f"Starting multi-ticker PBT: {request.algorithm} on {request.tickers} (resume={request.resume})")
    
    def run_search():
        try:
            result = orchestrator.run_multi_ticker_pbt(
                algorithm=request.algorithm,
                tickers=request.tickers,
                population_size=request.population_size,
                num_generations=request.num_generations,
                name=request.name,
                resume=request.resume,
                target_col=request.target_col,
                target_transform=request.target_transform,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )
            log.info(f"Multi-ticker PBT completed: {result.get('experiment_name')}")
        except Exception as e:
            log.error(f"Multi-ticker PBT failed: {e}")
    
    background_tasks.add_task(run_search)
    
    return {
        "status": "started",
        "experiment_type": "multi_ticker_pbt",
        "algorithm": request.algorithm,
        "tickers": request.tickers,
        "population_size": request.population_size,
        "num_generations": request.num_generations,
        "resume": request.resume
    }


@app.post("/search/bayesian")
async def start_bayesian_search(request: BayesianSearchRequest, background_tasks: BackgroundTasks):
    """
    Start a Bayesian Optimization search.
    
    Uses a Gaussian Process surrogate model to intelligently explore
    the hyperparameter space, with skip_duplicate=True to avoid
    re-testing identical configurations.
    
    This is ideal for finding optimal hyperparameters efficiently.
    
    Deduplication:
        - Automatically enabled via skip_duplicate=True
        - Set resume=true to continue a crashed/stopped experiment
    """
    log.info(f"Starting Bayesian search: {request.algorithm} on {request.tickers} (resume={request.resume})")
    
    def run_search():
        try:
            result = orchestrator.run_bayesian_search(
                algorithm=request.algorithm,
                tickers=request.tickers,
                num_samples=request.num_samples,
                name=request.name,
                resume=request.resume,
                target_col=request.target_col,
                target_transform=request.target_transform,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )
            log.info(f"Bayesian search completed: {result.get('experiment_name')}")
        except Exception as e:
            log.error(f"Bayesian search failed: {e}")
    
    background_tasks.add_task(run_search)
    
    return {
        "status": "started",
        "experiment_type": "bayesian",
        "algorithm": request.algorithm,
        "tickers": request.tickers,
        "num_samples": request.num_samples,
        "resume": request.resume,
        "skip_duplicate": settings.tune.skip_duplicate
    }


# ============================================================================
# Fingerprint/Deduplication Endpoints
# ============================================================================

@app.get("/fingerprint/stats")
async def get_fingerprint_stats():
    """
    Get fingerprint database statistics.
    
    Shows how many configurations have been tested and cached.
    """
    return fingerprint_db.get_stats()


@app.get("/fingerprint/{experiment_name}")
async def get_experiment_fingerprints(experiment_name: str):
    """
    Get all fingerprints for a specific experiment.
    """
    fingerprints = fingerprint_db.get_experiment_fingerprints(experiment_name)
    return {
        "experiment_name": experiment_name,
        "fingerprints": fingerprints,
        "count": len(fingerprints)
    }


@app.delete("/fingerprint/{experiment_name}")
async def delete_experiment_fingerprints(experiment_name: str):
    """
    Delete all fingerprints for an experiment.
    
    Use this if you want to re-run an experiment from scratch.
    """
    fingerprint_db.delete_experiment(experiment_name)
    return {
        "status": "deleted",
        "experiment_name": experiment_name
    }


@app.delete("/fingerprint")
async def clear_all_fingerprints():
    """
    Clear all fingerprints from the database.
    
    WARNING: This will allow all configs to be re-trained.
    Use with caution.
    """
    fingerprint_db.clear_all()
    return {
        "status": "cleared",
        "message": "All fingerprints have been deleted"
    }


@app.get("/experiments")
async def list_experiments():
    """List all experiments and their status."""
    experiments = []
    
    for name, results in orchestrator.results.items():
        try:
            df = results.get_dataframe()
            experiments.append({
                "name": name,
                "num_trials": len(df),
                "completed": len(df[df.get("status", "") == "completed"]) if "status" in df.columns else len(df),
                "best_metric": float(df[settings.tune.metric].max()) if settings.tune.metric in df.columns else None
            })
        except Exception as e:
            experiments.append({
                "name": name,
                "error": str(e)
            })
    
    return {"experiments": experiments}


@app.get("/experiments/{experiment_name}")
async def get_experiment_results(experiment_name: str):
    """Get detailed results for a specific experiment."""
    if experiment_name not in orchestrator.results:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_name} not found")
    
    results = orchestrator.results[experiment_name]
    
    try:
        best_result = results.get_best_result()
        df = results.get_dataframe()
        
        return {
            "experiment_name": experiment_name,
            "best_trial": {
                "config": best_result.config,
                "metrics": best_result.metrics,
                "checkpoint_path": str(best_result.checkpoint) if best_result.checkpoint else None
            },
            "all_trials": df.to_dict(orient="records"),
            "summary": {
                "total_trials": len(df),
                "completed": len(df[df.get("status", "") == "completed"]) if "status" in df.columns else len(df),
                "best_metric": float(df[settings.tune.metric].max()) if settings.tune.metric in df.columns else None,
                "mean_metric": float(df[settings.tune.metric].mean()) if settings.tune.metric in df.columns else None,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments/{experiment_name}/top/{n}")
async def get_top_models(experiment_name: str, n: int = 5):
    """Get top N models from an experiment."""
    models = orchestrator.get_best_models(experiment_name, n)
    
    if not models:
        raise HTTPException(status_code=404, detail=f"No models found for {experiment_name}")
    
    return {"experiment_name": experiment_name, "top_models": models}


# ============================================================================
# Ensemble Deployment Endpoints
# ============================================================================

@app.post("/ensemble/deploy")
async def deploy_ensemble(request: DeployEnsembleRequest):
    """
    Deploy a voting ensemble with specified models.
    """
    log.info(f"Deploying ensemble: {request.ensemble_name}")
    
    try:
        result = ensemble_manager.deploy_ensemble(
            ensemble_name=request.ensemble_name,
            model_paths=request.model_paths,
            model_ids=request.model_ids,
            voting=request.voting,
            threshold=request.threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ensemble/deploy-pbt-survivors")
async def deploy_pbt_survivors(request: DeployPBTSurvivorsRequest):
    """
    Deploy top N models from a PBT experiment as an ensemble.
    """
    log.info(f"Deploying PBT survivors as ensemble: {request.ensemble_name}")
    
    if request.experiment_name not in orchestrator.results:
        raise HTTPException(status_code=404, detail=f"Experiment {request.experiment_name} not found")
    
    try:
        # Get formatted results
        results = orchestrator.results[request.experiment_name]
        formatted = orchestrator._format_results(results, request.experiment_name)
        
        result = ensemble_manager.deploy_pbt_survivors(
            ensemble_name=request.ensemble_name,
            experiment_results=formatted,
            top_n=request.top_n,
            voting=request.voting,
            threshold=request.threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ensemble")
async def list_ensembles():
    """List all deployed ensembles."""
    return {"ensembles": ensemble_manager.list_ensembles()}


@app.get("/ensemble/{ensemble_name}")
async def get_ensemble_status(ensemble_name: str):
    """Get status of a specific ensemble."""
    status = ensemble_manager.get_ensemble_status(ensemble_name)
    
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    
    return status


@app.post("/ensemble/{ensemble_name}/predict")
async def ensemble_predict(ensemble_name: str, request: PredictRequest):
    """
    Get prediction from an ensemble.
    
    This is a convenience endpoint. For production, use Ray Serve's
    native endpoint at /{ensemble_name}.
    """
    if ensemble_name not in ensemble_manager.ensembles:
        raise HTTPException(status_code=404, detail=f"Ensemble {ensemble_name} not found")
    
    try:
        handle = ensemble_manager.ensembles[ensemble_name]
        result = await handle.remote({"features": request.features})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ensemble/{ensemble_name}")
async def delete_ensemble(ensemble_name: str):
    """Delete an ensemble and its models."""
    result = ensemble_manager.delete_ensemble(ensemble_name)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


# ============================================================================
# Dashboard UI
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Ray Orchestrator Dashboard",
        "ray_dashboard_url": f"http://localhost:{settings.ray.ray_dashboard_port}"
    })


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for running the server."""
    import uvicorn
    
    # Ensure data paths exist
    settings.data.ensure_paths()
    
    # Initialize Ray
    orchestrator.init_ray()
    
    log.info(f"Starting Ray Orchestrator API on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "ray_orchestrator.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
