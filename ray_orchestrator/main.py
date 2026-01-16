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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import ray

from .config import settings
from .data import get_available_symbols, get_symbol_date_range
from .tuner import orchestrator, SEARCH_SPACES
from .ensemble import ensemble_manager
from .fingerprint import fingerprint_db
from .streaming import create_preprocessing_pipeline, BarDataLoader
from .trainer import create_walk_forward_trainer

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("ray_orchestrator.api")

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
    num_gpus: float = 0.0  # Set to 1.0 for GPU acceleration
    actor_pool_size: int = 2


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
    num_samples: int = 50
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
                parquet_dir=str(settings.data.features_parquet_dir.parent / "parquet")
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


@app.get("/streaming/status")
async def get_streaming_status():
    """Get status of Ray Data streaming jobs."""
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
        loader = BarDataLoader(parquet_dir=str(settings.data.features_parquet_dir.parent / "parquet"))
        files = loader._discover_parquet_files()
        
        stats["data_sources"] = {
            "total_parquet_files": len(files),
            "sample_files": files[:10] if files else []
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
        loader = BarDataLoader(parquet_dir=str(settings.data.features_parquet_dir.parent / "parquet"))
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
    log.info(f"Starting walk-forward preprocessing: {request.symbols} from {request.start_date} to {request.end_date}")
    
    def run_preprocessing():
        try:
            from pathlib import Path
            
            # Create pipeline
            preprocessor = create_preprocessing_pipeline(
                parquet_dir=str(settings.data.features_parquet_dir.parent / "parquet")
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
                actor_pool_size=request.actor_pool_size
            ):
                fold_count += 1
                log.info(f"Processing {fold}")
                
                # Save train data
                if fold.train_ds:
                    train_path = f"{request.output_base_path}/fold_{fold.fold_id}/train"
                    fold.train_ds.write_parquet(train_path, try_create_dir=True)
                    log.info(f"Saved train data to {train_path}")
                
                # Save test data
                if fold.test_ds:
                    test_path = f"{request.output_base_path}/fold_{fold.fold_id}/test"
                    fold.test_ds.write_parquet(test_path, try_create_dir=True)
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
    
    Example:
        With 3-month train, 1-month test, 1-month step:
        - Trial 1 (alpha=0.01): Avg test RMSE across 5 folds = 0.0234
        - Trial 2 (alpha=0.1): Avg test RMSE across 5 folds = 0.0198
        - Trial 3 (alpha=1.0): Avg test RMSE across 5 folds = 0.0156 (best!)
    """
    log.info(f"Starting walk-forward training: {request.symbols} with {request.algorithm}")
    
    def run_training():
        try:
            # Create trainer
            trainer = create_walk_forward_trainer(
                parquet_dir=str(settings.data.features_parquet_dir.parent / "parquet")
            )
            
            # Run tuning
            results = trainer.run_walk_forward_tuning(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                train_months=request.train_months,
                test_months=request.test_months,
                step_months=request.step_months,
                algorithm=request.algorithm,
                param_space=request.param_space,
                num_samples=request.num_samples,
                context_symbols=request.context_symbols,
                windows=request.windows,
                resampling_timeframes=request.resampling_timeframes
            )
            
            # Log best result
            best = results.get_best_result()
            log.info(f"Training complete! Best config: {best.config}")
            log.info(f"Best metrics: test_rmse={best.metrics['test_rmse']:.6f}, "
                    f"test_r2={best.metrics['test_r2']:.4f}")
            
        except Exception as e:
            log.error(f"Walk-forward training failed: {e}", exc_info=True)
    
    background_tasks.add_task(run_training)
    
    return {
        "status": "started",
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
        "note": "Each trial is evaluated across all folds to ensure temporal robustness"
    }


# ============================================================================
# Dashboard UI
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the training dashboard UI."""
    return templates.TemplateResponse("training_dashboard.html", {"request": request})


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
