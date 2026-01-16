"""
Walk-forward training pipeline using Ray Tune.

Integrates the streaming preprocessing folds with hyperparameter tuning,
ensuring each trial is evaluated across multiple time-based folds.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path

import ray
from ray import tune
from ray.train import Checkpoint
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from .streaming import StreamingPreprocessor, BarDataLoader, Fold
from .config import settings

log = logging.getLogger(__name__)


class WalkForwardTrainer:
    """Train models using walk-forward validation with Ray Tune."""
    
    def __init__(self, preprocessor: StreamingPreprocessor):
        self.preprocessor = preprocessor
        self.folds: List[Fold] = []
    
    def create_trainable(
        self,
        algorithm: str = "elasticnet",
        target_col: str = "close",
        feature_cols: Optional[List[str]] = None,
        target_transform: str = "log_return"
    ):
        """
        Create a Ray Tune trainable function for walk-forward validation.
        
        Args:
            algorithm: Model type (elasticnet, ridge, lasso, randomforest)
            target_col: Column to predict
            feature_cols: Features to use (None = auto-select)
            target_transform: Target transformation (log_return, pct_change, raw)
            
        Returns:
            Trainable function for Ray Tune
        """
        folds = self.folds
        
        def train_on_folds(config: Dict) -> Dict:
            """
            Train on all folds and report average metrics.
            
            This is what Ray Tune will call for each hyperparameter configuration.
            """
            fold_metrics = []
            
            for fold in folds:
                # Train on this fold
                metrics = self._train_single_fold(
                    fold=fold,
                    config=config,
                    algorithm=algorithm,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    target_transform=target_transform
                )
                fold_metrics.append(metrics)
            
            # Average metrics across all folds
            avg_metrics = {
                "train_rmse": np.mean([m["train_rmse"] for m in fold_metrics]),
                "test_rmse": np.mean([m["test_rmse"] for m in fold_metrics]),
                "test_mae": np.mean([m["test_mae"] for m in fold_metrics]),
                "test_r2": np.mean([m["test_r2"] for m in fold_metrics]),
                "num_folds": len(fold_metrics),
            }
            
            # Report to Ray Tune
            return avg_metrics
        
        return train_on_folds
    
    def _train_single_fold(
        self,
        fold: Fold,
        config: Dict,
        algorithm: str,
        target_col: str,
        feature_cols: Optional[List[str]],
        target_transform: str
    ) -> Dict:
        """Train and evaluate on a single fold."""
        
        # Load fold data
        train_df = fold.train_ds.to_pandas() if fold.train_ds else pd.DataFrame()
        test_df = fold.test_ds.to_pandas() if fold.test_ds else pd.DataFrame()
        
        if train_df.empty or test_df.empty:
            log.warning(f"Empty data in {fold}")
            return {"train_rmse": np.inf, "test_rmse": np.inf, "test_mae": np.inf, "test_r2": -np.inf}
        
        # Prepare features and target
        if feature_cols is None:
            # Auto-select numeric features (exclude target and metadata)
            exclude_cols = {'ts', 'symbol', target_col, 'open', 'high', 'low', 'volume', 'vwap', 'trade_count'}
            feature_cols = [c for c in train_df.columns if c not in exclude_cols and train_df[c].dtype in [np.float64, np.int64]]
        
        # Create target variable
        if target_transform == "log_return":
            train_y = np.log(train_df[target_col] / train_df[target_col].shift(1)).dropna()
            test_y = np.log(test_df[target_col] / test_df[target_col].shift(1)).dropna()
        elif target_transform == "pct_change":
            train_y = train_df[target_col].pct_change().dropna()
            test_y = test_df[target_col].pct_change().dropna()
        else:  # raw
            train_y = train_df[target_col]
            test_y = test_df[target_col]
        
        # Align features with target (drop NaN rows)
        train_X = train_df[feature_cols].loc[train_y.index].dropna()
        train_y = train_y.loc[train_X.index]
        
        test_X = test_df[feature_cols].loc[test_y.index].dropna()
        test_y = test_y.loc[test_X.index]
        
        if len(train_X) == 0 or len(test_X) == 0:
            log.warning(f"No valid data after preprocessing in {fold}")
            return {"train_rmse": np.inf, "test_rmse": np.inf, "test_mae": np.inf, "test_r2": -np.inf}
        
        # Create model
        model = self._create_model(algorithm, config)
        
        # Train
        model.fit(train_X, train_y)
        
        # Evaluate
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)
        
        train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
        test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
        test_mae = mean_absolute_error(test_y, test_pred)
        test_r2 = r2_score(test_y, test_pred)
        
        return {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
        }
    
    def _create_model(self, algorithm: str, config: Dict):
        """Create model instance from algorithm name and config."""
        if algorithm == "elasticnet":
            return ElasticNet(
                alpha=config.get("alpha", 1.0),
                l1_ratio=config.get("l1_ratio", 0.5),
                max_iter=config.get("max_iter", 1000)
            )
        elif algorithm == "ridge":
            return Ridge(
                alpha=config.get("alpha", 1.0),
                max_iter=config.get("max_iter", 1000)
            )
        elif algorithm == "lasso":
            return Lasso(
                alpha=config.get("alpha", 1.0),
                max_iter=config.get("max_iter", 1000)
            )
        elif algorithm == "randomforest":
            return RandomForestRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 10),
                min_samples_split=config.get("min_samples_split", 2),
                random_state=42
            )
        elif algorithm == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", 0.1),
                subsample=config.get("subsample", 0.8),
                colsample_bytree=config.get("colsample_bytree", 0.8),
                random_state=42
            )
        elif algorithm == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", 0.1),
                subsample=config.get("subsample", 0.8),
                colsample_bytree=config.get("colsample_bytree", 0.8),
                random_state=42,
                verbosity=-1  # Suppress warnings
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def run_walk_forward_tuning(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        train_months: int = 3,
        test_months: int = 1,
        step_months: int = 1,
        algorithm: str = "elasticnet",
        param_space: Dict = None,
        num_samples: int = 50,
        context_symbols: Optional[List[str]] = None,
        windows: List[int] = [50, 200],
        resampling_timeframes: Optional[List[str]] = None,
    ) -> tune.ResultGrid:
        """
        Run hyperparameter tuning with walk-forward validation.
        
        This is the main entry point for training with proper backtesting.
        
        Args:
            symbols: Trading symbols
            start_date, end_date: Date range
            train_months, test_months, step_months: Fold configuration
            algorithm: Model type
            param_space: Hyperparameter search space
            num_samples: Number of trials
            context_symbols: Context symbols (QQQ, VIX)
            windows: SMA windows
            resampling_timeframes: Multi-timeframe features
            
        Returns:
            Ray Tune ResultGrid with all trial results
        """
        log.info("Starting walk-forward hyperparameter tuning")
        
        # Step 1: Generate and load folds
        log.info("Generating walk-forward folds...")
        self.folds = []
        
        for fold in self.preprocessor.create_walk_forward_pipeline(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            context_symbols=context_symbols,
            windows=windows,
            resampling_timeframes=resampling_timeframes,
            num_gpus=0.0,  # Set to 1.0 for GPU
            actor_pool_size=2
        ):
            self.folds.append(fold)
        
        log.info(f"Loaded {len(self.folds)} folds for training")
        
        # Validate that we have non-empty folds
        if not self.folds:
            raise ValueError(
                f"No folds generated! Check that data exists for date range {start_date} to {end_date}. "
                f"Use the /streaming/status endpoint to see available date ranges."
            )
        
        # Check if all folds are empty
        empty_folds = 0
        for fold in self.folds:
            try:
                train_count = fold.train_ds.count()
                test_count = fold.test_ds.count()
                if train_count == 0 or test_count == 0:
                    empty_folds += 1
                    log.warning(f"Empty fold detected: {fold} (train={train_count}, test={test_count})")
            except Exception as e:
                log.warning(f"Could not count fold data: {e}")
        
        if empty_folds == len(self.folds):
            raise ValueError(
                f"All {len(self.folds)} folds are empty! No data found for symbols {symbols} "
                f"in date range {start_date} to {end_date}. Check /streaming/status for available dates."
            )
        elif empty_folds > 0:
            log.warning(f"{empty_folds}/{len(self.folds)} folds are empty and will produce inf metrics")
        
        # Step 2: Define search space
        if param_space is None:
            param_space = self._default_param_space(algorithm)
        
        # Step 3: Create trainable
        trainable = self.create_trainable(
            algorithm=algorithm,
            target_col="close",
            target_transform="log_return"
        )
        
        # Step 4: Run tuning
        log.info(f"Running {num_samples} trials with {algorithm}")
        
        # Maximize CPU usage by running multiple trials in parallel
        # Each trial uses 1 CPU, so we can run as many concurrent trials as we have cores
        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric="test_rmse",
                mode="min",
                num_samples=num_samples,
                max_concurrent_trials=0,  # 0 = unlimited (use all available CPUs)
            ),
            run_config=ray.train.RunConfig(
                name=f"walk_forward_{algorithm}_{symbols[0]}",
                storage_path=str(settings.data.checkpoints_dir),
                # Limit checkpoint storage to save disk space
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=1,  # Only keep best checkpoint
                    checkpoint_score_attribute="test_rmse",
                    checkpoint_score_order="min",
                ),
            )
        )
        
        results = tuner.fit()
        
        log.info("Hyperparameter tuning complete")
        
        # Try to get best result, handle case where all trials failed
        try:
            best_result = results.get_best_result(metric="test_rmse", mode="min")
            log.info(f"Best config: {best_result.config}")
            log.info(f"Best test RMSE: {best_result.metrics['test_rmse']:.6f}")
        except RuntimeError as e:
            log.error(f"No valid trials completed successfully: {e}")
            log.error("All trials returned NaN/inf metrics. Check that your date range has actual data!")
            # Get a sample of results for debugging
            all_results = results.get_dataframe()
            log.error(f"Trial summary:\n{all_results[['test_rmse', 'test_mae', 'train_rmse']].describe()}")
            raise ValueError(
                "Training failed: All trials returned invalid metrics (NaN/inf). "
                "This typically means no data was found for the specified date range. "
                "Use /streaming/status to check available dates."
            ) from e
        
        return results
    
    def _default_param_space(self, algorithm: str) -> Dict:
        """Default hyperparameter search spaces."""
        if algorithm == "elasticnet":
            return {
                "alpha": tune.loguniform(1e-4, 1.0),
                "l1_ratio": tune.uniform(0.0, 1.0),
            }
        elif algorithm == "ridge":
            return {
                "alpha": tune.loguniform(1e-4, 100.0),
            }
        elif algorithm == "lasso":
            return {
                "alpha": tune.loguniform(1e-4, 10.0),
            }
        elif algorithm == "randomforest":
            return {
                "n_estimators": tune.choice([50, 100, 200]),
                "max_depth": tune.randint(5, 20),
                "min_samples_split": tune.randint(2, 10),
            }
        elif algorithm == "xgboost":
            return {
                "n_estimators": tune.choice([50, 100, 200, 300]),
                "max_depth": tune.randint(3, 10),
                "learning_rate": tune.loguniform(0.01, 0.3),
                "subsample": tune.uniform(0.6, 1.0),
                "colsample_bytree": tune.uniform(0.6, 1.0),
            }
        elif algorithm == "lightgbm":
            return {
                "n_estimators": tune.choice([50, 100, 200, 300]),
                "max_depth": tune.randint(3, 10),
                "learning_rate": tune.loguniform(0.01, 0.3),
                "subsample": tune.uniform(0.6, 1.0),
                "colsample_bytree": tune.uniform(0.6, 1.0),
            }
        else:
            return {}


def create_walk_forward_trainer(parquet_dir: str = "/app/data/parquet") -> WalkForwardTrainer:
    """Factory function to create a walk-forward trainer."""
    loader = BarDataLoader(parquet_dir=parquet_dir)
    preprocessor = StreamingPreprocessor(loader)
    return WalkForwardTrainer(preprocessor)
