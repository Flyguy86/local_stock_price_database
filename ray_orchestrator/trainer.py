"""
Walk-forward training pipeline using Ray Tune.

CRITICAL: All training must use pre-processed walk-forward folds from
/app/data/walk_forward_folds/ to ensure proper feature calculation isolation.

Each fold has features calculated ONLY on its training window to prevent
look-ahead bias. Direct loading of raw parquet data is prohibited for training.

Workflow:
1. Generate folds: POST /streaming/walk_forward (one-time preprocessing)
2. Load folds: Use load_fold_from_disk() in data.py
3. Train models: Ray Tune evaluates across all folds
4. Deploy: Best model from cross-fold validation
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path

import ray
from ray import tune
from ray.train import Checkpoint
from ray.air.integrations.mlflow import MLflowLoggerCallback
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from .streaming import StreamingPreprocessor, BarDataLoader, Fold
from .config import settings
from .mlflow_integration import MLflowTracker

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
        target_transform: str = "log_return",
        preprocessing_config: Optional[Dict] = None
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
            Saves per-fold models and feature lists to checkpoint.
            """
            import tempfile
            import joblib
            import time
            import os
            from datetime import datetime, timezone
            from pathlib import Path
            from ray.train import Checkpoint
            import sys
            import xgboost
            import lightgbm
            import sklearn
            import pandas as pd
            import numpy as np
            
            training_start_time = time.time()
            
            fold_metrics = []
            fold_models = {}  # Store models per fold
            fold_feature_lists = {}  # Store feature columns per fold
            fold_metadata = {}  # Store date ranges and row counts per fold
            fold_training_metrics = {}  # Store per-fold training performance
            
            for fold in folds:
                # Train on this fold with timing
                fold_start_time = time.time()
                metrics, model, features_used, train_rows, test_rows = self._train_single_fold(
                    fold=fold,
                    config=config,
                    algorithm=algorithm,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    target_transform=target_transform,
                    excluded_features=preprocessing_config.get("excluded_features") if preprocessing_config else None
                )
                fold_training_time = time.time() - fold_start_time
                
                fold_metrics.append(metrics)
                fold_models[fold.fold_id] = model
                fold_feature_lists[fold.fold_id] = features_used
                
                # Capture fold date ranges and row counts
                fold_metadata[f"fold_{fold.fold_id:03d}"] = {
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "train_rows": train_rows,
                    "test_rows": test_rows
                }
                
                # Store per-fold training metrics
                fold_training_metrics[f"fold_{fold.fold_id:03d}"] = {
                    "train_rmse": metrics["train_rmse"],
                    "test_rmse": metrics["test_rmse"],
                    "test_r2": metrics["test_r2"],
                    "test_mae": metrics["test_mae"],
                    "training_time_seconds": round(fold_training_time, 2)
                }
            
            # Calculate training duration
            total_training_time = time.time() - training_start_time
            
            # Average metrics across all folds
            avg_metrics = {
                "train_rmse": np.mean([m["train_rmse"] for m in fold_metrics]),
                "test_rmse": np.mean([m["test_rmse"] for m in fold_metrics]),
                "test_mae": np.mean([m["test_mae"] for m in fold_metrics]),
                "test_r2": np.mean([m["test_r2"] for m in fold_metrics]),
                "num_folds": len(fold_metrics),
            }
            
            # Calculate cross-fold consistency (variance)
            test_rmse_values = [m["test_rmse"] for m in fold_metrics]
            test_r2_values = [m["test_r2"] for m in fold_metrics]
            
            # Find best and worst folds
            best_fold_idx = np.argmin(test_rmse_values)
            worst_fold_idx = np.argmax(test_rmse_values)
            
            # Extract feature importance from first fold model (representative)
            feature_importance_data = None
            all_feature_importances = []
            if fold_models:
                first_model = next(iter(fold_models.values()))
                first_features = next(iter(fold_feature_lists.values()))
                
                try:
                    if hasattr(first_model, 'feature_importances_'):  # Tree-based models
                        importances = first_model.feature_importances_
                        # Create list of ALL features with importance
                        all_feature_importances = [
                            {
                                "name": first_features[i],
                                "importance": float(importances[i]),
                                "rank": rank + 1,
                                "is_context": any(ctx in first_features[i] for ctx in ['_QQQ', '_VIX', '_MSFT', '_SPY'])
                            }
                            for rank, i in enumerate(np.argsort(importances)[::-1])
                        ]
                        # Keep top 15 for metadata
                        feature_importance_data = {
                            "top_15_features": all_feature_importances[:15]
                        }
                    elif hasattr(first_model, 'coef_'):  # Linear models
                        coef = np.abs(first_model.coef_)
                        # Create list of ALL features with importance
                        all_feature_importances = [
                            {
                                "name": first_features[i],
                                "importance": float(coef[i]),
                                "rank": rank + 1,
                                "is_context": any(ctx in first_features[i] for ctx in ['_QQQ', '_VIX', '_MSFT', '_SPY'])
                            }
                            for rank, i in enumerate(np.argsort(coef)[::-1])
                        ]
                        # Keep top 15 for metadata
                        feature_importance_data = {
                            "top_15_features": all_feature_importances[:15]
                        }
                    
                    # Log top features to console
                    if all_feature_importances:
                        log.info("=" * 70)
                        log.info("TOP 15 FEATURES BY IMPORTANCE:")
                        for feat in all_feature_importances[:15]:
                            context_marker = " [CONTEXT]" if feat["is_context"] else ""
                            log.info(f"  {feat['rank']:2d}. {feat['name']:40s} {feat['importance']:.6f}{context_marker}")
                        
                        # Count context features in top 15
                        context_in_top15 = sum(1 for f in all_feature_importances[:15] if f["is_context"])
                        log.info(f"\n  📊 Context features in top 15: {context_in_top15}/15")
                        log.info("=" * 70)
                        
                except Exception as e:
                    log.warning(f"Could not extract feature importance: {e}")
            
            # Save checkpoint with per-fold models and metadata
            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                
                # Save each fold's model
                for fold_id, model in fold_models.items():
                    model_file = tmppath / f"fold_{fold_id:03d}_model.joblib"
                    joblib.dump(model, model_file)
                
                # Save feature lists as JSON
                import json
                features_file = tmppath / "feature_lists.json"
                with open(features_file, 'w') as f:
                    json.dump({str(k): v for k, v in fold_feature_lists.items()}, f, indent=2)
                
                # Save ALL feature importances to dedicated file
                if all_feature_importances:
                    importance_file = tmppath / "feature_importance.json"
                    with open(importance_file, 'w') as f:
                        json.dump({
                            "all_features": all_feature_importances,
                            "summary": {
                                "total_features": len(all_feature_importances),
                                "context_features_in_top15": sum(1 for f in all_feature_importances[:15] if f["is_context"]),
                                "context_features_in_top50": sum(1 for f in all_feature_importances[:50] if f["is_context"]),
                                "total_context_features": sum(1 for f in all_feature_importances if f["is_context"])
                            }
                        }, f, indent=2)
                
                # Save hyperparameters + top features + PBT config for MLflow visibility
                # NOTE: Ray Tune will also create params.json from this config automatically
                config_with_features = config.copy()
                
                # Add PBT-specific parameters for checkpoint tracking
                # These values reflect the PBT scheduler settings used during this trial
                from .config import settings
                config_with_features["pbt_perturbation_interval"] = settings.tune.pbt_perturbation_interval
                config_with_features["pbt_quantile_fraction"] = settings.tune.pbt_quantile_fraction
                config_with_features["pbt_resample_probability"] = settings.tune.pbt_resample_probability
                config_with_features["pbt_perturbation_factors"] = settings.tune.pbt_perturbation_factors
                config_with_features["checkpoint_frequency"] = settings.ray.checkpoint_frequency
                
                if feature_importance_data:
                    # Add top 5 feature names as params (MLflow friendly)
                    for i, feat in enumerate(feature_importance_data["top_15_features"][:5], 1):
                        config_with_features[f"top_feature_{i}"] = feat["name"]
                        config_with_features[f"top_feature_{i}_importance"] = feat["importance"]
                
                config_file = tmppath / "config.json"
                with open(config_file, 'w') as f:
                    json.dump(config_with_features, f, indent=2)
                
                # Build comprehensive metadata
                complete_metadata = {
                    # 1. Model algorithm & version
                    "model_info": {
                        "algorithm": algorithm,
                        "model_class": first_model.__class__.__name__ if fold_models else "Unknown",
                        "sklearn_version": sklearn.__version__,
                        "xgboost_version": xgboost.__version__,
                        "lightgbm_version": lightgbm.__version__,
                        "target_col": target_col,
                        "target_transform": target_transform
                    },
                    
                    # 2. Training timestamp & duration
                    "training_info": {
                        "trained_at": datetime.now(timezone.utc).isoformat(),
                        "total_training_time_seconds": round(total_training_time, 2),
                        "ray_version": ray.__version__,
                        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                    },
                    
                    # 3. Preprocessing configuration
                    "preprocessing_config": preprocessing_config or {},
                    
                    # 4. Fold date ranges and row counts
                    "fold_metadata": fold_metadata,
                    
                    # 5. Per-fold training metrics
                    "fold_training_metrics": fold_training_metrics,
                    
                    # 6. Feature importance (if available)
                    "feature_importance": feature_importance_data,
                    
                    # 7. Validation summary
                    "validation_summary": {
                        "overall_metrics": {
                            "avg_train_rmse": float(avg_metrics["train_rmse"]),
                            "avg_test_rmse": float(avg_metrics["test_rmse"]),
                            "avg_test_r2": float(avg_metrics["test_r2"]),
                            "avg_test_mae": float(avg_metrics["test_mae"]),
                            "best_fold": folds[best_fold_idx].fold_id if folds else None,
                            "worst_fold": folds[worst_fold_idx].fold_id if folds else None,
                            "num_folds": len(fold_metrics)
                        },
                        "cross_fold_consistency": {
                            "test_rmse_std": float(np.std(test_rmse_values)) if test_rmse_values else 0.0,
                            "test_rmse_min": float(np.min(test_rmse_values)) if test_rmse_values else 0.0,
                            "test_rmse_max": float(np.max(test_rmse_values)) if test_rmse_values else 0.0,
                            "test_r2_std": float(np.std(test_r2_values)) if test_r2_values else 0.0,
                            "test_r2_min": float(np.min(test_r2_values)) if test_r2_values else 0.0,
                            "test_r2_max": float(np.max(test_r2_values)) if test_r2_values else 0.0
                        }
                    }
                }
                
                # Save complete metadata
                metadata_file = tmppath / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(complete_metadata, f, indent=2)
                
                # Log to MLflow (if enabled)
                mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
                try:
                    mlflow_tracker = MLflowTracker(tracking_uri=mlflow_uri)
                    
                    # Get first fold's model and data for signature
                    first_model = next(iter(fold_models.values()))
                    first_features = next(iter(fold_feature_lists.values()))
                    
                    # Create sample X, y for signature (use first fold's test data)
                    first_fold = folds[0]
                    sample_df = first_fold.test_ds.to_pandas() if first_fold.test_ds else pd.DataFrame()
                    if not sample_df.empty:
                        X_sample = sample_df[first_features].fillna(0).replace([np.inf, -np.inf], 0).values[:100]
                        y_sample = sample_df[target_col].fillna(0).values[:100]
                        
                        # Build descriptive experiment name
                        primary_ticker = preprocessing_config.get('primary_ticker', 'unknown')
                        context_symbols = preprocessing_config.get('context_symbols', [])
                        num_features = len(first_features)
                        train_months = preprocessing_config.get('train_months', 3)
                        test_months = preprocessing_config.get('test_months', 1)
                        context_str = f"+{len(context_symbols)}ctx" if context_symbols else ""
                        experiment_name = f"wf_{algorithm}_{primary_ticker}{context_str}_f{num_features}_tr{train_months}m_te{test_months}m"
                        
                        # Log to MLflow
                        run_id = mlflow_tracker.log_training_run(
                            experiment_name=experiment_name,
                            model=first_model,
                            model_type=algorithm,
                            params=config,
                            metrics={
                                "avg_train_rmse": float(avg_metrics["train_rmse"]),
                                "avg_test_rmse": float(avg_metrics["test_rmse"]),
                                "avg_test_r2": float(avg_metrics["test_r2"]),
                                "avg_test_mae": float(avg_metrics["test_mae"]),
                                "num_folds": len(fold_metrics)
                            },
                            metadata=complete_metadata["model_info"] | complete_metadata["training_info"],
                            X_train=X_sample,
                            y_train=y_sample,
                            feature_names=first_features,
                            register_model=True
                        )
                        
                        # Calculate and log permutation importance
                        if len(X_sample) >= 10:  # Need enough samples
                            importance_df = mlflow_tracker.calculate_permutation_importance(
                                model=first_model,
                                X=X_sample,
                                y=y_sample,
                                feature_names=first_features,
                                n_repeats=5  # Keep low for speed
                            )
                            mlflow_tracker.log_permutation_importance(run_id, importance_df)
                        
                        # Run MLflow model evaluation (generates plots + comprehensive metrics)
                        if len(X_sample) >= 10:
                            mlflow_tracker.evaluate_model(
                                run_id=run_id,
                                model=first_model,
                                X_test=X_sample,
                                y_test=y_sample,
                                model_type="regressor"
                            )
                        
                        log.info(f"✅ Logged to MLflow: run_id={run_id}")
                except Exception as e:
                    log.warning(f"Failed to log to MLflow: {e}", exc_info=True)
                
                # Create checkpoint
                checkpoint = Checkpoint.from_directory(tmpdir)
                
                # Report to Ray Tune with checkpoint
                from ray import train as ray_train
                ray_train.report(avg_metrics, checkpoint=checkpoint)
            
            return avg_metrics
        
        return train_on_folds
    
    def _train_single_fold(
        self,
        fold: Fold,
        config: Dict,
        algorithm: str,
        target_col: str,
        feature_cols: Optional[List[str]],
        target_transform: str,
        excluded_features: Optional[List[str]] = None
    ) -> Dict:
        """Train and evaluate on a single fold."""
        
        # Load fold data
        train_df = fold.train_ds.to_pandas() if fold.train_ds else pd.DataFrame()
        test_df = fold.test_ds.to_pandas() if fold.test_ds else pd.DataFrame()
        
        # Capture row counts before any processing
        train_rows = len(train_df)
        test_rows = len(test_df)
        
        log.info(f"Fold {fold.fold_id}: train_df shape={train_df.shape}, test_df shape={test_df.shape}")
        
        if train_df.empty or test_df.empty:
            log.warning(f"Empty data in {fold}")
            return {"train_rmse": np.inf, "test_rmse": np.inf, "test_mae": np.inf, "test_r2": -np.inf}
        
        # Log data quality
        log.info(f"Fold {fold.fold_id} columns: {list(train_df.columns)}")
        log.info(f"Fold {fold.fold_id} dtypes:\n{train_df.dtypes}")
        log.info(f"Fold {fold.fold_id} sample:\n{train_df.head(3)}")
        
        # Prepare features and target
        if feature_cols is None:
            # Auto-select numeric features (exclude target and metadata)
            exclude_cols = {'ts', 'symbol', target_col, 'open', 'high', 'low', 'volume', 'vwap', 'trade_count'}
            feature_cols = [c for c in train_df.columns if c not in exclude_cols and train_df[c].dtype in [np.float64, np.int64]]
        
        # Filter out excluded features (for retraining workflow)
        if excluded_features:
            original_count = len(feature_cols)
            feature_cols = [c for c in feature_cols if c not in excluded_features]
            log.info(f"Fold {fold.fold_id} excluded {original_count - len(feature_cols)} features ({len(feature_cols)} remaining)")
        
        log.info(f"Fold {fold.fold_id} selected features ({len(feature_cols)}): {feature_cols[:10]}...")
        
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
        
        log.info(f"Fold {fold.fold_id} target created: train_y={len(train_y)}, test_y={len(test_y)}")
        log.info(f"Fold {fold.fold_id} train_y stats: min={train_y.min():.6f}, max={train_y.max():.6f}, mean={train_y.mean():.6f}")
        
        # Align features with target (drop NaN rows)
        train_X = train_df[feature_cols].loc[train_y.index].dropna()
        train_y = train_y.loc[train_X.index]
        
        test_X = test_df[feature_cols].loc[test_y.index].dropna()
        test_y = test_y.loc[test_X.index]
        
        log.info(f"Fold {fold.fold_id} after alignment: train_X={train_X.shape}, test_X={test_X.shape}")
        
        if len(train_X) == 0 or len(test_X) == 0:
            log.error(f"No valid data after preprocessing in {fold}")
            log.error(f"Original train_df had {len(train_df)} rows, train_y had {len(train_y)} rows")
            log.error(f"Feature NaN counts:\n{train_df[feature_cols].isna().sum()}")
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
        
        log.info(f"Fold {fold.fold_id} results: test_rmse={test_rmse:.6f}, test_r2={test_r2:.4f}")
        
        metrics = {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
        }
        
        # Return metrics, trained model, feature list, and row counts
        return metrics, model, feature_cols, train_rows, test_rows
    
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
                n_jobs=-1,  # Use all available CPU cores
                random_state=42
            )
        elif algorithm == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", 0.1),
                subsample=config.get("subsample", 0.8),
                colsample_bytree=config.get("colsample_bytree", 0.8),
                nthread=-1,  # Use all available CPU cores
                random_state=42
            )
        elif algorithm == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", 0.1),
                subsample=config.get("subsample", 0.8),
                colsample_bytree=config.get("colsample_bytree", 0.8),
                n_jobs=-1,  # Use all available CPU cores
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
        num_gpus: float = 0.0,
        actor_pool_size: Optional[int] = None,
        skip_empty_folds: bool = False,
        excluded_features: Optional[List[str]] = None,
        experiment_name: Optional[str] = None,
        disable_mlflow: bool = False,
        use_cached_folds: bool = True,
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
            skip_empty_folds: If True, skip empty folds with warnings; if False, fail on empty folds
            excluded_features: List of feature names to exclude from training (for retraining workflow)
            experiment_name: Custom experiment name (for retrain lineage tracking). If None, auto-generated.
            disable_mlflow: If True, skip MLflow logging (for isolated Ray jobs without network access)
            
        Returns:
            Ray Tune ResultGrid with all trial results
        """
        log.info("Starting walk-forward hyperparameter tuning")
        
        if excluded_features:
            log.info(f"Excluding {len(excluded_features)} features from training: {excluded_features[:5]}...")
        
        # Auto-detect CPU count if not specified
        import os
        try:
            cpu_count = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_count = os.cpu_count() or 4
        
        # CRITICAL: Reserve CPUs for training trials
        # Preprocessing uses half the CPUs, training uses the other half
        # This prevents deadlock where preprocessing actors block training trials
        if actor_pool_size is None:
            actor_pool_size = max(2, cpu_count // 2)  # Use half CPUs for preprocessing
        
        log.info(f"Using {actor_pool_size} parallel actors for preprocessing (CPUs available: {cpu_count})")
        log.info(f"Reserving {cpu_count - actor_pool_size} CPUs for training trials")
        
        # Step 1: Generate and load folds
        log.info("="*100)
        log.info("GENERATING WALK-FORWARD FOLDS")
        log.info(f"Configuration: {train_months} months train, {test_months} months test, {step_months} months step")
        log.info(f"Date Range: {start_date} to {end_date}")
        log.info(f"Cache Status: {'DISABLED (fresh calculation guaranteed)' if not use_cached_folds else 'ENABLED (may use cached data)'}")
        log.info("="*100)
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
            num_gpus=num_gpus,
            actor_pool_size=actor_pool_size,
            use_cached_folds=use_cached_folds
        ):
            self.folds.append(fold)
        
        log.info(f"Loaded {len(self.folds)} folds for training")
        
        # Log detailed fold splits
        log.info("="*80)
        log.info("TRAIN/TEST FOLD SPLITS")
        log.info("="*80)
        for fold in self.folds:
            train_rows = len(fold.X_train) if hasattr(fold, 'X_train') and fold.X_train is not None else 'unknown'
            test_rows = len(fold.X_test) if hasattr(fold, 'X_test') and fold.X_test is not None else 'unknown'
            log.info(f"Fold {fold.fold_id}:")
            log.info(f"  Train: {fold.train_start} to {fold.train_end} ({train_rows} rows)")
            log.info(f"  Test:  {fold.test_start} to {fold.test_end} ({test_rows} rows)")
        log.info("="*80)
        
        # CRITICAL: Force cleanup of preprocessing actors to free CPUs for training
        import gc
        gc.collect()  # Python garbage collection
        log.info("Preprocessing complete - forcing actor cleanup to free CPUs for training")
        
        # Validate that we have non-empty folds
        if not self.folds:
            raise ValueError(
                f"No folds generated! Check that data exists for date range {start_date} to {end_date}. "
                f"Use the /streaming/status endpoint to see available date ranges."
            )
        
        # PRE-VALIDATE ALL FOLDS: Fail-fast or skip based on skip_empty_folds parameter
        log.info(f"Validating all folds have sufficient data (skip_empty_folds={skip_empty_folds})...")
        validation_errors = []
        valid_folds = []
        skipped_folds = []
        
        for fold in self.folds:
            try:
                train_count = fold.train_ds.count()
                test_count = fold.test_ds.count()
                
                # Check for empty data
                fold_has_errors = False
                fold_error_msgs = []
                
                if train_count == 0:
                    msg = f"Fold {fold.fold_id} has ZERO training data (train: {fold.train_start} to {fold.train_end})"
                    fold_error_msgs.append(msg)
                    fold_has_errors = True
                elif train_count < 100:  # Minimum threshold for meaningful training
                    log.warning(
                        f"Fold {fold.fold_id} has only {train_count} training rows "
                        f"(train: {fold.train_start} to {fold.train_end}) - may not be enough"
                    )
                
                if test_count == 0:
                    msg = f"Fold {fold.fold_id} has ZERO test data (test: {fold.test_start} to {fold.test_end})"
                    fold_error_msgs.append(msg)
                    fold_has_errors = True
                
                # Handle empty fold based on skip_empty_folds flag
                if fold_has_errors:
                    if skip_empty_folds:
                        # Skip this fold with warning
                        for msg in fold_error_msgs:
                            log.warning(f"⚠️  SKIPPING: {msg}")
                        skipped_folds.append(fold)
                    else:
                        # Add to validation errors to fail later
                        validation_errors.extend(fold_error_msgs)
                else:
                    # Fold is valid
                    log.info(f"✓ Fold {fold.fold_id} validated: train={train_count:,} rows, test={test_count:,} rows")
                    valid_folds.append(fold)
                    
            except Exception as e:
                error_msg = f"Fold {fold.fold_id} validation failed: {e}"
                if skip_empty_folds:
                    log.warning(f"⚠️  SKIPPING: {error_msg}")
                    skipped_folds.append(fold)
                else:
                    validation_errors.append(error_msg)
        
        # Report skipped folds
        if skipped_folds:
            log.warning(f"⚠️  Skipped {len(skipped_folds)} empty folds (skip_empty_folds=True)")
            log.info(f"Continuing with {len(valid_folds)} valid folds")
        
        # Fail fast if any validation errors (when skip_empty_folds=False)
        if validation_errors:
            error_msg = "Fold validation failed:\\n" + "\\n".join(f"  - {err}" for err in validation_errors)
            error_msg += f"\\n\\nCheck parquet data exists for date range {start_date} to {end_date}"
            error_msg += "\\n\\nTo skip empty folds instead of failing, use skip_empty_folds=True"
            raise ValueError(error_msg)
        
        # Update folds list to only include valid folds when skipping
        if skip_empty_folds:
            self.folds = valid_folds
            if not self.folds:
                raise ValueError(
                    f"All {len(self.folds) + len(skipped_folds)} folds were empty! "
                    f"Check parquet data exists for date range {start_date} to {end_date}"
                )
        
        log.info(f"✓ Validated {len(self.folds)} folds for training")
        
        # Legacy empty fold check (should be caught above now)
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
        
        # Step 3: Create trainable with preprocessing config
        preprocessing_config = {
            "windows": windows,
            "resampling_timeframes": resampling_timeframes,
            "context_symbols": context_symbols,
            "actor_pool_size": actor_pool_size,
            "train_months": train_months,
            "test_months": test_months,
            "step_months": step_months,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "feature_engineering_version": self.preprocessor.feature_engineering_version,
            "excluded_features": excluded_features,
            "primary_ticker": symbols[0]  # For experiment naming
        }
        
        trainable = self.create_trainable(
            algorithm=algorithm,
            target_col="close",
            target_transform="log_return",
            preprocessing_config=preprocessing_config
        )
        
        # Step 4: Run tuning
        log.info(f"Running {num_samples} trials with {algorithm}")
        
        # Limit concurrent trials to prevent resource exhaustion
        # Reserve some CPUs for Ray overhead and cleanup
        max_concurrent = max(1, cpu_count - 1)  # Leave 1 CPU for Ray system
        log.info(f"Running up to {max_concurrent} concurrent trials (reserving 1 CPU for Ray)")
        
        # Build descriptive experiment name
        primary_symbol = symbols[0]
        context_str = f"+{len(context_symbols)}ctx" if context_symbols else ""
        # Use provided experiment_name or auto-generate one
        if experiment_name is None:
            experiment_name = f"wf_{algorithm}_{primary_symbol}{context_str}_pbt"
        
        log.info(f"Experiment name: {experiment_name}")
        
        # Setup MLflow callback for automatic trial logging
        callbacks = []
        if not disable_mlflow:
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            try:
                mlflow_callback = MLflowLoggerCallback(
                    tracking_uri=mlflow_uri,
                    experiment_name=experiment_name,
                    save_artifact=True,  # Save model artifacts to MLflow
                    tags={"algorithm": algorithm, "ticker": primary_symbol, "context_symbols": str(context_symbols)}
                )
                callbacks.append(mlflow_callback)
                log.info(f"✅ MLflow callback configured: {mlflow_uri}, experiment: {experiment_name}")
            except Exception as e:
                log.warning(f"⚠️ Failed to setup MLflow callback: {e}. Continuing without MLflow logging.")
        else:
            log.info("MLflow logging disabled for this training run")
        
        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric="test_rmse",
                mode="min",
                num_samples=num_samples,
                max_concurrent_trials=max_concurrent,  # Prevent deadlock by limiting concurrency
            ),
            run_config=ray.train.RunConfig(
                name=experiment_name,
                storage_path=str(settings.data.checkpoints_dir),
                callbacks=callbacks,  # Add MLflow callback
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
