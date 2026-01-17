#!/usr/bin/env python3
"""
Migrate existing Ray checkpoints to MLflow Model Registry.

Scans /app/data/ray_checkpoints/ for completed training runs and
registers them in MLflow so they appear in the Model Registry.

This script is idempotent - safe to run multiple times.
"""

import json
import os
import joblib  # Use joblib instead of pickle
from pathlib import Path
import logging

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def find_ray_checkpoints(checkpoints_dir: str = "/app/data/ray_checkpoints") -> list:
    """Find all Ray checkpoints with metadata.json."""
    checkpoints = []
    checkpoint_path = Path(checkpoints_dir)
    
    if not checkpoint_path.exists():
        log.warning(f"Checkpoints directory not found: {checkpoints_dir}")
        return []
    
    # Scan for experiment directories
    for exp_dir in checkpoint_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "backtest_results":
            continue
        
        # Scan for trial directories
        for trial_dir in exp_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            
            # Look for checkpoint subdirectories
            for ckpt_dir in trial_dir.iterdir():
                if not ckpt_dir.is_dir() or not ckpt_dir.name.startswith("checkpoint_"):
                    continue
                
                metadata_file = ckpt_dir / "metadata.json"
                if metadata_file.exists():
                    checkpoints.append(ckpt_dir)
    
    log.info(f"Found {len(checkpoints)} Ray checkpoints")
    return checkpoints


def checkpoint_already_migrated(checkpoint_path: Path) -> bool:
    """Check if checkpoint has already been migrated to MLflow."""
    marker_file = checkpoint_path / ".mlflow_migrated"
    return marker_file.exists()


def mark_checkpoint_migrated(checkpoint_path: Path):
    """Mark checkpoint as migrated to avoid duplicate imports."""
    marker_file = checkpoint_path / ".mlflow_migrated"
    marker_file.write_text(f"Migrated at: {mlflow.active_run().info.run_id if mlflow.active_run() else 'unknown'}")


def migrate_checkpoint_to_mlflow(checkpoint_path: Path):
    """Migrate a single Ray checkpoint to MLflow."""
    try:
        # Load metadata
        with open(checkpoint_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Extract info
        model_info = metadata.get("model_info", {})
        training_info = metadata.get("training_info", {})
        validation_summary = metadata.get("validation_summary", {})
        
        algorithm = model_info.get("algorithm", "unknown")
        ticker = training_info.get("primary_ticker", "unknown")
        experiment_name = f"walk_forward_{algorithm}_{ticker}"
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Load model (first fold)
        model_files = list(checkpoint_path.glob("fold_*_model.joblib"))
        if not model_files:
            log.warning(f"No model files found in {checkpoint_path}")
            return
        
        model = joblib.load(model_files[0])
        
        # Load feature list
        features_file = checkpoint_path / "feature_lists.json"
        if features_file.exists():
            with open(features_file, "r") as f:
                feature_lists = json.load(f)
                first_fold_features = list(feature_lists.values())[0] if feature_lists else []
        else:
            first_fold_features = []
        
        # Load config
        config_file = checkpoint_path / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            config = {}
        
        # Create MLflow run
        with mlflow.start_run(run_name=f"migrated_{checkpoint_path.parent.name}"):
            # Log parameters
            mlflow.log_params(config)
            mlflow.log_param("migrated", "true")
            mlflow.log_param("original_checkpoint", str(checkpoint_path))
            
            # Log metrics
            overall_metrics = validation_summary.get("overall_metrics", {})
            mlflow.log_metrics({
                "avg_train_rmse": overall_metrics.get("avg_train_rmse", 0.0),
                "avg_test_rmse": overall_metrics.get("avg_test_rmse", 0.0),
                "avg_test_r2": overall_metrics.get("avg_test_r2", 0.0),
                "avg_test_mae": overall_metrics.get("avg_test_mae", 0.0),
                "num_folds": overall_metrics.get("num_folds", 0)
            })
            
            # Log tags
            mlflow.set_tag("algorithm", algorithm)
            mlflow.set_tag("ticker", ticker)
            mlflow.set_tag("feature_engineering_version", model_info.get("feature_engineering_version", "unknown"))
            mlflow.set_tag("migrated_from_ray", "true")
            
            # Log model
            if algorithm in ["elasticnet", "ridge", "lasso", "randomforest"]:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=experiment_name
                )
            elif algorithm == "xgboost":
                mlflow.xgboost.log_model(
                    model,
                    "model",
                    registered_model_name=experiment_name
                )
            elif algorithm == "lightgbm":
                mlflow.lightgbm.log_model(
                    model,
                    "model",
                    registered_model_name=experiment_name
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=experiment_name
                )
            
            log.info(f"✅ Migrated {checkpoint_path.parent.name} to MLflow experiment: {experiment_name}")
            
            # Mark as migrated
            mark_checkpoint_migrated(checkpoint_path)
            
    except Exception as e:
        log.error(f"Failed to migrate {checkpoint_path}: {e}", exc_info=True)


def main():
    """Main migration entry point."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    log.info(f"Starting Ray checkpoint migration to MLflow ({mlflow_uri})...")
    
    # Find all checkpoints
    checkpoints = find_ray_checkpoints()
    
    if not checkpoints:
        log.info("No checkpoints to migrate")
        return
    
    # Migrate each checkpoint
    migrated = 0
    skipped = 0
    
    for checkpoint_path in checkpoints:
        if checkpoint_already_migrated(checkpoint_path):
            log.debug(f"Skipping already migrated: {checkpoint_path.parent.name}")
            skipped += 1
            continue
        
        migrate_checkpoint_to_mlflow(checkpoint_path)
        migrated += 1
    
    log.info(f"\n{'='*60}")
    log.info(f"Migration complete!")
    log.info(f"  Migrated: {migrated}")
    log.info(f"  Skipped (already migrated): {skipped}")
    log.info(f"  Total checkpoints: {len(checkpoints)}")
    log.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
