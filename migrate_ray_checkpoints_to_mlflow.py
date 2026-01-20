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


def checkpoint_already_migrated(checkpoint_path: Path, client: mlflow.tracking.MlflowClient = None) -> bool:
    """
    Check if checkpoint has already been migrated to MLflow.
    
    Uses two methods:
    1. Check for marker file (fast)
    2. Check MLflow for existing run with same checkpoint path (definitive)
    """
    # Method 1: Quick check via marker file
    marker_file = checkpoint_path / ".mlflow_migrated"
    if marker_file.exists():
        return True
    
    # Method 2: Query MLflow for runs with this checkpoint path
    if client:
        try:
            # Search for runs that have this checkpoint path as a tag
            checkpoint_str = str(checkpoint_path)
            all_experiments = client.search_experiments()
            
            for exp in all_experiments:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"tags.original_checkpoint = '{checkpoint_str}'"
                )
                if runs:
                    log.debug(f"Found existing MLflow run for {checkpoint_path.parent.name}")
                    # Create marker file so we don't query again
                    mark_checkpoint_migrated(checkpoint_path, runs[0].info.run_id)
                    return True
        except Exception as e:
            log.debug(f"Could not query MLflow for duplicates: {e}")
    
    return False


def mark_checkpoint_migrated(checkpoint_path: Path, run_id: str = None):
    """Mark checkpoint as migrated to avoid duplicate imports."""
    if run_id is None:
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else 'unknown'
    
    marker_file = checkpoint_path / ".mlflow_migrated"
    marker_file.write_text(f"Migrated at: {run_id}")


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
            
            # Log checkpoint artifacts (feature importance, metadata, etc.)
            log.info(f"  📊 Logging checkpoint artifacts...")
            
            # 1. Log feature importance if available
            feature_importance_file = checkpoint_path / "feature_importance.json"
            if feature_importance_file.exists():
                mlflow.log_artifact(str(feature_importance_file), "feature_analysis")
                log.info(f"    ✓ Logged feature_importance.json")
            else:
                log.warning(f"    ⚠️  feature_importance.json not found")
            
            # 2. Log feature lists (features used per fold)
            if features_file.exists():
                mlflow.log_artifact(str(features_file), "feature_analysis")
                log.info(f"    ✓ Logged feature_lists.json")
            
            # 3. Log full metadata (fold-by-fold metrics)
            metadata_file = checkpoint_path / "metadata.json"
            if metadata_file.exists():
                mlflow.log_artifact(str(metadata_file), "training_details")
                log.info(f"    ✓ Logged metadata.json")
            
            # 4. Log top features as parameters for quick filtering
            feature_importance_path = checkpoint_path / "feature_importance.json"
            if feature_importance_path.exists():
                try:
                    with open(feature_importance_path, "r") as f:
                        importance_data = json.load(f)
                    
                    top_features = importance_data.get("all_features", [])[:10]
                    for i, feat in enumerate(top_features, 1):
                        mlflow.log_param(f"top_{i}_feature", feat.get("name", "unknown"))
                        mlflow.log_metric(f"top_{i}_importance", feat.get("importance", 0.0))
                    
                    # Log context feature counts
                    summary = importance_data.get("summary", {})
                    if summary:
                        mlflow.log_param("context_features_in_top15", summary.get("context_features_in_top15", 0))
                        mlflow.log_param("context_features_in_top50", summary.get("context_features_in_top50", 0))
                        mlflow.log_param("total_context_features", summary.get("total_context_features", 0))
                    
                    log.info(f"    ✓ Logged top 10 features as params")
                except Exception as e:
                    log.warning(f"    ⚠️  Could not parse feature importance: {e}")
            
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
            
            # Get the run ID for verification
            run_id = mlflow.active_run().info.run_id
            
            log.info(f"✅ Migrated {checkpoint_path.parent.name} to MLflow experiment: {experiment_name}")
        
        # Verify model was registered (outside run context)
        log.info(f"🔍 Starting verification for {experiment_name}...")
        verify_model_registration(experiment_name, run_id, checkpoint_path)
        
        # Mark as migrated
        mark_checkpoint_migrated(checkpoint_path)
            
    except Exception as e:
        log.error(f"Failed to migrate {checkpoint_path}: {e}", exc_info=True)


def verify_model_registration(model_name: str, run_id: str, checkpoint_path: Path):
    """
    Verify that a model was successfully registered in MLflow.
    
    Args:
        model_name: Name of the registered model
        run_id: MLflow run ID
        checkpoint_path: Original checkpoint path for logging
    """
    log.info(f"🔍 VERIFY START: model_name='{model_name}', run_id={run_id[:8]}...")
    
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Wait a moment for registration to complete
        import time
        time.sleep(2)
        
        # First, list ALL registered models to see what's there
        all_registered = list(client.search_registered_models())
        log.info(f"   📊 Total registered model names in MLflow: {len(all_registered)}")
        if all_registered:
            log.info(f"   📋 Model names: {[rm.name for rm in all_registered[:5]]}")
        
        # Search for this specific registered model
        log.info(f"   🔎 Searching for model name: '{model_name}'")
        registered_models = list(client.search_registered_models(f"name='{model_name}'"))
        
        if not registered_models:
            log.error(f"❌ VERIFICATION FAILED: Model '{model_name}' not found in registry!")
            log.error(f"   Checkpoint: {checkpoint_path}")
            log.error(f"   Run ID: {run_id}")
            log.error(f"   Available models: {[rm.name for rm in all_registered]}")
            return False
        
        log.info(f"   ✓ Found registered model: {model_name}")
        
        # Find version associated with this run
        log.info(f"   🔎 Searching for version with run_id={run_id[:8]}...")
        versions = list(client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'"))
        
        if not versions:
            log.error(f"❌ VERIFICATION FAILED: No version found for run {run_id}")
            log.error(f"   Model: {model_name}")
            log.error(f"   Checkpoint: {checkpoint_path}")
            # List all versions for this model
            all_versions = list(client.search_model_versions(f"name='{model_name}'"))
            log.error(f"   Total versions for this model: {len(all_versions)}")
            return False
        
        version = versions[0]
        log.info(f"✅ VERIFIED: Model '{model_name}' v{version.version} registered successfully")
        log.info(f"   Stage: {version.current_stage}")
        log.info(f"   Run ID: {run_id[:8]}...")
        
        return True
        
    except Exception as e:
        log.error(f"❌ VERIFICATION ERROR for '{model_name}': {e}", exc_info=True)
        return False


def validate_mlflow_setup(mlflow_uri: str):
    """
    Validate MLflow configuration and database accessibility.
    
    Checks:
    - MLflow tracking URI is reachable
    - Backend database path is correct (4 slashes for absolute paths)
    - Database file exists and is writable
    - Can query registered models
    """
    log.info(f"🔍 Validating MLflow setup...")
    log.info(f"   Tracking URI: {mlflow_uri}")
    
    # Check if it's an HTTP URI
    if mlflow_uri.startswith("http"):
        try:
            import requests
            health_url = f"{mlflow_uri.rstrip('/')}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                log.info(f"   ✅ MLflow server is reachable at {mlflow_uri}")
            else:
                log.error(f"   ❌ MLflow server returned status {response.status_code}")
                return False
        except Exception as e:
            log.error(f"   ❌ Cannot reach MLflow server: {e}")
            return False
    
    # Check SQLite path format if direct database access
    elif mlflow_uri.startswith("sqlite"):
        # Validate path format
        if not mlflow_uri.startswith("sqlite:////"):
            log.error(f"   ❌ SQLite path should use 4 slashes for absolute paths!")
            log.error(f"      Current: {mlflow_uri}")
            log.error(f"      Should be: sqlite:////absolute/path/to/db.db")
            return False
        
        # Extract database path (remove sqlite://)
        db_path = mlflow_uri.replace("sqlite:///", "")
        db_file = Path(db_path)
        
        log.info(f"   Database path: {db_file}")
        
        if not db_file.parent.exists():
            log.error(f"   ❌ Database directory does not exist: {db_file.parent}")
            return False
        
        # Check if database exists or can be created
        if db_file.exists():
            log.info(f"   ✅ Database file exists ({db_file.stat().st_size / 1024:.1f} KB)")
            if not os.access(db_file, os.R_OK | os.W_OK):
                log.error(f"   ❌ Database file is not readable/writable!")
                return False
        else:
            log.info(f"   ℹ️  Database will be created at {db_file}")
    
    # Test MLflow client connection
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(mlflow_uri)
        
        # Try to list registered models
        models = list(client.search_registered_models())
        log.info(f"   ✅ MLflow client connected successfully")
        log.info(f"   📊 Currently {len(models)} registered model(s) in database")
        
        # Check backend store URI environment variable
        backend_uri = os.getenv("MLFLOW_BACKEND_STORE_URI")
        if backend_uri:
            log.info(f"   ℹ️  MLFLOW_BACKEND_STORE_URI env: {backend_uri}")
            if backend_uri.startswith("sqlite:///") and not backend_uri.startswith("sqlite:////"):
                log.warning(f"   ⚠️  Environment variable uses relative path (3 slashes)!")
                log.warning(f"      This may cause issues. Use 4 slashes for absolute paths.")
        
        return True
        
    except Exception as e:
        log.error(f"   ❌ Failed to connect to MLflow: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main migration entry point."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    log.info(f"Starting Ray checkpoint migration to MLflow ({mlflow_uri})...")
    
    # Validate MLflow setup before migration
    if not validate_mlflow_setup(mlflow_uri):
        log.error("❌ MLflow validation failed! Aborting migration.")
        log.error("Please check:")
        log.error("  1. MLflow server is running")
        log.error("  2. MLFLOW_BACKEND_STORE_URI uses 4 slashes (sqlite:////path)")
        log.error("  3. Database path exists and is writable")
        return
    
    log.info("")
    
    # Find all checkpoints
    checkpoints = find_ray_checkpoints()
    
    if not checkpoints:
        log.info("No Ray checkpoints found to migrate.")
        return
    
    # Create MLflow client for duplicate checking
    client = mlflow.tracking.MlflowClient(mlflow_uri)
    
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
