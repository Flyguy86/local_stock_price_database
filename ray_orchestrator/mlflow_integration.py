"""
MLflow integration for model tracking and registry.

Logs models, metrics, and parameters to MLflow for experiment tracking.
"""

import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

log = logging.getLogger(__name__)


class MLflowTracker:
    """Wrapper for MLflow experiment tracking and model registry."""
    
    def __init__(self, tracking_uri: str = "http://mlflow:5000"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        log.info(f"MLflow tracking URI set to: {tracking_uri}")
    
    def log_training_run(
        self,
        experiment_name: str,
        model: Any,
        model_type: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        metadata: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list,
        artifact_path: Optional[str] = None,
        register_model: bool = True
    ) -> str:
        """
        Log a complete training run to MLflow.
        
        Args:
            experiment_name: Name of the experiment
            model: Trained model instance
            model_type: Type of model (elasticnet, xgboost, etc.)
            params: Hyperparameters
            metrics: Training/validation metrics
            metadata: Additional metadata (fold info, feature version, etc.)
            X_train: Training features for signature inference
            y_train: Training targets for signature inference
            feature_names: List of feature names
            artifact_path: Optional path to additional artifacts
            register_model: Whether to register model in MLflow Model Registry
            
        Returns:
            run_id: MLflow run ID
        """
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            # Log hyperparameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log metadata as tags
            for key, value in metadata.items():
                mlflow.set_tag(key, str(value))
            
            # Log feature names
            mlflow.set_tag("feature_count", len(feature_names))
            mlflow.set_tag("feature_engineering_version", metadata.get("feature_engineering_version", "unknown"))
            
            # Infer model signature
            signature = infer_signature(X_train, y_train)
            
            # Log model based on type
            if model_type in ["elasticnet", "ridge", "lasso", "randomforest"]:
                model_info = mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=experiment_name if register_model else None
                )
            elif model_type == "xgboost":
                model_info = mlflow.xgboost.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=experiment_name if register_model else None
                )
            elif model_type == "lightgbm":
                model_info = mlflow.lightgbm.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=experiment_name if register_model else None
                )
            else:
                # Fallback to generic sklearn
                model_info = mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=experiment_name if register_model else None
                )
            
            # Log feature names as artifact (using log_dict instead of log_table for compatibility)
            try:
                feature_dict = {"features": feature_names}
                mlflow.log_dict(feature_dict, "features.json")
            except Exception as e:
                log.warning(f"Could not log features table: {e}")
            
            # Log additional artifacts if provided
            if artifact_path and Path(artifact_path).exists():
                mlflow.log_artifacts(artifact_path, artifact_path="additional")
            
            log.info(f"MLflow run logged: {run.info.run_id}")
            log.info(f"Model URI: {model_info.model_uri}")
            
            return run.info.run_id
    
    def calculate_permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Calculate permutation importance as a "polygraph test" for features.
        
        This shuffles each feature and measures the drop in model performance.
        Features that cause large drops are truly important.
        
        Args:
            model: Trained model
            X: Features (validation set)
            y: Targets (validation set)
            feature_names: List of feature names
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            
        Returns:
            DataFrame with importances sorted by mean decrease
        """
        log.info(f"Calculating permutation importance with {n_repeats} repeats...")
        
        perm_importance = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='neg_mean_squared_error'
        )
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        
        # Sort by mean importance (descending)
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        # Add interpretation
        importance_df['interpretation'] = importance_df['importance_mean'].apply(
            lambda x: '🔴 Critical' if x > 0.001 else
                     '🟡 Moderate' if x > 0.0001 else
                     '🟢 Minimal' if x > 0 else
                     '⚪ No impact'
        )
        
        log.info(f"Top 5 features by permutation importance:")
        for idx, row in importance_df.head(5).iterrows():
            log.info(f"  {row['rank']}. {row['feature']}: {row['importance_mean']:.6f} ± {row['importance_std']:.6f} ({row['interpretation']})")
        
        return importance_df
    
    def log_permutation_importance(
        self,
        run_id: str,
        importance_df: pd.DataFrame
    ):
        """
        Log permutation importance to an existing MLflow run.
        
        Args:
            run_id: MLflow run ID
            importance_df: DataFrame from calculate_permutation_importance()
        """
        with mlflow.start_run(run_id=run_id):
            # Log as dict artifact (using log_dict for compatibility)
            try:
                importance_dict = importance_df.to_dict(orient='records')
                mlflow.log_dict({"permutation_importance": importance_dict}, "permutation_importance.json")
            except Exception as e:
                log.warning(f"Could not log permutation importance table: {e}")
            
            # Log top features as metrics
            for idx, row in importance_df.head(10).iterrows():
                mlflow.log_metric(f"perm_importance_{row['feature']}", row['importance_mean'])
            
            log.info(f"Permutation importance logged to run {run_id}")
    
    def evaluate_model(
        self,
        run_id: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = "regressor"
    ):
        """
        Run MLflow model evaluation to generate comprehensive performance metrics.
        
        Uses mlflow.evaluate() to automatically compute metrics and generate plots:
        - Regression: RMSE, MAE, R², residuals plot, predicted vs actual
        - Additional: Feature importance, error distribution
        
        Args:
            run_id: MLflow run ID to log evaluation to
            model: Trained model instance
            X_test: Test features
            y_test: Test targets
            model_type: "regressor" or "classifier"
        """
        log.info(f"🔬 Running MLflow model evaluation...")
        
        try:
            with mlflow.start_run(run_id=run_id):
                # Create evaluation dataset
                eval_data = pd.DataFrame(X_test)
                eval_data['target'] = y_test
                
                # Define evaluators based on model type
                if model_type == "regressor":
                    evaluators = "default"  # RMSE, MAE, R², plots
                else:
                    evaluators = "default"  # Accuracy, precision, recall, etc.
                
                # Run evaluation (logs metrics + artifacts automatically)
                eval_result = mlflow.evaluate(
                    model=model,
                    data=eval_data,
                    targets='target',
                    model_type=model_type,
                    evaluators=evaluators,
                    evaluator_config={
                        "log_model_explainability": False,  # Skip SHAP for speed
                        "metric_prefix": "eval_"  # Prefix metrics to distinguish from training
                    }
                )
                
                log.info(f"✅ MLflow evaluation complete:")
                log.info(f"   📊 Metrics logged: {list(eval_result.metrics.keys())}")
                log.info(f"   📁 Artifacts logged: {list(eval_result.artifacts.keys())}")
                
                # Log custom time-series specific metrics
                y_pred = model.predict(X_test)
                residuals = y_test - y_pred
                
                # Additional metrics for financial time series
                mlflow.log_metrics({
                    "eval_mean_residual": float(np.mean(residuals)),
                    "eval_std_residual": float(np.std(residuals)),
                    "eval_max_error": float(np.max(np.abs(residuals))),
                    "eval_mape": float(np.mean(np.abs(residuals / (y_test + 1e-9))) * 100),  # Mean Absolute Percentage Error
                })
                
                log.info(f"   ✓ Custom time-series metrics logged")
                
        except Exception as e:
            log.warning(f"⚠️  MLflow evaluation failed: {e}")
            log.debug(f"Evaluation error details:", exc_info=True)
    
    def get_registered_models(self) -> list:
        """
        Get list of all registered models.
        
        Returns:
            List of model metadata for ALL versions
        """
        try:
            log.info(f"🔍 get_registered_models() - tracking_uri: {self.tracking_uri}")
            
            # Use REST API directly since search_registered_models() has issues
            import requests
            response = requests.post(
                f"{self.tracking_uri}/api/2.0/mlflow/registered-models/search",
                json={"max_results": 1000}
            )
            response.raise_for_status()
            data = response.json()
            
            registered_models_data = data.get("registered_models", [])
            log.info(f"🔍 Found {len(registered_models_data)} registered model names via REST API")
            
            if not registered_models_data:
                log.warning(f"⚠️ No registered models found in MLflow!")
                return []
            
            models = []
            client = mlflow.tracking.MlflowClient(self.tracking_uri)
            
            for rm_data in registered_models_data:
                model_name = rm_data["name"]
                log.info(f"🔍 Processing model: {model_name}")
                
                # Get ALL versions for this model
                all_versions = client.search_model_versions(f"name='{model_name}'")
                version_list = list(all_versions)
                log.info(f"🔍   Found {len(version_list)} versions for {model_name}")
                
                for version in version_list:
                    models.append({
                        "name": version.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "run_id": version.run_id,
                        "description": rm_data.get("description", ""),
                        "created_at": version.creation_timestamp
                    })
            
            log.info(f"🔍 Returning {len(models)} total model versions")
            return models
        except Exception as e:
            log.error(f"❌ Error in get_registered_models(): {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return []
    
    def load_model(self, model_name: str, version: Optional[str] = None, stage: Optional[str] = None):
        """
        Load a model from MLflow Model Registry.
        
        Args:
            model_name: Registered model name
            version: Model version (e.g., "1", "2")
            stage: Model stage ("None", "Staging", "Production")
            
        Returns:
            Loaded model
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        log.info(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        return model
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ):
        """
        Transition a model to a new stage (None, Staging, Production, Archived).
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage
        """
        client = mlflow.tracking.MlflowClient(self.tracking_uri)
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        log.info(f"Transitioned {model_name} v{version} to {stage}")
