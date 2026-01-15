"""
Ray Serve Ensemble Deployment for Production Inference.

Deploys multiple trained models as a voting ensemble that:
- Runs inference in parallel across all models
- Aggregates predictions using hard or soft voting
- Provides real-time trading signals
- Supports dynamic model swapping without downtime
"""

import logging
from typing import Optional, Any
from pathlib import Path
import asyncio

import numpy as np
import pandas as pd
import joblib

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

from .config import settings

log = logging.getLogger("ray_orchestrator.ensemble")


@serve.deployment(
    name="TradingModel",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5},
)
class TradingModel:
    """
    Individual trading model deployment.
    
    Loads a trained model checkpoint and provides inference.
    Can be dynamically scaled and updated.
    """
    
    def __init__(self, model_path: str, model_id: str = "unknown"):
        """
        Initialize with a trained model.
        
        Args:
            model_path: Path to .joblib model file
            model_id: Unique identifier for this model
        """
        self.model_id = model_id
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            log.info(f"Loaded model {self.model_id} from {self.model_path}")
        except Exception as e:
            log.error(f"Failed to load model {self.model_id}: {e}")
            self.model = None
    
    def reload(self, new_path: Optional[str] = None):
        """
        Reload model (for hot-swapping).
        
        Args:
            new_path: Optional new model path
        """
        if new_path:
            self.model_path = new_path
        self._load_model()
        return {"status": "reloaded", "model_id": self.model_id}
    
    async def predict(self, features: dict) -> dict:
        """
        Generate prediction for input features.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dict with prediction and confidence
        """
        if self.model is None:
            return {
                "model_id": self.model_id,
                "error": "Model not loaded",
                "prediction": None,
                "confidence": 0.0
            }
        
        try:
            # Convert dict to DataFrame
            df = pd.DataFrame([features])
            
            # Get prediction
            pred = self.model.predict(df)
            
            # Get probability if available
            confidence = 0.5
            if hasattr(self.model, "predict_proba"):
                try:
                    proba = self.model.predict_proba(df)
                    confidence = float(np.max(proba))
                except:
                    pass
            elif hasattr(self.model, "named_steps"):
                # For pipelines, try to get from the model step
                model_step = self.model.named_steps.get("model")
                if hasattr(model_step, "predict_proba"):
                    try:
                        proba = model_step.predict_proba(
                            self.model.named_steps["preprocessor"].transform(df)
                        )
                        confidence = float(np.max(proba))
                    except:
                        pass
            
            return {
                "model_id": self.model_id,
                "prediction": float(pred[0]),
                "confidence": confidence,
                "error": None
            }
            
        except Exception as e:
            log.error(f"Prediction error in {self.model_id}: {e}")
            return {
                "model_id": self.model_id,
                "prediction": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_info(self) -> dict:
        """Get model information."""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "loaded": self.model is not None,
            "type": type(self.model).__name__ if self.model else None
        }


@serve.deployment(
    name="VotingEnsemble",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
)
class VotingEnsemble:
    """
    Ensemble controller that aggregates predictions from multiple models.
    
    Supports:
    - Hard voting (majority rule)
    - Soft voting (probability weighted average)
    - Dynamic model management (add/remove models at runtime)
    """
    
    def __init__(
        self,
        model_handles: list[DeploymentHandle],
        voting: str = "soft",
        threshold: float = 0.7
    ):
        """
        Initialize ensemble with model handles.
        
        Args:
            model_handles: List of TradingModel deployment handles
            voting: "hard" or "soft" voting strategy
            threshold: Confidence threshold for soft voting
        """
        self.model_handles = model_handles
        self.voting = voting
        self.threshold = threshold
        self.model_weights = {i: 1.0 for i in range(len(model_handles))}
        
        log.info(f"Ensemble initialized with {len(model_handles)} models, voting={voting}")
    
    async def __call__(self, request: dict) -> dict:
        """
        Generate ensemble prediction.
        
        Args:
            request: Dict with "features" key containing feature values
            
        Returns:
            Ensemble prediction with signal and confidence
        """
        features = request.get("features", request)
        
        # Query all models in parallel
        prediction_tasks = [
            handle.predict.remote(features)
            for handle in self.model_handles
        ]
        
        try:
            predictions = await asyncio.gather(*[
                asyncio.wrap_future(task.future())
                for task in prediction_tasks
            ])
        except Exception as e:
            log.error(f"Ensemble prediction failed: {e}")
            return {
                "signal": "ERROR",
                "confidence": 0.0,
                "error": str(e),
                "individual_predictions": []
            }
        
        # Filter out failed predictions
        valid_predictions = [p for p in predictions if p.get("prediction") is not None]
        
        if not valid_predictions:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "No valid predictions",
                "individual_predictions": predictions
            }
        
        # Aggregate based on voting strategy
        if self.voting == "hard":
            signal, confidence = self._hard_vote(valid_predictions)
        else:
            signal, confidence = self._soft_vote(valid_predictions)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "num_models": len(valid_predictions),
            "individual_predictions": predictions,
            "voting_method": self.voting,
            "threshold": self.threshold
        }
    
    def _hard_vote(self, predictions: list[dict]) -> tuple[str, float]:
        """
        Hard voting: majority rule.
        
        Each model votes BUY (pred > 0) or SELL (pred <= 0).
        Majority wins.
        """
        votes = []
        for p in predictions:
            pred = p.get("prediction", 0)
            votes.append(1 if pred > 0 else 0)
        
        buy_votes = sum(votes)
        total = len(votes)
        
        buy_ratio = buy_votes / total if total > 0 else 0.5
        
        if buy_ratio > 0.5:
            return "BUY", buy_ratio
        elif buy_ratio < 0.5:
            return "SELL", 1 - buy_ratio
        else:
            return "HOLD", 0.5
    
    def _soft_vote(self, predictions: list[dict]) -> tuple[str, float]:
        """
        Soft voting: weighted average of predictions.
        
        Uses confidence-weighted average. Only acts if above threshold.
        """
        weighted_sum = 0.0
        weight_total = 0.0
        
        for i, p in enumerate(predictions):
            pred = p.get("prediction", 0)
            conf = p.get("confidence", 0.5)
            weight = self.model_weights.get(i, 1.0) * conf
            
            weighted_sum += pred * weight
            weight_total += weight
        
        avg_pred = weighted_sum / weight_total if weight_total > 0 else 0
        
        # Calculate consensus confidence
        preds = [p.get("prediction", 0) for p in predictions]
        agreement = 1 - np.std(preds) / (abs(np.mean(preds)) + 1e-9) if preds else 0
        confidence = min(1.0, max(0.0, agreement))
        
        # Determine signal
        if avg_pred > 0 and confidence >= self.threshold:
            return "BUY", confidence
        elif avg_pred < 0 and confidence >= self.threshold:
            return "SELL", confidence
        else:
            return "HOLD", confidence
    
    def update_weights(self, weights: dict[int, float]):
        """Update model weights for soft voting."""
        self.model_weights.update(weights)
        log.info(f"Updated model weights: {self.model_weights}")
    
    def set_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.threshold = threshold
        log.info(f"Updated threshold: {self.threshold}")
    
    def get_status(self) -> dict:
        """Get ensemble status."""
        return {
            "num_models": len(self.model_handles),
            "voting": self.voting,
            "threshold": self.threshold,
            "model_weights": self.model_weights
        }


class EnsembleManager:
    """
    Manager for deploying and managing model ensembles.
    
    Handles:
    - Deploying new ensembles from checkpoints
    - Adding/removing models at runtime
    - Monitoring ensemble health
    """
    
    def __init__(self):
        self.ensembles = {}
        self.models = {}
    
    def init_ray_serve(self):
        """Initialize Ray Serve."""
        if not ray.is_initialized():
            ray.init(
                address=settings.ray.ray_address if settings.ray.ray_address != "auto" else None,
                namespace=settings.ray.ray_namespace,
                ignore_reinit_error=True
            )
        
        serve.start(
            http_options={"host": settings.serve.serve_host, "port": settings.serve.serve_port}
        )
        log.info(f"Ray Serve started on {settings.serve.serve_host}:{settings.serve.serve_port}")
    
    def deploy_ensemble(
        self,
        ensemble_name: str,
        model_paths: list[str],
        model_ids: Optional[list[str]] = None,
        voting: str = "soft",
        threshold: float = 0.7
    ) -> dict:
        """
        Deploy a new voting ensemble.
        
        Args:
            ensemble_name: Unique name for the ensemble
            model_paths: List of paths to model checkpoints
            model_ids: Optional list of model IDs (defaults to indices)
            voting: "hard" or "soft" voting
            threshold: Confidence threshold for soft voting
            
        Returns:
            Deployment info
        """
        self.init_ray_serve()
        
        if model_ids is None:
            model_ids = [f"model_{i}" for i in range(len(model_paths))]
        
        # Deploy individual models
        model_handles = []
        for path, model_id in zip(model_paths, model_ids):
            model_app = TradingModel.bind(model_path=path, model_id=model_id)
            handle = serve.run(model_app, name=f"{ensemble_name}_{model_id}")
            model_handles.append(handle)
            self.models[f"{ensemble_name}_{model_id}"] = handle
        
        # Deploy ensemble
        ensemble_app = VotingEnsemble.bind(
            model_handles=model_handles,
            voting=voting,
            threshold=threshold
        )
        
        ensemble_handle = serve.run(ensemble_app, name=ensemble_name, route_prefix=f"/{ensemble_name}")
        self.ensembles[ensemble_name] = ensemble_handle
        
        log.info(f"Deployed ensemble {ensemble_name} with {len(model_paths)} models")
        
        return {
            "ensemble_name": ensemble_name,
            "num_models": len(model_paths),
            "voting": voting,
            "threshold": threshold,
            "endpoint": f"http://{settings.serve.serve_host}:{settings.serve.serve_port}/{ensemble_name}"
        }
    
    def deploy_pbt_survivors(
        self,
        ensemble_name: str,
        experiment_results: dict,
        top_n: int = 5,
        voting: str = "soft",
        threshold: float = 0.7
    ) -> dict:
        """
        Deploy top N models from a PBT experiment as an ensemble.
        
        Args:
            ensemble_name: Unique name for the ensemble
            experiment_results: Results from TuneOrchestrator
            top_n: Number of top models to include
            voting: Voting strategy
            threshold: Confidence threshold
            
        Returns:
            Deployment info
        """
        # Extract top N checkpoint paths
        top_trials = experiment_results.get("top_5", [])[:top_n]
        
        model_paths = []
        model_ids = []
        
        for i, trial in enumerate(top_trials):
            checkpoint_path = trial.get("checkpoint_path")
            if checkpoint_path:
                # Checkpoints are directories; model is inside
                model_file = Path(checkpoint_path) / "model.joblib"
                if model_file.exists():
                    model_paths.append(str(model_file))
                    model_ids.append(f"pbt_survivor_{i}")
        
        if not model_paths:
            return {"error": "No valid checkpoints found in experiment results"}
        
        return self.deploy_ensemble(
            ensemble_name=ensemble_name,
            model_paths=model_paths,
            model_ids=model_ids,
            voting=voting,
            threshold=threshold
        )
    
    def add_model_to_ensemble(
        self,
        ensemble_name: str,
        model_path: str,
        model_id: str
    ) -> dict:
        """Add a new model to an existing ensemble."""
        if ensemble_name not in self.ensembles:
            return {"error": f"Ensemble {ensemble_name} not found"}
        
        # Deploy new model
        model_app = TradingModel.bind(model_path=model_path, model_id=model_id)
        handle = serve.run(model_app, name=f"{ensemble_name}_{model_id}")
        
        self.models[f"{ensemble_name}_{model_id}"] = handle
        
        log.info(f"Added {model_id} to ensemble {ensemble_name}")
        
        # Note: To actually add to the ensemble's handle list would require
        # redeploying the ensemble. For production, consider using a 
        # centralized model registry that the ensemble queries.
        
        return {
            "status": "added",
            "model_id": model_id,
            "ensemble_name": ensemble_name,
            "note": "Redeploy ensemble to include new model in voting"
        }
    
    def remove_model(self, ensemble_name: str, model_id: str) -> dict:
        """Remove a model from the system."""
        key = f"{ensemble_name}_{model_id}"
        
        if key in self.models:
            serve.delete(key)
            del self.models[key]
            log.info(f"Removed model {model_id} from {ensemble_name}")
            return {"status": "removed", "model_id": model_id}
        
        return {"error": f"Model {key} not found"}
    
    def delete_ensemble(self, ensemble_name: str) -> dict:
        """Delete an entire ensemble and its models."""
        if ensemble_name not in self.ensembles:
            return {"error": f"Ensemble {ensemble_name} not found"}
        
        # Remove ensemble
        serve.delete(ensemble_name)
        del self.ensembles[ensemble_name]
        
        # Remove associated models
        models_to_remove = [k for k in self.models if k.startswith(f"{ensemble_name}_")]
        for key in models_to_remove:
            serve.delete(key)
            del self.models[key]
        
        log.info(f"Deleted ensemble {ensemble_name} with {len(models_to_remove)} models")
        
        return {
            "status": "deleted",
            "ensemble_name": ensemble_name,
            "models_removed": len(models_to_remove)
        }
    
    def list_ensembles(self) -> list[dict]:
        """List all deployed ensembles."""
        return [
            {
                "name": name,
                "endpoint": f"http://{settings.serve.serve_host}:{settings.serve.serve_port}/{name}"
            }
            for name in self.ensembles.keys()
        ]
    
    def get_ensemble_status(self, ensemble_name: str) -> dict:
        """Get status of an ensemble."""
        if ensemble_name not in self.ensembles:
            return {"error": f"Ensemble {ensemble_name} not found"}
        
        # Get status from ensemble
        handle = self.ensembles[ensemble_name]
        try:
            status = ray.get(handle.get_status.remote())
            return status
        except Exception as e:
            return {"error": str(e)}


# Global manager instance
ensemble_manager = EnsembleManager()
