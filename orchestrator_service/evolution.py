"""
Evolution Engine - Orchestrates the Train → Prune → Simulate loop.
"""
import asyncio
import uuid
import logging
import httpx
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel

from .db import db
from .fingerprint import compute_fingerprint
from .criteria import evaluate_holy_grail, format_criteria_result, HolyGrailCriteria

log = logging.getLogger("orchestrator.evolution")


class EvolutionConfig(BaseModel):
    """Configuration for an evolution run."""
    seed_model_id: Optional[str] = None         # Start from existing model
    seed_features: Optional[List[str]] = None   # Or start fresh with feature list
    symbol: str
    algorithm: str = "RandomForest"
    target_col: str = "close"
    hyperparameters: Dict[str, Any] = {}
    target_transform: str = "log_return"
    max_generations: int = 4                    # User-configurable, default 4
    data_options: Optional[str] = None
    timeframe: str = "1m"
    
    # Simulation grid
    thresholds: List[float] = [0.0001, 0.0003, 0.0005, 0.0007]
    regime_configs: List[Dict[str, Any]] = [
        {"regime_gmm": [0]},
        {"regime_gmm": [1]},
        {"regime_vix": [0, 1]}
    ]
    
    # Holy Grail criteria
    sqn_min: float = 3.0
    sqn_max: float = 5.0
    profit_factor_min: float = 2.0
    profit_factor_max: float = 4.0
    trade_count_min: int = 200
    trade_count_max: int = 10000


@dataclass
class EvolutionState:
    """Tracks state during an evolution run."""
    run_id: str
    config: EvolutionConfig
    current_model_id: Optional[str] = None
    current_generation: int = 0
    current_features: List[str] = field(default_factory=list)
    parent_sqn: float = 0.0
    promoted: bool = False
    stopped_reason: Optional[str] = None


class EvolutionEngine:
    """
    Orchestrates the recursive evolution loop.
    
    Loop:
    1. Get feature importance from training service
    2. Prune features with importance <= 0
    3. Compute fingerprint, check for existing model
    4. Train new model or reuse existing
    5. Queue simulations with priority
    6. Wait for completion, evaluate results
    7. If promoted or no pruning possible, stop; else recurse
    """
    
    def __init__(
        self,
        training_url: str = "http://training:8200",
        simulation_url: str = "http://simulation:8300"
    ):
        self.training_url = training_url.rstrip("/")
        self.simulation_url = simulation_url.rstrip("/")
        self.http_client: Optional[httpx.AsyncClient] = None
    
    async def start(self):
        """Initialize HTTP client."""
        self.http_client = httpx.AsyncClient(timeout=120.0)
    
    async def stop(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
    
    async def run_evolution(self, config: EvolutionConfig) -> Dict[str, Any]:
        """
        Execute a full evolution run.
        
        Returns:
            Dict with run_id, generations_completed, promoted status, best results
        """
        run_id = str(uuid.uuid4())
        state = EvolutionState(run_id=run_id, config=config)
        
        log.info(f"Starting evolution run {run_id} for {config.symbol}")
        log.info(f"Max generations: {config.max_generations}")
        
        # Create run record
        await db.create_evolution_run(
            run_id=run_id,
            seed_model_id=config.seed_model_id,
            symbol=config.symbol,
            max_generations=config.max_generations,
            config=config.model_dump()
        )
        
        try:
            await db.update_evolution_run(run_id, status="RUNNING", step_status="Initializing")
            
            # Initialize: get seed model's features or use provided
            if config.seed_model_id:
                await db.update_evolution_run(run_id, step_status="Loading seed model features")
                state.current_model_id = config.seed_model_id
                state.current_features = await self._get_model_features(config.seed_model_id)
            elif config.seed_features:
                state.current_features = config.seed_features
                await db.update_evolution_run(run_id, step_status=f"Training initial model with {len(state.current_features)} features")
                # Train initial model with seed features
                state.current_model_id = await self._train_model(
                    state, 
                    parent_model_id=None,
                    features=state.current_features
                )
            else:
                raise ValueError("Must provide seed_model_id or seed_features")
            
            # Evolution loop
            while state.current_generation < config.max_generations:
                gen_label = f"Gen {state.current_generation}/{config.max_generations}"
                log.info(f"=== Generation {state.current_generation} ===")
                log.info(f"Current model: {state.current_model_id}")
                log.info(f"Current features count: {len(state.current_features)}")
                
                # 1. Get feature importance
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Getting feature importance")
                try:
                    importance = await self._get_feature_importance(state.current_model_id)
                    log.info(f"Step 1: Got importance for {len(importance)} features")
                except Exception as e:
                    log.error(f"Step 1 FAILED: Could not get feature importance: {e}")
                    state.stopped_reason = f"feature_importance_error: {e}"
                    break
                
                if not importance:
                    log.warning("Step 1: No importance data returned")
                    state.stopped_reason = "Could not get feature importance"
                    break
                
                # 2. Prune features with importance == 0
                try:
                    pruned, remaining = self._prune_features(importance)
                    log.info(f"Step 2: Pruning complete - {len(pruned)} removed, {len(remaining)} remaining")
                except Exception as e:
                    log.error(f"Step 2 FAILED: Pruning error: {e}")
                    state.stopped_reason = f"pruning_error: {e}"
                    break
                
                if not pruned:
                    log.info("Step 2: No features to prune (all have non-zero importance), evolution complete")
                    state.stopped_reason = "no_features_to_prune"
                    break
                
                if not remaining:
                    log.warning("Step 2: All features would be pruned, stopping")
                    log.warning(f"  Pruned features: {pruned[:10]}{'...' if len(pruned) > 10 else ''}")
                    state.stopped_reason = "all_features_pruned"
                    break
                
                log.info(f"Step 2: Pruned {len(pruned)} zero-importance features: {pruned}")
                log.info(f"Step 2: Remaining {len(remaining)} features: {remaining[:10]}{'...' if len(remaining) > 10 else ''}")
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Pruned {len(pruned)}, keeping {len(remaining)} features")
                
                # 3. Compute fingerprint
                try:
                    fingerprint = compute_fingerprint(
                        features=remaining,
                        hyperparams=config.hyperparameters,
                        target_transform=config.target_transform,
                        symbol=config.symbol,
                        target_col=config.target_col
                    )
                    log.info(f"Step 3: Computed fingerprint: {fingerprint[:16]}...")
                except Exception as e:
                    log.error(f"Step 3 FAILED: Fingerprint error: {e}")
                    state.stopped_reason = f"fingerprint_error: {e}"
                    break
                
                # 4. Check for existing model
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Checking fingerprint cache")
                try:
                    existing_model_id = await db.get_model_by_fingerprint(fingerprint)
                    if existing_model_id:
                        log.info(f"Step 4: Fingerprint match! Reusing model {existing_model_id}")
                    else:
                        log.info("Step 4: No existing model, will train new")
                except Exception as e:
                    log.error(f"Step 4 FAILED: Fingerprint lookup error: {e}")
                    existing_model_id = None  # Continue with training
                
                if existing_model_id:
                    child_model_id = existing_model_id
                    await db.update_evolution_run(run_id, step_status=f"{gen_label}: Reusing cached model")
                else:
                    # Step 5: Train new model
                    await db.update_evolution_run(run_id, step_status=f"{gen_label}: Training model ({len(remaining)} features)")
                    try:
                        log.info(f"Step 5: Training new model with {len(remaining)} features...")
                        child_model_id = await self._train_model(
                            state,
                            parent_model_id=state.current_model_id,
                            features=remaining
                        )
                        log.info(f"Step 5: Training complete, model ID: {child_model_id}")
                    except Exception as e:
                        log.error(f"Step 5 FAILED: Training error: {e}")
                        state.stopped_reason = f"training_error: {e}"
                        break
                    
                    # Record fingerprint
                    try:
                        await db.insert_fingerprint(
                            fingerprint=fingerprint,
                            model_id=child_model_id,
                            features=remaining,
                            hyperparams=config.hyperparameters,
                            target_transform=config.target_transform,
                            symbol=config.symbol
                        )
                        log.info("Step 5: Fingerprint recorded")
                    except Exception as e:
                        log.warning(f"Step 5: Failed to record fingerprint (non-fatal): {e}")
                
                # Step 6: Record evolution lineage
                try:
                    await db.insert_evolution_log(
                        log_id=str(uuid.uuid4()),
                        run_id=run_id,
                        parent_model_id=state.current_model_id,
                        child_model_id=child_model_id,
                        generation=state.current_generation,
                        parent_sqn=state.parent_sqn,
                        pruned_features=pruned,
                        remaining_features=remaining,
                        pruning_reason="importance_zero"
                    )
                    log.info(f"Step 6: Evolution lineage recorded")
                except Exception as e:
                    log.warning(f"Step 6: Failed to record lineage (non-fatal): {e}")
                
                # Step 7: Queue simulations
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Queueing simulations")
                try:
                    log.info(f"Step 7: Queueing simulations for model {child_model_id}")
                    await self._queue_simulations(state, child_model_id)
                    log.info("Step 7: Simulations queued")
                except Exception as e:
                    log.error(f"Step 7 FAILED: Queue simulations error: {e}")
                    state.stopped_reason = f"simulation_queue_error: {e}"
                    break
                
                # Step 8: Wait for simulations and evaluate
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Waiting for simulation results...")
                try:
                    log.info("Step 8: Waiting for simulation results...")
                    best_result = await self._wait_and_evaluate(state, child_model_id)
                    log.info(f"Step 8: Got result - SQN: {best_result.get('sqn', 'N/A') if best_result else 'None'}")
                except Exception as e:
                    log.error(f"Step 8 FAILED: Simulation evaluation error: {e}")
                    best_result = None
                
                if best_result:
                    state.parent_sqn = best_result.get("sqn", 0)
                    await db.update_evolution_run(run_id, step_status=f"{gen_label}: Evaluating results (SQN={state.parent_sqn:.2f})")
                    
                    # Check Holy Grail criteria
                    criteria = HolyGrailCriteria(
                        sqn_min=config.sqn_min,
                        sqn_max=config.sqn_max,
                        profit_factor_min=config.profit_factor_min,
                        profit_factor_max=config.profit_factor_max,
                        trade_count_min=config.trade_count_min,
                        trade_count_max=config.trade_count_max
                    )
                    evaluation = evaluate_holy_grail(best_result, criteria)
                    log.info(format_criteria_result(evaluation))
                    
                    if evaluation.meets_all:
                        state.promoted = True
                        await self._record_promotion(state, child_model_id, best_result, evaluation)
                        log.info(f"🎯 Model {child_model_id} PROMOTED!")
                
                # Update state for next iteration
                state.current_model_id = child_model_id
                state.current_features = remaining
                state.current_generation += 1
                
                await db.update_evolution_run(
                    run_id,
                    current_generation=state.current_generation,
                    best_sqn=state.parent_sqn,
                    best_model_id=state.current_model_id
                )
            
            # Final status
            final_status = "PROMOTED" if state.promoted else "COMPLETED"
            await db.update_evolution_run(
                run_id,
                status=final_status,
                promoted=state.promoted
            )
            
            log.info(f"Evolution run {run_id} finished: {final_status}")
            
        except Exception as e:
            import traceback
            log.error(f"Evolution run {run_id} failed: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            await db.update_evolution_run(run_id, status="FAILED")
            raise
        
        return {
            "run_id": run_id,
            "generations_completed": state.current_generation,
            "final_model_id": state.current_model_id,
            "promoted": state.promoted,
            "best_sqn": state.parent_sqn,
            "stopped_reason": state.stopped_reason
        }
    
    async def _get_model_features(self, model_id: str) -> List[str]:
        """Get feature list from training service."""
        resp = await self.http_client.get(
            f"{self.training_url}/api/model/{model_id}/config"
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("features", [])
    
    async def _get_feature_importance(self, model_id: str) -> Dict[str, float]:
        """Get feature importance scores from training service."""
        try:
            resp = await self.http_client.get(
                f"{self.training_url}/api/model/{model_id}/importance"
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("importance", {})
        except Exception as e:
            log.error(f"Failed to get importance for {model_id}: {e}")
            return {}
    
    def _prune_features(
        self, 
        importance: Dict[str, float]
    ) -> tuple[List[str], List[str]]:
        """
        Prune features with importance == 0 (exactly zero).
        Small negative values are kept as they still contribute.
        
        Returns:
            Tuple of (pruned_features, remaining_features)
        """
        pruned = []
        remaining = []
        
        # Log the importance values we received
        log.info(f"Feature importance received: {len(importance)} features")
        for feature, score in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
            log.debug(f"  {feature}: {score}")
        
        for feature, score in importance.items():
            # Only prune features with exactly zero importance
            if score == 0:
                pruned.append(feature)
            else:
                remaining.append(feature)
        
        log.info(f"Pruning result: {len(pruned)} zero-importance pruned, {len(remaining)} remaining")
        
        return pruned, remaining
    
    async def _train_model(
        self,
        state: EvolutionState,
        parent_model_id: Optional[str],
        features: Optional[List[str]] = None
    ) -> str:
        """Train a new model via training service."""
        config = state.config
        
        payload = {
            "symbol": config.symbol,
            "algorithm": config.algorithm,
            "target_col": config.target_col,
            "hyperparameters": config.hyperparameters,
            "target_transform": config.target_transform,
            "data_options": config.data_options,
            "timeframe": config.timeframe,
            "parent_model_id": parent_model_id,
            "feature_whitelist": features
        }
        
        log.info(f"Training model with {len(features or [])} features...")
        log.debug(f"Training payload: {payload}")
        resp = await self.http_client.post(
            f"{self.training_url}/train",
            json=payload
        )
        if resp.status_code != 200:
            log.error(f"Training service returned {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        data = resp.json()
        model_id = data["id"]
        
        # Poll for completion
        while True:
            await asyncio.sleep(5)
            resp = await self.http_client.get(f"{self.training_url}/models")
            models = resp.json()
            model = next((m for m in models if m["id"] == model_id), None)
            
            if model:
                status = model.get("status")
                if status == "completed":
                    log.info(f"Model {model_id} training completed")
                    return model_id
                elif status == "failed":
                    raise RuntimeError(f"Model training failed: {model.get('error_message')}")
        
        return model_id
    
    async def _queue_simulations(
        self,
        state: EvolutionState,
        model_id: str
    ) -> int:
        """Queue simulation jobs with priority."""
        config = state.config
        jobs = []
        batch_id = str(uuid.uuid4())
        
        for threshold in config.thresholds:
            for regime_config in config.regime_configs:
                job = {
                    "id": str(uuid.uuid4()),
                    "batch_id": batch_id,
                    "run_id": state.run_id,
                    "model_id": model_id,
                    "generation": state.current_generation,
                    "parent_sqn": state.parent_sqn,
                    "params": {
                        "model_id": model_id,
                        "ticker": config.symbol,
                        "threshold": threshold,
                        "regime_config": regime_config,
                        "z_score_threshold": 3.0,
                        "use_trading_bot": False,
                        "use_volume_normalization": True
                    }
                }
                jobs.append(job)
        
        count = await db.enqueue_jobs(jobs)
        log.info(f"Queued {count} simulation jobs with priority {state.parent_sqn:.2f}")
        return count
    
    async def _wait_and_evaluate(
        self,
        state: EvolutionState,
        model_id: str,
        poll_interval: float = 10.0,
        timeout: float = 3600.0
    ) -> Optional[Dict[str, Any]]:
        """Wait for simulations to complete and return best result."""
        elapsed = 0.0
        
        while elapsed < timeout:
            pending = await db.get_pending_job_count(state.run_id)
            if pending == 0:
                break
            
            log.info(f"Waiting for {pending} simulations...")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        # Get completed jobs for this generation
        completed = await db.get_completed_jobs(
            state.run_id, 
            generation=state.current_generation
        )
        
        if not completed:
            log.warning("No completed simulations found")
            return None
        
        # Find best by SQN
        best = None
        best_sqn = -999
        
        for job in completed:
            result = job.get("result", {})
            if isinstance(result, str):
                import json
                result = json.loads(result)
            
            sqn = result.get("sqn", 0)
            if sqn > best_sqn:
                best_sqn = sqn
                best = result
        
        log.info(f"Best SQN this generation: {best_sqn:.2f}")
        return best
    
    async def _record_promotion(
        self,
        state: EvolutionState,
        model_id: str,
        result: Dict[str, Any],
        evaluation
    ) -> None:
        """Record a promoted model."""
        await db.insert_promoted_model(
            promoted_id=str(uuid.uuid4()),
            model_id=model_id,
            run_id=state.run_id,
            job_id=result.get("job_id", ""),
            generation=state.current_generation,
            sqn=evaluation.sqn,
            profit_factor=evaluation.profit_factor,
            trade_count=evaluation.trade_count,
            weekly_consistency=evaluation.weekly_consistency,
            ticker=state.config.symbol,
            regime_config=result.get("regime_config", {}),
            threshold=result.get("threshold", 0),
            full_result=result
        )
        
        await db.update_evolution_run(
            state.run_id,
            promoted=True,
            best_model_id=model_id,
            best_sqn=evaluation.sqn
        )


# Singleton engine
engine = EvolutionEngine()
