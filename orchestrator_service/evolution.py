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
    symbol: str                                  # Primary symbol for training
    simulation_tickers: Optional[List[str]] = None  # Tickers to run simulations on (defaults to [symbol])
    algorithm: str = "RandomForest"
    target_col: str = "close"
    hyperparameters: Dict[str, Any] = {}
    target_transform: str = "log_return"
    max_generations: int = 4                    # User-configurable, default 4
    data_options: Optional[str] = None
    timeframe: str = "1m"
    
    # Grid search for regularization (ElasticNet, Ridge, Lasso)
    alpha_grid: Optional[List[float]] = None    # L2 penalty: [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    l1_ratio_grid: Optional[List[float]] = None # L1/L2 mix: [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    
    # Pruning strategy: prune bottom X% of features each generation
    prune_fraction: float = 0.25                # Prune bottom 25% each gen
    min_features: int = 5                       # Never go below this many features
    
    # Simulation grid - full grid search across all combinations
    thresholds: List[float] = [0.0001, 0.0003, 0.0005, 0.0007]
    z_score_thresholds: List[float] = [0, 2.0, 2.5, 3.0, 3.5]  # Z-score cutoffs (0 = no filter)
    regime_configs: List[Dict[str, Any]] = [
        {"regime_vix": [0]},                    # VIX 0: Bear Volatile (Crash)
        {"regime_vix": [1]},                    # VIX 1: Bear Quiet (Drift Down)
        {"regime_vix": [2]},                    # VIX 2: Bull Volatile (Melt Up)
        {"regime_vix": [3]},                    # VIX 3: Bull Quiet (Best)
        {"regime_gmm": [0]},                    # GMM 0: Low volatility cluster
        {"regime_gmm": [1]},                    # GMM 1: High volatility cluster
        {}                                       # No regime filter (all conditions)
    ]
    
    # Holy Grail criteria
    sqn_min: float = 3.0
    sqn_max: float = 5.0
    profit_factor_min: float = 2.0
    profit_factor_max: float = 4.0
    trade_count_min: int = 200
    trade_count_max: int = 10000
    
    def get_simulation_tickers(self) -> List[str]:
        """Get tickers to simulate on. Defaults to [symbol] if not specified."""
        if self.simulation_tickers and len(self.simulation_tickers) > 0:
            return self.simulation_tickers
        return [self.symbol]


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
                
                # ================================================================
                # STEP A: Run simulations on CURRENT model (before pruning)
                # ================================================================
                # This ensures we always evaluate every trained model
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Queueing simulations")
                try:
                    log.info(f"Step A: Queueing simulations for current model {state.current_model_id}")
                    await self._queue_simulations(state, state.current_model_id)
                    log.info("Step A: Simulations queued")
                except Exception as e:
                    log.error(f"Step A FAILED: Queue simulations error: {e}")
                    state.stopped_reason = f"simulation_queue_error: {e}"
                    break
                
                # Wait for simulations and evaluate
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Waiting for simulation results...")
                try:
                    log.info("Step A: Waiting for simulation results...")
                    best_result = await self._wait_and_evaluate(state, state.current_model_id)
                    log.info(f"Step A: Got result - SQN: {best_result.get('sqn', 'N/A') if best_result else 'None'}")
                except Exception as e:
                    log.error(f"Step A FAILED: Simulation evaluation error: {e}")
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
                        await self._record_promotion(state, state.current_model_id, best_result, evaluation)
                        log.info(f"🎯 Model {state.current_model_id} PROMOTED!")
                        break  # Stop evolution - we found a winner!
                    
                    # Update best results in run record
                    await db.update_evolution_run(
                        run_id,
                        best_sqn=state.parent_sqn,
                        best_model_id=state.current_model_id
                    )
                
                # ================================================================
                # STEP B: Get feature importance for pruning
                # ================================================================
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Getting feature importance")
                try:
                    importance = await self._get_feature_importance(state.current_model_id)
                    log.info(f"Step B: Got importance for {len(importance)} features")
                except Exception as e:
                    log.error(f"Step B FAILED: Could not get feature importance: {e}")
                    state.stopped_reason = f"feature_importance_error: {e}"
                    break
                
                if not importance:
                    log.warning("Step B: No importance data returned")
                    state.stopped_reason = "no_importance_data"
                    break
                
                # ================================================================
                # STEP C: Prune bottom X% of features by importance
                # ================================================================
                try:
                    pruned, remaining = self._prune_features(
                        importance, 
                        prune_fraction=config.prune_fraction,
                        min_features=config.min_features
                    )
                    log.info(f"Step C: Pruning complete - {len(pruned)} removed, {len(remaining)} remaining")
                except Exception as e:
                    log.error(f"Step C FAILED: Pruning error: {e}")
                    state.stopped_reason = f"pruning_error: {e}"
                    break
                
                # Check if we can continue pruning
                # Note: We've already simulated the current model in Step A, so results are saved
                if not pruned:
                    log.info("Step C: No features to prune (at min_features limit or all equal importance)")
                    log.info("Step C: Current model has been evaluated, cannot create child generation")
                    state.stopped_reason = "no_features_to_prune"
                    break  # Can't create a new generation without pruning
                
                if not remaining or len(remaining) < config.min_features:
                    log.warning(f"Step C: Would go below min_features ({config.min_features})")
                    log.info("Step C: Current model has been evaluated, stopping at feature limit")
                    state.stopped_reason = "min_features_reached"
                    break  # Can't train a valid model with fewer features
                
                log.info(f"Step C: Pruned {len(pruned)} low-importance features")
                log.info(f"Step C: Remaining {len(remaining)} features: {remaining[:10]}{'...' if len(remaining) > 10 else ''}")
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Pruned {len(pruned)}, keeping {len(remaining)} features")
                
                # ================================================================
                # STEP D: Compute fingerprint for child model
                # ================================================================
                try:
                    fingerprint = compute_fingerprint(
                        features=remaining,
                        hyperparams=config.hyperparameters,
                        target_transform=config.target_transform,
                        symbol=config.symbol,
                        target_col=config.target_col,
                        alpha_grid=config.alpha_grid,
                        l1_ratio_grid=config.l1_ratio_grid,
                        regime_configs=config.regime_configs
                    )
                    log.info(f"Step D: Computed fingerprint: {fingerprint[:16]}...")
                except Exception as e:
                    log.error(f"Step D FAILED: Fingerprint error: {e}")
                    state.stopped_reason = f"fingerprint_error: {e}"
                    break
                
                # Check for existing model with same fingerprint
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Checking fingerprint cache")
                try:
                    existing_model_id = await db.get_model_by_fingerprint(fingerprint)
                    if existing_model_id:
                        log.info(f"Step D: Fingerprint match! Reusing model {existing_model_id}")
                    else:
                        log.info("Step D: No existing model, will train new")
                except Exception as e:
                    log.error(f"Step D FAILED: Fingerprint lookup error: {e}")
                    existing_model_id = None  # Continue with training
                
                if existing_model_id:
                    child_model_id = existing_model_id
                    await db.update_evolution_run(run_id, step_status=f"{gen_label}: Reusing cached model")
                else:
                    # ================================================================
                    # STEP E: Train child model with pruned features
                    # ================================================================
                    await db.update_evolution_run(run_id, step_status=f"{gen_label}: Training child model ({len(remaining)} features)")
                    try:
                        log.info(f"Step E: Training child model with {len(remaining)} features...")
                        child_model_id = await self._train_model(
                            state,
                            parent_model_id=state.current_model_id,
                            features=remaining
                        )
                        log.info(f"Step E: Training complete, child model ID: {child_model_id}")
                    except Exception as e:
                        log.error(f"Step E FAILED: Training error: {e}")
                        state.stopped_reason = f"training_error: {e}"
                        break
                    
                    # Record fingerprint with full config for reproducibility
                    try:
                        full_config = {
                            **config.hyperparameters,
                            "alpha_grid": config.alpha_grid,
                            "l1_ratio_grid": config.l1_ratio_grid,
                            "regime_configs": config.regime_configs
                        }
                        await db.insert_fingerprint(
                            fingerprint=fingerprint,
                            model_id=child_model_id,
                            features=remaining,
                            hyperparams=full_config,
                            target_transform=config.target_transform,
                            symbol=config.symbol
                        )
                        log.info("Step E: Fingerprint recorded")
                    except Exception as e:
                        log.warning(f"Step E: Failed to record fingerprint (non-fatal): {e}")
                
                # ================================================================
                # STEP F: Record evolution lineage
                # ================================================================
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
                    log.info(f"Step F: Evolution lineage recorded")
                except Exception as e:
                    log.warning(f"Step F: Failed to record lineage (non-fatal): {e}")
                
                # ================================================================
                # Update state: child becomes current for next iteration
                # Child model will be simulated in Step A of next iteration
                # ================================================================
                state.current_model_id = child_model_id
                state.current_features = remaining
                state.current_generation += 1
                
                await db.update_evolution_run(
                    run_id,
                    current_generation=state.current_generation,
                    best_sqn=state.parent_sqn,
                    best_model_id=state.current_model_id
                )
                
                log.info(f"Step F: Advancing to Generation {state.current_generation}, child model {child_model_id} will be simulated next iteration")
            
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
        importance: Dict[str, float],
        prune_fraction: float = 0.25,
        min_features: int = 5
    ) -> tuple[List[str], List[str]]:
        """
        Prune the bottom X% of features by absolute importance.
        
        Strategy:
        1. First, remove any features with exactly 0 importance
        2. Then, prune the bottom prune_fraction of remaining features
        3. Always keep at least min_features
        
        Returns:
            Tuple of (pruned_features, remaining_features)
        """
        pruned = []
        remaining = []
        
        # Log the importance values we received
        log.info(f"Feature importance received: {len(importance)} features")
        for feature, score in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
            log.info(f"  Top importance: {feature}: {score:.6f}")
        
        # Step 1: Separate zero-importance features
        zero_importance = [f for f, s in importance.items() if s == 0]
        non_zero = {f: s for f, s in importance.items() if s != 0}
        
        log.info(f"Zero-importance features: {len(zero_importance)}")
        log.info(f"Non-zero importance features: {len(non_zero)}")
        
        # Step 2: Sort non-zero features by absolute importance
        sorted_features = sorted(non_zero.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Step 3: Calculate how many to keep
        total_non_zero = len(sorted_features)
        num_to_keep = max(min_features, int(total_non_zero * (1 - prune_fraction)))
        
        # Step 4: Split into remaining and pruned
        remaining = [f for f, _ in sorted_features[:num_to_keep]]
        importance_pruned = [f for f, _ in sorted_features[num_to_keep:]]
        
        # Combine zero-importance and low-importance pruned
        pruned = zero_importance + importance_pruned
        
        log.info(f"Pruning strategy: keep top {num_to_keep}/{total_non_zero} non-zero features")
        log.info(f"Pruning result: {len(pruned)} pruned ({len(zero_importance)} zero + {len(importance_pruned)} low), {len(remaining)} remaining")
        
        if importance_pruned:
            log.info(f"Lowest importance features pruned: {importance_pruned[:5]}{'...' if len(importance_pruned) > 5 else ''}")
        
        return pruned, remaining
    
    async def _train_model(
        self,
        state: EvolutionState,
        parent_model_id: Optional[str],
        features: Optional[List[str]] = None,
        timeout: float = 600.0,  # 10 minute timeout
        poll_interval: float = 5.0
    ) -> str:
        """Train a new model via training service."""
        config = state.config
        run_id = state.run_id
        
        payload = {
            "symbol": config.symbol,
            "algorithm": config.algorithm,
            "target_col": config.target_col,
            "hyperparameters": config.hyperparameters,
            "target_transform": config.target_transform,
            "data_options": config.data_options,
            "timeframe": config.timeframe,
            "parent_model_id": parent_model_id,
            "feature_whitelist": features,
            "alpha_grid": config.alpha_grid,       # Grid search: L2 penalty values
            "l1_ratio_grid": config.l1_ratio_grid  # Grid search: L1/L2 mix values
        }
        
        log.info(f"Training model with {len(features or [])} features...")
        log.debug(f"Training payload: {payload}")
        
        try:
            resp = await self.http_client.post(
                f"{self.training_url}/train",
                json=payload
            )
        except Exception as e:
            await db.update_evolution_run(run_id, step_status=f"Training request failed: {e}")
            raise RuntimeError(f"Failed to connect to training service: {e}")
        
        if resp.status_code != 200:
            error_msg = f"Training service returned {resp.status_code}: {resp.text}"
            log.error(error_msg)
            await db.update_evolution_run(run_id, step_status=f"Training failed: HTTP {resp.status_code}")
            raise RuntimeError(error_msg)
        
        data = resp.json()
        model_id = data["id"]
        log.info(f"Training job started: {model_id}")
        
        # Poll for completion with timeout
        elapsed = 0.0
        last_status = None
        
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
            try:
                resp = await self.http_client.get(f"{self.training_url}/models")
                if resp.status_code != 200:
                    log.warning(f"Training service models endpoint returned {resp.status_code}")
                    await db.update_evolution_run(run_id, step_status=f"Training poll failed: HTTP {resp.status_code}")
                    continue
                    
                models = resp.json()
                model = next((m for m in models if m["id"] == model_id), None)
                
                if not model:
                    log.warning(f"Model {model_id} not found in training service (may have restarted)")
                    await db.update_evolution_run(run_id, step_status=f"Training: Model not found (retry {int(elapsed)}s)")
                    continue
                
                status = model.get("status")
                
                if status != last_status:
                    log.info(f"Model {model_id} status: {status}")
                    await db.update_evolution_run(run_id, step_status=f"Training: {status} ({int(elapsed)}s)")
                    last_status = status
                
                if status == "completed":
                    log.info(f"Model {model_id} training completed after {int(elapsed)}s")
                    return model_id
                elif status == "failed":
                    error_msg = model.get('error_message', 'Unknown error')
                    await db.update_evolution_run(run_id, step_status=f"Training failed: {error_msg[:50]}")
                    raise RuntimeError(f"Model training failed: {error_msg}")
                    
            except httpx.RequestError as e:
                log.warning(f"Training service unreachable: {e}")
                await db.update_evolution_run(run_id, step_status=f"Training: Service unreachable ({int(elapsed)}s)")
        
        # Timeout reached
        await db.update_evolution_run(run_id, step_status=f"Training timeout after {int(timeout)}s")
        raise RuntimeError(f"Training timeout after {timeout}s for model {model_id}")
    
    async def _queue_simulations(
        self,
        state: EvolutionState,
        model_id: str
    ) -> int:
        """
        Queue simulation jobs with priority - full grid search.
        
        Grid dimensions:
        - Thresholds: Signal strength cutoffs (e.g., 0.0001 to 0.001)
        - Z-score thresholds: Volatility-adjusted signal filtering (e.g., 2.0 to 3.5)
        - Regime configs: Market condition filters (GMM clusters, VIX regimes)
        - Simulation tickers: Run on multiple tickers to test generalization
        
        Total jobs = len(tickers) * len(thresholds) * len(z_scores) * len(regimes)
        Example: 2 tickers * 4 thresholds * 4 z-scores * 4 regimes = 128 simulations per model
        """
        config = state.config
        jobs = []
        batch_id = str(uuid.uuid4())
        
        # Get simulation tickers (defaults to training symbol if not specified)
        sim_tickers = config.get_simulation_tickers()
        
        # Full 4D grid search: tickers × thresholds × z-scores × regimes
        for ticker in sim_tickers:
            for threshold in config.thresholds:
                for z_score in config.z_score_thresholds:
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
                                "ticker": ticker,
                                "threshold": threshold,
                                "z_score_threshold": z_score,
                                "regime_config": regime_config,
                                "use_trading_bot": False,
                                "use_volume_normalization": True
                            }
                        }
                        jobs.append(job)
        
        count = await db.enqueue_jobs(jobs)
        grid_dims = f"{len(sim_tickers)} tickers × {len(config.thresholds)} thresholds × {len(config.z_score_thresholds)} z-scores × {len(config.regime_configs)} regimes"
        log.info(f"Queued {count} simulation jobs ({grid_dims}) with priority {state.parent_sqn:.2f}")
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
        run_id = state.run_id
        total_jobs = None
        
        while elapsed < timeout:
            pending = await db.get_pending_job_count(run_id)
            completed_count = await db.get_completed_job_count(run_id, state.current_generation)
            
            if total_jobs is None and (pending > 0 or completed_count > 0):
                total_jobs = pending + completed_count
            
            if pending == 0:
                break
            
            # Update status with progress
            progress = f"{completed_count}/{total_jobs}" if total_jobs else f"{pending} pending"
            await db.update_evolution_run(run_id, step_status=f"Simulating: {progress} ({int(elapsed)}s)")
            
            log.info(f"Waiting for {pending} simulations... ({completed_count} completed)")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        if elapsed >= timeout:
            await db.update_evolution_run(run_id, step_status=f"Simulation timeout after {int(timeout)}s")
            log.warning(f"Simulation timeout after {timeout}s")
        
        # Get completed jobs for this generation
        completed = await db.get_completed_jobs(
            run_id, 
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
