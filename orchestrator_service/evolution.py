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
from .fingerprint import compute_fingerprint, compute_simulation_fingerprint
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
    
    # Grid search for XGBoost
    max_depth_grid: Optional[List[int]] = None
    min_child_weight_grid: Optional[List[int]] = None
    reg_lambda_grid: Optional[List[float]] = None
    learning_rate_grid: Optional[List[float]] = None
    
    # Grid search for LightGBM
    num_leaves_grid: Optional[List[int]] = None
    min_data_in_leaf_grid: Optional[List[int]] = None
    lambda_l2_grid: Optional[List[float]] = None
    lgbm_learning_rate_grid: Optional[List[float]] = None
    
    # Grid search for RandomForest
    rf_max_depth_grid: Optional[List[Any]] = None  # Can include None
    min_samples_split_grid: Optional[List[int]] = None
    min_samples_leaf_grid: Optional[List[int]] = None
    n_estimators_grid: Optional[List[int]] = None
    
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
    
    # Progress tracking
    models_trained: int = 0
    models_total: int = 0
    simulations_completed: int = 0
    simulations_total: int = 0
    
    # Track all trained models for simulation phase
    trained_models: List[Dict[str, Any]] = field(default_factory=list)  # [{model_id, generation, features_count}]


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
            
            # Calculate total work upfront
            # NOTE: Currently using GridSearchCV which returns best model only
            # If we implement full grid enumeration, update this calculation
            state.models_total = config.max_generations  # One model per generation (best from grid search)
            
            # Simulation grid calculation
            sim_tickers = config.simulation_tickers or [config.symbol]
            num_tickers = len(sim_tickers)
            num_thresholds = len(config.thresholds)
            num_z_scores = len(config.z_score_thresholds)
            num_regimes = len(config.regime_configs)
            sims_per_model = num_tickers * num_thresholds * num_z_scores * num_regimes
            
            # Total sims = sims_per_model × number of models we'll train
            state.simulations_total = sims_per_model * config.max_generations
            
            log.info(f"Work estimate: {state.models_total} models, {state.simulations_total} simulations")
            log.info(f"Simulation grid: {num_tickers} tickers × {num_thresholds} thresholds × {num_z_scores} z-scores × {num_regimes} regimes = {sims_per_model} per model")
            
            # Update database with totals
            await db.update_evolution_run(
                run_id,
                models_total=state.models_total,
                simulations_total=state.simulations_total
            )
            
            # Initialize: get seed model's features or use provided
            if config.seed_model_id:
                await db.update_evolution_run(run_id, step_status="Loading seed model features")
                state.current_model_id = config.seed_model_id
                state.current_features = await self._get_model_features(config.seed_model_id)
                
                # Validate seed model before adding to trained models
                try:
                    seed_importance = await self._get_feature_importance(config.seed_model_id)
                    non_zero_count = sum(1 for v in seed_importance.values() if v != 0)
                    
                    if non_zero_count == 0:
                        log.error(f"Seed model {config.seed_model_id} has ALL ZERO importance features")
                        state.stopped_reason = "seed_model_invalid_all_zero_importance"
                        await db.update_evolution_run(
                            run_id,
                            status="FAILED",
                            step_status=f"Seed model has all zero importance - cannot proceed: {state.stopped_reason}"
                        )
                        return {
                            "run_id": run_id,
                            "generations_completed": 0,
                            "models_trained": 0,
                            "final_model_id": config.seed_model_id,
                            "promoted": False,
                            "best_sqn": 0.0,
                            "stopped_reason": state.stopped_reason
                        }
                    
                    log.info(f"Seed model validation passed - {non_zero_count}/{len(seed_importance)} features have non-zero importance")
                except Exception as e:
                    log.warning(f"Could not validate seed model importance: {e}")
                
                # Add seed model to trained models list
                state.trained_models.append({
                    "model_id": config.seed_model_id,
                    "generation": 0,
                    "features_count": len(state.current_features)
                })
            elif config.seed_features:
                state.current_features = config.seed_features
                await db.update_evolution_run(run_id, step_status=f"Training initial model with {len(state.current_features)} features")
                # Train initial model with seed features
                state.current_model_id = await self._train_model(
                    state, 
                    parent_model_id=None,
                    features=state.current_features
                )
                # Update model counter
                state.models_trained += 1
                await db.update_evolution_run(run_id, models_trained=state.models_trained)
                
                # Validate initial trained model
                try:
                    initial_importance = await self._get_feature_importance(state.current_model_id)
                    non_zero_count = sum(1 for v in initial_importance.values() if v != 0)
                    
                    if non_zero_count == 0:
                        log.error(f"Initial trained model {state.current_model_id} has ALL ZERO importance features")
                        state.stopped_reason = "initial_model_invalid_all_zero_importance"
                        await db.update_evolution_run(
                            run_id,
                            status="FAILED",
                            step_status=f"Initial model has all zero importance - cannot proceed: {state.stopped_reason}"
                        )
                        return {
                            "run_id": run_id,
                            "generations_completed": 0,
                            "models_trained": 0,
                            "final_model_id": state.current_model_id,
                            "promoted": False,
                            "best_sqn": 0.0,
                            "stopped_reason": state.stopped_reason
                        }
                    
                    log.info(f"Initial model validation passed - {non_zero_count}/{len(initial_importance)} features have non-zero importance")
                except Exception as e:
                    log.warning(f"Could not validate initial model importance: {e}")
                
                # Add to trained models list
                state.trained_models.append({
                    "model_id": state.current_model_id,
                    "generation": 0,
                    "features_count": len(state.current_features)
                })
            else:
                raise ValueError("Must provide seed_model_id or seed_features")
            
            # ================================================================
            # PHASE 1: TRAIN ALL MODELS (GENERATIONAL PRUNING)
            # ================================================================
            log.info("=" * 60)
            log.info("PHASE 1: TRAINING ALL MODELS")
            log.info("=" * 60)
            
            # Evolution loop - train models through progressive feature pruning
            while state.current_generation < config.max_generations:
                gen_label = f"Gen {state.current_generation}/{config.max_generations}"
                log.info(f"=== Generation {state.current_generation} ===")
                log.info(f"Current model: {state.current_model_id}")
                log.info(f"Current features count: {len(state.current_features)}")
                
                # ================================================================
                # STEP A: Get feature importance from CURRENT model
                # ================================================================
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Getting feature importance")
                try:
                    importance = await self._get_feature_importance(state.current_model_id)
                    log.info(f"Step A: Got importance for {len(importance)} features from training")
                except Exception as e:
                    log.error(f"Step A FAILED: Could not get feature importance: {e}")
                    state.stopped_reason = f"feature_importance_error: {e}"
                    break
                
                if not importance:
                    log.warning("Step A: No importance data returned")
                    state.stopped_reason = "no_importance_data"
                    break
                
                # ================================================================
                # STEP B: Prune low-importance features
                # ================================================================
                try:
                    pruned, remaining = self._prune_features(
                        importance, 
                        prune_fraction=config.prune_fraction,
                        min_features=config.min_features
                    )
                    log.info(f"Step B: Pruning complete - {len(pruned)} removed, {len(remaining)} remaining")
                except Exception as e:
                    log.error(f"Step B FAILED: Pruning error: {e}")
                    state.stopped_reason = f"pruning_error: {e}"
                    break
                
                # Check if we can continue pruning
                if not pruned:
                    log.info("Step B: No features to prune (at min_features limit or all equal importance)")
                    log.info("Step B: Current model is valid but cannot create child generation")
                    log.info(f"Step B: Saving current model {state.current_model_id} as final model")
                    
                    # The current model is already trained and valid - just can't be pruned further
                    # Make sure it's in the trained_models list if not already there
                    current_model_exists = any(
                        m["model_id"] == state.current_model_id 
                        for m in state.trained_models
                    )
                    
                    if not current_model_exists:
                        log.info(f"Step B: Adding current model to trained_models list")
                        state.trained_models.append({
                            "model_id": state.current_model_id,
                            "generation": state.current_generation,
                            "features_count": len(state.current_features)
                        })
                    
                    state.stopped_reason = "no_features_to_prune"
                    break
                
                if not remaining or len(remaining) < config.min_features:
                    log.warning(f"Step B: Would go below min_features ({config.min_features})")
                    log.info("Step B: Stopping at feature limit")
                    log.info(f"Step B: Saving current model {state.current_model_id} as final model")
                    
                    # The current model is already trained and valid
                    current_model_exists = any(
                        m["model_id"] == state.current_model_id 
                        for m in state.trained_models
                    )
                    
                    if not current_model_exists:
                        log.info(f"Step B: Adding current model to trained_models list")
                        state.trained_models.append({
                            "model_id": state.current_model_id,
                            "generation": state.current_generation,
                            "features_count": len(state.current_features)
                        })
                    
                    state.stopped_reason = "min_features_reached"
                    break
                
                log.info(f"Step B: Pruned {len(pruned)} low-importance features")
                log.info(f"Step B: Remaining {len(remaining)} features: {remaining[:10]}{'...' if len(remaining) > 10 else ''}")
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Pruned {len(pruned)}, keeping {len(remaining)} features")
                
                # ================================================================
                # STEP C: Compute fingerprint for child model
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
                    log.info(f"Step C: Computed fingerprint: {fingerprint[:16]}...")
                except Exception as e:
                    log.error(f"Step C FAILED: Fingerprint error: {e}")
                    state.stopped_reason = f"fingerprint_error: {e}"
                    break
                
                # Check for existing model with same fingerprint
                await db.update_evolution_run(run_id, step_status=f"{gen_label}: Checking fingerprint cache")
                try:
                    existing_model_id = await db.get_model_by_fingerprint(fingerprint)
                    if existing_model_id:
                        log.info(f"Step C: Fingerprint match! Reusing model {existing_model_id}")
                    else:
                        log.info("Step C: No existing model, will train new")
                except Exception as e:
                    log.error(f"Step C FAILED: Fingerprint lookup error: {e}")
                    existing_model_id = None  # Continue with training
                
                if existing_model_id:
                    child_model_id = existing_model_id
                    await db.update_evolution_run(run_id, step_status=f"{gen_label}: Reusing cached model")
                    # Single cached model - add to list
                    state.current_generation += 1
                    state.trained_models.append({
                        "model_id": child_model_id,
                        "generation": state.current_generation,
                        "features_count": len(remaining)
                    })
                    state.models_trained += 1
                    await db.update_evolution_run(run_id, models_trained=state.models_trained)
                else:
                    # ================================================================
                    # STEP D: Train child model(s) with pruned features
                    # NOTE: For ElasticNet with grid search, training service currently
                    # uses GridSearchCV which returns only the BEST model.
                    # TODO: Update training service to save ALL grid combinations as separate models
                    # ================================================================
                    await db.update_evolution_run(
                        run_id, 
                        step_status=f"{gen_label}: Training model ({len(remaining)} features)"
                    )
                    
                    try:
                        log.info(f"Step D: Training model with {len(remaining)} features...")
                        child_model_id = await self._train_model(
                            state,
                            parent_model_id=state.current_model_id,
                            features=remaining
                        )
                        log.info(f"Step D: Training complete, model ID: {child_model_id}")
                        
                        # Validate model - check if it has any non-zero importance features
                        try:
                            child_importance = await self._get_feature_importance(child_model_id)
                            non_zero_count = sum(1 for v in child_importance.values() if v != 0)
                            
                            if non_zero_count == 0:
                                log.error(f"Step D: Model {child_model_id} has ALL ZERO importance features - SKIPPING")
                                log.warning(f"Step D: Bad model will not be simulated. Continuing to next iteration.")
                                await db.update_evolution_run(
                                    run_id, 
                                    step_status=f"{gen_label}: Model failed (all zero importance) - skipping"
                                )
                                # Don't add to trained_models, don't increment generation
                                # Continue training loop to try next generation
                                state.current_model_id = child_model_id  # Still update for importance extraction
                                state.current_features = remaining
                                continue  # Skip to next iteration without incrementing generation
                            
                            log.info(f"Step D: Model validation passed - {non_zero_count}/{len(child_importance)} features have non-zero importance")
                            
                        except Exception as e:
                            log.warning(f"Step D: Could not validate model importance (continuing anyway): {e}")
                        
                        # Add single model (training service returns best from grid search)
                        state.current_generation += 1
                        state.trained_models.append({
                            "model_id": child_model_id,
                            "generation": state.current_generation,
                            "features_count": len(remaining)
                        })
                        state.models_trained += 1
                        await db.update_evolution_run(run_id, models_trained=state.models_trained)
                        
                    except Exception as e:
                        log.error(f"Step D FAILED: Training error: {e}")
                        state.stopped_reason = f"training_error: {e}"
                        break
                    
                    # Record fingerprint with full config for reproducibility
                    # For grid search, we record one fingerprint for the base feature set
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
                        log.info("Step D: Fingerprint recorded")
                    except Exception as e:
                        log.warning(f"Step D: Failed to record fingerprint (non-fatal): {e}")
                
                # Record evolution lineage (use first child model as representative)
                representative_model = state.trained_models[-1]["model_id"] if state.trained_models else child_model_id
                try:
                    await db.insert_evolution_log(
                        log_id=str(uuid.uuid4()),
                        run_id=run_id,
                        parent_model_id=state.current_model_id,
                        child_model_id=representative_model,
                        generation=state.current_generation,
                        parent_sqn=0.0,  # Will be filled in after simulations
                        pruned_features=pruned,
                        remaining_features=remaining,
                        pruning_reason="importance_based"
                    )
                    log.info(f"Evolution lineage recorded: Gen {state.current_generation} ({len([m for m in state.trained_models if m['generation'] == state.current_generation])} models)")
                except Exception as e:
                    log.warning(f"Failed to record lineage (non-fatal): {e}")
                
                # Update state for next iteration - use first model from current generation
                # Feature importance will be similar across grid search variants
                state.current_model_id = representative_model
                state.current_features = remaining
                
            # ================================================================
            # PHASE 2: SIMULATE ALL MODELS
            # ================================================================
            
            # Check if this is train-only mode (no simulations requested)
            if not config.simulation_tickers or len(config.simulation_tickers) == 0:
                log.info("=" * 60)
                log.info("TRAIN-ONLY MODE: Skipping simulations")
                log.info("=" * 60)
                log.info(f"Successfully trained {len(state.trained_models)} models")
                
                # Mark as COMPLETED for train-only mode
                final_reason = state.stopped_reason or "train_only_complete"
                await db.update_evolution_run(
                    run_id,
                    status="COMPLETED",
                    step_status=f"Train-only complete: {len(state.trained_models)} models trained, 0 simulations run. Reason: {final_reason}"
                )
                
                return {
                    "run_id": run_id,
                    "generations_completed": len(state.trained_models) - 1,
                    "models_trained": len(state.trained_models),
                    "final_model_id": state.current_model_id,
                    "promoted": False,  # No promotions in train-only mode
                    "best_sqn": 0.0,
                    "stopped_reason": final_reason
                }
            
            # Safety check: Don't proceed to simulation if no valid models were trained
            if len(state.trained_models) == 0:
                log.error("No valid models to simulate - all models had zero importance")
                state.stopped_reason = "no_valid_models_all_zero_importance"
                await db.update_evolution_run(
                    run_id,
                    status="FAILED",
                    step_status=f"Phase 1 complete - no valid models to simulate: {state.stopped_reason}"
                )
                return {
                    "run_id": run_id,
                    "generations_completed": 0,
                    "models_trained": 0,
                    "final_model_id": state.current_model_id,
                    "promoted": False,
                    "best_sqn": 0.0,
                    "stopped_reason": state.stopped_reason
                }
            
            log.info("=" * 60)
            log.info(f"PHASE 2: SIMULATING ALL MODELS ({len(state.trained_models)} models)")
            log.info("=" * 60)
            
            await db.update_evolution_run(run_id, step_status="Phase 2: Queueing all simulations")
            
            # Queue simulations for all trained models
            total_queued = 0
            for i, model_info in enumerate(state.trained_models):
                model_id = model_info["model_id"]
                generation = model_info["generation"]
                log.info(f"Queueing simulations for model {i+1}/{len(state.trained_models)}: {model_id} (Gen {generation})")
                
                # Temporarily set generation for simulation tagging
                temp_gen = state.current_generation
                state.current_generation = generation
                
                try:
                    num_queued = await self._queue_simulations(state, model_id)
                    total_queued += num_queued
                    log.info(f"  Queued {num_queued} simulations for model {model_id}")
                except Exception as e:
                    log.error(f"  Failed to queue simulations for model {model_id}: {e}")
                
                state.current_generation = temp_gen
            
            log.info(f"Total simulations queued: {total_queued}")
            await db.update_evolution_run(run_id, step_status=f"Phase 2: Waiting for {total_queued} simulations")
            
            # Wait for ALL simulations across ALL models
            log.info("Waiting for all simulations to complete...")
            best_overall_result = None
            best_overall_sqn = -999
            best_overall_model_id = None
            
            # Poll for completion of all simulations
            elapsed = 0.0
            poll_interval = 10.0
            timeout = 7200.0  # 2 hours for all simulations
            
            while elapsed < timeout:
                pending = await db.get_pending_job_count(run_id)
                
                # Update simulation progress
                total_completed = 0
                for model_info in state.trained_models:
                    generation = model_info["generation"]
                    completed_count = await db.get_completed_job_count(run_id, generation)
                    total_completed += completed_count
                
                state.simulations_completed = total_completed
                await db.update_evolution_run(run_id, simulations_completed=total_completed)
                
                if pending == 0:
                    log.info(f"All simulations complete ({total_completed} total)")
                    break
                
                progress_pct = int((total_completed / state.simulations_total) * 100) if state.simulations_total > 0 else 0
                await db.update_evolution_run(
                    run_id, 
                    step_status=f"Phase 2: Simulations {total_completed}/{state.simulations_total} ({progress_pct}%) - {int(elapsed)}s"
                )
                
                log.info(f"Waiting for {pending} simulations... ({total_completed}/{state.simulations_total} completed)")
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
            
            if elapsed >= timeout:
                state.stopped_reason = "simulation_timeout"
                log.warning(f"Simulation phase timeout after {timeout}s")
            
            # Evaluate all results to find best model + config combination
            log.info("Evaluating all simulation results...")
            await db.update_evolution_run(run_id, step_status="Phase 2: Evaluating all results")
            
            for i, model_info in enumerate(state.trained_models):
                model_id = model_info["model_id"]
                generation = model_info["generation"]
                
                # Get all completed jobs for this model's generation
                completed = await db.get_completed_jobs(run_id, generation=generation)
                
                if not completed:
                    log.warning(f"No completed simulations for model {model_id} (Gen {generation})")
                    continue
                
                # Find best result for this model
                for job in completed:
                    result = job.get("result", {})
                    if isinstance(result, str):
                        import json
                        result = json.loads(result)
                    
                    sqn = result.get("sqn", 0)
                    if sqn > best_overall_sqn:
                        best_overall_sqn = sqn
                        best_overall_result = result
                        best_overall_model_id = model_id
                
                log.info(f"Model {i+1}/{len(state.trained_models)} (Gen {generation}): Best SQN = {best_overall_sqn:.2f}")
            
            log.info(f"Overall best result: Model {best_overall_model_id}, SQN = {best_overall_sqn:.2f}")
            
            # Check if best result meets Holy Grail criteria
            if best_overall_result:
                criteria = HolyGrailCriteria(
                    sqn_min=config.sqn_min,
                    sqn_max=config.sqn_max,
                    profit_factor_min=config.profit_factor_min,
                    profit_factor_max=config.profit_factor_max,
                    trade_count_min=config.trade_count_min,
                    trade_count_max=config.trade_count_max
                )
                evaluation = evaluate_holy_grail(best_overall_result, criteria)
                log.info(format_criteria_result(evaluation))
                
                if evaluation.meets_all:
                    state.promoted = True
                    await self._record_promotion(state, best_overall_model_id, best_overall_result, evaluation)
                    log.info(f"🎯 Model {best_overall_model_id} PROMOTED!")
                
                # Update best results
                await db.update_evolution_run(
                    run_id,
                    best_sqn=best_overall_sqn,
                    best_model_id=best_overall_model_id
                )
                
                state.parent_sqn = best_overall_sqn
            
            # Final status update
            final_reason = state.stopped_reason or ("promoted" if state.promoted else "no_promotion")
            await db.update_evolution_run(
                run_id,
                status="COMPLETED" if state.promoted else "FAILED",
                step_status=f"Complete: {len(state.trained_models)} models trained, {state.simulations_completed} simulations run. Reason: {final_reason}"
            )
            
            log.info(f"Evolution run {run_id} finished: {'COMPLETED' if state.promoted else 'FAILED'} (reason: {state.stopped_reason or 'no_promotion'})")
            
            return {
                "run_id": run_id,
                "generations_completed": len(state.trained_models) - 1,  # Subtract initial model
                "models_trained": len(state.trained_models),
                "final_model_id": best_overall_model_id or state.current_model_id,
                "promoted": state.promoted,
                "best_sqn": state.parent_sqn,
                "stopped_reason": state.stopped_reason
            }
        
        except Exception as e:
            log.error(f"Evolution run failed with exception: {e}", exc_info=True)
            await db.update_evolution_run(
                run_id,
                status="FAILED",
                step_status=f"Exception: {str(e)}"
            )
            raise
    
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
    
    async def _get_grid_search_models(self, base_model_id: str) -> List[str]:
        """
        Get all model IDs from a grid search training job.
        For ElasticNet with grid search, the training service creates multiple models.
        Returns list of model IDs in the grid.
        """
        try:
            # Query training service for all models from this training job
            resp = await self.http_client.get(f"{self.training_url}/models")
            resp.raise_for_status()
            models = resp.json()
            
            # Filter models that belong to this grid search
            # The base_model_id is the job_id, grid models have parent_model_id or similar
            # For now, we'll get models that were created around the same time
            # A better approach would be for training service to return grid_models with the base model
            
            # Fallback: if training service doesn't support grid model retrieval,
            # just return the base model
            # TODO: Update training service to return grid models
            log.warning(f"Grid search model retrieval not fully implemented, using base model only")
            return [base_model_id]
            
        except Exception as e:
            log.error(f"Failed to get grid search models for {base_model_id}: {e}")
            return [base_model_id]
    
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
            # ElasticNet grids
            "alpha_grid": config.alpha_grid,
            "l1_ratio_grid": config.l1_ratio_grid,
            # XGBoost grids
            "max_depth_grid": config.max_depth_grid,
            "min_child_weight_grid": config.min_child_weight_grid,
            "reg_lambda_grid": config.reg_lambda_grid,
            "learning_rate_grid": config.learning_rate_grid,
            # LightGBM grids
            "num_leaves_grid": config.num_leaves_grid,
            "min_data_in_leaf_grid": config.min_data_in_leaf_grid,
            "lambda_l2_grid": config.lambda_l2_grid,
            "lgbm_learning_rate_grid": config.lgbm_learning_rate_grid,
            # RandomForest grids
            "rf_max_depth_grid": config.rf_max_depth_grid,
            "min_samples_split_grid": config.min_samples_split_grid,
            "min_samples_leaf_grid": config.min_samples_leaf_grid,
            "n_estimators_grid": config.n_estimators_grid,
            # Control flag to save ALL models from grid
            "save_all_grid_models": True
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
                columns_initial = model.get("columns_initial")
                columns_remaining = model.get("columns_remaining")
                
                # Build status message with column info if available
                status_msg = f"Training: {status}"
                if columns_initial and columns_remaining:
                    dropped = columns_initial - columns_remaining
                    status_msg += f" ({columns_remaining}/{columns_initial} cols, dropped {dropped})"
                status_msg += f" ({int(elapsed)}s)"
                
                if status != last_status:
                    log.info(f"Model {model_id} status: {status}")
                    await db.update_evolution_run(run_id, step_status=status_msg)
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
        Checks simulation fingerprints to avoid duplicate work.
        
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
        skipped_count = 0
        
        # Get model details and compute model fingerprint for simulation fingerprinting
        try:
            model_features = await self._get_model_features(model_id)
            # Parse data_options for train/test window
            data_opts = {}
            if config.data_options:
                import json
                data_opts = json.loads(config.data_options) if isinstance(config.data_options, str) else config.data_options
            train_window = data_opts.get("train_window", 20000)
            test_window = data_opts.get("test_window", 1000)
            
            # Compute MODEL fingerprint (same config = same fingerprint even if retrained)
            model_fingerprint = compute_fingerprint(
                features=model_features,
                hyperparams=config.hyperparameters,
                target_transform=config.target_transform,
                symbol=config.symbol,
                target_col=config.target_col,
                alpha_grid=config.alpha_grid,
                l1_ratio_grid=config.l1_ratio_grid,
                regime_configs=config.regime_configs
            )
            log.info(f"Model fingerprint for simulations: {model_fingerprint[:16]}...")
        except Exception as e:
            log.warning(f"Failed to get model details for fingerprinting: {e}. Queueing all simulations without dedup.")
            model_features = state.current_features
            train_window = 20000
            test_window = 1000
            # Fallback fingerprint
            model_fingerprint = compute_fingerprint(
                features=model_features,
                hyperparams=config.hyperparameters,
                target_transform=config.target_transform,
                symbol=config.symbol,
                target_col=config.target_col,
                alpha_grid=config.alpha_grid,
                l1_ratio_grid=config.l1_ratio_grid,
                regime_configs=config.regime_configs
            )
        
        # Get simulation tickers (defaults to training symbol if not specified)
        sim_tickers = config.get_simulation_tickers()
        
        # Full 4D grid search: tickers × thresholds × z-scores × regimes
        for ticker in sim_tickers:
            for threshold in config.thresholds:
                for z_score in config.z_score_thresholds:
                    for regime_config in config.regime_configs:
                        # Compute simulation fingerprint using MODEL fingerprint
                        # This ensures same model config + same sim params = reusable result
                        sim_fingerprint = compute_simulation_fingerprint(
                            model_fingerprint=model_fingerprint,
                            target_ticker=config.symbol,
                            simulation_ticker=ticker,
                            threshold=threshold,
                            z_score_threshold=z_score,
                            regime_config=regime_config,
                            train_window=train_window,
                            test_window=test_window
                        )
                        
                        # Check if this simulation already ran
                        existing = await db.get_simulation_by_fingerprint(sim_fingerprint)
                        if existing and existing.get("result_sqn") is not None:
                            log.info(f"Simulation fingerprint {sim_fingerprint[:16]}... already exists (SQN={existing.get('result_sqn'):.2f}), skipping")
                            skipped_count += 1
                            continue
                        
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
                                "use_volume_normalization": True,
                                # Store fingerprints for saving after completion
                                "simulation_fingerprint": sim_fingerprint,
                                "model_fingerprint": model_fingerprint,
                                "target_ticker": config.symbol,
                                "train_window": train_window,
                                "test_window": test_window
                            }
                        }
                        jobs.append(job)
        
        count = await db.enqueue_jobs(jobs)
        grid_dims = f"{len(sim_tickers)} tickers × {len(config.thresholds)} thresholds × {len(config.z_score_thresholds)} z-scores × {len(config.regime_configs)} regimes"
        total_possible = len(sim_tickers) * len(config.thresholds) * len(config.z_score_thresholds) * len(config.regime_configs)
        log.info(f"Queued {count} simulation jobs ({grid_dims}), skipped {skipped_count} already tested, with priority {state.parent_sqn:.2f}")
        log.info(f"Fingerprint deduplication: {skipped_count}/{total_possible} simulations reused from cache")
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
        gen_sims_start = state.simulations_completed  # Track sims before this generation
        
        while elapsed < timeout:
            pending = await db.get_pending_job_count(run_id)
            completed_count = await db.get_completed_job_count(run_id, state.current_generation)
            
            if total_jobs is None and (pending > 0 or completed_count > 0):
                total_jobs = pending + completed_count
            
            # Update cumulative simulation counter
            if completed_count > 0:
                state.simulations_completed = gen_sims_start + completed_count
                await db.update_evolution_run(run_id, simulations_completed=state.simulations_completed)
            
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
            log.warning(f"No completed simulations found for generation {state.current_generation} (model {model_id})")
            log.warning(f"Checked after {int(elapsed)}s - pending jobs remaining: {pending}")
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
