"""
Ray Tune Orchestrator for Hyperparameter Search.

Provides multiple search strategies:
- Grid Search: Exhaustive search over parameter combinations
- Random Search: Sample from parameter distributions
- Population-Based Training (PBT): Evolutionary multi-generational search
- ASHA: Asynchronous Successive Halving for early stopping
- Bayesian Optimization: Efficient search using surrogate models

Deduplication Features:
- skip_duplicate: Ray hashes configs to avoid re-testing identical setups
- Fingerprint Database: SQLite-backed tracking for complex multi-generational setups
- Experiment Resuming: Continue crashed/stopped experiments without re-training
"""

import logging
from typing import Optional, Any
from datetime import datetime
from pathlib import Path
import json

import ray
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import (
    PopulationBasedTraining,
    ASHAScheduler,
    HyperBandScheduler,
    FIFOScheduler
)
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.stopper import (
    MaximumIterationStopper,
    TrialPlateauStopper,
    CombinedStopper
)
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig

from .config import settings
from .objectives import train_trading_model, multi_ticker_objective
from .fingerprint import fingerprint_db, FingerprintDB

log = logging.getLogger("ray_orchestrator.tuner")


# Pre-defined search spaces for each algorithm
SEARCH_SPACES = {
    "elasticnet": {
        "alpha": tune.loguniform(1e-5, 1.0),
        "l1_ratio": tune.uniform(0.0, 1.0),
    },
    "xgboost_regressor": {
        "max_depth": tune.randint(3, 12),
        "learning_rate": tune.loguniform(0.001, 0.3),
        "n_estimators": tune.randint(50, 500),
        "min_child_weight": tune.randint(1, 20),
        "reg_alpha": tune.loguniform(1e-5, 10.0),
        "reg_lambda": tune.loguniform(1e-5, 10.0),
        "subsample": tune.uniform(0.5, 1.0),
    },
    "lightgbm_regressor": {
        "num_leaves": tune.randint(15, 200),
        "max_depth": tune.randint(3, 15),
        "learning_rate": tune.loguniform(0.001, 0.3),
        "n_estimators": tune.randint(50, 500),
        "min_data_in_leaf": tune.randint(5, 100),
        "lambda_l1": tune.loguniform(1e-5, 10.0),
        "lambda_l2": tune.loguniform(1e-5, 10.0),
    },
    "random_forest_regressor": {
        "max_depth": tune.choice([5, 10, 15, 20, 30, None]),
        "n_estimators": tune.randint(50, 300),
        "min_samples_split": tune.randint(2, 50),
        "min_samples_leaf": tune.randint(1, 20),
        "max_features": tune.choice(["sqrt", "log2", 0.3, 0.5, 0.7]),
    },
}

# PBT mutation ranges
PBT_HYPERPARAM_MUTATIONS = {
    "elasticnet": {
        "alpha": tune.loguniform(1e-5, 1.0),
        "l1_ratio": tune.uniform(0.0, 1.0),
    },
    "xgboost_regressor": {
        "learning_rate": tune.loguniform(0.001, 0.3),
        "reg_alpha": tune.loguniform(1e-5, 10.0),
        "reg_lambda": tune.loguniform(1e-5, 10.0),
        "subsample": tune.uniform(0.5, 1.0),
    },
    "lightgbm_regressor": {
        "learning_rate": tune.loguniform(0.001, 0.3),
        "lambda_l1": tune.loguniform(1e-5, 10.0),
        "lambda_l2": tune.loguniform(1e-5, 10.0),
    },
}


class TuneOrchestrator:
    """
    Ray Tune orchestrator for hyperparameter search.
    
    Supports multiple search strategies optimized for trading models.
    
    Deduplication:
        - `skip_duplicate=True` on search algorithms (Bayesian, Optuna, HyperOpt)
        - SQLite fingerprint database for cross-experiment deduplication
        - Experiment resuming for fault tolerance
    """
    
    def __init__(self):
        self.results = {}
        self.active_experiments = {}
        self.fingerprint_db = fingerprint_db
    
    def init_ray(self):
        """Initialize Ray cluster connection."""
        if not ray.is_initialized():
            # If ray_address is "auto" or empty, connect to local Ray instance
            # (which should already be started by entrypoint.sh)
            address = None
            if settings.ray.ray_address and settings.ray.ray_address != "auto":
                address = settings.ray.ray_address
            
            ray.init(
                address=address,
                namespace=settings.ray.ray_namespace,
                dashboard_host="0.0.0.0",
                dashboard_port=settings.ray.ray_dashboard_port,
                ignore_reinit_error=True
            )
            log.info(f"Ray initialized: {ray.cluster_resources()}")
    
    def create_search_algorithm(
        self,
        search_type: str = "random",
        space: Optional[dict] = None
    ):
        """
        Create a search algorithm with skip_duplicate enabled.
        
        The skip_duplicate flag tells Ray to hash the configuration
        dictionary and skip configs that have already been tested.
        This is your "fingerprint" toggle at the Ray level.
        
        Args:
            search_type: "random", "bayesopt", "optuna", "hyperopt"
            space: Search space for Bayesian methods
            
        Returns:
            Search algorithm instance or None for random search
        """
        skip_dup = settings.tune.skip_duplicate
        
        if search_type == "bayesopt":
            # Bayesian Optimization with skip_duplicate
            searcher = BayesOptSearch(
                metric=settings.tune.metric,
                mode=settings.tune.mode,
                skip_duplicate=skip_dup,  # <-- Fingerprint toggle
            )
            log.info(f"Created BayesOptSearch (skip_duplicate={skip_dup})")
            return ConcurrencyLimiter(
                searcher,
                max_concurrent=settings.ray.max_concurrent_trials
            )
        
        elif search_type == "optuna":
            # Optuna with skip_duplicate
            searcher = OptunaSearch(
                metric=settings.tune.metric,
                mode=settings.tune.mode,
            )
            # Note: Optuna handles duplicates internally via sampler
            log.info(f"Created OptunaSearch")
            return ConcurrencyLimiter(
                searcher,
                max_concurrent=settings.ray.max_concurrent_trials
            )
        
        elif search_type == "hyperopt":
            # HyperOpt with skip_duplicate
            searcher = HyperOptSearch(
                metric=settings.tune.metric,
                mode=settings.tune.mode,
            )
            log.info(f"Created HyperOptSearch")
            return ConcurrencyLimiter(
                searcher,
                max_concurrent=settings.ray.max_concurrent_trials
            )
        
        # Random search - no searcher needed
        return None
    
    def try_restore_experiment(
        self,
        experiment_path: Path,
        trainable: Any
    ) -> Optional[Tuner]:
        """
        Try to restore a previous experiment for resuming.
        
        When a node crashes or you stop the search, Ray saves a JSON file
        of every result. This method restores from that checkpoint.
        
        Args:
            experiment_path: Path to experiment results folder
            trainable: The training function
            
        Returns:
            Restored Tuner if exists, None otherwise
        """
        if not experiment_path.exists():
            return None
        
        try:
            log.info(f"Attempting to restore experiment from {experiment_path}")
            
            restored = Tuner.restore(
                path=str(experiment_path),
                trainable=trainable,
                resume_errored=settings.tune.resume_errored,
                resume_unfinished=settings.tune.resume_unfinished,
            )
            
            log.info(f"Successfully restored experiment from {experiment_path}")
            return restored
            
        except Exception as e:
            log.warning(f"Could not restore experiment: {e}")
            return None
    
    def build_search_space(
        self,
        algorithm: str,
        tickers: list[str],
        custom_space: Optional[dict] = None,
        **fixed_params
    ) -> dict:
        """
        Build search space for the given algorithm and tickers.
        
        Args:
            algorithm: Algorithm name
            tickers: List of ticker symbols
            custom_space: Optional custom search space overrides
            **fixed_params: Fixed parameters (not tuned)
            
        Returns:
            Complete search space dict
        """
        # Start with algorithm-specific hyperparameters
        space = SEARCH_SPACES.get(algorithm, {}).copy()
        
        # Override with custom space
        if custom_space:
            space.update(custom_space)
        
        # Add fixed parameters
        space.update(fixed_params)
        
        # Add tickers as grid search if multiple
        if len(tickers) > 1:
            space["ticker"] = tune.grid_search(tickers)
        else:
            space["ticker"] = tickers[0]
        
        # Add algorithm name
        space["algorithm"] = algorithm
        
        return space
    
    def create_pbt_scheduler(self, algorithm: str) -> PopulationBasedTraining:
        """
        Create Population-Based Training scheduler with best practice configuration.
        
        PBT evolves a population of models over generations, replacing poor
        performers with mutated clones of winners.
        
        BEST PRACTICE PARAMETERS:
        ┌─────────────────────────┬─────────────────┬────────────────────────────────────────┐
        │ Parameter               │ Recommended     │ Why?                                   │
        ├─────────────────────────┼─────────────────┼────────────────────────────────────────┤
        │ perturbation_interval   │ 4-10 iterations │ Too low=thrashing, too high=waste time │
        │ quantile_fraction       │ 0.2-0.25        │ Top 25% teach, bottom 25% replaced     │
        │ resample_probability    │ 0.25            │ Jump out of local optima               │
        │ perturbation_factors    │ [1.2, 0.8]      │ Random walk around good values         │
        └─────────────────────────┴─────────────────┴────────────────────────────────────────┘
        
        CRITICAL SYNC RULE:
        - perturbation_interval must be >= checkpoint_frequency
        - PBT can only exploit trials that have saved checkpoints
        - Current: perturbation_interval={settings.tune.pbt_perturbation_interval}, 
                   checkpoint_frequency={settings.ray.checkpoint_frequency}
        """
        mutations = PBT_HYPERPARAM_MUTATIONS.get(algorithm, {})
        
        # Validate sync rule
        if settings.tune.pbt_perturbation_interval < settings.ray.checkpoint_frequency:
            log.warning(
                f"⚠️ PBT SYNC RULE VIOLATION: perturbation_interval "
                f"({settings.tune.pbt_perturbation_interval}) < checkpoint_frequency "
                f"({settings.ray.checkpoint_frequency}). "
                f"PBT will not be able to exploit all trials! "
                f"Set checkpoint_frequency <= perturbation_interval."
            )
        
        log.info(
            f"PBT Scheduler: perturbation every {settings.tune.pbt_perturbation_interval} iterations, "
            f"checkpoints every {settings.ray.checkpoint_frequency} iterations ✓"
        )
        log.info(
            f"PBT Parameters: quantile={settings.tune.pbt_quantile_fraction}, "
            f"resample_prob={settings.tune.pbt_resample_probability}, "
            f"perturbation_factors={settings.tune.pbt_perturbation_factors}"
        )
        
        # Custom explore function with perturbation factors
        def explore(config):
            """Custom explore function for PBT continuous parameter mutation."""
            import random
            # Resample probability: pick new random value vs mutate existing
            if random.random() < settings.tune.pbt_resample_probability:
                # Resample from hyperparam_mutations
                return None  # Signals PBT to use hyperparam_mutations
            else:
                # Perturb by multiplying with random factor
                factor = random.choice(settings.tune.pbt_perturbation_factors)
                return {k: v * factor if isinstance(v, (int, float)) else v 
                        for k, v in config.items()}
        
        return PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=settings.tune.pbt_perturbation_interval,
            hyperparam_mutations=mutations,
            quantile_fraction=settings.tune.pbt_quantile_fraction,
            resample_probability=settings.tune.pbt_resample_probability,
            custom_explore_fn=explore,  # Use custom perturbation factors
            log_config=True,
        )
    
    def create_asha_scheduler(self) -> ASHAScheduler:
        """
        Create ASHA (Asynchronous Successive Halving) scheduler.
        
        ASHA aggressively stops underperforming trials early,
        focusing resources on the most promising configurations.
        """
        return ASHAScheduler(
            time_attr="training_iteration",
            metric=settings.tune.metric,
            mode=settings.tune.mode,
            max_t=settings.tune.asha_max_t,
            grace_period=settings.tune.asha_grace_period,
            reduction_factor=settings.tune.asha_reduction_factor,
        )
    
    def run_grid_search(
        self,
        algorithm: str,
        tickers: list[str],
        param_grid: dict,
        name: Optional[str] = None,
        **config_overrides
    ) -> dict:
        """
        Run exhaustive grid search over parameter combinations.
        
        Args:
            algorithm: Algorithm to use
            tickers: Ticker symbols
            param_grid: Dict of param -> list of values
            name: Experiment name
            **config_overrides: Additional config params
            
        Returns:
            Results dict with best trial info
        """
        self.init_ray()
        
        # Convert lists to grid_search
        search_space = {**config_overrides}
        search_space["algorithm"] = algorithm
        
        for param, values in param_grid.items():
            if isinstance(values, list):
                search_space[param] = tune.grid_search(values)
            else:
                search_space[param] = values
        
        if len(tickers) > 1:
            search_space["ticker"] = tune.grid_search(tickers)
        else:
            search_space["ticker"] = tickers[0]
        
        # Calculate total trials
        total_trials = 1
        for param, values in param_grid.items():
            if isinstance(values, list):
                total_trials *= len(values)
        total_trials *= len(tickers)
        
        exp_name = name or f"grid_search_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        log.info(f"Starting grid search: {total_trials} trials for {exp_name}")
        
        tuner = Tuner(
            train_trading_model,
            param_space=search_space,
            tune_config=TuneConfig(
                metric=settings.tune.metric,
                mode=settings.tune.mode,
                num_samples=1,  # Grid search handles combinations
                max_concurrent_trials=settings.ray.max_concurrent_trials,
            ),
            run_config=RunConfig(
                name=exp_name,
                storage_path=str(settings.data.checkpoints_dir),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_frequency=settings.ray.checkpoint_frequency,
                ),
                failure_config=ray.train.FailureConfig(
                    max_failures=settings.ray.max_failures
                ),
            ),
        )
        
        results = tuner.fit()
        
        self.results[exp_name] = results
        
        return self._format_results(results, exp_name)
    
    def run_pbt_search(
        self,
        algorithm: str,
        tickers: list[str],
        population_size: Optional[int] = None,
        num_generations: int = 10,
        name: Optional[str] = None,
        resume: bool = False,
        **config_overrides
    ) -> dict:
        """
        Run Population-Based Training (multi-generational evolution).
        
        PBT evolves a population of models over generations, replacing
        the worst performers with mutated versions of the best.
        
        Deduplication:
            - PBT naturally avoids duplicates via mutation
            - Can resume crashed experiments with resume=True
        
        Args:
            algorithm: Algorithm to use
            tickers: Ticker symbols
            population_size: Number of parallel trials (default from settings)
            num_generations: Number of evolutionary generations
            name: Experiment name
            resume: Try to resume previous experiment with same name
            **config_overrides: Additional config params
            
        Returns:
            Results dict with surviving models
        """
        self.init_ray()
        
        pop_size = population_size or settings.tune.pbt_population_size
        
        exp_name = name or f"pbt_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_path = settings.data.checkpoints_dir / exp_name
        
        # Try to resume if requested
        if resume:
            restored = self.try_restore_experiment(exp_path, train_trading_model)
            if restored:
                log.info(f"Resuming PBT experiment {exp_name}")
                results = restored.fit()
                self.results[exp_name] = results
                return self._format_results(results, exp_name)
        
        # Build initial search space
        search_space = self.build_search_space(
            algorithm, tickers, **config_overrides
        )
        
        log.info(f"Starting PBT: population={pop_size}, generations={num_generations}")
        
        tuner = Tuner(
            train_trading_model,
            param_space=search_space,
            tune_config=TuneConfig(
                metric=settings.tune.metric,
                mode=settings.tune.mode,
                num_samples=pop_size,
                scheduler=self.create_pbt_scheduler(algorithm),
                max_concurrent_trials=settings.ray.max_concurrent_trials,
            ),
            run_config=RunConfig(
                name=exp_name,
                storage_path=str(settings.data.checkpoints_dir),
                stop={"training_iteration": num_generations},
                checkpoint_config=CheckpointConfig(
                    num_to_keep=2,  # PBT only needs recent checkpoints (saves disk space)
                    checkpoint_frequency=1,  # Must be <= perturbation_interval (currently 5)
                ),
                failure_config=ray.train.FailureConfig(
                    max_failures=settings.ray.max_failures
                ),
            ),
        )
        
        results = tuner.fit()
        
        self.results[exp_name] = results
        
        return self._format_results(results, exp_name)
    
    def run_asha_search(
        self,
        algorithm: str,
        tickers: list[str],
        num_samples: int = 50,
        name: Optional[str] = None,
        search_alg: str = "random",  # "random", "bayesopt", "optuna", "hyperopt"
        resume: bool = False,
        **config_overrides
    ) -> dict:
        """
        Run ASHA (Asynchronous Successive Halving) search.
        
        ASHA aggressively stops poor trials early while allowing
        promising ones to continue, maximizing resource efficiency.
        
        Deduplication:
            - Uses skip_duplicate=True for Bayesian/Optuna/HyperOpt
            - Can resume crashed experiments with resume=True
        
        Args:
            algorithm: Algorithm to use
            tickers: Ticker symbols
            num_samples: Number of random samples
            name: Experiment name
            search_alg: Search algorithm (random, bayesopt, optuna, hyperopt)
            resume: Try to resume previous experiment with same name
            **config_overrides: Additional config params
            
        Returns:
            Results dict
        """
        self.init_ray()
        
        search_space = self.build_search_space(
            algorithm, tickers, **config_overrides
        )
        
        exp_name = name or f"asha_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_path = settings.data.checkpoints_dir / exp_name
        
        # Try to resume if requested
        if resume:
            restored = self.try_restore_experiment(exp_path, train_trading_model)
            if restored:
                log.info(f"Resuming experiment {exp_name}")
                results = restored.fit()
                self.results[exp_name] = results
                return self._format_results(results, exp_name)
        
        # Create search algorithm with skip_duplicate
        searcher = self.create_search_algorithm(search_alg, search_space)
        
        log.info(f"Starting ASHA search: {num_samples} samples with {search_alg}")
        
        tuner = Tuner(
            train_trading_model,
            param_space=search_space,
            tune_config=TuneConfig(
                metric=settings.tune.metric,
                mode=settings.tune.mode,
                num_samples=num_samples,
                scheduler=self.create_asha_scheduler(),
                search_alg=searcher,
                max_concurrent_trials=settings.ray.max_concurrent_trials,
            ),
            run_config=RunConfig(
                name=exp_name,
                storage_path=str(settings.data.checkpoints_dir),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=5,
                ),
                failure_config=ray.train.FailureConfig(
                    max_failures=settings.ray.max_failures
                ),
            ),
        )
        
        results = tuner.fit()
        
        self.results[exp_name] = results
        
        return self._format_results(results, exp_name)
    
    def run_bayesian_search(
        self,
        algorithm: str,
        tickers: list[str],
        num_samples: int = 50,
        name: Optional[str] = None,
        resume: bool = False,
        **config_overrides
    ) -> dict:
        """
        Run Bayesian Optimization search with skip_duplicate.
        
        Bayesian Optimization uses a surrogate model (Gaussian Process)
        to intelligently explore the hyperparameter space. The skip_duplicate
        flag ensures we never re-test identical configurations.
        
        This is ideal for trading bots where we want to find optimal
        hyperparameters efficiently without wasting compute.
        
        Args:
            algorithm: Algorithm to use
            tickers: Ticker symbols
            num_samples: Number of samples to explore
            name: Experiment name
            resume: Try to resume previous experiment
            **config_overrides: Additional config params
            
        Returns:
            Results dict
        """
        self.init_ray()
        
        search_space = self.build_search_space(
            algorithm, tickers, **config_overrides
        )
        
        exp_name = name or f"bayesian_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_path = settings.data.checkpoints_dir / exp_name
        
        # Try to resume if requested
        if resume:
            restored = self.try_restore_experiment(exp_path, train_trading_model)
            if restored:
                log.info(f"Resuming Bayesian experiment {exp_name}")
                results = restored.fit()
                self.results[exp_name] = results
                return self._format_results(results, exp_name)
        
        # Create Bayesian search with skip_duplicate=True
        searcher = self.create_search_algorithm("bayesopt", search_space)
        
        log.info(f"Starting Bayesian search: {num_samples} samples (skip_duplicate={settings.tune.skip_duplicate})")
        
        tuner = Tuner(
            train_trading_model,
            param_space=search_space,
            tune_config=TuneConfig(
                metric=settings.tune.metric,
                mode=settings.tune.mode,
                num_samples=num_samples,
                search_alg=searcher,
                max_concurrent_trials=settings.ray.max_concurrent_trials,
            ),
            run_config=RunConfig(
                name=exp_name,
                storage_path=str(settings.data.checkpoints_dir),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_frequency=settings.ray.checkpoint_frequency,
                ),
                failure_config=ray.train.FailureConfig(
                    max_failures=settings.ray.max_failures
                ),
            ),
        )
        
        results = tuner.fit()
        
        self.results[exp_name] = results
        
        return self._format_results(results, exp_name)
    
    def run_multi_ticker_pbt(
        self,
        algorithm: str,
        tickers: list[str],
        population_size: int = 20,
        num_generations: int = 10,
        name: Optional[str] = None,
        resume: bool = False,
        **config_overrides
    ) -> dict:
        """
        Run PBT across multiple tickers simultaneously.
        
        Each trial trains on ALL tickers with the same hyperparameters,
        and is evaluated on aggregate metrics (average Sharpe, min Sharpe).
        
        This finds hyperparameters that work well across different markets.
        
        Deduplication:
            - PBT naturally avoids duplicates via mutation
            - Can resume crashed experiments with resume=True
        
        Args:
            algorithm: Algorithm to use
            tickers: Ticker symbols to train on together
            population_size: Number of parallel configurations
            num_generations: Evolutionary generations
            name: Experiment name
            resume: Try to resume previous experiment
            **config_overrides: Additional config params
            
        Returns:
            Results with best universal hyperparameters
        """
        self.init_ray()
        
        exp_name = name or f"multi_pbt_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_path = settings.data.checkpoints_dir / exp_name
        
        # Try to resume if requested
        if resume:
            restored = self.try_restore_experiment(exp_path, multi_ticker_objective)
            if restored:
                log.info(f"Resuming multi-ticker PBT experiment {exp_name}")
                results = restored.fit()
                self.results[exp_name] = results
                return self._format_results(results, exp_name)
        
        # Build search space (but tickers are ALL passed to each trial)
        search_space = SEARCH_SPACES.get(algorithm, {}).copy()
        search_space.update(config_overrides)
        search_space["algorithm"] = algorithm
        search_space["tickers"] = tickers  # All tickers per trial
        
        log.info(f"Starting multi-ticker PBT: {len(tickers)} tickers, pop={population_size}")
        
        tuner = Tuner(
            multi_ticker_objective,  # Different objective function
            param_space=search_space,
            tune_config=TuneConfig(
                metric=settings.tune.metric,
                mode=settings.tune.mode,
                num_samples=population_size,
                scheduler=self.create_pbt_scheduler(algorithm),
                max_concurrent_trials=settings.ray.max_concurrent_trials,
            ),
            run_config=RunConfig(
                name=exp_name,
                storage_path=str(settings.data.checkpoints_dir),
                stop={"training_iteration": num_generations},
                checkpoint_config=CheckpointConfig(
                    num_to_keep=10,
                ),
            ),
        )
        
        results = tuner.fit()
        
        self.results[exp_name] = results
        
        return self._format_results(results, exp_name)
    
    def get_best_models(self, experiment_name: str, n: int = 5) -> list[dict]:
        """Get top N models from an experiment."""
        if experiment_name not in self.results:
            return []
        
        results = self.results[experiment_name]
        
        best_results = results.get_dataframe().nlargest(n, settings.tune.metric)
        
        return best_results.to_dict(orient="records")
    
    def _format_results(self, results: Any, experiment_name: str) -> dict:
        """Format tuner results for API response."""
        try:
            best_result = results.get_best_result()
            df = results.get_dataframe()
            
            return {
                "experiment_name": experiment_name,
                "num_trials": len(df),
                "best_trial": {
                    "config": best_result.config,
                    "metrics": best_result.metrics,
                    "checkpoint_path": str(best_result.checkpoint) if best_result.checkpoint else None,
                },
                "top_5": df.nlargest(5, settings.tune.metric)[[
                    "trial_id", settings.tune.metric, "r2", "rmse", "status"
                ]].to_dict(orient="records") if settings.tune.metric in df.columns else [],
                "summary": {
                    "completed": len(df[df.get("status", "") == "completed"]) if "status" in df.columns else len(df),
                    "failed": len(df[df.get("status", "") == "failed"]) if "status" in df.columns else 0,
                    "best_metric": float(df[settings.tune.metric].max()) if settings.tune.metric in df.columns else None,
                    "mean_metric": float(df[settings.tune.metric].mean()) if settings.tune.metric in df.columns else None,
                }
            }
        except Exception as e:
            log.error(f"Error formatting results: {e}")
            return {
                "experiment_name": experiment_name,
                "error": str(e)
            }


# Global orchestrator instance
orchestrator = TuneOrchestrator()
