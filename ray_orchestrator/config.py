"""
Configuration for Ray Orchestrator.

Environment variables:
- RAY_ADDRESS: Ray cluster address (default: local)
- FEATURES_PARQUET_DIR: Path to feature parquet files
- MODELS_DIR: Path to save model artifacts
- DATABASE_URL: PostgreSQL connection string for metadata
- RAY_DASHBOARD_PORT: Port for Ray dashboard (default: 8265)
- SERVE_PORT: Port for Ray Serve API (default: 8000)
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class RaySettings(BaseSettings):
    """Ray cluster and resource settings."""
    
    # Ray cluster
    ray_address: str = "auto"  # "auto" for local, "ray://<ip>:10001" for cluster
    ray_dashboard_port: int = 8265
    ray_namespace: str = "trading_bot"
    
    # Resource allocation
    num_cpus_per_trial: float = 1.0
    num_gpus_per_trial: float = 0.0
    max_concurrent_trials: int = 8
    
    # Fault tolerance
    max_failures: int = 3
    checkpoint_frequency: int = 5  # Every N iterations
    
    class Config:
        env_prefix = "RAY_"


class DataSettings(BaseSettings):
    """Data paths and loading settings."""
    
    features_parquet_dir: Path = Path("/app/data/features_parquet")
    source_duckdb_path: Path = Path("/app/data/duckdb/local.db")
    models_dir: Path = Path("/app/data/models/ray")
    checkpoints_dir: Path = Path("/app/data/ray_checkpoints")
    
    # Database
    database_url: str = "postgresql://postgres:postgres@db:5432/training"
    
    class Config:
        env_file = ".env"
    
    def ensure_paths(self):
        """Create directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)


class TuneSettings(BaseSettings):
    """Ray Tune hyperparameter search settings."""
    
    # Search algorithm
    search_algorithm: str = "pbt"  # pbt, asha, hyperopt, optuna
    
    # Population-Based Training (PBT)
    pbt_population_size: int = 20
    pbt_perturbation_interval: int = 5  # Generations between perturbations
    pbt_quantile_fraction: float = 0.25  # Bottom 25% replaced
    pbt_resample_probability: float = 0.25
    
    # Early stopping (ASHA)
    asha_max_t: int = 100  # Max training iterations
    asha_grace_period: int = 10  # Min iterations before stopping
    asha_reduction_factor: int = 3
    
    # General
    metric: str = "sharpe_ratio"  # Optimize for this metric
    mode: str = "max"  # "max" or "min"
    num_samples: int = 50  # Total trials
    
    # Deduplication settings
    skip_duplicate: bool = True  # Skip configs already tested (fingerprint)
    float_precision: int = 5  # Decimal places for float hashing
    use_fingerprint_db: bool = True  # Use SQLite fingerprint database
    
    # Experiment resuming
    resume_errored: bool = True  # Resume only errored/unfinished trials
    resume_unfinished: bool = True  # Continue unfinished experiments
    
    class Config:
        env_prefix = "TUNE_"


class ServeSettings(BaseSettings):
    """Ray Serve deployment settings."""
    
    serve_port: int = 8000
    serve_host: str = "0.0.0.0"
    
    # Ensemble settings
    ensemble_voting: str = "soft"  # "hard" or "soft"
    ensemble_threshold: float = 0.7  # Confidence threshold for soft voting
    ensemble_min_models: int = 3
    
    # Autoscaling
    min_replicas: int = 1
    max_replicas: int = 10
    target_num_ongoing_requests_per_replica: int = 5
    
    class Config:
        env_prefix = "SERVE_"


class Settings(BaseSettings):
    """Main settings combining all sub-settings."""
    
    ray: RaySettings = RaySettings()
    data: DataSettings = DataSettings()
    tune: TuneSettings = TuneSettings()
    serve: ServeSettings = ServeSettings()
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8100
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"


settings = Settings()
