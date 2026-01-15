"""
Fingerprint Database for Deduplication.

Provides a SQLite-backed system for tracking completed training configurations,
enabling the "skip duplicate" feature across experiments and sessions.

This is the "Global Fingerprint" approach for complex multi-generational setups
where you're swapping datasets and tickers.

Usage:
    fp = FingerprintDB()
    
    # Before heavy training
    if fp.exists(config):
        return fp.get_cached_result(config)
    
    # After training
    fp.record(config, metrics)
"""

import hashlib
import json
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
from contextlib import contextmanager

from .config import settings

log = logging.getLogger("ray_orchestrator.fingerprint")


class FingerprintDB:
    """
    SQLite database for tracking completed training configurations.
    
    Creates a fingerprint (hash) of the config dict, allowing us to
    skip already-tested configurations across experiments.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize fingerprint database.
        
        Args:
            db_path: Path to SQLite database (defaults to checkpoints_dir/fingerprints.db)
        """
        self.db_path = db_path or settings.data.checkpoints_dir / "fingerprints.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.precision = settings.tune.float_precision
        self._init_db()
    
    def _init_db(self):
        """Create the fingerprints table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    fingerprint TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    result_json TEXT,
                    experiment_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'completed'
                )
            """)
            # Index for quick lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment 
                ON fingerprints(experiment_name)
            """)
            conn.commit()
    
    @contextmanager
    def _connect(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def generate_fingerprint(self, config: dict) -> str:
        """
        Generate a fingerprint (hash) for a config dictionary.
        
        Handles:
        - Nested dicts (flattened to key/value pairs)
        - Float precision (rounds to N decimal places)
        - Consistent ordering (sorted keys)
        
        Args:
            config: Configuration dictionary
            
        Returns:
            SHA256 hash string
        """
        # Normalize the config
        normalized = self._normalize_config(config)
        
        # Create deterministic JSON string
        config_str = json.dumps(normalized, sort_keys=True, default=str)
        
        # Hash it
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _normalize_config(self, config: dict, prefix: str = "") -> dict:
        """
        Normalize config for consistent hashing.
        
        - Flattens nested dicts: {"params": {"lr": 0.1}} -> {"params/lr": 0.1}
        - Rounds floats to specified precision
        - Sorts arrays
        """
        normalized = {}
        
        for key, value in config.items():
            full_key = f"{prefix}/{key}" if prefix else key
            
            if isinstance(value, dict):
                # Flatten nested dicts
                nested = self._normalize_config(value, full_key)
                normalized.update(nested)
            elif isinstance(value, float):
                # Round floats to precision
                normalized[full_key] = round(value, self.precision)
            elif isinstance(value, list):
                # Sort lists for consistency
                normalized[full_key] = sorted([
                    round(v, self.precision) if isinstance(v, float) else v
                    for v in value
                ])
            else:
                normalized[full_key] = value
        
        return normalized
    
    def exists(self, config: dict) -> bool:
        """
        Check if a config has already been tested.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if fingerprint exists in database
        """
        fingerprint = self.generate_fingerprint(config)
        
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM fingerprints WHERE fingerprint = ?",
                (fingerprint,)
            )
            return cursor.fetchone() is not None
    
    def get_cached_result(self, config: dict) -> Optional[dict]:
        """
        Get cached result for a config if it exists.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Result dict if exists, None otherwise
        """
        fingerprint = self.generate_fingerprint(config)
        
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT result_json FROM fingerprints WHERE fingerprint = ?",
                (fingerprint,)
            )
            row = cursor.fetchone()
            
            if row and row["result_json"]:
                return json.loads(row["result_json"])
        
        return None
    
    def record(
        self,
        config: dict,
        result: Optional[dict] = None,
        experiment_name: Optional[str] = None,
        status: str = "completed"
    ):
        """
        Record a completed configuration.
        
        Args:
            config: Configuration dictionary
            result: Result metrics dict
            experiment_name: Name of the experiment
            status: Trial status (completed, failed, running)
        """
        fingerprint = self.generate_fingerprint(config)
        config_json = json.dumps(config, default=str)
        result_json = json.dumps(result, default=str) if result else None
        
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO fingerprints 
                (fingerprint, config_json, result_json, experiment_name, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                fingerprint,
                config_json,
                result_json,
                experiment_name,
                datetime.utcnow().isoformat(),
                status
            ))
            conn.commit()
        
        log.debug(f"Recorded fingerprint {fingerprint} for {experiment_name}")
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    COUNT(DISTINCT experiment_name) as experiments
                FROM fingerprints
            """)
            row = cursor.fetchone()
            
            return {
                "total_fingerprints": row["total"],
                "completed": row["completed"],
                "failed": row["failed"],
                "experiments": row["experiments"]
            }
    
    def get_experiment_fingerprints(self, experiment_name: str) -> list[str]:
        """Get all fingerprints for an experiment."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT fingerprint FROM fingerprints WHERE experiment_name = ?",
                (experiment_name,)
            )
            return [row["fingerprint"] for row in cursor.fetchall()]
    
    def delete_experiment(self, experiment_name: str):
        """Delete all fingerprints for an experiment."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM fingerprints WHERE experiment_name = ?",
                (experiment_name,)
            )
            conn.commit()
    
    def clear_all(self):
        """Clear all fingerprints (use with caution)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM fingerprints")
            conn.commit()
        log.warning("Cleared all fingerprints from database")


def check_and_skip_duplicate(
    config: dict,
    experiment_name: Optional[str] = None
) -> Optional[dict]:
    """
    Helper function for use in training objectives.
    
    Call this at the start of your train function to check if
    this config has already been tested.
    
    Example:
        def train_trading_model(config):
            cached = check_and_skip_duplicate(config)
            if cached:
                # Report cached results and return early
                ray.train.report(cached)
                return
            
            # ... heavy training logic ...
    
    Args:
        config: Training configuration
        experiment_name: Optional experiment name
        
    Returns:
        Cached result dict if exists, None otherwise
    """
    if not settings.tune.use_fingerprint_db:
        return None
    
    fp = FingerprintDB()
    
    if fp.exists(config):
        result = fp.get_cached_result(config)
        if result:
            log.info(f"Skipping duplicate config (fingerprint: {fp.generate_fingerprint(config)[:8]}...)")
            return result
    
    return None


def record_trial_result(
    config: dict,
    result: dict,
    experiment_name: Optional[str] = None
):
    """
    Record a trial result to the fingerprint database.
    
    Call this after training completes successfully.
    
    Args:
        config: Training configuration
        result: Result metrics
        experiment_name: Optional experiment name
    """
    if not settings.tune.use_fingerprint_db:
        return
    
    fp = FingerprintDB()
    fp.record(config, result, experiment_name)


# Global instance for convenience
fingerprint_db = FingerprintDB()
