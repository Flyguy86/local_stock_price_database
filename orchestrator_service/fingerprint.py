"""
Fingerprint calculation for model deduplication.
Creates a deterministic hash from model configuration.
"""
import hashlib
import json
from typing import Dict, Any, List


def compute_fingerprint(
    features: List[str],
    hyperparams: Dict[str, Any],
    target_transform: str,
    symbol: str,
    target_col: str = "close",
    alpha_grid: List[float] = None,
    l1_ratio_grid: List[float] = None,
    regime_configs: List[Dict[str, Any]] = None,
    timeframe: str = "1m",
    train_window: int = 20000,
    test_window: int = 1000,
    context_symbols: List[str] = None,
    cv_folds: int = 5,
    cv_strategy: str = "time_series_split"
) -> str:
    """
    Compute SHA-256 fingerprint from model configuration.
    
    The fingerprint uniquely identifies a model configuration:
    - Same features + params + transform + symbol = same fingerprint
    - Any difference = different fingerprint
    
    Includes ALL factors that affect model training:
    - Features: Which columns are used
    - Hyperparameters: Algorithm settings
    - Target: Column, transform (log_return vs pct_change)
    - Symbol: Training ticker
    - Data: Timeframe, train/test windows
    - Context: Additional symbols for feature engineering
    - Cross-validation: Folds and strategy
    - Grid search: Alpha, L1 ratio, regime configs
    
    Args:
        features: List of feature column names
        hyperparams: Model hyperparameters dict
        target_transform: Target transformation (none, log_return, pct_change, log)
        symbol: Trading symbol
        target_col: Target column name
        alpha_grid: Grid search values for alpha (L2 penalty)
        l1_ratio_grid: Grid search values for l1_ratio (L1/L2 mix)
        regime_configs: Grid search regime filter configurations
        timeframe: Bar timeframe (1m, 5m, 1h, etc.)
        train_window: Training window size in bars
        test_window: Test window size in bars
        context_symbols: Additional symbols used for feature engineering
        cv_folds: Number of cross-validation folds
        cv_strategy: CV strategy (time_series_split, walk_forward, etc.)
    
    Returns:
        64-character hex SHA-256 hash
    """
    # Normalize the payload for consistent hashing
    payload = {
        "features": sorted(features),
        "hyperparams": _normalize_params(hyperparams),
        "target_transform": target_transform,
        "symbol": symbol.upper(),
        "target_col": target_col,
        "timeframe": timeframe,
        "train_window": train_window,
        "test_window": test_window,
        "context_symbols": sorted(context_symbols) if context_symbols else [],
        "cv_folds": cv_folds,
        "cv_strategy": cv_strategy,
        "alpha_grid": sorted(alpha_grid) if alpha_grid else None,
        "l1_ratio_grid": sorted(l1_ratio_grid) if l1_ratio_grid else None,
        "regime_configs": _normalize_regime_configs(regime_configs) if regime_configs else None
    }
    
    # Create deterministic JSON string
    json_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    
    # Compute SHA-256
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize hyperparameters for consistent hashing.
    - Sort keys
    - Convert floats to consistent precision
    - Remove None values
    """
    normalized = {}
    for key in sorted(params.keys()):
        value = params[key]
        if value is None:
            continue
        if isinstance(value, float):
            # Round to 8 decimal places for consistency
            normalized[key] = round(value, 8)
        elif isinstance(value, dict):
            normalized[key] = _normalize_params(value)
        elif isinstance(value, list):
            normalized[key] = sorted(value) if all(isinstance(x, (str, int, float)) for x in value) else value
        else:
            normalized[key] = value
    return normalized


def _normalize_regime_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize regime configs for consistent hashing.
    - Sort by keys
    - Sort regime value lists
    - Sort config list by JSON representation
    """
    normalized = []
    for cfg in configs:
        norm_cfg = {}
        for key in sorted(cfg.keys()):
            value = cfg[key]
            if isinstance(value, list):
                norm_cfg[key] = sorted(value)
            else:
                norm_cfg[key] = value
        normalized.append(norm_cfg)
    
    # Sort configs by JSON representation for consistent ordering
    return sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True))


def compute_simulation_fingerprint(
    model_fingerprint: str,
    target_ticker: str,
    simulation_ticker: str,
    threshold: float,
    z_score_threshold: float,
    regime_config: Dict[str, Any],
    train_window: int,
    test_window: int
) -> str:
    """
    Compute SHA-256 fingerprint for a simulation configuration.
    
    This uniquely identifies a simulation run to avoid duplicate testing.
    Uses MODEL FINGERPRINT (not model_id) so that:
    - Same model config (features + hyperparams) = same model fingerprint
    - Same model fingerprint + same simulation params = same simulation fingerprint
    - Simulations can be reused even if model is retrained with identical config
    
    Includes all factors that affect simulation results:
    - Model fingerprint (captures features, params, transform)
    - Data fold (train/test windows)
    - Trading strategy (threshold, z-score, regime filter)
    - Tickers (training target vs simulation ticker)
    
    Args:
        model_fingerprint: SHA-256 hash of model configuration (from compute_fingerprint)
        target_ticker: Symbol model was trained on
        simulation_ticker: Symbol being simulated (may differ for generalization tests)
        threshold: Trading signal threshold
        z_score_threshold: Outlier filter cutoff (0 = disabled)
        regime_config: Market regime filter dict
        train_window: Training fold size (e.g., 20000 bars)
        test_window: Test fold size (e.g., 1000 bars)
    
    Returns:
        64-character hex SHA-256 hash
    """
    payload = {
        "model_fingerprint": model_fingerprint,  # Use model fingerprint, not model_id
        "target_ticker": target_ticker.upper(),
        "simulation_ticker": simulation_ticker.upper(),
        "threshold": round(threshold, 8),
        "z_score_threshold": round(z_score_threshold, 8),
        "regime_config": _normalize_params(regime_config) if regime_config else {},
        "train_window": train_window,
        "test_window": test_window
    }
    
    # Create deterministic JSON string
    json_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    
    # Compute SHA-256
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def fingerprint_matches(fp1: str, fp2: str) -> bool:
    """Check if two fingerprints match (case-insensitive)."""
    return fp1.lower() == fp2.lower()
