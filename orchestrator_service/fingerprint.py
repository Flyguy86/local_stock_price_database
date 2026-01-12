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
    target_col: str = "close"
) -> str:
    """
    Compute SHA-256 fingerprint from model configuration.
    
    The fingerprint uniquely identifies a model configuration:
    - Same features + params + transform + symbol = same fingerprint
    - Any difference = different fingerprint
    
    Args:
        features: List of feature column names
        hyperparams: Model hyperparameters dict
        target_transform: Target transformation (none, log_return, pct_change)
        symbol: Trading symbol
        target_col: Target column name
    
    Returns:
        64-character hex SHA-256 hash
    """
    # Normalize the payload for consistent hashing
    payload = {
        "features": sorted(features),
        "hyperparams": _normalize_params(hyperparams),
        "target_transform": target_transform,
        "symbol": symbol.upper(),
        "target_col": target_col
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


def fingerprint_matches(fp1: str, fp2: str) -> bool:
    """Check if two fingerprints match (case-insensitive)."""
    return fp1.lower() == fp2.lower()
