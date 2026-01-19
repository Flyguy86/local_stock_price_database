#!/usr/bin/env python3
"""
List all trained models from Ray checkpoints with their metrics.
"""
import json
from pathlib import Path
from typing import List, Dict

def list_trained_models(checkpoints_dir: str = "/app/data/ray_checkpoints") -> List[Dict]:
    """
    Scan Ray checkpoints and extract model information.
    
    Returns:
        List of dicts with model info: checkpoint_path, metrics, params, etc.
    """
    models = []
    checkpoint_path = Path(checkpoints_dir)
    
    if not checkpoint_path.exists():
        return []
    
    # Scan for experiment directories
    for exp_dir in checkpoint_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "backtest_results":
            continue
        
        # Scan for trial directories
        for trial_dir in exp_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            
            # Look for checkpoint subdirectories
            for ckpt_dir in trial_dir.iterdir():
                if not ckpt_dir.is_dir() or not ckpt_dir.name.startswith("checkpoint_"):
                    continue
                
                metadata_file = ckpt_dir / "metadata.json"
                if not metadata_file.exists():
                    continue
                
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract key information
                    model_info = metadata.get("model_info", {})
                    training_info = metadata.get("training_info", {})
                    validation_summary = metadata.get("validation_summary", {})
                    overall_metrics = validation_summary.get("overall_metrics", {})
                    
                    # Load config
                    config_file = ckpt_dir / "config.json"
                    config = {}
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                    
                    models.append({
                        "checkpoint_path": str(ckpt_dir),
                        "trial_name": trial_dir.name,
                        "experiment_name": exp_dir.name,
                        "algorithm": model_info.get("algorithm", "unknown"),
                        "ticker": training_info.get("primary_ticker", "unknown"),
                        "date_range": training_info.get("date_range", {}),
                        "feature_version": model_info.get("feature_engineering_version", "unknown"),
                        "metrics": {
                            "avg_train_rmse": overall_metrics.get("avg_train_rmse", 0.0),
                            "avg_test_rmse": overall_metrics.get("avg_test_rmse", 0.0),
                            "avg_test_r2": overall_metrics.get("avg_test_r2", 0.0),
                            "avg_test_mae": overall_metrics.get("avg_test_mae", 0.0),
                            "num_folds": overall_metrics.get("num_folds", 0)
                        },
                        "params": config,
                        "created_at": metadata.get("created_at", "")
                    })
                    
                except Exception as e:
                    continue
    
    # Sort by test RMSE (best first)
    models.sort(key=lambda x: x["metrics"]["avg_test_rmse"])
    
    return models


if __name__ == "__main__":
    models = list_trained_models()
    print(f"Found {len(models)} trained models:\n")
    
    for i, model in enumerate(models[:10], 1):
        print(f"{i}. {model['algorithm'].upper()} - {model['ticker']}")
        print(f"   Test RMSE: {model['metrics']['avg_test_rmse']:.6f}")
        print(f"   Test R²: {model['metrics']['avg_test_r2']:.4f}")
        print(f"   Path: {model['checkpoint_path']}")
        print()
