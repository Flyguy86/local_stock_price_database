"""
VectorBT-based backtesting for model validation.

This module provides fast, vectorized backtesting to validate trained models
before deploying them to production. Uses walk-forward methodology to ensure
realistic out-of-sample performance.

Workflow:
1. Load best model from Ray Tune results
2. Generate predictions on test folds
3. Convert predictions to buy/sell signals
4. Run VectorBT simulation with realistic costs
5. Calculate performance metrics (Sharpe, drawdown, win rate)
6. Compare to buy & hold baseline
7. Aggregate results across all folds

For production portfolio trading, use Backtrader module (future implementation).
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import ray
from ray import tune

# VectorBT
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    logging.warning("VectorBT not installed. Install with: pip install vectorbt>=0.26.0")

from .streaming import Fold
from .data import load_fold_from_disk, get_available_folds
from .config import settings

log = logging.getLogger(__name__)


class ModelBacktester:
    """
    Fast model validation using VectorBT.
    
    Loads trained models and backtests them on walk-forward folds
    to determine real-world profitability.
    """
    
    def __init__(
        self,
        experiment_name: str,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize backtester.
        
        Args:
            experiment_name: Name of Ray Tune experiment (e.g., "walk_forward_xgboost_GOOGL")
            checkpoint_dir: Path to Ray Tune checkpoints (defaults to config)
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT required for backtesting. Install with: pip install vectorbt")
        
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir or settings.data.checkpoints_dir
        self.best_trial = None
        self.best_config = None
        self.models = {}  # fold_id -> trained model
        
    def load_best_model(self) -> Dict:
        """
        Load the best performing model from Ray Tune results.
        
        Returns:
            Dict with model metadata and config
        """
        exp_dir = Path(self.checkpoint_dir) / self.experiment_name
        
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"Experiment not found: {exp_dir}\n"
                f"Run training first via POST /train/walk_forward"
            )
        
        # Find best trial by test_rmse
        best_rmse = float('inf')
        best_trial_dir = None
        
        for trial_dir in exp_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            
            result_file = trial_dir / "result.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                test_rmse = result.get("test_rmse", float('inf'))
                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    best_trial_dir = trial_dir
        
        if best_trial_dir is None:
            raise ValueError(f"No valid trials found in {exp_dir}")
        
        # Load best trial config
        params_file = best_trial_dir / "params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                self.best_config = json.load(f)
        
        result_file = best_trial_dir / "result.json"
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        self.best_trial = {
            "trial_id": best_trial_dir.name,  # Trial directory name
            "trial_dir": str(best_trial_dir),
            "config": self.best_config,
            "test_rmse": result.get("test_rmse"),
            "test_r2": result.get("test_r2"),
            "test_mae": result.get("test_mae"),
            "train_rmse": result.get("train_rmse"),
            "num_folds": result.get("num_folds"),
            "metrics": {
                "test_rmse": result.get("test_rmse"),
                "test_r2": result.get("test_r2"),
                "test_mae": result.get("test_mae"),
                "train_rmse": result.get("train_rmse"),
                "num_folds": result.get("num_folds")
            }
        }
        
        log.info(f"✅ Loaded best model: test_rmse={best_rmse:.6f}, R²={result.get('test_r2', 0):.4f}")
        log.info(f"   Trial: {best_trial_dir.name}, Config: {self.best_config}")
        
        return self.best_trial
    
    def _load_fold_model_from_checkpoint(self, fold_id: int) -> Tuple:
        """
        Load fold-specific model and feature list from checkpoint.
        
        Args:
            fold_id: Fold number to load
            
        Returns:
            Tuple of (model, feature_cols) or (None, None) if not found
        """
        import joblib
        import json
        
        if not self.best_trial:
            log.warning("No best trial loaded - call load_best_model() first")
            return None, None
        
        # Find best checkpoint directory
        exp_dir = Path(self.checkpoint_dir) / self.experiment_name
        if not exp_dir.exists():
            log.warning(f"Experiment directory not found: {exp_dir}")
            return None, None
        
        # Find best trial checkpoint
        trial_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
        if not trial_dirs:
            log.warning(f"No trial directories found in {exp_dir}")
            return None, None
        
        # Sort by test_rmse to find best trial
        best_trial_dir = None
        best_rmse = float('inf')
        for trial_dir in trial_dirs:
            result_file = trial_dir / "result.json"
            if result_file.exists():
                with open(result_file) as f:
                    result = json.load(f)
                    rmse = result.get("test_rmse", float('inf'))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_trial_dir = trial_dir
        
        if not best_trial_dir:
            log.warning("Could not find best trial directory")
            return None, None
        
        # Look for checkpoint subdirectory
        checkpoint_dirs = list(best_trial_dir.glob("checkpoint_*"))
        if not checkpoint_dirs:
            log.warning(f"No checkpoint directories found in {best_trial_dir}")
            return None, None
        
        checkpoint_dir = checkpoint_dirs[0]  # Use first/only checkpoint
        
        # Load fold-specific model
        model_file = checkpoint_dir / f"fold_{fold_id:03d}_model.joblib"
        if not model_file.exists():
            log.warning(f"Fold model not found: {model_file}")
            return None, None
        
        model = joblib.load(model_file)
        
        # Load feature list
        features_file = checkpoint_dir / "feature_lists.json"
        if features_file.exists():
            with open(features_file) as f:
                feature_lists = json.load(f)
                feature_cols = feature_lists.get(str(fold_id))
        else:
            log.warning(f"Feature list not found: {features_file}")
            feature_cols = None
        
        log.info(f"✅ Loaded fold {fold_id} model from checkpoint with {len(feature_cols) if feature_cols else '?'} features")
        
        return model, feature_cols
    
    def generate_signals(
        self,
        predictions: np.ndarray,
        threshold: float = 0.0,
        min_confidence: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert model predictions to buy/sell signals.
        
        Args:
            predictions: Predicted log returns
            threshold: Prediction threshold for signal generation
            min_confidence: Minimum prediction magnitude to trade (filters noise)
        
        Returns:
            Tuple of (entry_signals, exit_signals) boolean arrays
        """
        # Entry signal: predicted return > threshold
        entries = predictions > threshold
        
        # Exit signal: predicted return < -threshold (or reverse position)
        exits = predictions < -threshold
        
        # Optional: Filter low-confidence predictions
        if min_confidence is not None:
            confidence_mask = np.abs(predictions) >= min_confidence
            entries = entries & confidence_mask
            exits = exits & confidence_mask
        
        return entries, exits
    
    def backtest_fold(
        self,
        fold_id: int,
        symbol: str,
        model=None,
        fees: float = 0.001,
        slippage: float = 0.0005,
        signal_threshold: float = 0.0,
        min_confidence: Optional[float] = None
    ) -> Dict:
        """
        Backtest a single fold.
        
        Args:
            fold_id: Fold number
            symbol: Trading symbol
            model: Trained model (if None, loads from checkpoint)
            fees: Transaction fee (0.001 = 0.1%)
            slippage: Slippage (0.0005 = 0.05%)
            signal_threshold: Prediction threshold for signals
            min_confidence: Minimum prediction magnitude
        
        Returns:
            Dict with fold backtest results
        """
        # Load fold data
        train_df, test_df = load_fold_from_disk(fold_id, symbol)
        
        if test_df.empty:
            log.warning(f"Fold {fold_id} has no test data")
            return {"fold_id": fold_id, "error": "No test data"}
        
        # Try to load model and features from checkpoint if not provided
        if model is None:
            model, feature_cols = self._load_fold_model_from_checkpoint(fold_id)
        else:
            # Use backtester's feature extraction logic
            feature_cols = None
        
        # Prepare features
        if feature_cols is None:
            # Fallback: match training feature logic (same as data.py)
            exclude_cols = {'target', 'symbol', 'date', 'source', 'ts', 'timestamp'}
            numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        log.info(f"Fold {fold_id}: Using {len(feature_cols)} features")
        
        # Get predictions
        X_test = test_df[feature_cols].dropna()
        predictions = model.predict(X_test)
        
        # Align with price data
        prices = test_df['close'].loc[X_test.index]
        
        # Generate signals
        entries, exits = self.generate_signals(
            predictions,
            threshold=signal_threshold,
            min_confidence=min_confidence
        )
        
        # Run VectorBT backtest
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            fees=fees,
            slippage=slippage,
            freq='1min',
            init_cash=10000  # Starting capital
        )
        
        # Calculate metrics
        total_return = portfolio.total_return()
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        win_rate = portfolio.trades.win_rate()
        
        # Buy & hold baseline
        buy_hold_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        
        results = {
            "fold_id": fold_id,
            "num_trades": len(portfolio.trades.records),
            "total_return": total_return,
            "buy_hold_return": buy_hold_return,
            "alpha": total_return - buy_hold_return,  # Excess return
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_predictions": len(predictions),
            "avg_prediction": np.mean(predictions),
            "prediction_std": np.std(predictions),
        }
        
        log.info(
            f"Fold {fold_id}: Return={total_return:.2%}, "
            f"Sharpe={sharpe_ratio:.2f}, Trades={len(portfolio.trades.records)}, "
            f"Win Rate={win_rate:.2%}"
        )
        
        return results
    
    @ray.remote
    def backtest_fold_remote(self, *args, **kwargs):
        """Ray remote wrapper for parallel fold backtesting."""
        return self.backtest_fold(*args, **kwargs)
    
    def backtest_all_folds(
        self,
        symbol: str,
        model,
        parallel: bool = True,
        **backtest_kwargs
    ) -> pd.DataFrame:
        """
        Backtest across all available folds.
        
        Args:
            symbol: Trading symbol
            model: Trained model
            parallel: Use Ray parallelization
            **backtest_kwargs: Args passed to backtest_fold
        
        Returns:
            DataFrame with results per fold
        """
        folds = get_available_folds(symbol)
        
        if not folds:
            raise ValueError(f"No folds found for {symbol}")
        
        log.info(f"Backtesting {len(folds)} folds for {symbol}...")
        
        if parallel and ray.is_initialized():
            # Parallel backtesting with Ray
            futures = [
                self.backtest_fold_remote.remote(
                    self, fold_id, symbol, model, **backtest_kwargs
                )
                for fold_id in folds
            ]
            results = ray.get(futures)
        else:
            # Sequential backtesting
            results = [
                self.backtest_fold(fold_id, symbol, model, **backtest_kwargs)
                for fold_id in folds
            ]
        
        df = pd.DataFrame(results)
        return df
    
    def aggregate_results(self, fold_results: pd.DataFrame) -> Dict:
        """
        Aggregate backtest results across all folds.
        
        Args:
            fold_results: DataFrame from backtest_all_folds
        
        Returns:
            Dict with aggregated metrics
        """
        # Remove error folds
        valid_results = fold_results[~fold_results.get('error').notna()].copy()
        
        if len(valid_results) == 0:
            raise ValueError("No valid fold results")
        
        # Calculate consistency score (lower std of returns = more consistent)
        return_consistency = 1 / (1 + valid_results['total_return'].std())
        trade_consistency = 1 / (1 + valid_results['num_trades'].std())
        
        aggregated = {
            "num_folds": len(valid_results),
            "avg_return": valid_results['total_return'].mean(),
            "median_return": valid_results['total_return'].median(),
            "std_return": valid_results['total_return'].std(),
            "min_return": valid_results['total_return'].min(),
            "max_return": valid_results['total_return'].max(),
            "avg_sharpe": valid_results['sharpe_ratio'].mean(),
            "avg_max_drawdown": valid_results['max_drawdown'].mean(),
            "avg_win_rate": valid_results['win_rate'].mean(),
            "total_trades": valid_results['num_trades'].sum(),
            "avg_trades_per_fold": valid_results['num_trades'].mean(),
            "profitable_folds": (valid_results['total_return'] > 0).sum(),
            "profitability_rate": (valid_results['total_return'] > 0).mean(),
            "avg_buy_hold_return": valid_results['buy_hold_return'].mean(),
            "avg_alpha": valid_results['alpha'].mean(),  # Average excess return
            "return_consistency_score": return_consistency,
            "trade_consistency_score": trade_consistency,
        }
        
        # Overall assessment
        if aggregated['avg_return'] > 0 and aggregated['avg_sharpe'] > 1.0:
            aggregated['assessment'] = "🟢 EXCELLENT - Deploy to production"
        elif aggregated['avg_return'] > 0 and aggregated['avg_sharpe'] > 0.5:
            aggregated['assessment'] = "🟡 GOOD - Consider with risk management"
        elif aggregated['avg_return'] > aggregated['avg_buy_hold_return']:
            aggregated['assessment'] = "🟠 MARGINAL - Beats buy & hold but low Sharpe"
        else:
            aggregated['assessment'] = "🔴 POOR - Does not beat buy & hold"
        
        return aggregated


def create_backtester(experiment_name: str) -> ModelBacktester:
    """Factory function to create backtester instance."""
    return ModelBacktester(experiment_name=experiment_name)
