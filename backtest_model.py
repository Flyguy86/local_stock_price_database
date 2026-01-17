#!/usr/bin/env python3
"""
Backtest a trained model on a different ticker.

Usage:
    python backtest_model.py \
        --checkpoint /app/data/ray_checkpoints/walk_forward_elasticnet_GOOGL/train_on_folds_0afef_00051.../checkpoint_000000 \
        --ticker AAPL \
        --start-date 2024-01-01 \
        --end-date 2024-12-31
    
    OR with MLflow:
    python backtest_model.py \
        --checkpoint models:/walk_forward_elasticnet_GOOGL/1 \
        --ticker AAPL \
        --start-date 2024-01-01 \
        --end-date 2024-12-31
"""

import argparse
import json
import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

import ray
from ray.data import read_parquet

from ray_orchestrator.streaming import StreamingPreprocessor, BarDataLoader


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load model and metadata from Ray checkpoint or MLflow."""
    
    # Check if this is an MLflow URI (models:/...)
    if checkpoint_path.startswith("models:/"):
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        
        # Load model from MLflow
        model = mlflow.pyfunc.load_model(checkpoint_path)
        
        # Get run metadata
        model_name, version = checkpoint_path.replace("models:/", "").split("/")
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version(model_name, version)
        run = client.get_run(mv.run_id)
        
        metadata = {
            "model_info": {
                "model_type": run.data.tags.get("algorithm", "unknown"),
                "feature_engineering_version": run.data.tags.get("feature_engineering_version", "unknown")
            },
            "training_info": {
                "primary_ticker": model_name.split("_")[-1] if "_" in model_name else "unknown"
            },
            "validation_summary": {
                "best_test_r2": run.data.metrics.get("avg_test_r2", 0.0),
                "best_test_rmse": run.data.metrics.get("avg_test_rmse", 0.0)
            }
        }
        
        print("\n=== MLFLOW MODEL METADATA ===")
        print(f"Model: {model_name} v{version}")
        print(f"Model Type: {metadata['model_info']['model_type']}")
        print(f"Feature Version: {metadata['model_info']['feature_engineering_version']}")
        print(f"Train Ticker: {metadata['training_info']['primary_ticker']}")
        print(f"Best Test R2: {metadata['validation_summary']['best_test_r2']:.4f}")
        print(f"Best Test RMSE: {metadata['validation_summary']['best_test_rmse']:.6f}")
        
        return {"model": model, "metadata": metadata}
    
    else:
        # Original Ray checkpoint loading
        ckpt_dir = Path(checkpoint_path)
        
        # Load model
        with open(ckpt_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load metadata
        with open(ckpt_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        print("\n=== CHECKPOINT METADATA ===")
        print(f"Model Type: {metadata['model_info']['model_type']}")
        print(f"Feature Version: {metadata['model_info']['feature_engineering_version']}")
        print(f"Train Ticker: {metadata['training_info']['primary_ticker']}")
        print(f"Best Test R2: {metadata['validation_summary']['best_test_r2']:.4f}")
        print(f"Best Test RMSE: {metadata['validation_summary']['best_test_rmse']:.6f}")
        
        return {"model": model, "metadata": metadata}


def generate_features(ticker: str, start_date: str, end_date: str, context_symbols: list) -> pd.DataFrame:
    """
    Generate features for a ticker using StreamingPreprocessor with Ray Data.
    
    CRITICAL: Always uses Ray Data to ensure features are calculated exactly
    the same way as during training. This is the single source of truth for
    feature engineering.
    """
    print(f"\n=== GENERATING FEATURES FOR {ticker} ===")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Using Ray Data: TRUE (ensures consistency with training)")
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Initialize data loader and preprocessor
    loader = BarDataLoader(parquet_dir="/app/data/parquet")
    preprocessor = StreamingPreprocessor(loader)
    
    # Load primary ticker data for date range
    primary_ds = loader.load_all_bars(symbols=[ticker])
    
    # Filter to date range using Ray Data
    def filter_dates(batch):
        df = pd.DataFrame(batch)
        df['ts'] = pd.to_datetime(df['ts'])
        mask = (df['ts'] >= start_date) & (df['ts'] <= end_date)
        return {k: v[mask] for k, v in df.to_dict('series').items()}
    
    primary_ds = primary_ds.map_batches(filter_dates, batch_format="numpy")
    
    # Load context symbols
    context_datasets = {}
    for ctx_symbol in context_symbols:
        ctx_ds = loader.load_all_bars(symbols=[ctx_symbol])
        ctx_ds = ctx_ds.map_batches(filter_dates, batch_format="numpy")
        context_datasets[ctx_symbol] = ctx_ds
        print(f"Loaded context: {ctx_symbol}")
    
    # Process through feature engineering pipeline
    # This uses the EXACT SAME code path as training (calculate_indicators_gpu)
    def process_batch(batch):
        df = pd.DataFrame(batch)
        if df.empty:
            return df
        
        # Apply full indicator calculation (same as training)
        df = preprocessor.calculate_indicators_gpu(
            df,
            windows=[50, 200],  # Match training defaults
            zscore_window=200,   # Match training defaults
            drop_warmup=True
        )
        
        # Join context features if available
        for ctx_symbol, ctx_ds in context_datasets.items():
            ctx_df = ctx_ds.to_pandas()
            if not ctx_df.empty:
                df = preprocessor._calculate_context_features(
                    primary_df=df,
                    context_df=ctx_df,
                    context_symbol=ctx_symbol,
                    windows=[50, 200]
                )
        
        return df
    
    feature_ds = primary_ds.map_batches(process_batch, batch_format="pandas")
    
    # Convert to pandas for final backtesting
    df = feature_ds.to_pandas()
    df = df.sort_values('ts').reset_index(drop=True)
    
    print(f"Generated {len(df)} rows with {len(df.columns)} features")
    print(f"Feature engineering version: {preprocessor.feature_engineering_version}")
    return df


def backtest(model, features_df: pd.DataFrame, target_col: str = 'target_1bar_fwd_ret') -> dict:
    """Run backtest and calculate metrics."""
    print("\n=== RUNNING BACKTEST ===")
    
    # Drop rows with missing target
    df = features_df.dropna(subset=[target_col]).copy()
    
    # Separate features and target
    y = df[target_col].values
    feature_cols = [c for c in df.columns if c not in ['ts', 'symbol', target_col]]
    X = df[feature_cols].values
    
    # Handle any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # Direction accuracy
    y_direction = np.sign(y)
    pred_direction = np.sign(y_pred)
    direction_acc = np.mean(y_direction == pred_direction)
    
    results = {
        "n_samples": len(df),
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "direction_accuracy": direction_acc,
        "predictions": pd.DataFrame({
            "ts": df['ts'].values,
            "actual": y,
            "predicted": y_pred,
            "error": y - y_pred
        })
    }
    
    print(f"Samples: {results['n_samples']}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.6f}")
    print(f"Direction Accuracy: {direction_acc:.2%}")
    
    return results


def plot_results(results: dict, ticker: str, save_path: str = None):
    """Plot backtest results."""
    df = results['predictions']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Actual vs Predicted
    axes[0].plot(df['ts'], df['actual'], label='Actual', alpha=0.7, linewidth=1)
    axes[0].plot(df['ts'], df['predicted'], label='Predicted', alpha=0.7, linewidth=1)
    axes[0].set_title(f'{ticker} - Actual vs Predicted Returns')
    axes[0].set_ylabel('Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Prediction Error
    axes[1].plot(df['ts'], df['error'], color='red', alpha=0.5, linewidth=1)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].set_title('Prediction Error (Actual - Predicted)')
    axes[1].set_ylabel('Error')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Scatter Plot
    axes[2].scatter(df['actual'], df['predicted'], alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(df['actual'].min(), df['predicted'].min())
    max_val = max(df['actual'].max(), df['predicted'].max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect Prediction')
    
    axes[2].set_xlabel('Actual Return')
    axes[2].set_ylabel('Predicted Return')
    axes[2].set_title(f'Scatter Plot (R² = {results["r2"]:.4f})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Backtest a trained model on a different ticker')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint directory')
    parser.add_argument('--ticker', required=True, help='Ticker symbol to backtest')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--context-symbols', default='QQQ,SPY,VIX', help='Comma-separated context symbols')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output-dir', default='/app/data/backtest_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse context symbols
    context_symbols = [s.strip() for s in args.context_symbols.split(',') if s.strip()]
    
    # Load checkpoint
    checkpoint_data = load_checkpoint(args.checkpoint)
    model = checkpoint_data['model']
    metadata = checkpoint_data['metadata']
    
    # Generate features
    features_df = generate_features(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        context_symbols=context_symbols
    )
    
    # Run backtest
    results = backtest(model, features_df)
    
    # Save results
    results_file = output_dir / f"backtest_{args.ticker}_{args.start_date}_{args.end_date}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "ticker": args.ticker,
            "date_range": {"start": args.start_date, "end": args.end_date},
            "checkpoint": args.checkpoint,
            "trained_on": metadata['training_info']['primary_ticker'],
            "metrics": {
                "n_samples": results['n_samples'],
                "rmse": results['rmse'],
                "r2": results['r2'],
                "mae": results['mae'],
                "direction_accuracy": results['direction_accuracy']
            }
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Save predictions
    pred_file = output_dir / f"predictions_{args.ticker}_{args.start_date}_{args.end_date}.csv"
    results['predictions'].to_csv(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")
    
    # Plot if requested
    if args.plot:
        plot_file = output_dir / f"backtest_plot_{args.ticker}_{args.start_date}_{args.end_date}.png"
        plot_results(results, args.ticker, save_path=str(plot_file))
    
    print("\n=== BACKTEST COMPLETE ===")


if __name__ == "__main__":
    main()
