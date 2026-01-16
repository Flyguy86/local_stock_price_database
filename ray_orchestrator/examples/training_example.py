"""
Complete walk-forward training example.

Demonstrates the full pipeline from data preprocessing to model training
with proper walk-forward validation.
"""

import logging
import ray
from ray_orchestrator.trainer import create_walk_forward_trainer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    """Run complete walk-forward training pipeline."""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address="auto", namespace="trading_bot", ignore_reinit_error=True)
    
    log.info("="*80)
    log.info("WALK-FORWARD TRAINING PIPELINE")
    log.info("="*80)
    
    # Create trainer
    trainer = create_walk_forward_trainer(parquet_dir="/app/data/parquet")
    
    # Run hyperparameter tuning with walk-forward validation
    log.info("\nStarting hyperparameter tuning with walk-forward validation...")
    log.info("This will:")
    log.info("  1. Generate time-based folds (e.g., Fold 1: Train Jan-Mar, Test Apr)")
    log.info("  2. Preprocess each fold independently (no look-ahead)")
    log.info("  3. Test each hyperparameter config across ALL folds")
    log.info("  4. Report average metrics across folds\n")
    
    results = trainer.run_walk_forward_tuning(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-06-30",
        train_months=3,
        test_months=1,
        step_months=1,
        algorithm="elasticnet",
        num_samples=20,  # Try 20 different hyperparameter configurations
        context_symbols=["QQQ"],  # Add market context
        windows=[50, 200],  # SMA-50 and SMA-200
        resampling_timeframes=["5min", "15min"]  # Multi-timeframe features
    )
    
    # Get best result
    best = results.get_best_result()
    
    log.info("\n" + "="*80)
    log.info("RESULTS")
    log.info("="*80)
    log.info(f"\nBest hyperparameters:")
    for key, value in best.config.items():
        log.info(f"  {key}: {value}")
    
    log.info(f"\nBest performance (averaged across all folds):")
    log.info(f"  Test RMSE: {best.metrics['test_rmse']:.6f}")
    log.info(f"  Test MAE:  {best.metrics['test_mae']:.6f}")
    log.info(f"  Test R²:   {best.metrics['test_r2']:.4f}")
    log.info(f"  Num Folds: {best.metrics['num_folds']}")
    
    # Show all trial results
    log.info(f"\n\nAll {len(results)} trials:")
    df = results.get_dataframe()
    df_sorted = df.sort_values("test_rmse")
    
    log.info("\nTop 5 configurations:")
    for idx, row in df_sorted.head(5).iterrows():
        log.info(f"  Trial {idx}: RMSE={row['test_rmse']:.6f}, "
                f"R²={row['test_r2']:.4f}, "
                f"alpha={row.get('config/alpha', 'N/A')}, "
                f"l1_ratio={row.get('config/l1_ratio', 'N/A')}")
    
    log.info("\n" + "="*80)
    log.info("NEXT STEPS")
    log.info("="*80)
    log.info("""
    1. Deploy the best model to production
    2. Run live paper trading simulation
    3. Monitor performance metrics
    4. Retrain periodically with new data
    """)
    
    log.info("\nDone!")


if __name__ == "__main__":
    main()
