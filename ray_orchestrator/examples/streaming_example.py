"""
Example usage of Ray Data streaming preprocessing with walk-forward validation.

Run this inside the ray_orchestrator container:
    docker exec -it ray_orchestrator python -m ray_orchestrator.examples.streaming_example
"""

import logging
import ray
from ray_orchestrator.streaming import create_preprocessing_pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def example_walk_forward():
    """
    Example: Walk-forward preprocessing with proper fold isolation.
    
    This is the RECOMMENDED approach for balanced backtesting.
    Each fold calculates indicators independently to prevent look-ahead bias.
    """
    log.info("\n" + "="*80)
    log.info("WALK-FORWARD PREPROCESSING (Recommended for Balanced Backtesting)")
    log.info("="*80)
    
    preprocessor = create_preprocessing_pipeline(parquet_dir="/app/data/parquet")
    
    # Generate walk-forward folds
    # Example: 3-month train, 1-month test, step forward 1 month
    fold_count = 0
    
    for fold in preprocessor.create_walk_forward_pipeline(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-06-30",
        train_months=3,
        test_months=1,
        step_months=1,
        context_symbols=["QQQ"],  # Add QQQ for context
        windows=[50, 200],  # SMA-50 and SMA-200
        resampling_timeframes=["5min", "15min"],  # Multi-timeframe
        num_gpus=0.0,  # Set to 1.0 when running on GPU
        actor_pool_size=2
    ):
        fold_count += 1
        log.info(f"\n{fold}")
        
        # Train data stats
        if fold.train_ds:
            train_count = fold.train_ds.count()
            log.info(f"  Train rows: {train_count}")
            
            # Show sample (indicators properly start with NaN for warm-up)
            train_sample = fold.train_ds.take(3)
            log.info("  Train sample:")
            for row in train_sample:
                log.info(f"    {row['ts']}: close={row.get('close', 'N/A'):.2f}, "
                        f"sma50={row.get('sma_50', 'N/A')}, "
                        f"sma200={row.get('sma_200', 'N/A')}, "
                        f"rsi={row.get('rsi_14', 'N/A')}, "
                        f"macd={row.get('macd', 'N/A')}")
        
        # Test data stats
        if fold.test_ds:
            test_count = fold.test_ds.count()
            log.info(f"  Test rows: {test_count}")
            
            # Show sample
            test_sample = fold.test_ds.take(3)
            log.info("  Test sample:")
            for row in test_sample:
                log.info(f"    {row['ts']}: close={row.get('close', 'N/A'):.2f}, "
                        f"sma50={row.get('sma_50', 'N/A')}, "
                        f"sma200={row.get('sma_200', 'N/A')}")
        
        # Here you would:
        # 1. Train a model on fold.train_ds
        # 2. Test it on fold.test_ds
        # 3. Record metrics
        
        # Break after first fold for demo (remove in production)
        if fold_count >= 2:
            log.info(f"\n... (showing first 2 folds only)")
            break
    
    log.info(f"\nProcessed {fold_count} folds total")


def example_simple_preview():
    """Example: Quick preview of raw data."""
    log.info("\n" + "="*80)
    log.info("SIMPLE DATA PREVIEW")
    log.info("="*80)
    
    preprocessor = create_preprocessing_pipeline(parquet_dir="/app/data/parquet")
    
    # Load and preview data for specific symbols
    ds = preprocessor.loader.load_all_bars(symbols=["AAPL"], parallelism=2)
    
    log.info(f"Loaded {ds.count()} total rows")
    log.info("\nSchema:")
    print(ds.schema())
    
    log.info("\nFirst 5 rows:")
    for row in ds.take(5):
        print(f"{row['ts']}: O={row['open']:.2f} H={row['high']:.2f} "
              f"L={row['low']:.2f} C={row['close']:.2f} V={row['volume']}")


def main():
    """Main entry point."""
    
    # Initialize Ray if not already
    if not ray.is_initialized():
        ray.init(address="auto", namespace="trading_bot", ignore_reinit_error=True)
    
    log.info("Ray cluster resources:")
    print(ray.cluster_resources())
    
    # Run examples
    try:
        # Example 1: Simple preview
        example_simple_preview()
        
        # Example 2: Walk-forward preprocessing (RECOMMENDED)
        example_walk_forward()
        
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
    
    log.info("\n" + "="*80)
    log.info("DONE!")
    log.info("="*80)


if __name__ == "__main__":
    main()
