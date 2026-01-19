"""
Ray Data streaming preprocessing pipeline.

Reads 1-minute bar data from parquet files and applies transformations
for ML training pipelines using Ray Data's streaming engine.

Key features:
- Walk-forward fold-based processing (no look-ahead bias)
- GPU-accelerated indicator calculation
- Multi-timeframe resampling (1min -> 5min, 15min, etc.)
- Context features from related symbols (QQQ, VIX)
- Strict indicator reset at fold boundaries
"""

# =============================================================================
# FEATURE ENGINEERING VERSION TRACKING
# =============================================================================
# CRITICAL: When modifying feature calculations, increment this version number
# and document the changes below. This version is saved in checkpoint metadata
# to ensure reproducibility and track feature evolution over time.
#
# VERSION HISTORY:
# - v3.1 (2026-01-17): Implemented comprehensive 3-phase normalization pipeline for all
#                      10 major technical indicators. Phase 1 preserves indicator physics
#                      (raw calculation on actual prices). Phase 3 adds simple centering
#                      for bounded indicators (0-100 → -1 to +1). Phase 4 adds rolling
#                      z-score (zscore_window=200) for regime-adaptive normalization.
#                      Affected indicators: Stochastic (k/d), RSI-14, MACD (signal/diff),
#                      Bollinger Bands (upper/mid/lower/width), ATR-14 (raw/pct), OBV,
#                      SMAs (all windows + volume_ma + volatility), EMAs (all windows),
#                      Volume Ratio. Models can now choose raw, _norm, or _zscore variants.
#                      Ensures features are comparable across stocks, timeframes, and
#                      volatility regimes. Added _rolling_zscore() helper method.
#
# - v3.0 (2026-01-17): Converted OHLC prices to log returns to prevent absolute
#                      price leakage. Keeps close_raw for VectorBT simulation.
#                      Drops raw open/high/low/close columns, adds 
#                      open_log_return, high_log_return, low_log_return,
#                      close_log_return. Target is future close log return.
#
# - v2.1 (2025-12-10): Added sin/cos time encoding for cyclical features,
#                      market session indicators (is_market_open, is_morning),
#                      enhanced volatility features with multiple windows.
#
# - v2.0 (2025-11-15): Added multi-timeframe resampling (5min, 15min, 1H),
#                      context symbol features (QQQ, VIX relative indicators),
#                      OBV (On Balance Volume), VWAP distance.
#
# - v1.0 (2025-10-01): Initial feature set with SMA, EMA, RSI, MACD, Bollinger
#                      Bands, ATR, Stochastic Oscillator, volume indicators.
#
# =============================================================================

FEATURE_ENGINEERING_VERSION = "v3.1"

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
import pyarrow as pa
import ray
from ray.data import Dataset
from ray.data.preprocessors import Chain
import pandas as pd
import numpy as np
import duckdb

# Disable Ray Data progress bars to avoid ANSI escape codes in logs
from ray.data import DataContext
ctx = DataContext.get_current()
ctx.enable_progress_bars = False

# Enable verbose Ray Data execution logs for debugging
logging.getLogger("ray.data").setLevel(logging.INFO)

# Optimize Ray Data for CPU utilization
ctx.execution_options.resource_limits.cpu = None  # Use all available CPUs
ctx.execution_options.preserve_order = False  # Allow reordering for better parallelism
ctx.target_max_block_size = 512 * 1024 * 1024  # 512MB blocks (larger = more efficient)

log = logging.getLogger(__name__)


@dataclass
class Fold:
    """Represents a single walk-forward fold."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_ds: Optional[Dataset] = None
    test_ds: Optional[Dataset] = None
    
    def __repr__(self):
        return f"Fold({self.fold_id}: train={self.train_start}→{self.train_end}, test={self.test_start}→{self.test_end})"


class BarDataLoader:
    """Load and stream bar data using DuckDB to handle Hive-partitioned parquet files."""
    
    def __init__(self, parquet_dir: str = "/app/data/parquet"):
        self.parquet_dir = Path(parquet_dir)
        log.info(f"Initialized BarDataLoader with path: {self.parquet_dir}")
    
    def load_all_bars(
        self,
        symbols: Optional[List[str]] = None,
        parallelism: int = 10
    ) -> Dataset:
        """
        Load bar data for all or specific symbols using DuckDB.
        
        Parquet files are in Hive partition format: Ticker/dt=YYYY-MM-DD/bars.parquet
        
        Args:
            symbols: List of ticker symbols to load. If None, loads all.
            parallelism: Number of parallel read tasks
            
        Returns:
            Ray Dataset of bar data
        """
        try:
            # Use DuckDB in read-only mode to prevent accidental writes
            con = duckdb.connect(":memory:", read_only=False)  # Memory DB can't be read-only
            
            if symbols:
                # Build WHERE clause for specific symbols
                symbol_list = "','".join(symbols)
                query = f"""
                SELECT * FROM read_parquet('{self.parquet_dir}/**/bars.parquet', 
                    hive_partitioning=true, 
                    union_by_name=true)
                WHERE symbol IN ('{symbol_list}')
                """
            else:
                # Load all symbols
                query = f"""
                SELECT * FROM read_parquet('{self.parquet_dir}/**/bars.parquet', 
                    hive_partitioning=true,
                    union_by_name=true)
                """
            
            log.info(f"Loading data with DuckDB query: {query}")
            df = con.execute(query).fetchdf()
            con.close()
            
            if df.empty:
                log.warning(f"No data found for symbols {symbols}")
                return ray.data.from_pandas(pd.DataFrame())
            
            log.info(f"Loaded {len(df)} rows for {df['symbol'].nunique()} symbols")
            
            # Convert to Ray Dataset
            ds = ray.data.from_pandas(df)
            return ds
            
        except Exception as e:
            log.error(f"Error loading parquet data: {e}")
            return ray.data.from_pandas(pd.DataFrame())
    
    def load_symbol(self, symbol: str, parallelism: int = 2) -> Dataset:
        """Load data for a single symbol."""
        return self.load_all_bars(symbols=[symbol], parallelism=parallelism)
    
    def _discover_parquet_files(self, symbols: Optional[List[str]] = None) -> List[str]:
        """
        Discover all parquet files in Hive-partitioned structure.
        
        Structure: Ticker/dt=YYYY-MM-DD/bars.parquet
        """
        if not self.parquet_dir.exists():
            log.error(f"Parquet directory DOES NOT EXIST: {self.parquet_dir}")
            log.error(f"Absolute path: {self.parquet_dir.absolute()}")
            return []
        
        log.info(f"Searching for Hive-partitioned parquet files in: {self.parquet_dir.absolute()}")
        
        # Find all bars.parquet files
        files = list(self.parquet_dir.rglob("bars.parquet"))
        
        if symbols:
            # Filter by symbol directories
            filtered_files = []
            for file in files:
                # Check if any parent directory matches a symbol
                symbol_dir = file.parent.parent.name  # e.g., AAPL from AAPL/dt=2024-01-01/bars.parquet
                if symbol_dir in symbols or symbol_dir.upper() in [s.upper() for s in symbols]:
                    filtered_files.append(str(file))
            files = filtered_files
        else:
            files = [str(f) for f in files]
        
        log.info(f"Total parquet files discovered: {len(files)}")
        if len(files) > 0:
            log.info(f"Sample files: {files[:3]}")
        
        return sorted(files)


class StreamingPreprocessor:
    """Streaming preprocessing pipeline using Ray Data with walk-forward folds."""
    
    def __init__(self, loader: BarDataLoader):
        self.loader = loader
        self.feature_engineering_version = FEATURE_ENGINEERING_VERSION
    
    def check_cached_folds(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        train_months: int = 3,
        test_months: int = 1,
        step_months: int = 1
    ) -> tuple[int, int]:
        """
        Check how many folds are available in cache vs need to be computed.
        
        Returns:
            Tuple of (cached_count, total_count)
        """
        from .config import settings
        
        # Generate fold dates to check
        folds = self.generate_walk_forward_folds(
            start_date=start_date,
            end_date=end_date,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months
        )
        
        cached_count = 0
        for fold in folds:
            fold_dir = settings.data.walk_forward_folds_dir / symbol / f"fold_{fold.fold_id:03d}"
            train_path = fold_dir / "train"
            test_path = fold_dir / "test"
            
            if train_path.exists() and test_path.exists():
                cached_count += 1
        
        total_count = len(folds)
        log.info(f"Fold cache status for {symbol}: {cached_count}/{total_count} available ({cached_count/total_count*100:.0f}%)")
        
        return cached_count, total_count
    
    def generate_walk_forward_folds(
        self,
        start_date: str,
        end_date: str,
        train_months: int = 3,
        test_months: int = 1,
        step_months: int = 1
    ) -> List[Fold]:
        """
        Generate walk-forward validation folds.
        
        Example: train_months=3, test_months=1, step_months=1
        - Fold 1: Train Jan-Mar, Test Apr
        - Fold 2: Train Feb-Apr, Test May
        - Fold 3: Train Mar-May, Test Jun
        
        Args:
            start_date: Starting date (YYYY-MM-DD)
            end_date: Ending date (YYYY-MM-DD)
            train_months: Number of months for training
            test_months: Number of months for testing
            step_months: How many months to step forward each fold
            
        Returns:
            List of Fold objects with date ranges
        """
        from datetime import datetime, timedelta
        
        # VALIDATION: Never allow training on current month data
        # Stop at last day of previous month to ensure complete data
        now = datetime.now()
        max_allowed_date = (now.replace(day=1) - timedelta(days=1)).date()  # Last day of previous month
        requested_end = pd.Timestamp(end_date).date()
        
        if requested_end > max_allowed_date:
            raise ValueError(
                f"Cannot train on current month or future data. "
                f"Requested end_date: {end_date}, "
                f"Max allowed (last month): {max_allowed_date.strftime('%Y-%m-%d')}"
            )
        
        log.info(f"Date validation passed: end_date={end_date}, max_allowed={max_allowed_date}")
        
        folds = []
        fold_id = 1
        
        current = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        while current + pd.DateOffset(months=train_months + test_months) <= end:
            train_start = current
            train_end = current + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
            
            fold = Fold(
                fold_id=fold_id,
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d")
            )
            folds.append(fold)
            
            log.info(f"Created {fold}")
            
            # Step forward
            current += pd.DateOffset(months=step_months)
            fold_id += 1
        
        log.info(f"Generated {len(folds)} walk-forward folds")
        return folds
    
    def _try_load_cached_fold(self, fold: Fold, symbol: str) -> Optional[Fold]:
        """
        Try to load a pre-computed fold from disk cache.
        
        Returns:
            Fold with populated datasets if cache exists, None otherwise
        """
        from .config import settings
        
        fold_dir = settings.data.walk_forward_folds_dir / symbol / f"fold_{fold.fold_id:03d}"
        
        if not fold_dir.exists():
            return None
        
        train_path = fold_dir / "train"
        test_path = fold_dir / "test"
        
        if not train_path.exists() or not test_path.exists():
            log.warning(f"Incomplete cached fold at {fold_dir}, will recompute")
            return None
        
        try:
            # Load train and test datasets using Ray Data
            fold.train_ds = ray.data.read_parquet(str(train_path))
            fold.test_ds = ray.data.read_parquet(str(test_path))
            
            # Verify datasets have data
            train_count = fold.train_ds.count()
            test_count = fold.test_ds.count()
            
            if train_count == 0 or test_count == 0:
                log.warning(f"Cached fold {fold.fold_id} has empty datasets, will recompute")
                return None
            
            log.info(f"Loaded cached fold {fold.fold_id}: train={train_count:,} rows, test={test_count:,} rows")
            return fold
            
        except Exception as e:
            log.warning(f"Failed to load cached fold {fold.fold_id}: {e}, will recompute")
            return None
    
    def _save_fold_to_cache(self, fold: Fold, symbol: str):
        """
        Save a computed fold to disk cache for future runs.
        
        Saves both train and test datasets as partitioned parquet files.
        """
        from .config import settings
        
        fold_dir = settings.data.walk_forward_folds_dir / symbol / f"fold_{fold.fold_id:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = fold_dir / "train"
        test_path = fold_dir / "test"
        
        try:
            # Save train dataset
            if fold.train_ds:
                fold.train_ds.write_parquet(
                    str(train_path),
                    try_create_dir=True
                )
                log.info(f"Saved train data to {train_path}")
            
            # Save test dataset
            if fold.test_ds:
                fold.test_ds.write_parquet(
                    str(test_path),
                    try_create_dir=True
                )
                log.info(f"Saved test data to {test_path}")
            
            log.info(f"✓ Cached fold {fold.fold_id} to {fold_dir}")
            
        except Exception as e:
            log.warning(f"Failed to cache fold {fold.fold_id}: {e}")
    
    def load_fold_data(
        self,
        fold: Fold,
        symbols: List[str],
        context_symbols: Optional[List[str]] = None
    ) -> Fold:
        """
        Load data for a specific fold with strict date filtering.
        
        Args:
            fold: Fold object with date ranges
            symbols: Primary trading symbols
            context_symbols: Context symbols (QQQ, VIX, etc.)
            
        Returns:
            Fold with populated train_ds and test_ds
        """
        log.info(f"Loading data for {fold}")
        
        # Load train data
        train_ds = self._load_date_range(
            symbols=symbols,
            start_date=fold.train_start,
            end_date=fold.train_end
        )
        
        # Load test data
        test_ds = self._load_date_range(
            symbols=symbols,
            start_date=fold.test_start,
            end_date=fold.test_end
        )
        
        # Load context symbols if specified (parallelized for speed)
        if context_symbols:
            # Load train and test context in parallel using Ray
            @ray.remote
            def load_context_async(ctx_symbols, start, end):
                return self._load_date_range(
                    symbols=ctx_symbols,
                    start_date=start,
                    end_date=end
                )
            
            train_context_future = load_context_async.remote(context_symbols, fold.train_start, fold.train_end)
            test_context_future = load_context_async.remote(context_symbols, fold.test_start, fold.test_end)
            
            # Wait for both to complete
            train_context, test_context = ray.get([train_context_future, test_context_future])
            
            log.info(f"Loaded {len(context_symbols)} context symbols in parallel")
            
            # Join on timestamp
            train_ds = self._join_context_features(train_ds, train_context, symbols[0])
            test_ds = self._join_context_features(test_ds, test_context, symbols[0])
        
        fold.train_ds = train_ds
        fold.test_ds = test_ds
        
        return fold
    
    def _load_date_range(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dataset:
        """Load data for specific date range."""
        ds = self.loader.load_all_bars(symbols=symbols)
        
        # Check if dataset is empty
        try:
            count = ds.count()
            if count == 0:
                log.warning(f"No data found for symbols {symbols}")
                return ds
        except Exception as e:
            log.warning(f"Error checking dataset: {e}")
            return ds
        
        # Filter by date range
        # Market data is in US/Eastern timezone (NYSE hours)
        start_dt = pd.to_datetime(start_date).tz_localize('US/Eastern')
        end_dt = pd.to_datetime(end_date).tz_localize('US/Eastern') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        def filter_dates(batch: pd.DataFrame) -> pd.DataFrame:
            if batch.empty:
                return batch
            # Ensure ts column is datetime
            if not pd.api.types.is_datetime64_any_dtype(batch['ts']):
                batch['ts'] = pd.to_datetime(batch['ts'])
            
            # Convert to US/Eastern if it's in a different timezone or naive
            if batch['ts'].dt.tz is None:
                # Assume naive timestamps are already in Eastern time
                batch['ts'] = batch['ts'].dt.tz_localize('US/Eastern')
            elif str(batch['ts'].dt.tz) != 'US/Eastern':
                # Convert from other timezones to Eastern
                batch['ts'] = batch['ts'].dt.tz_convert('US/Eastern')
            
            mask = (batch['ts'] >= start_dt) & (batch['ts'] <= end_dt)
            filtered = batch[mask]
            log.debug(f"Filtered {len(batch)} rows to {len(filtered)} rows for date range {start_date} to {end_date}")
            return filtered
        
        return ds.map_batches(filter_dates, batch_format="pandas")
    
    def _join_context_features(
        self,
        primary_ds: Dataset,
        context_ds: Dataset,
        primary_symbol: str
    ) -> Dataset:
        """
        Join primary symbol data with context symbols (QQQ, VIX) to create cross-sectional features.
        
        This enables models to distinguish:
        - Stock-specific moves vs market-wide moves
        - Beta-adjusted returns (abnormal performance)
        - Volatility regime context (VIX-based)
        
        Features created:
        1. **Relative Strength**: close_ratio = close_AAPL / close_QQQ
        2. **Beta-60**: Rolling 60-bar beta vs market proxy (QQQ)
        3. **Beta-Adjusted Returns**: residual_return = return - (beta * market_return)
        4. **VIX Regime**: vix_zscore, high_vix_regime (VIX > 20)
        5. **Context Indicators**: Suffixed with symbol name (rsi_14_QQQ, macd_VIX)
        
        Args:
            primary_ds: Primary symbol dataset (e.g., AAPL)
            context_ds: Context symbols dataset (e.g., QQQ, VIX)
            primary_symbol: Primary trading symbol name
            
        Returns:
            Dataset with joined context features
            
        Example:
            # Input: AAPL data + [QQQ, VIX] context
            # Output: AAPL data with:
            #   - close_ratio_QQQ = close_AAPL / close_QQQ
            #   - beta_60_QQQ = rolling_beta(AAPL, QQQ)
            #   - rsi_14_QQQ, macd_QQQ (context indicators)
            #   - vix_zscore, high_vix_regime (if VIX present)
        """
        # Convert datasets to pandas for complex join operations
        # This is acceptable because we're processing fold-sized chunks (2-3 months)
        # not the entire multi-year dataset
        try:
            log.info(f"Converting datasets to pandas for context feature merge (primary: {primary_symbol})")
            primary_pdf = primary_ds.to_pandas()
            context_pdf = context_ds.to_pandas()
            
            if context_pdf.empty:
                log.warning(f"Context dataset is empty, skipping context merge for {primary_symbol}")
                return primary_ds
            
            pre_merge_rows = len(primary_pdf)
            pre_merge_cols = len(primary_pdf.columns)
            
            # Get unique context symbols
            context_symbols_in_df = context_pdf['symbol'].unique()
            log.info(f"Merging {len(context_symbols_in_df)} context symbols: {list(context_symbols_in_df)}")
            
            # Process each context symbol separately
            for ctx_sym in context_symbols_in_df:
                ctx_data = context_pdf[context_pdf['symbol'] == ctx_sym].copy()
                
                # Rename columns to avoid collisions (except 'ts')
                ctx_cols_to_rename = [c for c in ctx_data.columns if c not in ['ts', 'symbol']]
                rename_dict = {c: f"{c}_{ctx_sym}" for c in ctx_cols_to_rename}
                ctx_data = ctx_data.rename(columns=rename_dict)
                
                # Select only renamed columns + ts for merge
                merge_cols = ['ts'] + [f"{c}_{ctx_sym}" for c in ctx_cols_to_rename]
                ctx_data = ctx_data[merge_cols]
                
                # Ensure timestamps are datetime
                if not pd.api.types.is_datetime64_any_dtype(primary_pdf['ts']):
                    primary_pdf['ts'] = pd.to_datetime(primary_pdf['ts'])
                if not pd.api.types.is_datetime64_any_dtype(ctx_data['ts']):
                    ctx_data['ts'] = pd.to_datetime(ctx_data['ts'])
                
                # Merge on timestamp (left join to keep all primary data)
                # This ensures we don't drop primary data if context has gaps
                before_merge = len(primary_pdf)
                primary_pdf = pd.merge(
                    primary_pdf,
                    ctx_data,
                    on='ts',
                    how='left',
                    suffixes=('', f'_dup_{ctx_sym}')
                )
                after_merge = len(primary_pdf)
                
                # Verify no rows were dropped (left join should preserve all rows)
                if after_merge != before_merge:
                    log.error(f"CRITICAL: Row count changed during merge! {before_merge} → {after_merge}")
                    raise ValueError(f"Context merge corrupted data: row count changed")
                
                # Forward-fill context features to handle missing timestamps
                # This is safe because we're only filling gaps, not creating future data
                context_feature_cols = [c for c in primary_pdf.columns if c.endswith(f"_{ctx_sym}")]
                nan_before = primary_pdf[context_feature_cols].isna().sum().sum()
                primary_pdf[context_feature_cols] = primary_pdf[context_feature_cols].ffill()
                nan_after = primary_pdf[context_feature_cols].isna().sum().sum()
                
                log.info(f"  Merged {ctx_sym}: {len(ctx_data)} context rows → {len(context_feature_cols)} features "
                        f"(filled {nan_before - nan_after} NaNs via forward-fill)")
            
            # Verify timestamps are still monotonically increasing (no future leakage)
            if not primary_pdf['ts'].is_monotonic_increasing:
                log.error("CRITICAL: Timestamps are not monotonically increasing after merge!")
                raise ValueError("Context merge broke timestamp ordering - potential future leakage!")
            
            post_merge_rows = len(primary_pdf)
            post_merge_cols = len(primary_pdf.columns)
            added_cols = post_merge_cols - pre_merge_cols
            
            log.info(f"Context merge complete for {primary_symbol}: "
                    f"{pre_merge_rows} rows preserved, "
                    f"{added_cols} context features added "
                    f"({pre_merge_cols} → {post_merge_cols} total columns)")
            
            # Drop metadata columns before validation (not used for training)
            # Common metadata: is_backfilled, symbol, date, partition columns
            metadata_patterns = ['is_backfilled', 'symbol', 'date', 'dt=']
            metadata_cols = [
                c for c in primary_pdf.columns 
                if any(pattern in c for pattern in metadata_patterns)
            ]
            if metadata_cols:
                log.info(f"Dropping {len(metadata_cols)} metadata columns before validation: {metadata_cols}")
                validation_df = primary_pdf.drop(columns=metadata_cols)
            else:
                validation_df = primary_pdf
            
            # === DATA VALIDATION: Check for NaN/null values after merge ===
            self._validate_data_quality(
                df=validation_df,
                stage="after_context_merge",
                symbol=primary_symbol,
                allow_nan_threshold=0.05  # Allow up to 5% NaNs (will be handled by imputation)
            )
            
            # Convert back to Ray Dataset
            return ray.data.from_pandas(primary_pdf)
            
        except Exception as e:
            log.error(f"Failed to merge context features for {primary_symbol}: {e}", exc_info=True)
            log.warning(f"Returning primary dataset without context features due to merge error")
            return primary_ds
    
    def _calculate_context_features(
        self,
        primary_df: pd.DataFrame,
        context_df: pd.DataFrame,
        context_symbol: str,
        windows: List[int] = [50, 200]
    ) -> pd.DataFrame:
        """
        Calculate cross-sectional features from context symbol data.
        
        This is called during indicator calculation if context data is available.
        
        Args:
            primary_df: Primary symbol DataFrame with indicators already calculated
            context_df: Context symbol DataFrame (QQQ, VIX, etc.)
            context_symbol: Name of context symbol ("QQQ", "VIX")
            windows: Window sizes for rolling calculations
            
        Returns:
            DataFrame with added context features
        """
        if context_df.empty or primary_df.empty:
            return primary_df
        
        # Ensure both have timestamps
        if 'ts' not in primary_df.columns or 'ts' not in context_df.columns:
            log.warning(f"Missing timestamp column, skipping context features for {context_symbol}")
            return primary_df
        
        # Ensure timestamps are datetime
        primary_df['ts'] = pd.to_datetime(primary_df['ts'])
        context_df['ts'] = pd.to_datetime(context_df['ts'])
        
        # Calculate indicators on context symbol first
        context_indicators = self.calculate_indicators_gpu(
            batch=context_df.copy(),
            windows=windows,
            drop_warmup=False  # Keep all rows for alignment
        )
        
        # Suffix all context columns (except 'ts') with context symbol name
        context_cols = [col for col in context_indicators.columns if col != 'ts']
        rename_dict = {col: f"{col}_{context_symbol}" for col in context_cols}
        context_indicators = context_indicators.rename(columns=rename_dict)
        
        # Timestamp-aligned join (inner join to ensure exact matches)
        # This automatically drops any rows where timestamps don't align
        merged = pd.merge(
            primary_df,
            context_indicators,
            on='ts',
            how='left'  # Use left join to keep all primary data, fill NaN for missing context
        )
        
        # Forward-fill context features to handle missing timestamps
        context_feature_cols = [col for col in merged.columns if col.endswith(f"_{context_symbol}")]
        merged[context_feature_cols] = merged[context_feature_cols].ffill()
        
        # Calculate relative features (only if we have price data)
        if f'close_{context_symbol}' in merged.columns and 'close' in merged.columns:
            # Relative price strength
            merged[f'close_ratio_{context_symbol}'] = merged['close'] / (merged[f'close_{context_symbol}'] + 1e-9)
            merged[f'close_ratio_{context_symbol}_zscore'] = self._rolling_zscore(
                merged[f'close_ratio_{context_symbol}'],
                window=200
            )
            
            # Relative SMA strength (for each window)
            for window in windows:
                sma_col = f'sma_{window}'
                context_sma_col = f'{sma_col}_{context_symbol}'
                if sma_col in merged.columns and context_sma_col in merged.columns:
                    merged[f'{sma_col}_ratio_{context_symbol}'] = merged[sma_col] / (merged[context_sma_col] + 1e-9)
        
        # Calculate beta (rolling covariance / variance) if we have returns
        if f'returns_{context_symbol}' in merged.columns and 'returns' in merged.columns:
            # 60-bar rolling beta
            returns_primary = merged['returns']
            returns_context = merged[f'returns_{context_symbol}']
            
            # Rolling covariance and variance
            rolling_cov = returns_primary.rolling(window=60).cov(returns_context)
            rolling_var = returns_context.rolling(window=60).var()
            
            merged[f'beta_60_{context_symbol}'] = rolling_cov / (rolling_var + 1e-9)
            merged[f'beta_60_{context_symbol}'] = merged[f'beta_60_{context_symbol}'].fillna(1.0)  # Default beta = 1.0
            
            # Beta-adjusted returns (residual = actual - expected)
            expected_return = merged[f'beta_60_{context_symbol}'] * returns_context
            merged[f'residual_return_{context_symbol}'] = returns_primary - expected_return
            
            # Z-score of residual returns (identifies abnormal moves)
            merged[f'residual_return_{context_symbol}_zscore'] = self._rolling_zscore(
                merged[f'residual_return_{context_symbol}'],
                window=60
            )
        
        # VIX-specific regime features (if context symbol is VIX or VIXY)
        if context_symbol.upper() in ['VIX', 'VIXY']:
            if f'close_{context_symbol}' in merged.columns:
                vix_close = merged[f'close_{context_symbol}']
                
                # VIX z-score (is VIX elevated?)
                merged['vix_zscore'] = self._rolling_zscore(vix_close, window=60)
                
                # High VIX regime (VIX > 20 = fear/uncertainty)
                merged['high_vix_regime'] = (vix_close > 20).astype(int)
                
                # VIX spike detection (VIX jumped > 1 std dev)
                vix_returns = vix_close.pct_change()
                vix_spike_threshold = vix_returns.rolling(window=20).std()
                merged['vix_spike'] = (vix_returns > vix_spike_threshold).astype(int)
                
                # Log return of VIX (measures fear velocity)
                merged['vix_log_return'] = np.log(vix_close / vix_close.shift(1))
                merged['vix_log_return'] = merged['vix_log_return'].fillna(0.0)
        
        log.debug(f"Added context features from {context_symbol}: {len(context_feature_cols)} base features + relative/beta features")
        
        # Validate data quality after context feature calculation
        nan_counts = merged[context_feature_cols].isna().sum()
        high_nan_cols = nan_counts[nan_counts > len(merged) * 0.1].index.tolist()  # >10% NaN
        if high_nan_cols:
            log.warning(f"Context features with >10% NaN after calculation: {high_nan_cols}")
        
        return merged
    
    def calculate_indicators_gpu(
        self,
        batch: pd.DataFrame,
        windows: List[int] = [50, 200],
        resampling_timeframes: Optional[List[str]] = None,
        drop_warmup: bool = True,
        zscore_window: int = 200
    ) -> pd.DataFrame:
        """
        GPU-accelerated indicator calculation with strict no-look-ahead.
        
        IMPORTANT: All indicators are calculated on BARS (rows), not calendar days.
        - 200-bar SMA = 200 data points (e.g., 200 minutes for 1-min data)
        - For 1-min data: 200 bars = ~3.3 hours of trading
        - For daily data: 200 bars = 200 trading days
        
        This ensures indicators work correctly regardless of timeframe.
        With walk-forward folds (2-3 months of 1-min data), we have plenty
        of bars (~40,000 per month) for any indicator window.
        
        This function is designed to run on a SINGLE fold's data, ensuring
        that SMAs reset at the beginning of each fold and cannot peek across splits.
        
        Indicators calculated:
        - Time features: time_sin, time_cos, day_of_week_sin, day_of_week_cos
        - Returns: returns, log_returns
        - Price features: price_range, price_range_pct
        - Trend: SMA (custom windows + 20), EMA (custom windows + 12, 26)
        - Momentum: RSI-14, Stochastic Oscillator (%K, %D)
        - Volatility: ATR-14, Bollinger Bands (upper, mid, lower)
        - Trend: MACD, MACD Signal, MACD Diff
        - Volume: Volume MA, Volume Ratio, OBV
        - VWAP Distance
        
        Args:
            batch: DataFrame with OHLCV data
            windows: SMA/EMA window sizes in BARS (default: [50, 200])
            resampling_timeframes: Multi-timeframe aggregations (5min, 15min, etc.)
            drop_warmup: Drop rows where indicators are NaN (warm-up period)
            zscore_window: Window for rolling z-score normalization (default: 200)
            
        Returns:
            DataFrame with calculated indicators (raw + normalized versions)
        """
        # Handle empty batches
        if batch.empty:
            log.debug("Empty batch received, returning as-is")
            return batch
        
        # Ensure timestamp is datetime
        batch['ts'] = pd.to_datetime(batch['ts'])
        batch = batch.sort_values('ts')
        
        # Time-based features with sin/cos encoding
        hour = batch['ts'].dt.hour
        minute = batch['ts'].dt.minute
        day_of_week = batch['ts'].dt.dayofweek
        
        # Calculate minutes since midnight (0-1439)
        minutes_of_day = hour * 60 + minute
        
        # Sin/Cos encoding for cyclical time (minutes in a day: 0-1439)
        batch['time_sin'] = np.sin(2 * np.pi * minutes_of_day / 1440)
        batch['time_cos'] = np.cos(2 * np.pi * minutes_of_day / 1440)
        
        # Sin/Cos encoding for day of week (0-6)
        batch['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        batch['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Raw time features (optional, for tree-based models)
        batch['hour'] = hour
        batch['day_of_week'] = day_of_week
        batch['day_of_month'] = batch['ts'].dt.day
        batch['month'] = batch['ts'].dt.month
        
        # Market session features
        batch['is_market_open'] = ((hour >= 9) & (hour < 16)).astype(int)
        batch['is_morning'] = ((hour >= 9) & (hour < 12)).astype(int)
        
        # Basic returns (already have no look-ahead)
        batch['returns'] = batch['close'].pct_change()
        batch['log_returns'] = np.log(batch['close'] / batch['close'].shift(1))
        
        # Price features
        batch['price_range'] = batch['high'] - batch['low']
        batch['price_range_pct'] = batch['price_range'] / batch['close']
        
        # SMAs - Phase 1: Raw calculation on actual prices
        for window in windows:
            batch[f'sma_{window}'] = batch['close'].rolling(window=window, min_periods=window).mean()
            batch[f'volume_ma_{window}'] = batch['volume'].rolling(window=window, min_periods=window).mean()
            
            # Volatility (standard deviation of returns)
            batch[f'volatility_{window}'] = batch['returns'].rolling(window=window, min_periods=window).std()
            
            # Distance from SMA (Phase 3: Already normalized as % deviation)
            batch[f'dist_sma_{window}'] = (batch['close'] - batch[f'sma_{window}']) / batch[f'sma_{window}']
        
        # Phase 4: Rolling Z-Score for raw SMA values (in price units)
        for window in windows:
            batch[f'sma_{window}_zscore'] = self._rolling_zscore(batch[f'sma_{window}'], window=zscore_window)
            batch[f'volume_ma_{window}_zscore'] = self._rolling_zscore(batch[f'volume_ma_{window}'], window=zscore_window)
            batch[f'volatility_{window}_zscore'] = self._rolling_zscore(batch[f'volatility_{window}'], window=zscore_window)
        
        # Additional standard SMAs from feature_service
        if 20 not in windows:
            batch['sma_20'] = batch['close'].rolling(window=20, min_periods=20).mean()
            batch['sma_20_zscore'] = self._rolling_zscore(batch['sma_20'], window=zscore_window)
        
        # EMA (Exponential Moving Average) - Phase 1: Raw calculation
        for window in windows:
            batch[f'ema_{window}'] = batch['close'].ewm(span=window, min_periods=window).mean()
        
        # Phase 4: Rolling Z-Score for raw EMA values
        for window in windows:
            batch[f'ema_{window}_zscore'] = self._rolling_zscore(batch[f'ema_{window}'], window=zscore_window)
        
        # Additional standard EMAs from feature_service
        if 12 not in windows:
            batch['ema_12'] = batch['close'].ewm(span=12, min_periods=12).mean()
            batch['ema_12_zscore'] = self._rolling_zscore(batch['ema_12'], window=zscore_window)
        if 26 not in windows:
            batch['ema_26'] = batch['close'].ewm(span=26, min_periods=26).mean()
            batch['ema_26_zscore'] = self._rolling_zscore(batch['ema_26'], window=zscore_window)
        
        # RSI (Relative Strength Index) - Phase 1: Raw calculation on price changes
        batch['rsi_14'] = self._calculate_rsi(batch['close'], period=14)
        
        # Phase 3: Normalization (center at 0, range -1 to 1)
        # RSI is 0-100 scale, we center at 50
        batch['rsi_norm'] = (batch['rsi_14'] - 50) / 50
        
        # Phase 4: Rolling Z-Score (adaptive to market regime)
        batch['rsi_zscore'] = self._rolling_zscore(batch['rsi_14'], window=zscore_window)
        
        # Stochastic Oscillator (Phase 1: Raw calculation on actual prices)
        batch['stoch_k'], batch['stoch_d'] = self._calculate_stochastic(
            high=batch['high'],
            low=batch['low'],
            close=batch['close'],
            k_period=14,
            d_period=3
        )
        
        # Phase 3: Normalization (center at 0, range -1 to 1)
        # Stochastic is 0-100 scale, we center at 50
        batch['stoch_k_norm'] = (batch['stoch_k'] - 50) / 50
        batch['stoch_d_norm'] = (batch['stoch_d'] - 50) / 50
        
        # Phase 4: Rolling Z-Score (adaptive to market regime)
        batch['stoch_k_zscore'] = self._rolling_zscore(batch['stoch_k'], window=zscore_window)
        batch['stoch_d_zscore'] = self._rolling_zscore(batch['stoch_d'], window=zscore_window)
        
        # MACD (Phase 1: Raw calculation on actual prices)
        batch['macd'], batch['macd_signal'] = self._calculate_macd(batch['close'])
        batch['macd_diff'] = batch['macd'] - batch['macd_signal']
        
        # Phase 4: Rolling Z-Score (MACD values vary by stock price, need normalization)
        # MACD and signal are in price units (e.g., $2 for AAPL), not comparable across stocks
        batch['macd_zscore'] = self._rolling_zscore(batch['macd'], window=zscore_window)
        batch['macd_signal_zscore'] = self._rolling_zscore(batch['macd_signal'], window=zscore_window)
        batch['macd_diff_zscore'] = self._rolling_zscore(batch['macd_diff'], window=zscore_window)
        
        # Bollinger Bands (Phase 1: Raw calculation on actual prices)
        batch['bb_upper'], batch['bb_mid'], batch['bb_lower'] = self._calculate_bollinger_bands(
            close=batch['close'],
            window=20,
            std_dev=2
        )
        
        # Phase 3: Derived features (position within bands, band width)
        # BB Position: Where is price within the bands? (0 = at lower, 1 = at upper)
        bb_range = batch['bb_upper'] - batch['bb_lower']
        batch['bb_position'] = (batch['close'] - batch['bb_lower']) / (bb_range + 1e-9)
        
        # BB Width: How volatile is the market? (wider bands = higher volatility)
        batch['bb_width'] = bb_range
        
        # BB Width as % of price (normalized version)
        batch['bb_width_pct'] = bb_range / batch['close']
        
        # Phase 4: Rolling Z-Score (for raw band values in price units)
        batch['bb_upper_zscore'] = self._rolling_zscore(batch['bb_upper'], window=zscore_window)
        batch['bb_mid_zscore'] = self._rolling_zscore(batch['bb_mid'], window=zscore_window)
        batch['bb_lower_zscore'] = self._rolling_zscore(batch['bb_lower'], window=zscore_window)
        batch['bb_width_zscore'] = self._rolling_zscore(batch['bb_width'], window=zscore_window)
        
        # Average True Range (ATR) - Phase 1: Raw calculation on actual price ranges
        batch['atr_14'] = self._calculate_atr(
            high=batch['high'],
            low=batch['low'],
            close=batch['close'],
            period=14
        )
        
        # Phase 3: ATR as % of price (normalized, comparable across stocks)
        batch['atr_pct'] = batch['atr_14'] / batch['close']
        
        # Phase 4: Rolling Z-Score (adaptive to volatility regimes)
        batch['atr_zscore'] = self._rolling_zscore(batch['atr_14'], window=zscore_window)
        batch['atr_pct_zscore'] = self._rolling_zscore(batch['atr_pct'], window=zscore_window)
        
        # On Balance Volume (OBV) - Phase 1: Raw cumulative calculation
        batch['obv'] = self._calculate_obv(
            close=batch['close'],
            volume=batch['volume']
        )
        
        # Phase 4: Rolling Z-Score (OBV is cumulative, grows unbounded)
        # Z-score makes it comparable across different time periods and stocks
        batch['obv_zscore'] = self._rolling_zscore(batch['obv'], window=zscore_window)
        
        # Multi-timeframe resampling
        if resampling_timeframes:
            for tf in resampling_timeframes:
                batch = self._add_resampled_features(batch, timeframe=tf)
        
        # Volume indicators
        # Volume Ratio: Current volume vs 20-period average (Phase 3: already normalized as ratio)
        batch['volume_ratio'] = batch['volume'] / batch['volume'].rolling(window=20, min_periods=20).mean()
        
        # Phase 4: Rolling Z-Score (helps identify extreme volume spikes)
        batch['volume_ratio_zscore'] = self._rolling_zscore(batch['volume_ratio'], window=zscore_window)
        
        # VWAP Distance (already normalized as ratio)
        batch['vwap_dist'] = (batch['close'] - batch['vwap']) / batch['vwap'] if 'vwap' in batch.columns else 0
        
        # Drop warm-up period if requested
        if drop_warmup:
            # Find the maximum window size to know when all indicators are "warm"
            max_window = max(windows) if windows else 200
            batch = batch.iloc[max_window:].reset_index(drop=True)
            log.debug(f"Dropped {max_window} warm-up rows, {len(batch)} rows remain")
        
        return batch
    
    def _rolling_zscore(self, series: pd.Series, window: int = 200) -> pd.Series:
        """
        Calculate rolling z-score for adaptive normalization.
        
        Z-score = (value - rolling_mean) / rolling_std
        
        This normalizes indicators relative to recent market regime,
        making them comparable across different volatility periods.
        
        Args:
            series: Input series to normalize
            window: Rolling window size (default: 200 bars)
            
        Returns:
            Z-score normalized series
        """
        rolling_mean = series.rolling(window=window, min_periods=window).mean()
        rolling_std = series.rolling(window=window, min_periods=window).std()
        
        # Add small epsilon to prevent division by zero
        zscore = (series - rolling_mean) / (rolling_std + 1e-9)
        
        return zscore
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
        ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, min_periods=signal).mean()
        return macd, signal_line
    
    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        %K = 100 * (Close - Low14) / (High14 - Low14)
        %D = 3-period SMA of %K
        """
        # Calculate rolling min/max
        low_min = low.rolling(window=k_period, min_periods=k_period).min()
        high_max = high.rolling(window=k_period, min_periods=k_period).max()
        
        # %K calculation
        stoch_k = 100 * (close - low_min) / (high_max - low_min)
        
        # %D is simple moving average of %K
        stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
        
        return stoch_k, stoch_d
    
    def _calculate_bollinger_bands(
        self,
        close: pd.Series,
        window: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Returns: (upper_band, middle_band, lower_band)
        """
        # Middle band is SMA
        bb_mid = close.rolling(window=window, min_periods=window).mean()
        
        # Standard deviation
        std = close.rolling(window=window, min_periods=window).std()
        
        # Upper and lower bands
        bb_upper = bb_mid + (std * std_dev)
        bb_lower = bb_mid - (std * std_dev)
        
        return bb_upper, bb_mid, bb_lower
    
    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        True Range = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
        ATR = SMA of True Range over period
        """
        # Previous close
        prev_close = close.shift(1)
        
        # True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the moving average of True Range
        atr = true_range.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    def _calculate_obv(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate On Balance Volume (OBV).
        
        If close > prev_close: OBV = prev_OBV + volume
        If close < prev_close: OBV = prev_OBV - volume
        If close == prev_close: OBV = prev_OBV
        """
        # Calculate price changes
        price_change = close.diff()
        
        # Volume direction: +1 if price up, -1 if price down, 0 if unchanged
        volume_direction = np.sign(price_change)
        
        # Signed volume
        signed_volume = volume * volume_direction
        
        # Cumulative sum
        obv = signed_volume.cumsum()
        
        return obv
    
    def _add_resampled_features(
        self,
        batch: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Resample 1-min bars to higher timeframes (5min, 15min, 1H).
        
        Creates features like close_5min, sma50_15min, etc.
        """
        batch['ts'] = pd.to_datetime(batch['ts'])
        batch_resampled = batch.set_index('ts').resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        # Calculate SMA on resampled data
        batch_resampled[f'close_{timeframe}'] = batch_resampled['close']
        batch_resampled[f'sma50_{timeframe}'] = batch_resampled['close'].rolling(50, min_periods=50).mean()
        
        # Merge back to original data (forward-fill)
        batch = batch.merge(
            batch_resampled[['ts', f'close_{timeframe}', f'sma50_{timeframe}']],
            on='ts',
            how='left'
        )
        batch[[f'close_{timeframe}', f'sma50_{timeframe}']] = batch[[f'close_{timeframe}', f'sma50_{timeframe}']].ffill()
        
        return batch
    
    def process_fold_with_gpu(
        self,
        fold: Fold,
        num_gpus: float = 1.0,
        actor_pool_size: Optional[int] = None,
        windows: List[int] = [50, 200],
        resampling_timeframes: Optional[List[str]] = None
    ) -> Fold:
        """
        Process a fold with GPU acceleration.
        
        This ensures indicators are calculated ONLY on fold data,
        preventing any look-ahead bias across train/test splits.
        
        Args:
            fold: Fold with loaded datasets
            num_gpus: Number of GPUs per actor
            actor_pool_size: Number of actors in pool
            windows: SMA window sizes
            resampling_timeframes: Multi-timeframe aggregations
            
        Returns:
            Fold with processed datasets
        """
        import os
        
        # Auto-detect CPUs if not specified
        if actor_pool_size is None:
            actor_pool_size = os.cpu_count() or 4
        
        log.info(f"Processing {fold} with {actor_pool_size} parallel actors (GPU={num_gpus > 0})")
        
        # Optimize batch size based on actor count (larger batches = less overhead)
        # Each actor processes one batch at a time, so larger batches are more efficient
        optimal_batch_size = 50000 if actor_pool_size >= 8 else 25000
        
        # Create a wrapper function for map_batches
        def process_batch(batch):
            return self.calculate_indicators_gpu(
                batch,
                windows=windows,
                resampling_timeframes=resampling_timeframes,
                drop_warmup=True
            )
        
        # Process train data
        if fold.train_ds:
            if num_gpus > 0:
                # GPU acceleration with concurrency
                fold.train_ds = fold.train_ds.map_batches(
                    process_batch,
                    batch_format="pandas",
                    batch_size=optimal_batch_size,
                    concurrency=actor_pool_size,
                    num_gpus=num_gpus,
                    num_cpus=1  # Reserve 1 CPU per GPU actor
                )
                log.info(f"Using {actor_pool_size} GPU actors (batch_size={optimal_batch_size}) for train data")
            elif actor_pool_size > 1:
                # CPU parallelism (no GPU)
                fold.train_ds = fold.train_ds.map_batches(
                    process_batch,
                    batch_format="pandas",
                    batch_size=optimal_batch_size,
                    concurrency=actor_pool_size,
                    num_cpus=1,  # Reserve 1 CPU per actor
                    zero_copy_batch=True  # Reduce memory copying
                )
                log.info(f"Using {actor_pool_size} CPU actors (batch_size={optimal_batch_size}) for train data")
            else:
                # Single-threaded CPU processing
                fold.train_ds = fold.train_ds.map_batches(
                    process_batch,
                    batch_format="pandas",
                    batch_size=optimal_batch_size
                )
            log.info(f"Processed train data for {fold}")
        
        # Process test data (separate calculation, no leakage!)
        if fold.test_ds:
            if num_gpus > 0:
                # GPU acceleration with concurrency
                fold.test_ds = fold.test_ds.map_batches(
                    process_batch,
                    batch_format="pandas",
                    batch_size=optimal_batch_size,
                    concurrency=actor_pool_size,
                    num_gpus=num_gpus,
                    num_cpus=1
                )
                log.info(f"Using {actor_pool_size} GPU actors (batch_size={optimal_batch_size}) for test data")
            elif actor_pool_size > 1:
                # CPU parallelism (no GPU)
                fold.test_ds = fold.test_ds.map_batches(
                    process_batch,
                    batch_format="pandas",
                    batch_size=optimal_batch_size,
                    concurrency=actor_pool_size,
                    num_cpus=1,
                    zero_copy_batch=True
                )
                log.info(f"Using {actor_pool_size} CPU actors (batch_size={optimal_batch_size}) for test data")
            else:
                # Single-threaded CPU processing
                fold.test_ds = fold.test_ds.map_batches(
                    process_batch,
                    batch_format="pandas",
                    batch_size=optimal_batch_size
                )
            log.info(f"Processed test data for {fold}")
        
        # === FINAL VALIDATION: Check both train and test datasets ===
        if fold.train_ds:
            train_sample = fold.train_ds.take(1000)  # Sample for validation
            if train_sample:
                train_df = pd.DataFrame(train_sample)
                # Drop metadata columns before validation
                metadata_patterns = ['is_backfilled', 'symbol', 'date', 'dt=']
                metadata_cols = [
                    c for c in train_df.columns 
                    if any(pattern in c for pattern in metadata_patterns)
                ]
                if metadata_cols:
                    train_df = train_df.drop(columns=metadata_cols)
                
                self._validate_data_quality(
                    df=train_df,
                    stage="after_indicator_calculation_train",
                    symbol="train_fold",
                    allow_nan_threshold=0.02  # Stricter after indicators
                )
        
        if fold.test_ds:
            test_sample = fold.test_ds.take(1000)  # Sample for validation
            if test_sample:
                test_df = pd.DataFrame(test_sample)
                # Drop metadata columns before validation
                metadata_patterns = ['is_backfilled', 'symbol', 'date', 'dt=']
                metadata_cols = [
                    c for c in test_df.columns 
                    if any(pattern in c for pattern in metadata_patterns)
                ]
                if metadata_cols:
                    test_df = test_df.drop(columns=metadata_cols)
                
                self._validate_data_quality(
                    df=test_df,
                    stage="after_indicator_calculation_test",
                    symbol="test_fold",
                    allow_nan_threshold=0.02  # Stricter after indicators
                )
        
        return fold
    
    def _validate_data_quality(
        self,
        df: pd.DataFrame,
        stage: str,
        symbol: str,
        allow_nan_threshold: float = 0.05
    ) -> None:
        """
        Validate data quality after processing stages.
        
        Args:
            df: DataFrame to validate
            stage: Processing stage name (for logging)
            symbol: Symbol being processed
            allow_nan_threshold: Maximum allowed NaN percentage (0.0 to 1.0)
            
        Raises:
            ValueError: If data quality issues exceed thresholds
        """
        if df.empty:
            log.warning(f"Validation [{stage}] for {symbol}: DataFrame is empty")
            return
        
        total_cells = df.shape[0] * df.shape[1]
        nan_count = df.isna().sum().sum()
        nan_pct = nan_count / total_cells if total_cells > 0 else 0
        
        log.info(f"""Data Quality Validation [{stage}] for {symbol}:
          Rows: {len(df):,}
          Columns: {len(df.columns)}
          Total cells: {total_cells:,}
          NaN cells: {nan_count:,} ({nan_pct*100:.2f}%)
          Threshold: {allow_nan_threshold*100:.1f}%""")
        
        # Check overall NaN percentage
        if nan_pct > allow_nan_threshold:
            # Identify columns with high NaN percentage
            nan_by_col = df.isna().sum()
            high_nan_cols = nan_by_col[nan_by_col > len(df) * 0.1].sort_values(ascending=False)
            
            error_msg = (
                f"VALIDATION FAILED [{stage}]: NaN percentage {nan_pct*100:.2f}% "
                f"exceeds threshold {allow_nan_threshold*100:.1f}%\n"
                f"Columns with >10% NaN (top 10):\n"
            )
            for col, count in high_nan_cols.head(10).items():
                col_pct = count / len(df) * 100
                error_msg += f"  - {col}: {count:,}/{len(df):,} ({col_pct:.1f}%)\n"
            
            # Add suggestion
            metadata_keywords = ['is_backfilled', 'symbol', 'date', 'partition', 'dt=']
            likely_metadata = [
                col for col in high_nan_cols.index[:10] 
                if any(kw in col.lower() for kw in metadata_keywords)
            ]
            if likely_metadata:
                error_msg += f"\nNote: Columns {likely_metadata} appear to be metadata and should be dropped before validation.\n"
            
            log.error(error_msg)
            raise ValueError(error_msg)
        
        # Check for columns that are entirely NaN
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            log.warning(f"Columns with 100% NaN (will be dropped): {all_nan_cols}")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            log.warning(f"Columns with infinite values: {inf_counts}")
            # Replace inf with NaN for downstream handling
            for col in inf_counts:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                log.info(f"Replaced {inf_counts[col]} infinite values with NaN in {col}")
        
        # Summary
        if nan_pct <= allow_nan_threshold and not inf_counts:
            log.info(f"✅ Validation [{stage}] PASSED for {symbol}")
        else:
            log.warning(f"⚠️  Validation [{stage}] passed with warnings for {symbol}")
    
    def add_basic_features(self, ds: Dataset) -> Dataset:
        """Add basic technical features using Ray Data map."""
        
        def compute_returns(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            """Compute returns and log returns."""
            close = batch["close"]
            
            # Simple returns
            batch["returns"] = np.concatenate([[np.nan], np.diff(close) / close[:-1]])
            
            # Log returns
            batch["log_returns"] = np.concatenate([[np.nan], np.diff(np.log(close))])
            
            # Price range
            batch["price_range"] = batch["high"] - batch["low"]
            batch["price_range_pct"] = batch["price_range"] / batch["close"]
            
            return batch
        
        return ds.map_batches(
            compute_returns,
            batch_format="numpy",
            batch_size=1000
        )
    
    def add_rolling_features(
        self,
        ds: Dataset,
        windows: List[int] = [5, 10, 20]
    ) -> Dataset:
        """Add rolling window features (SMA, volatility, etc.)."""
        
        def compute_rolling(batch: pd.DataFrame) -> pd.DataFrame:
            """Compute rolling statistics."""
            for window in windows:
                # Simple Moving Average
                batch[f"sma_{window}"] = batch["close"].rolling(window).mean()
                
                # Rolling volatility
                if "returns" in batch.columns:
                    batch[f"volatility_{window}"] = (
                        batch["returns"].rolling(window).std()
                    )
                
                # Rolling volume
                batch[f"volume_ma_{window}"] = batch["volume"].rolling(window).mean()
            
            return batch
        
        return ds.map_batches(
            compute_rolling,
            batch_format="pandas",
            batch_size=1000
        )
    
    def add_time_features(self, ds: Dataset) -> Dataset:
        """
        Extract time-based features including sin/cos encoding for cyclical time.
        
        Time is encoded as minutes since midnight (0-1439), then transformed into
        sin/cos pairs to capture the circular nature of time where 23:59 is close to 00:01.
        
        Formula:
            minutes_of_day = hour * 60 + minute
            time_sin = sin(2π * minutes_of_day / 1440)
            time_cos = cos(2π * minutes_of_day / 1440)
        
        This allows linear models to understand that late evening (1439 min) is
        adjacent to early morning (0 min) in the time cycle.
        """
        
        def extract_time_features(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            """Extract hour, day of week, and cyclical encodings."""
            timestamps = pd.to_datetime(batch["ts"])
            
            # Basic time components
            hour = timestamps.hour.values
            minute = timestamps.minute.values
            day_of_week = timestamps.dayofweek.values
            
            # Calculate minutes since midnight (0-1439)
            minutes_of_day = hour * 60 + minute
            
            # Sin/Cos encoding for cyclical time (minutes in a day: 0-1439)
            # Creates a circular feature space where 23:59 (1439 min) is close to 00:01 (1 min)
            batch["time_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
            batch["time_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)
            
            # Sin/Cos encoding for day of week (0-6)
            # Monday (0) and Sunday (6) are adjacent in the cycle
            batch["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
            batch["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Keep raw values for non-linear models (optional)
            batch["hour"] = hour
            batch["day_of_week"] = day_of_week
            batch["day_of_month"] = timestamps.day.values
            batch["month"] = timestamps.month.values
            
            # Market session features
            batch["is_market_open"] = (
                (hour >= 9) & (hour < 16)
            ).astype(int)
            
            batch["is_morning"] = (
                (hour >= 9) & (hour < 12)
            ).astype(int)
            
            return batch
        
        return ds.map_batches(
            extract_time_features,
            batch_format="numpy",
            batch_size=1000
        )
    
    def filter_market_hours(self, ds: Dataset) -> Dataset:
        """Filter to only market hours (9:30 AM - 4:00 PM ET)."""
        
        def is_market_hours(batch: pd.DataFrame) -> pd.DataFrame:
            timestamps = pd.to_datetime(batch["ts"])
            mask = (
                (timestamps.dt.hour >= 9) & (timestamps.dt.hour < 16) &
                (timestamps.dt.dayofweek < 5)  # Weekdays only
            )
            return batch[mask]
        
        return ds.map_batches(
            is_market_hours,
            batch_format="pandas",
            batch_size=1000
        )
    
    def create_training_pipeline(
        self,
        symbols: Optional[List[str]] = None,
        market_hours_only: bool = True,
        rolling_windows: List[int] = [5, 10, 20]
    ) -> Dataset:
        """
        Create complete preprocessing pipeline for training.
        
        DEPRECATED: Use create_walk_forward_pipeline for proper train/test splits.
        
        Args:
            symbols: Symbols to process (None = all)
            market_hours_only: Filter to market hours only
            rolling_windows: Window sizes for rolling features
            
        Returns:
            Preprocessed Ray Dataset ready for training
        """
        log.warning("create_training_pipeline is deprecated. Use create_walk_forward_pipeline instead.")
        
        log.info("Starting streaming preprocessing pipeline")
        
        # Step 1: Load data
        ds = self.loader.load_all_bars(symbols=symbols)
        
        # Step 2: Add basic features
        ds = self.add_basic_features(ds)
        log.info("Added basic features")
        
        # Step 3: Add time features
        ds = self.add_time_features(ds)
        log.info("Added time features")
        
        # Step 4: Filter to market hours if requested
        if market_hours_only:
            ds = self.filter_market_hours(ds)
            log.info("Filtered to market hours")
        
        # Step 5: Add rolling features (grouped by symbol)
        # Note: This requires grouping which we'll handle per-symbol
        ds = self.add_rolling_features(ds, windows=rolling_windows)
        log.info("Added rolling features")
        
        log.info("Preprocessing pipeline complete")
        return ds
    
    def create_walk_forward_pipeline(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        train_months: int = 3,
        test_months: int = 1,
        step_months: int = 1,
        context_symbols: Optional[List[str]] = None,
        windows: List[int] = [50, 200],
        resampling_timeframes: Optional[List[str]] = None,
        num_gpus: float = 0.0,  # Set to 1.0 when running on GPU
        actor_pool_size: Optional[int] = None,  # None = auto-detect all CPUs
        use_cached_folds: bool = True,  # NEW: Use pre-computed folds if available
        save_folds: bool = True  # NEW: Save computed folds to disk
    ) -> Generator[Fold, None, None]:
        """
        Create walk-forward validation pipeline with GPU acceleration and caching.
        
        This is the RECOMMENDED way to preprocess data for balanced backtesting.
        Each fold calculates indicators independently, preventing look-ahead bias.
        
        **PERFORMANCE OPTIMIZATION**: Set `use_cached_folds=True` to load pre-processed
        folds from disk instead of recalculating features. This saves 90%+ time on
        repeated training runs.
        
        Args:
            symbols: Primary trading symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            train_months: Training window size
            test_months: Test window size
            step_months: Step size for rolling window
            context_symbols: Context symbols like QQQ, VIX
            windows: SMA window sizes
            resampling_timeframes: Multi-timeframe aggregations (e.g., ['5min', '15min'])
            num_gpus: GPUs per actor (0 for CPU, 1.0 for GPU)
            actor_pool_size: Number of parallel actors
            use_cached_folds: Load pre-computed folds from disk if available (FAST)
            save_folds: Save computed folds to disk for future runs
            
        Yields:
            Fold objects with processed train_ds and test_ds
            
        Example:
            ```python
            preprocessor = StreamingPreprocessor(loader)
            
            # First run: calculates and saves folds (slow)
            for fold in preprocessor.create_walk_forward_pipeline(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                context_symbols=["QQQ", "VIX"],
                use_cached_folds=True,  # Will use cache if exists
                save_folds=True  # Save for next time
            ):
                pass
            
            # Subsequent runs: loads from cache (fast!)
            # 90%+ time savings by reusing pre-computed features
            ```
        """
        log.info(f"Creating walk-forward pipeline: {symbols} from {start_date} to {end_date}")
        
        # Generate folds
        folds = self.generate_walk_forward_folds(
            start_date=start_date,
            end_date=end_date,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months
        )
        
        # Process each fold
        for fold in folds:
            # Check if cached fold exists
            if use_cached_folds and len(symbols) == 1:
                cached_fold = self._try_load_cached_fold(fold, symbols[0])
                if cached_fold is not None:
                    log.info(f"✓ Using cached fold {fold.fold_id} for {symbols[0]} (skipping recalculation)")
                    yield cached_fold
                    continue
            
            # Cache miss or caching disabled - compute from scratch
            log.info(f"Computing fold {fold.fold_id} from scratch...")
            
            # Load data
            fold = self.load_fold_data(
                fold=fold,
                symbols=symbols,
                context_symbols=context_symbols
            )
            
            # Validate fold has data before processing
            try:
                train_count = fold.train_ds.count() if fold.train_ds else 0
                test_count = fold.test_ds.count() if fold.test_ds else 0
                
                if train_count == 0 or test_count == 0:
                    raise ValueError(
                        f"Fold {fold.fold_id} has no data:\n"
                        f"  Train: {fold.train_start} to {fold.train_end} ({train_count} rows)\n"
                        f"  Test: {fold.test_start} to {fold.test_end} ({test_count} rows)\n"
                        f"  Check that parquet files exist for these dates in {self.loader.parquet_dir}"
                    )
                
                log.info(f"Fold {fold.fold_id} validation passed: train={train_count:,} rows, test={test_count:,} rows")
            except Exception as e:
                log.error(f"Failed to validate {fold}: {e}")
                raise
            
            # Calculate indicators with GPU
            fold = self.process_fold_with_gpu(
                fold=fold,
                num_gpus=num_gpus,
                actor_pool_size=actor_pool_size,
                windows=windows,
                resampling_timeframes=resampling_timeframes
            )
            
            # Save fold to disk for future runs
            if save_folds and len(symbols) == 1:
                self._save_fold_to_cache(fold, symbols[0])
            
            yield fold
    
    def save_processed_data(
        self,
        ds: Dataset,
        output_path: str,
        partition_by: Optional[List[str]] = None
    ):
        """Save processed dataset to parquet."""
        log.info(f"Saving processed data to {output_path}")
        
        ds.write_parquet(
            output_path,
            partition_cols=partition_by or ["symbol"],
            try_create_dir=True
        )
        
        log.info(f"Saved to {output_path}")


def create_preprocessing_pipeline(
    parquet_dir: str = "/app/data/parquet",
    symbols: Optional[List[str]] = None
) -> StreamingPreprocessor:
    """
    Factory function to create a preprocessing pipeline.
    
    Example usage:
        preprocessor = create_preprocessing_pipeline()
        ds = preprocessor.create_training_pipeline(symbols=["AAPL", "MSFT"])
        ds.show(5)
    """
    loader = BarDataLoader(parquet_dir=parquet_dir)
    return StreamingPreprocessor(loader)
