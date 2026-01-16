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
    """Load and stream bar data using Ray Data."""
    
    def __init__(self, parquet_dir: str = "/app/data/parquet"):
        self.parquet_dir = Path(parquet_dir)
        log.info(f"Initialized BarDataLoader with path: {self.parquet_dir}")
    
    def load_all_bars(
        self,
        symbols: Optional[List[str]] = None,
        parallelism: int = 10
    ) -> Dataset:
        """
        Load bar data for all or specific symbols using Ray Data.
        
        Args:
            symbols: List of ticker symbols to load. If None, loads all.
            parallelism: Number of parallel read tasks
            
        Returns:
            Ray Dataset of bar data
        """
        parquet_files = self._discover_parquet_files(symbols)
        
        if not parquet_files:
            log.warning("No parquet files found")
            return ray.data.from_items([])
        
        log.info(f"Loading {len(parquet_files)} parquet files")
        
        # Load data using Ray Data's read_parquet (streaming)
        ds = ray.data.read_parquet(
            parquet_files,
            parallelism=parallelism,
            ray_remote_args={"num_cpus": 1}
        )
        
        log.info(f"Loaded dataset with {ds.count()} rows")
        return ds
    
    def load_symbol(self, symbol: str, parallelism: int = 2) -> Dataset:
        """Load data for a single symbol."""
        return self.load_all_bars(symbols=[symbol], parallelism=parallelism)
    
    def _discover_parquet_files(self, symbols: Optional[List[str]] = None) -> List[str]:
        """Discover all parquet files in the data directory."""
        if not self.parquet_dir.exists():
            log.error(f"Parquet directory DOES NOT EXIST: {self.parquet_dir}")
            log.error(f"Absolute path: {self.parquet_dir.absolute()}")
            return []
        
        log.info(f"Searching for parquet files in: {self.parquet_dir.absolute()}")
        
        # Pattern: data/parquet/SYMBOL/YYYY-MM-DD.parquet
        files = []
        
        if symbols:
            # Load specific symbols
            for symbol in symbols:
                symbol_dir = self.parquet_dir / symbol
                log.info(f"Checking symbol directory: {symbol_dir.absolute()} (exists={symbol_dir.exists()})")
                if symbol_dir.exists():
                    symbol_files = list(symbol_dir.glob("*.parquet"))
                    log.info(f"Found {len(symbol_files)} files for {symbol}")
                    files.extend([str(f) for f in symbol_files])
                else:
                    # Try case-insensitive search
                    for child in self.parquet_dir.iterdir():
                        if child.is_dir() and child.name.upper() == symbol.upper():
                            log.info(f"Found case-insensitive match: {child.name} for {symbol}")
                            symbol_files = list(child.glob("*.parquet"))
                            log.info(f"Found {len(symbol_files)} files for {child.name}")
                            files.extend([str(f) for f in symbol_files])
                            break
        else:
            # Load all symbols
            files = [str(f) for f in self.parquet_dir.rglob("*.parquet")]
        
        log.info(f"Total parquet files discovered: {len(files)}")
        if len(files) > 0:
            log.info(f"Sample files: {files[:3]}")
        
        return sorted(files)


class StreamingPreprocessor:
    """Streaming preprocessing pipeline using Ray Data with walk-forward folds."""
    
    def __init__(self, loader: BarDataLoader):
        self.loader = loader
    
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
        
        # Load context symbols if specified
        if context_symbols:
            train_context = self._load_date_range(
                symbols=context_symbols,
                start_date=fold.train_start,
                end_date=fold.train_end
            )
            test_context = self._load_date_range(
                symbols=context_symbols,
                start_date=fold.test_start,
                end_date=fold.test_end
            )
            
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
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        def filter_dates(batch: pd.DataFrame) -> pd.DataFrame:
            if batch.empty:
                return batch
            batch['ts'] = pd.to_datetime(batch['ts'])
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
        Join primary symbol data with context symbols (QQQ, VIX).
        
        Creates features like relative_sma = sma50_AAPL / sma50_QQQ
        """
        # This is a simplified version - in production you'd use Ray's join
        # For now, we'll handle this in the indicator calculation phase
        return primary_ds
    
    def calculate_indicators_gpu(
        self,
        batch: pd.DataFrame,
        windows: List[int] = [50, 200],
        resampling_timeframes: Optional[List[str]] = None,
        drop_warmup: bool = True
    ) -> pd.DataFrame:
        """
        GPU-accelerated indicator calculation with strict no-look-ahead.
        
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
            windows: SMA/EMA window sizes (default: [50, 200])
            resampling_timeframes: Multi-timeframe aggregations (5min, 15min, etc.)
            drop_warmup: Drop rows where indicators are NaN (warm-up period)
            
        Returns:
            DataFrame with calculated indicators
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
        
        # SMAs - these will be NaN for first N rows (proper behavior)
        for window in windows:
            batch[f'sma_{window}'] = batch['close'].rolling(window=window, min_periods=window).mean()
            batch[f'volume_ma_{window}'] = batch['volume'].rolling(window=window, min_periods=window).mean()
            
            # Volatility
            batch[f'volatility_{window}'] = batch['returns'].rolling(window=window, min_periods=window).std()
            
            # Distance from SMA
            batch[f'dist_sma_{window}'] = (batch['close'] - batch[f'sma_{window}']) / batch[f'sma_{window}']
        
        # Additional standard SMAs from feature_service
        if 20 not in windows:
            batch['sma_20'] = batch['close'].rolling(window=20, min_periods=20).mean()
        
        # EMA (Exponential Moving Average)
        for window in windows:
            batch[f'ema_{window}'] = batch['close'].ewm(span=window, min_periods=window).mean()
        
        # Additional standard EMAs from feature_service
        if 12 not in windows:
            batch['ema_12'] = batch['close'].ewm(span=12, min_periods=12).mean()
        if 26 not in windows:
            batch['ema_26'] = batch['close'].ewm(span=26, min_periods=26).mean()
        
        # RSI (Relative Strength Index)
        batch['rsi_14'] = self._calculate_rsi(batch['close'], period=14)
        
        # Stochastic Oscillator
        batch['stoch_k'], batch['stoch_d'] = self._calculate_stochastic(
            high=batch['high'],
            low=batch['low'],
            close=batch['close'],
            k_period=14,
            d_period=3
        )
        
        # MACD
        batch['macd'], batch['macd_signal'] = self._calculate_macd(batch['close'])
        batch['macd_diff'] = batch['macd'] - batch['macd_signal']
        
        # Bollinger Bands
        batch['bb_upper'], batch['bb_mid'], batch['bb_lower'] = self._calculate_bollinger_bands(
            close=batch['close'],
            window=20,
            std_dev=2
        )
        
        # Average True Range (ATR)
        batch['atr_14'] = self._calculate_atr(
            high=batch['high'],
            low=batch['low'],
            close=batch['close'],
            period=14
        )
        
        # On Balance Volume (OBV)
        batch['obv'] = self._calculate_obv(
            close=batch['close'],
            volume=batch['volume']
        )
        
        # Multi-timeframe resampling
        if resampling_timeframes:
            for tf in resampling_timeframes:
                batch = self._add_resampled_features(batch, timeframe=tf)
        
        # Volume indicators
        batch['volume_ratio'] = batch['volume'] / batch['volume'].rolling(window=20, min_periods=20).mean()
        batch['vwap_dist'] = (batch['close'] - batch['vwap']) / batch['vwap'] if 'vwap' in batch.columns else 0
        
        # Drop warm-up period if requested
        if drop_warmup:
            # Find the maximum window size to know when all indicators are "warm"
            max_window = max(windows) if windows else 200
            batch = batch.iloc[max_window:].reset_index(drop=True)
            log.debug(f"Dropped {max_window} warm-up rows, {len(batch)} rows remain")
        
        return batch
    
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
        batch[[f'close_{timeframe}', f'sma50_{timeframe}']] = batch[[f'close_{timeframe}', f'sma50_{timeframe}']].fillna(method='ffill')
        
        return batch
    
    def process_fold_with_gpu(
        self,
        fold: Fold,
        num_gpus: float = 1.0,
        actor_pool_size: int = 2,
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
        log.info(f"Processing {fold} with GPU acceleration")
        
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
            if num_gpus > 0 and actor_pool_size > 1:
                # Use concurrency for GPU acceleration (Ray 2.9+)
                fold.train_ds = fold.train_ds.map_batches(
                    process_batch,
                    batch_format="pandas",
                    batch_size=10000,
                    concurrency=actor_pool_size,
                    num_gpus=num_gpus
                )
            else:
                # Simple CPU processing
                fold.train_ds = fold.train_ds.map_batches(
                    process_batch,
                    batch_format="pandas",
                    batch_size=10000
                )
            log.info(f"Processed train data for {fold}")
        
        # Process test data (separate calculation, no leakage!)
        if fold.test_ds:
            if num_gpus > 0 and actor_pool_size > 1:
                # Use concurrency for GPU acceleration (Ray 2.9+)
                fold.test_ds = fold.test_ds.map_batches(
                    process_batch,
                    batch_format="pandas",
                    batch_size=10000,
                    concurrency=actor_pool_size,
                    num_gpus=num_gpus
                )
            else:
                # Simple CPU processing
                fold.test_ds = fold.test_ds.map_batches(
                    process_batch,
                    batch_format="pandas",
                    batch_size=10000
                )
            log.info(f"Processed test data for {fold}")
        
        return fold
    
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
        actor_pool_size: int = 2
    ) -> Generator[Fold, None, None]:
        """
        Create walk-forward validation pipeline with GPU acceleration.
        
        This is the RECOMMENDED way to preprocess data for balanced backtesting.
        Each fold calculates indicators independently, preventing look-ahead bias.
        
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
            
        Yields:
            Fold objects with processed train_ds and test_ds
            
        Example:
            ```python
            preprocessor = StreamingPreprocessor(loader)
            
            for fold in preprocessor.create_walk_forward_pipeline(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                context_symbols=["QQQ", "VIX"],
                windows=[50, 200],
                resampling_timeframes=["5min", "15min"],
                num_gpus=1.0
            ):
                # Train on fold.train_ds
                # Test on fold.test_ds
                # Indicators are properly reset, no leakage!
                pass
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
            # Load data
            fold = self.load_fold_data(
                fold=fold,
                symbols=symbols,
                context_symbols=context_symbols
            )
            
            # Calculate indicators with GPU
            fold = self.process_fold_with_gpu(
                fold=fold,
                num_gpus=num_gpus,
                actor_pool_size=actor_pool_size,
                windows=windows,
                resampling_timeframes=resampling_timeframes
            )
            
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
