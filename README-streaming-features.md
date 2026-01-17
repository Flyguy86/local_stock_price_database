# Streaming Feature Engineering Pipeline

This document explains the **3-Phase Feature Engineering Pipeline** used in `ray_orchestrator/streaming.py` to ensure machine learning models receive properly normalized, comparable features across different stocks, timeframes, and market regimes.

## Table of Contents
- [Philosophy: Why Normalize?](#philosophy-why-normalize)
- [The 3-Phase Pipeline](#the-3-phase-pipeline)
- [Feature Inventory](#feature-inventory)
- [Implemented Features](#implemented-features)
  - [Stochastic Oscillator](#stochastic-oscillator)
  - [RSI-14](#rsi-14-relative-strength-index)
  - [MACD](#macd-moving-average-convergence-divergence)
  - [Bollinger Bands](#bollinger-bands)
  - [ATR-14](#atr-14-average-true-range)
  - [OBV](#obv-on-balance-volume)
  - [SMAs/EMAs/Volatility](#smasemas-volatility)
  - [Volume Ratio](#volume-ratio)
- [Already-Normalized Features](#already-normalized-features-no-additional-processing)
  - [Time Features](#time-features-cyclical-encoding)
  - [Returns](#returns-symmetric-by-design)
  - [Price Features](#price-features-already-ratio-based)
  - [VWAP Distance](#vwap-distance-already-ratio-based)
  - [Distance from SMA](#distance-from-sma-already--deviation)
- [Configuration](#configuration)
- [Feature Selection Guide](#feature-selection-guide)
- [Validation Checklist](#validation-checklist)
- [References](#references)

---

## Philosophy: Why Normalize?

**Problem**: Raw technical indicators have vastly different scales:
- **RSI**: 0-100
- **Stochastic**: 0-100  
- **MACD**: -2 to +2 (for $150 stock), -50 to +50 (for $500 stock)
- **Log Returns**: -0.05 to +0.05
- **SMAs**: $50 (for low-priced stock) vs $500 (for high-priced stock)

**Without normalization**, models will:
- Over-weight large-scale features (like raw SMAs)
- Under-weight small-scale features (like log returns)
- Fail to generalize across different stocks (AAPL vs GOOGL)
- Struggle with changing market regimes (low vs high volatility)

**Solution**: The 3-Phase Pipeline ensures all features are on comparable scales.

---

## The 3-Phase Pipeline

### Phase 1: Raw Indicator Calculation
**Purpose**: Calculate indicators using the **physics** of price action.

**Why raw prices?**
- Technical indicators depend on actual price ranges (High-Low)
- SMAs need real prices to show support/resistance levels
- Bollinger Bands need actual volatility in price units

**Example** (Stochastic):
```python
# %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
low_min = batch['low'].rolling(window=14).min()
high_max = batch['high'].rolling(window=14).max()
stoch_k = 100 * (batch['close'] - low_min) / (high_max - low_min)
```

**Output**: `stoch_k` (0-100 scale)

---

### Phase 2: Target Generation (Log Returns)
**Purpose**: Model predicts **relative changes**, not absolute prices.

**Why log returns?**
- Symmetric: -5% and +5% have equal magnitude
- Stock-agnostic: Works for $50 and $500 stocks
- Stationary: No trend bias from rising/falling markets

**Example**:
```python
# Target: 15-minute forward log return
batch['target'] = np.log(batch['close'].shift(-15) / batch['close'])
```

**Output**: `target` (-0.10 to +0.10 typical range)

---

### Phase 3: Simple Normalization
**Purpose**: Center bounded indicators (0-100 scale) to a symmetric range.

**Why?**
- Makes 0-100 indicators comparable to log returns (-1 to +1)
- Centers at neutral point (50 becomes 0)
- Helps linear models understand "neutral" vs "extreme"

**Example** (Stochastic):
```python
# Center at 50, scale to -1 to +1
batch['stoch_k_norm'] = (batch['stoch_k'] - 50) / 50
```

**Output**: 
- `stoch_k_norm = -1`: Oversold (was 0)
- `stoch_k_norm = 0`: Neutral (was 50)
- `stoch_k_norm = +1`: Overbought (was 100)

---

### Phase 4: Rolling Z-Score (Adaptive Normalization)
**Purpose**: Normalize to **current market regime** (volatility-adaptive).

**Why?**
- Low volatility period: MACD of ±0.5 is significant
- High volatility period: MACD of ±0.5 is noise
- Z-score makes them comparable: "2 standard deviations above recent mean"

**Formula**:
```python
rolling_mean = series.rolling(window=200).mean()
rolling_std = series.rolling(window=200).std()
zscore = (series - rolling_mean) / (rolling_std + 1e-9)
```

**Example** (MACD):
```python
batch['macd_zscore'] = self._rolling_zscore(batch['macd'], window=200)
```

**Output**:
- `macd_zscore = 0`: At recent average
- `macd_zscore = +2`: 2 standard deviations above recent mean (bullish)
- `macd_zscore = -2`: 2 standard deviations below recent mean (bearish)

**Hyperparameter**: `zscore_window=200` (configurable in `calculate_indicators_gpu()`)

---

## Feature Inventory

All features calculated in `calculate_indicators_gpu()`:

### ✅ Normalized Features (Phase 3 & 4 Applied) - v3.1 COMPLETE
1. **Stochastic Oscillator** - `stoch_k`, `stoch_d` + normalized + z-score ✅
2. **MACD** - `macd`, `macd_signal`, `macd_diff` + z-score ✅
3. **Bollinger Bands** - `bb_upper/mid/lower`, `bb_position`, `bb_width_pct` + z-score ✅
4. **ATR-14** - `atr_14`, `atr_pct` + z-score ✅
5. **OBV** - `obv` + z-score ✅
6. **RSI-14** - `rsi_14`, `rsi_norm` + z-score ✅
7. **SMAs** (50, 200, 20) - `sma_X`, `dist_sma_X` (already %), `volume_ma_X` + z-score ✅
8. **EMAs** (50, 200, 12, 26) - `ema_X` + z-score ✅
9. **Volatility** - `volatility_X` + z-score ✅
10. **Volume Ratio** - `volume_ratio` (already ratio) + z-score ✅

**Status**: All 10 major technical indicators normalized with 3-phase pipeline (v3.1)

### ✅ Already Normalized (No Changes Needed)
- **Time Features** - `time_sin`, `time_cos`, `day_of_week_sin/cos` (cyclical encoding)
- **Returns** - `returns`, `log_returns` (already symmetric)
- **Price Range %** - `price_range_pct` (already ratio)
- **Distance from SMA** - `dist_sma_X` (already % deviation)
- **VWAP Distance** - `vwap_dist` (already ratio)

---

## Implemented Features

### Stochastic Oscillator

**Indicator Purpose**: Momentum oscillator comparing closing price to price range over N periods.

**Phase 1: Raw Calculation**
```python
# Uses raw OHLC prices (physics of indicator)
low_min = batch['low'].rolling(window=14, min_periods=14).min()
high_max = batch['high'].rolling(window=14, min_periods=14).max()
stoch_k = 100 * (batch['close'] - low_min) / (high_max - low_min)
stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
```
**Output**: 
- `stoch_k`: 0-100 scale (%K line)
- `stoch_d`: 0-100 scale (%D line, smoothed %K)

**Phase 3: Simple Normalization**
```python
# Center at 0, range -1 to +1
batch['stoch_k_norm'] = (batch['stoch_k'] - 50) / 50
batch['stoch_d_norm'] = (batch['stoch_d'] - 50) / 50
```
**Output**:
- `stoch_k_norm`: -1 (oversold) to +1 (overbought)
- `stoch_d_norm`: -1 (oversold) to +1 (overbought)

**Phase 4: Rolling Z-Score**
```python
# Adaptive to recent volatility (200-bar window)
batch['stoch_k_zscore'] = self._rolling_zscore(batch['stoch_k'], window=200)
batch['stoch_d_zscore'] = self._rolling_zscore(batch['stoch_d'], window=200)
```
**Output**:
- `stoch_k_zscore`: How many standard deviations from recent mean
- `stoch_d_zscore`: How many standard deviations from recent mean

**Usage**:
- **Raw** (`stoch_k`): For visual analysis, classic 20/80 overbought/oversold levels
- **Normalized** (`stoch_k_norm`): For linear models expecting -1 to +1 range
- **Z-Score** (`stoch_k_zscore`): For regime-adaptive signals (accounts for changing volatility)

---

### RSI-14 (Relative Strength Index)

**Indicator Purpose**: Momentum oscillator measuring speed and magnitude of price changes.

**Phase 1: Raw Calculation**
```python
# Calculate average gains and losses over 14 periods
delta = batch['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```
**Output**: 
- `rsi_14`: 0-100 scale (0 = extremely oversold, 100 = extremely overbought)

**Phase 3: Simple Normalization**
```python
# Center at 0, range -1 to +1
batch['rsi_norm'] = (batch['rsi_14'] - 50) / 50
```
**Output**:
- `rsi_norm`: -1 (oversold) to +1 (overbought), 0 = neutral

**Phase 4: Rolling Z-Score**
```python
# Adaptive to recent volatility (200-bar window)
batch['rsi_zscore'] = self._rolling_zscore(batch['rsi_14'], window=200)
```
**Output**:
- `rsi_zscore`: How many standard deviations from recent mean

**Usage**:
- **Raw** (`rsi_14`): Classic signals
  - `< 30`: Oversold (potential buy)
  - `> 70`: Overbought (potential sell)
  - `50`: Neutral (no momentum bias)
- **Normalized** (`rsi_norm`): For models expecting symmetric range
  - `-0.6`: Oversold (was 30)
  - `+0.4`: Overbought (was 70)
  - `0`: Neutral (was 50)
- **Z-Score** (`rsi_zscore`): Regime-adaptive momentum
  - `> +2`: Extremely strong momentum vs recent history
  - `< -2`: Extremely weak momentum vs recent history

**Why Z-Score Matters**:
- Trending market: RSI might stay 55-75 for weeks (new "normal")
- Range-bound market: RSI oscillates 30-70 regularly
- Z-score adapts: "Is RSI=60 unusual **right now**?"
- Makes momentum signals comparable across different market regimes

---

### MACD (Moving Average Convergence Divergence)

**Indicator Purpose**: Trend-following momentum indicator showing relationship between two EMAs.

**Phase 1: Raw Calculation**
```python
# Uses raw close prices to calculate EMAs
ema_fast = batch['close'].ewm(span=12, min_periods=12).mean()
ema_slow = batch['close'].ewm(span=26, min_periods=26).mean()
macd = ema_fast - ema_slow
signal = macd.ewm(span=9, min_periods=9).mean()
macd_diff = macd - signal
```
**Output**:
- `macd`: In price units (e.g., $2.50 for AAPL, $50 for GOOGL)
- `macd_signal`: Signal line (9-period EMA of MACD)
- `macd_diff`: Histogram (MACD - Signal)

**Phase 3: Skipped**
- No simple normalization (MACD has no fixed bounds like 0-100)
- Values are stock-specific and regime-dependent

**Phase 4: Rolling Z-Score**
```python
# Normalize to recent volatility (200-bar window)
batch['macd_zscore'] = self._rolling_zscore(batch['macd'], window=200)
batch['macd_signal_zscore'] = self._rolling_zscore(batch['macd_signal'], window=200)
batch['macd_diff_zscore'] = self._rolling_zscore(batch['macd_diff'], window=200)
```
**Output**:
- `macd_zscore`: How strong is current MACD vs recent history
- `macd_signal_zscore`: Normalized signal line
- `macd_diff_zscore`: Normalized histogram (most important for crossovers)

**Usage**:
- **Raw** (`macd`, `macd_diff`): Classic crossover signals (MACD > Signal = bullish)
- **Z-Score** (`macd_diff_zscore`): Regime-adaptive crossovers
  - `macd_diff_zscore > +2`: Strong bullish crossover (2 std devs above mean)
  - `macd_diff_zscore < -2`: Strong bearish crossover (2 std devs below mean)

**Why Z-Score Matters**:
- Low volatility: MACD diff of +0.50 might be significant
- High volatility: MACD diff of +2.00 might be normal noise
- Z-score tells you: "Is this +0.50 meaningful in current regime?"

---

### Bollinger Bands

**Indicator Purpose**: Volatility bands showing standard deviation channels around moving average.

**Phase 1: Raw Calculation**
```python
# Uses raw close prices to calculate bands
bb_mid = batch['close'].rolling(window=20, min_periods=20).mean()
std = batch['close'].rolling(window=20, min_periods=20).std()
bb_upper = bb_mid + (std * 2)
bb_lower = bb_mid - (std * 2)
```
**Output**:
- `bb_upper`: Upper band (in price units, e.g., $152)
- `bb_mid`: Middle band / 20-period SMA (e.g., $150)
- `bb_lower`: Lower band (e.g., $148)

**Phase 3: Derived Features**
```python
# Position within bands (0-1 scale)
bb_range = batch['bb_upper'] - batch['bb_lower']
batch['bb_position'] = (batch['close'] - batch['bb_lower']) / (bb_range + 1e-9)

# Band width in price units
batch['bb_width'] = bb_range

# Band width as % of price (normalized)
batch['bb_width_pct'] = bb_range / batch['close']
```
**Output**:
- `bb_position`: 0 (at lower band) to 1 (at upper band), 0.5 = middle
- `bb_width`: Band width in price units (e.g., $4)
- `bb_width_pct`: Band width as % of price (e.g., 0.027 = 2.7%)

**Phase 4: Rolling Z-Score**
```python
# Normalize raw band values and width
batch['bb_upper_zscore'] = self._rolling_zscore(batch['bb_upper'], window=200)
batch['bb_mid_zscore'] = self._rolling_zscore(batch['bb_mid'], window=200)
batch['bb_lower_zscore'] = self._rolling_zscore(batch['bb_lower'], window=200)
batch['bb_width_zscore'] = self._rolling_zscore(batch['bb_width'], window=200)
```
**Output**:
- `bb_upper/mid/lower_zscore`: How bands compare to recent history
- `bb_width_zscore`: Is current volatility high or low vs recent regime?

**Usage**:
- **Raw Bands** (`bb_upper/mid/lower`): Visual support/resistance levels
- **BB Position** (`bb_position`): Where is price within bands?
  - `< 0.2`: Near lower band (oversold)
  - `0.4-0.6`: Middle range (neutral)
  - `> 0.8`: Near upper band (overbought)
- **BB Width %** (`bb_width_pct`): Volatility measurement
  - `< 2%`: Low volatility (squeeze)
  - `2-4%`: Normal volatility
  - `> 4%`: High volatility (expansion)
- **BB Width Z-Score** (`bb_width_zscore`): Is this volatility extreme?
  - `< -2`: Extremely tight bands (expect breakout)
  - `> +2`: Extremely wide bands (expect consolidation)

**Why BB Position Matters**:
- Stock-agnostic: Works for $50 and $500 stocks
- Tells relative position without knowing absolute price
- Combines mean-reversion signal with trend strength

---

### SMAs (Simple Moving Averages)

**Indicator Purpose**: Trend-following indicator showing average price over N periods.

**Phase 1: Raw Calculation**
```python
# Calculate SMA for each window size (e.g., 50, 200 bars)
for window in windows:  # [50, 200]
    batch[f'sma_{window}'] = batch['close'].rolling(window=window).mean()
    batch[f'volume_ma_{window}'] = batch['volume'].rolling(window=window).mean()
```
**Output**:
- `sma_50`, `sma_200`: In price units (e.g., $148.50)
- `volume_ma_50`, `volume_ma_200`: In share units (e.g., 1,250,000 shares)

**Phase 3: Derived Features (Already Normalized)**
```python
# Distance from SMA as % deviation (already stock-agnostic)
batch[f'dist_sma_{window}'] = (batch['close'] - batch[f'sma_{window}']) / batch[f'sma_{window}']
```
**Output**:
- `dist_sma_50`: % above/below 50-period SMA (e.g., 0.02 = 2% above)
- `dist_sma_200`: % above/below 200-period SMA (e.g., -0.05 = 5% below)

**Phase 4: Rolling Z-Score**
```python
# Normalize raw SMA values (200-bar window)
for window in windows:
    batch[f'sma_{window}_zscore'] = self._rolling_zscore(batch[f'sma_{window}'], window=200)
    batch[f'volume_ma_{window}_zscore'] = self._rolling_zscore(batch[f'volume_ma_{window}'], window=200)
```
**Output**:
- `sma_50_zscore`, `sma_200_zscore`: How SMA compares to recent trend
- `volume_ma_50_zscore`: How average volume compares to recent levels

**Usage**:
- **Raw** (`sma_50`, `sma_200`): Classic support/resistance levels
  - Golden Cross: SMA-50 > SMA-200 (bullish)
  - Death Cross: SMA-50 < SMA-200 (bearish)
- **Distance** (`dist_sma_50`): Mean reversion signals
  - `> 0.05`: 5% above SMA (potentially overbought)
  - `< -0.05`: 5% below SMA (potentially oversold)
- **Z-Score** (`sma_50_zscore`): Trend strength
  - `> +2`: SMA rising much faster than usual (strong uptrend)
  - `< -2`: SMA falling much faster than usual (strong downtrend)

**Why Z-Score Matters**:
- Raw SMA: $150 (AAPL) vs $500 (GOOGL) - not comparable
- Z-score: "Is SMA rising faster than normal for **this stock**?"
- Makes trend signals comparable across different stocks and timeframes

---

### EMAs (Exponential Moving Averages)

**Indicator Purpose**: Trend-following indicator giving more weight to recent prices.

**Phase 1: Raw Calculation**
```python
# Calculate EMA for each window size (e.g., 12, 26, 50, 200)
for window in windows:
    batch[f'ema_{window}'] = batch['close'].ewm(span=window, min_periods=window).mean()
```
**Output**:
- `ema_12`, `ema_26`, `ema_50`, `ema_200`: In price units (e.g., $149.25)

**Phase 3: Skipped**
- Could calculate `dist_ema_X` similar to SMAs if needed
- Currently focusing on raw values and z-scores

**Phase 4: Rolling Z-Score**
```python
# Normalize raw EMA values (200-bar window)
for window in windows:
    batch[f'ema_{window}_zscore'] = self._rolling_zscore(batch[f'ema_{window}'], window=200)
```
**Output**:
- `ema_12_zscore`, `ema_26_zscore`: How EMA compares to recent trend

**Usage**:
- **Raw** (`ema_12`, `ema_26`): Responsive trend signals
  - EMA-12 > EMA-26: Short-term bullish momentum
  - EMA-12 < EMA-26: Short-term bearish momentum
  - Used in MACD calculation (MACD = EMA-12 - EMA-26)
- **Z-Score** (`ema_12_zscore`): Trend acceleration
  - `> +2`: EMA rising unusually fast (momentum surge)
  - `< -2`: EMA falling unusually fast (momentum collapse)

**EMA vs SMA**:
- **EMA**: More responsive to recent prices, better for short-term trends
- **SMA**: Smoother, less whipsaw, better for long-term trends
- Both get z-scores for regime-adaptive comparison

---

### Volatility (Rolling Standard Deviation)

**Indicator Purpose**: Measure of price variability/risk over N periods.

**Phase 1: Raw Calculation**
```python
# Calculate standard deviation of returns for each window
for window in windows:  # [50, 200]
    batch[f'volatility_{window}'] = batch['returns'].rolling(window=window).std()
```
**Output**:
- `volatility_50`, `volatility_200`: Standard deviation (e.g., 0.015 = 1.5% daily volatility)

**Phase 4: Rolling Z-Score**
```python
# Normalize volatility to recent regime (200-bar window)
for window in windows:
    batch[f'volatility_{window}_zscore'] = self._rolling_zscore(batch[f'volatility_{window}'], window=200)
```
**Output**:
- `volatility_50_zscore`: How volatile vs recent history

**Usage**:
- **Raw** (`volatility_50`): Risk measurement
  - `< 0.01`: Low volatility (1% daily moves)
  - `0.01-0.03`: Normal volatility (1-3% daily moves)
  - `> 0.03`: High volatility (>3% daily moves)
- **Z-Score** (`volatility_50_zscore`): Regime detection
  - `> +2`: Extremely high volatility (VIX spike, expect mean reversion)
  - `< -2`: Extremely low volatility (volatility compression, expect breakout)

**Why Z-Score Matters**:
- Low volatility stock: 0.5% daily moves is "high volatility"
- High volatility stock: 0.5% daily moves is "low volatility"
- Z-score: "Is current volatility unusual for **this stock**?"
- Critical for position sizing and risk management

---

### Volume Ratio

**Indicator Purpose**: Relative volume measurement comparing current volume to recent average.

**Phase 1: Skipped**
- We go directly to ratio calculation (Phase 3)

**Phase 3: Ratio Calculation (Already Normalized)**
```python
# Current volume vs 20-period moving average
volume_ma_20 = batch['volume'].rolling(window=20, min_periods=20).mean()
batch['volume_ratio'] = batch['volume'] / volume_ma_20
```
**Output**:
- `volume_ratio`: Ratio (e.g., 1.5 = 50% above average, 0.5 = 50% below average)

**Phase 4: Rolling Z-Score**
```python
# Identify extreme volume anomalies (200-bar window)
batch['volume_ratio_zscore'] = self._rolling_zscore(batch['volume_ratio'], window=200)
```
**Output**:
- `volume_ratio_zscore`: How unusual is current volume activity

**Usage**:
- **Raw Ratio** (`volume_ratio`): Relative volume strength
  - `< 0.5`: Very light volume (50% below average)
  - `0.8-1.2`: Normal volume range
  - `> 2.0`: Heavy volume (2x average)
  - `> 3.0`: Exceptional volume (potential breakout/breakdown)
- **Z-Score** (`volume_ratio_zscore`): Extreme volume detection
  - `> +2`: Unusually high volume spike (institutional activity, news event)
  - `< -2`: Unusually low volume (consolidation, lack of interest)

**Why Both Versions Matter**:
- **Ratio**: Easy interpretation (2x average volume = volume_ratio of 2.0)
- **Z-Score**: Accounts for volatility in volume patterns
  - Stock with erratic volume: ratio of 2.0 might be normal (z-score = 0)
  - Stock with stable volume: ratio of 2.0 is highly unusual (z-score = +3)
- Z-score answers: "Is this volume spike meaningful for **this stock's** normal behavior?"

**Trading Implications**:
- High volume_ratio + price breakout = Strong confirmation
- High volume_ratio + no price movement = Distribution/Accumulation
- Low volume_ratio during trend = Weak trend (likely reversal)
- Use z-score to filter false signals from naturally volatile volume patterns

---

### OBV (On Balance Volume)

**Indicator Purpose**: Cumulative volume indicator tracking buying/selling pressure.

**Phase 1: Raw Calculation**
```python
# Cumulative volume weighted by price direction
price_change = batch['close'].diff()
volume_direction = np.sign(price_change)  # +1 if up, -1 if down, 0 if unchanged
signed_volume = batch['volume'] * volume_direction
obv = signed_volume.cumsum()
```
**Output**:
- `obv`: Cumulative value (grows unbounded, e.g., 1,250,000 shares)

**Phase 3: Skipped**
- No simple normalization (OBV is cumulative and unbounded)
- Value depends on stock's trading history and volume levels

**Phase 4: Rolling Z-Score**
```python
# Normalize cumulative values to recent trend (200-bar window)
batch['obv_zscore'] = self._rolling_zscore(batch['obv'], window=200)
```
**Output**:
- `obv_zscore`: How strong is current OBV vs recent trend

**Usage**:
- **Raw** (`obv`): Classic divergence analysis
  - Price makes new high, OBV doesn't → bearish divergence
  - Price makes new low, OBV doesn't → bullish divergence
- **Z-Score** (`obv_zscore`): Regime-adaptive volume pressure
  - `obv_zscore > +2`: Extremely strong buying pressure
  - `obv_zscore < -2`: Extremely strong selling pressure
  - `obv_zscore near 0`: Volume pressure in line with recent average

**Why Z-Score Matters**:
- OBV for low-volume stock: 100,000 cumulative shares
- OBV for high-volume stock: 50,000,000 cumulative shares
- Raw values not comparable, z-score normalizes to "how unusual is this?"
- Makes volume pressure comparable across stocks with different average volumes

---

## Configuration

### Hyperparameters

All configurable in `calculate_indicators_gpu()`:

```python
def calculate_indicators_gpu(
    self,
    batch: pd.DataFrame,
    windows: List[int] = [50, 200],          # SMA/EMA window sizes
    resampling_timeframes: Optional[List[str]] = None,  # Multi-timeframe features
    drop_warmup: bool = True,                # Drop NaN rows from warm-up period
    zscore_window: int = 200                 # Rolling window for z-score normalization
) -> pd.DataFrame:
```

**Key Parameters**:
- `windows`: SMA/EMA calculation windows (bars, not days)
  - Default: `[50, 200]` (50-bar and 200-bar SMAs)
  - Example: For 1-min data, 200 bars = 3.3 hours of trading
- `zscore_window`: Rolling window for adaptive normalization
  - Default: `200` bars
  - Higher = smoother, less reactive to recent changes
  - Lower = more adaptive to regime shifts
- `resampling_timeframes`: Multi-timeframe features
  - Example: `["5T", "15T", "1H"]` for 5-min, 15-min, 1-hour aggregations

### Feature Version Tracking

Current version: **v3.1** (in progress)

Defined in `ray_orchestrator/streaming.py`:
```python
FEATURE_ENGINEERING_VERSION = "v3.1"
```

**Version History**:
- **v3.1** (in progress): Adding Phase 3 & 4 normalization to all indicators
- **v3.0** (2026-01-17): Converted OHLC to log returns, prevent price leakage
- **v2.1** (2025-12-10): Sin/cos time encoding, market session features
- **v2.0** (2025-11-15): Multi-timeframe resampling, context symbols (QQQ/VIX)
- **v1.0** (2025-10-01): Initial feature set

**Update Instructions**:
When modifying feature calculations:
1. Increment `FEATURE_ENGINEERING_VERSION`
2. Add entry to version history in `streaming.py` header comments
3. Document changes in this README
4. Version automatically saved to checkpoint `metadata.json`

---

## Best Practices

### For Model Training

**Feature Selection Strategy**:
1. **Start with normalized versions** (`_norm` suffix)
   - Consistent scale across features
   - Works well with linear models
   
2. **Add z-score versions for robustness** (`_zscore` suffix)
   - Adapts to changing market regimes
   - Reduces overfitting to specific volatility periods
   
3. **Keep raw versions for interpretability**
   - Classic technical analysis signals
   - Easier to explain to traders

**Example Feature Set**:
```python
features = [
    # Normalized oscillators
    'stoch_k_norm', 'stoch_d_norm',
    'rsi_norm',  # (to be added)
    
    # Z-score trend indicators  
    'macd_diff_zscore',
    'bb_width_zscore',
    
    # Derived features
    'bb_position',
    'bb_width_pct',
    
    # Time features (already normalized)
    'time_sin', 'time_cos',
    'day_of_week_sin', 'day_of_week_cos',
    
    # Returns (already normalized)
    'log_returns'
]
```

### For Debugging

**Check Feature Distributions**:
```python
# Raw features should vary by stock/regime
assert batch['macd'].std() > 0.1  # Has variation
assert batch['macd'].min() != batch['macd'].max()  # Not constant

# Normalized features should be comparable
assert -3 < batch['stoch_k_norm'].mean() < 3  # Roughly centered
assert 0.5 < batch['stoch_k_norm'].std() < 2  # Reasonable spread

# Z-scores should be centered around 0
assert -1 < batch['macd_zscore'].mean() < 1  # Mean near 0
assert 0.8 < batch['macd_zscore'].std() < 1.2  # Std near 1
```

---

## Already-Normalized Features (No Additional Processing)

These features are **already properly normalized** and don't require Phase 3 or Phase 4 processing:

### Time Features (Cyclical Encoding)
**Purpose**: Encode time as cyclical features (9:30 AM and 9:30 AM next day are "close").

**Why cyclical encoding?**
- Raw hour (0-23) treats midnight and 11 PM as far apart (23 units)
- Reality: Midnight and 11 PM are only 1 hour apart
- Sin/Cos encoding creates continuous circle: `sin(0°) ≈ sin(360°)`

**Implementation**:
```python
# Minutes since midnight (0-1439)
minutes_of_day = batch['ts'].dt.hour * 60 + batch['ts'].dt.minute

# Sin/Cos encoding (maps to unit circle)
batch['time_sin'] = np.sin(2 * np.pi * minutes_of_day / 1440)
batch['time_cos'] = np.cos(2 * np.pi * minutes_of_day / 1440)

# Day of week (0-6, Monday=0)
day_of_week = batch['ts'].dt.dayofweek
batch['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
batch['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
```

**Output Range**: All values in [-1, 1]
- `time_sin`, `time_cos`: Market open (9:30 AM) vs close (4:00 PM)
- `day_of_week_sin`, `day_of_week_cos`: Monday effect vs Friday effect

**Why No Z-Score?**
- Already bounded and symmetric [-1, 1]
- Cyclical encoding ensures continuity (no "edge" at midnight)
- Models learn time patterns directly from sin/cos pairs

---

### Returns (Symmetric by Design)
**Purpose**: Measure % change in price.

**Why already normalized?**
- **Symmetric**: -5% down = +5% up in magnitude
- **Stock-agnostic**: Works for $10 penny stock and $1000 growth stock
- **Stationary**: No trend bias from absolute price levels

**Implementation**:
```python
# Simple returns (pct_change)
batch['returns'] = batch['close'].pct_change()

# Log returns (more symmetric, preferred for modeling)
batch['log_returns'] = np.log(batch['close'] / batch['close'].shift(1))
```

**Typical Range**:
- `returns`: -0.05 to +0.05 (±5% per bar)
- `log_returns`: -0.05 to +0.05 (nearly identical for small changes)

**Why No Z-Score?**
- Returns are **already the normalization** of prices
- Target variable (`target = log(future_close / close)`) uses same scale
- Adding z-score would remove signal (market has drift/momentum)
- Models learn directly from return distribution

---

### Price Features (Already Ratio-Based)
**Purpose**: Measure intrabar volatility.

**Implementation**:
```python
# Absolute range (in $)
batch['price_range'] = batch['high'] - batch['low']

# Range as % of close price (normalized)
batch['price_range_pct'] = batch['price_range'] / batch['close']
```

**Why `price_range_pct` is normalized?**
- `price_range = $5` means different things for:
  - $50 stock: 10% volatility (high)
  - $500 stock: 1% volatility (low)
- `price_range_pct = 0.10` means **10% intrabar range** for any stock
- Comparable across all stocks and price levels

**Typical Range**: 0.001 to 0.10 (0.1% to 10% intrabar volatility)

**Why No Z-Score?**
- Already normalized as ratio (% of price)
- Comparable across stocks and timeframes
- Models learn volatility patterns directly
- Z-score would obscure absolute volatility levels

---

### VWAP Distance (Already Ratio-Based)
**Purpose**: Measure price deviation from VWAP (Volume Weighted Average Price).

**Implementation**:
```python
# Distance from VWAP as % deviation
batch['vwap_dist'] = (batch['close'] - batch['vwap']) / batch['vwap']
```

**Why already normalized?**
- `vwap_dist = +0.02`: Price is 2% above VWAP (bullish)
- `vwap_dist = -0.02`: Price is 2% below VWAP (bearish)
- Works identically for $50 and $500 stocks
- Measures **relative** deviation, not absolute $

**Typical Range**: -0.05 to +0.05 (±5% from VWAP)

**Why No Z-Score?**
- Already ratio-based (% deviation)
- VWAP is recalculated daily (resets, no cumulative drift)
- Models learn mean-reversion signals directly
- Absolute deviation from VWAP has trading significance

---

### Distance from SMA (Already % Deviation)
**Purpose**: Measure price position relative to moving average.

**Implementation**:
```python
# For each SMA window (50, 200, etc.)
batch[f'dist_sma_{window}'] = (batch['close'] - batch[f'sma_{window}']) / batch[f'sma_{window}']
```

**Why already normalized?**
- `dist_sma_50 = +0.10`: Price is 10% above SMA-50 (strong uptrend)
- `dist_sma_50 = -0.10`: Price is 10% below SMA-50 (strong downtrend)
- Works for any stock price level
- Captures trend strength as % deviation

**Typical Range**: -0.20 to +0.20 (±20% deviation from SMA)

**Why No Z-Score?**
- Already % deviation (ratio-based)
- Trend strength has absolute meaning (10% above SMA is significant)
- Models learn trend-following signals directly
- Z-score would obscure trend magnitude

---

### Summary: Already-Normalized Features

| Feature | Normalization Method | Range | Why No Z-Score? |
|---------|---------------------|-------|------------------|
| `time_sin`, `time_cos` | Cyclical encoding | [-1, 1] | Already bounded, continuous |
| `day_of_week_sin/cos` | Cyclical encoding | [-1, 1] | Already bounded, continuous |
| `returns` | % change | ~[-0.05, 0.05] | Already the normalization of price |
| `log_returns` | Log ratio | ~[-0.05, 0.05] | Target uses same scale |
| `price_range_pct` | Ratio (% of price) | [0, 0.10] | Comparable across stocks |
| `vwap_dist` | % deviation | ~[-0.05, 0.05] | Ratio-based, daily reset |
| `dist_sma_X` | % deviation | ~[-0.20, 0.20] | Ratio-based, trend magnitude matters |

**Key Principle**: If a feature is **already a ratio or %**, it's normalized. If it's in **price units** or **unbounded**, it needs z-score.

---

## References

- **Main implementation**: [ray_orchestrator/streaming.py](ray_orchestrator/streaming.py)
- **Feature version**: `FEATURE_ENGINEERING_VERSION = "v3.1"` (2026-01-17)
- **Checkpoint metadata**: Saved to `metadata.json` in Ray Tune checkpoints
- **Helper method**: `_rolling_zscore()` for Phase 4 normalization
- **Configuration**: `zscore_window=200` parameter in `calculate_indicators_gpu()`

---

## Feature Selection Guide

### For Different Model Types:

**Tree-Based Models** (XGBoost, LightGBM, Random Forest):
- Use **raw** or **_norm** variants
- Trees handle different scales via splits
- Example: `stoch_k`, `macd`, `bb_upper`

**Linear Models** (Ridge, Lasso, ElasticNet):
- Use **_norm** or **_zscore** variants
- Linear models require comparable scales
- Example: `stoch_k_norm`, `macd_zscore`, `rsi_norm`

**Neural Networks** (MLPs, LSTMs, Transformers):
- Use **_zscore** variants
- Regime-adaptive features improve generalization
- Example: `stoch_k_zscore`, `macd_zscore`, `bb_width_zscore`

**Feature Selection Best Practices**:
1. Start with z-score variants for all indicators
2. Add already-normalized features (time_sin/cos, returns, dist_sma_X)
3. Remove raw price-unit features (sma_50, ema_200) unless using trees
4. Keep both `_norm` and `_zscore` variants initially, prune via feature importance
5. Monitor feature correlation (remove redundant variants)

---

## Validation Checklist

Before training, verify normalization:

```python
import pandas as pd

# Load processed features
df = pd.read_parquet('features.parquet')

# Check z-score features
for col in df.columns:
    if '_zscore' in col:
        mean = df[col].mean()
        std = df[col].std()
        print(f"{col}: mean={mean:.3f}, std={std:.3f}")
        assert -0.2 < mean < 0.2, f"{col} mean not near 0"
        assert 0.8 < std < 1.2, f"{col} std not near 1"

# Check bounded features
for col in ['time_sin', 'time_cos', 'day_of_week_sin', 'day_of_week_cos']:
    assert -1.1 < df[col].min() < -0.9, f"{col} min out of range"
    assert 0.9 < df[col].max() < 1.1, f"{col} max out of range"

# Check norm features
for col in df.columns:
    if '_norm' in col and 'log' not in col:
        assert -1.5 < df[col].min() < -0.5, f"{col} min out of range"
        assert 0.5 < df[col].max() < 1.5, f"{col} max out of range"

print("✅ All normalization checks passed!")
```

---

**Last Updated**: 2026-01-17 (v3.1 - Comprehensive 3-phase normalization complete)
