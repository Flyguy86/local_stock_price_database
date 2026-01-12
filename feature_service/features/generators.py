import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features including sin/cos encoding for cyclical time representation.
    
    Time is encoded as minutes since midnight (0-1439), then transformed into
    sin/cos pairs to capture the circular nature of time where 23:59 is close to 00:01.
    
    Formula:
        minutes_of_day = hour * 60 + minute
        time_sin = sin(2π * minutes_of_day / 1440)
        time_cos = cos(2π * minutes_of_day / 1440)
    
    This allows linear models to understand that late evening (1439 min) is
    adjacent to early morning (0 min) in the time cycle.
    """
    df = df.copy()
    
    # Extract time components from timestamp
    df['hour'] = df['ts'].dt.hour
    df['minute'] = df['ts'].dt.minute
    df['day_of_week'] = df['ts'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Calculate minutes since midnight (0-1439)
    df['minutes_of_day'] = df['hour'] * 60 + df['minute']
    
    # Sin/Cos encoding for cyclical time (minutes in a day: 0-1439)
    # This creates a circular feature space where 23:59 (1439 min) is close to 00:01 (1 min)
    df['time_sin'] = np.sin(2 * np.pi * df['minutes_of_day'] / 1440)
    df['time_cos'] = np.cos(2 * np.pi * df['minutes_of_day'] / 1440)
    
    # Sin/Cos encoding for day of week (0-6)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Drop intermediate columns (keep only encoded features)
    df = df.drop(columns=['hour', 'minute', 'minutes_of_day', 'day_of_week'])
    
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators using the 'ta' library.
    """
    df = df.copy()
    
    # Trend Indicators
    df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
    
    # MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Momentum Indicators
    df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Volatility Indicators
    bb = BollingerBands(close=df['close'], window=20)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    
    df['atr_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    # Volume Indicators
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    
    return df