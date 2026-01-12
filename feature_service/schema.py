FEATURE_BARS_DDL = """
CREATE TABLE IF NOT EXISTS feature_bars (
    symbol VARCHAR,
    ts TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    trade_count INTEGER,
    vwap DOUBLE,
    
    -- Technical Indicators
    sma_20 DOUBLE,
    ema_12 DOUBLE,
    ema_26 DOUBLE,
    macd DOUBLE,
    macd_signal DOUBLE,
    macd_diff DOUBLE,
    rsi_14 DOUBLE,
    stoch_k DOUBLE,
    stoch_d DOUBLE,
    bb_upper DOUBLE,
    bb_lower DOUBLE,
    bb_mid DOUBLE,
    atr_14 DOUBLE,
    obv DOUBLE,
    
    -- Time Features (Cyclical Encoding)
    time_sin DOUBLE,
    time_cos DOUBLE,
    day_of_week_sin DOUBLE,
    day_of_week_cos DOUBLE,
    
    PRIMARY KEY (symbol, ts)
)
"""