RAW_BARS_TABLE = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT,
    ts TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    vwap DOUBLE,
    trade_count BIGINT,
    source TEXT,
    PRIMARY KEY(symbol, ts)
);
"""

FEATURE_META_TABLE = """
CREATE TABLE IF NOT EXISTS feature_metadata (
    feature_name TEXT,
    version TEXT,
    params JSON,
    hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(feature_name, version)
);
"""

def table_blueprints() -> list[str]:
    return [RAW_BARS_TABLE, FEATURE_META_TABLE]
