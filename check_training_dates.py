#!/usr/bin/env python3
"""Check available date ranges in parquet data."""

import duckdb
from pathlib import Path

parquet_dir = Path("/workspaces/local_stock_price_database/data/parquet")

# Connect to DuckDB
con = duckdb.connect(":memory:")

# Query all symbols and their date ranges
query = f"""
SELECT 
    symbol,
    MIN(ts) as first_date,
    MAX(ts) as last_date,
    COUNT(*) as total_rows,
    COUNT(DISTINCT DATE_TRUNC('day', ts)) as num_days
FROM read_parquet('{parquet_dir}/**/bars.parquet', 
    hive_partitioning=true, 
    union_by_name=true)
GROUP BY symbol
ORDER BY symbol
"""

print("=" * 80)
print("AVAILABLE DATA RANGES")
print("=" * 80)

df = con.execute(query).fetchdf()

for _, row in df.iterrows():
    print(f"\n{row['symbol']}:")
    print(f"  First: {row['first_date']}")
    print(f"  Last:  {row['last_date']}")
    print(f"  Rows:  {row['total_rows']:,}")
    print(f"  Days:  {row['num_days']}")

print("\n" + "=" * 80)
print("RECOMMENDED DATE RANGES FOR TRAINING")
print("=" * 80)

# Find common date range across all symbols
query2 = f"""
SELECT 
    MAX(first_date) as common_start,
    MIN(last_date) as common_end
FROM (
    SELECT 
        symbol,
        MIN(ts) as first_date,
        MAX(ts) as last_date
    FROM read_parquet('{parquet_dir}/**/bars.parquet', 
        hive_partitioning=true, 
        union_by_name=true)
    GROUP BY symbol
)
"""

common = con.execute(query2).fetchdf().iloc[0]
print(f"\nCommon date range (all symbols overlap):")
print(f"  Start: {common['common_start']}")
print(f"  End:   {common['common_end']}")
print(f"\nUse these dates in the training UI to ensure all symbols have data!")

con.close()
