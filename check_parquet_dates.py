#!/usr/bin/env python3
"""
Quick script to check available date ranges in parquet files.
Run this to see what dates you actually have data for.
"""

import os
from pathlib import Path
import duckdb

def check_parquet_dates(parquet_dir: str = "/workspaces/local_stock_price_database/data/parquet"):
    """Check date ranges available in parquet files."""
    
    parquet_path = Path(parquet_dir)
    
    if not parquet_path.exists():
        print(f"❌ Parquet directory not found: {parquet_dir}")
        return
    
    # Find all parquet files
    parquet_files = list(parquet_path.rglob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ No parquet files found in {parquet_dir}")
        return
    
    print(f"✅ Found {len(parquet_files)} parquet file(s)")
    print()
    
    # Use DuckDB to query all files at once
    con = duckdb.connect(":memory:")
    
    # Register all parquet files
    query = f"""
    SELECT 
        MIN(ts) as earliest_date,
        MAX(ts) as latest_date,
        COUNT(DISTINCT symbol) as symbol_count,
        COUNT(*) as total_rows
    FROM read_parquet('{parquet_dir}/**/*.parquet')
    """
    
    try:
        result = con.execute(query).fetchone()
        
        if result:
            earliest, latest, symbols, rows = result
            print(f"📊 Data Summary:")
            print(f"   Earliest date: {earliest}")
            print(f"   Latest date:   {latest}")
            print(f"   Symbols:       {symbols}")
            print(f"   Total rows:    {rows:,}")
            print()
            
            # Get per-symbol breakdown
            symbol_query = f"""
            SELECT 
                symbol,
                MIN(ts) as first_date,
                MAX(ts) as last_date,
                COUNT(*) as row_count
            FROM read_parquet('{parquet_dir}/**/*.parquet')
            GROUP BY symbol
            ORDER BY symbol
            """
            
            print(f"📈 Per-Symbol Breakdown:")
            for row in con.execute(symbol_query).fetchall():
                symbol, first, last, count = row
                print(f"   {symbol:8} {first} → {last}  ({count:,} rows)")
            
    except Exception as e:
        print(f"❌ Error querying parquet files: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        check_parquet_dates(sys.argv[1])
    else:
        check_parquet_dates()
