#!/usr/bin/env python3
"""Check if parquet files exist and have options column."""
import duckdb
from pathlib import Path

parquet_dir = Path("/app/data/features_parquet")

print(f"Checking: {parquet_dir}")
print(f"Exists: {parquet_dir.exists()}")

if parquet_dir.exists():
    symbols = [d.name for d in parquet_dir.iterdir() if d.is_dir()]
    print(f"Symbols: {symbols}")
    
    if symbols:
        # Check first symbol
        first_symbol = symbols[0]
        first_dir = parquet_dir / first_symbol
        parquet_files = list(first_dir.rglob("*.parquet"))
        print(f"\n{first_symbol} has {len(parquet_files)} parquet files")
        
        if parquet_files:
            # Query one file
            sample = parquet_files[0]
            print(f"Sample file: {sample}")
            
            conn = duckdb.connect(":memory:")
            result = conn.execute(f"SELECT * FROM read_parquet('{sample}') LIMIT 1").fetchdf()
            print(f"\nColumns: {list(result.columns)}")
            print(f"\nSample row:")
            print(result)
            
            # Check for options column
            if 'options' in result.columns:
                query = f"SELECT DISTINCT options FROM read_parquet('{parquet_dir}/**/*.parquet', union_by_name=true)"
                print(f"\nQuerying all parquet files for distinct options...")
                options = conn.execute(query).fetchall()
                print(f"Found options: {options}")
            else:
                print("\n❌ No 'options' column found!")
            
            conn.close()
