import duckdb
from pathlib import Path
import json
import time

def scan_features():
    path = "data/features_parquet/**/*.parquet"
    print(f"Scanning {path}...")
    
    start = time.time()
    try:
        # We need to extract symbol from the path or the content
        # If accessing via hive partitioning, we might get 'symbol' column if we point to root?
        # Let's try pointing to the root directory with hive partitioning
        
        # Method 1: Query global wild card
        # Note: 'symbol' might not be in the parquet file if it was partition-key but not written to file? 
        # In pipeline.py `write_features`: 
        # `merged.to_parquet(file_path, index=False)`. 
        # `merged` comes from `df` which has `symbol` column added in `duckdb_client.py`? 
        # pipeline.py -> clean_bars -> engineer_features.
        # `write_features` calls `ensure_dest_schema`.
        # `ensure_dest_schema` defines table with `symbol`.
        # When writing to parquet, if `symbol` is in DataFrame, it's in the file.
        
        # Let's check a sample parquet file's schema.
        con = duckdb.connect()
        
        # We use a limit to fail fast if slow
        res = con.execute(f"SELECT DISTINCT symbol, options FROM '{path}'").fetchall()
        
        mapping = {}
        for sym, opt in res:
            if opt not in mapping:
                mapping[opt] = []
            mapping[opt].append(sym)
            
        print(f"Found {len(res)} combinations.")
        for opt, syms in mapping.items():
            print(f"Option: {opt[:50]}... -> {len(syms)} symbols: {syms[:5]}...")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Time taken: {time.time() - start:.4f}s")

if __name__ == "__main__":
    scan_features()
