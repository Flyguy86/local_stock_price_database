import pandas as pd
import duckdb
from pathlib import Path
import os

# Find a parquet file
base_dir = Path("data/features_parquet")
parquet_files = list(base_dir.glob("**/*.parquet"))

if not parquet_files:
    print("No parquet files found")
    exit()

target_file = parquet_files[0]
print(f"Inspecting {target_file}")

try:
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{target_file}' LIMIT 1000").df()
    
    print("Columns:", df.columns.tolist())
    
    # Check for object/string columns and unique values
    for col in df.columns:
        if df[col].dtype == 'object':
            uniques = df[col].unique()
            print(f"Column '{col}' unique sample: {uniques[:10]}")
            
except Exception as e:
    print(f"Error: {e}")
