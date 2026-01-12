#!/usr/bin/env python3
"""
Quick debug script to check what options values exist in parquet files.
"""
import duckdb
from pathlib import Path

parquet_path = Path("/workspaces/local_stock_price_database/data/features_parquet")

print(f"Checking parquet files in: {parquet_path}")
print(f"Directory exists: {parquet_path.exists()}\n")

try:
    # Connect to in-memory DuckDB
    conn = duckdb.connect(":memory:")
    
    # Query to check first file's columns
    print("=== Checking Schema ===")
    sample_query = f"SELECT * FROM '{parquet_path}/**/*.parquet' LIMIT 1"
    sample_df = conn.execute(sample_query).df()
    print(f"Columns in parquet files: {list(sample_df.columns)}")
    print(f"Has 'options' column: {'options' in sample_df.columns}\n")
    
    if 'options' in sample_df.columns:
        # Query distinct options values
        print("=== Querying Distinct Options ===")
        options_query = f"""
        SELECT DISTINCT options, COUNT(*) as row_count 
        FROM '{parquet_path}/**/*.parquet' 
        GROUP BY options 
        ORDER BY options
        """
        result = conn.execute(options_query).df()
        print(result)
        
        # Check for NULL/empty
        print("\n=== Checking for NULL/Empty ===")
        null_query = f"""
        SELECT 
            SUM(CASE WHEN options IS NULL THEN 1 ELSE 0 END) as null_count,
            SUM(CASE WHEN options = '' THEN 1 ELSE 0 END) as empty_count,
            SUM(CASE WHEN options IS NOT NULL AND options != '' THEN 1 ELSE 0 END) as non_empty_count,
            COUNT(*) as total_count
        FROM '{parquet_path}/**/*.parquet'
        """
        null_result = conn.execute(null_query).df()
        print(null_result)
    else:
        print("❌ No 'options' column found - this is legacy data")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
