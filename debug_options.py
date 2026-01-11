
import duckdb
import glob
import os

base_path = "/workspaces/local_stock_price_database/data/features_parquet"
files = glob.glob(f"{base_path}/**/*.parquet", recursive=True)

print(f"Found {len(files)} parquet files.")

if not files:
    exit()

print("--- Inspecting Samples ---")
# Check first file and maybe one deep in inside logic
samples = [files[0], files[-1]]
for f in samples:
    print(f"File: {f}")
    try:
        rel = duckdb.query(f"SELECT * FROM read_parquet('{f}') LIMIT 1")
        print(rel.columns)
        if 'options' in rel.columns:
            print(f"Options value: {rel.project('options').fetchone()}")
        else:
            print("No 'options' column found.")
    except Exception as e:
        print(e)
    print("-" * 20)

print("\n--- Running Global Group By ---")
try:
    sql = f"""
    SELECT 
        options,
        count(*)
    FROM read_parquet('{base_path}/**/*.parquet', filename=true, union_by_name=true)
    GROUP BY options
    """
    res = duckdb.query(sql).fetchall()
    for row in res:
        print(f"Option: {row[0]} | Count: {row[1]}")
except Exception as e:
    print(f"Global Query Failed: {e}")
