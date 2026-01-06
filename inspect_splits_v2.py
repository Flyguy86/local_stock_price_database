import duckdb
import pandas as pd

try:
    con = duckdb.connect()
    # Read unique values from data_split
    splits = con.execute("SELECT DISTINCT data_split FROM 'data/features_parquet/**/*.parquet'").fetchall()
    print("Unique data_split values:", splits)
    
    # Check if there is a 'fold' column
    cols = con.execute("SELECT * FROM 'data/features_parquet/**/*.parquet' LIMIT 1").df().columns
    print("Columns:", cols)
except Exception as e:
    print(e)