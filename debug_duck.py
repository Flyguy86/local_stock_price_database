
import duckdb
import pandas as pd
from pathlib import Path

# Try to match the logic in core.py
try:
    path = Path("/workspaces/local_stock_price_database/data/features_parquet/AAPL")
    if not path.exists():
        print(f"Path does not exist: {path}")
        exit(1)
        
    query = f"SELECT * FROM '{path}/**/*.parquet' LIMIT 5"
    print(f"Query: {query}")
    
    df = duckdb.query(query).to_df()
    print("Success!")
    print(df.head())
    print("Columns:", df.columns)
except Exception as e:
    print(f"Error: {e}")
