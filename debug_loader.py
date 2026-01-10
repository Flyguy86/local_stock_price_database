
import duckdb
from pathlib import Path
import os
import pandas as pd

# Mock settings
class Settings:
    features_parquet_dir = Path("/workspaces/local_stock_price_database/data/features_parquet")

settings = Settings()

def test_load(symbol, options_filter=None):
    path = settings.features_parquet_dir / symbol
    print(f"Path exists: {path.exists()}")
    
    query = f"SELECT * FROM '{path}/**/*.parquet'"
    
    if options_filter:
        print(f"Filtering by options: {options_filter}")
        query += f" WHERE options = '{options_filter}'"
        
    print(f"Query: {query}")
    try:
        df = duckdb.query(query).to_df()
        print(f"Rows returned: {len(df)}")
        if not df.empty:
            print("Columns:", df.columns)
            if 'options' in df.columns:
                print("Unique options:", df['options'].unique())
            print("Head:", df.head(1))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_load("GOOGL")
    test_load("GOOGL", options_filter="some_missing_option")
