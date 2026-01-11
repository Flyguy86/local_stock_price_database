
import logging
import time
import duckdb
from pathlib import Path

# Mock config
class Settings:
    features_parquet_dir = Path("/workspaces/local_stock_price_database/data/features_parquet")

settings = Settings()
log = logging.getLogger("debug_scanner")
logging.basicConfig(level=logging.INFO)

def get_feature_map():
    base_path = settings.features_parquet_dir
    if not base_path.exists():
        print(f"Feature path does not exist: {base_path}")
        return {}
    
    glob_pattern = str(base_path / "**/*.parquet")
    print(f"Scanning pattern: {glob_pattern}")
    
    sql = f"""
    SELECT 
        options,
        count(*) as count,
        array_agg(DISTINCT regexp_extract(filename, 'features_parquet/([^/]+)/', 1)) as symbols
    FROM read_parquet('{glob_pattern}', filename=true, union_by_name=true)
    GROUP BY options
    """
    
    try:
        res = duckdb.query(sql).fetchall()
        print(f"Found {len(res)} configurations.")
        for row in res:
            opt = row[0]
            count = row[1]
            syms = row[2]
            print(f"Option: {opt} | Count: {count} | Symbols: {syms}")
            
    except Exception as e:
        print(f"Failed to scan: {e}")

if __name__ == "__main__":
    get_feature_map()
