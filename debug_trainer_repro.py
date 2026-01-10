
import sys
import os
import logging

# Setup paths
sys.path.append("/workspaces/local_stock_price_database")

from training_service.trainer import train_model_task, start_training
from training_service.db import db
import uuid
import pandas as pd
import numpy as np

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Mock DB interactions to avoid pollution (optional, but trainer writes to DB)
# For this verify script, we can just let it write to the db or mock it.
# Let's just let it run, we can delete the model later.

def test_1m_training():
    symbol = "SPY" # Adjust if needed
    algorithm = "random_forest_regressor"
    target_col = "open" # As requested by user "Open/Close together"
    timeframe = "1m"
    params = {"n_estimators": 10} # Fast
    
    # Try to find a symbol that exists
    # We can check data/features_parquet
    
    print(f"--- Testing {symbol} {timeframe} {target_col} ---")
    
    tid = str(uuid.uuid4())
    try:
        train_model_task(
            training_id=tid,
            symbol=symbol,
            algorithm=algorithm,
            target_col=target_col,
            params=params,
            data_options=None,
            timeframe=timeframe,
            parent_model_id=None,
            feature_whitelist=None
        )
        print("Training Success!")
    except Exception as e:
        print(f"Training Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_1m_training()
