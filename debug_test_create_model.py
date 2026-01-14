#!/usr/bin/env python3
"""Debug script to trace test_create_model_record issue."""
import asyncio
import os
import sys
from pathlib import Path

# Set test database URL
TEST_POSTGRES_URL = "postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test"
os.environ["POSTGRES_URL"] = TEST_POSTGRES_URL
os.environ["TEST_POSTGRES_URL"] = TEST_POSTGRES_URL

# Add project root
sys.path.insert(0, str(Path(__file__).parent))


async def main():
    import asyncpg
    import uuid
    import json
    
    print("=" * 60)
    print("DEBUG: test_create_model_record trace")
    print("=" * 60)
    
    # Step 1: Create test database if needed
    print("\n1. Connecting to postgres...")
    try:
        conn = await asyncpg.connect(TEST_POSTGRES_URL.replace('/strategy_factory_test', '/postgres'))
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", 'strategy_factory_test')
        if not exists:
            await conn.execute('CREATE DATABASE "strategy_factory_test"')
            print("   Created strategy_factory_test database")
        else:
            print("   Database already exists")
        await conn.close()
    except Exception as e:
        print(f"   Error: {e}")
    
    # Step 2: Create pool (simulating db_tables fixture)
    print("\n2. Creating test pool...")
    pool = await asyncpg.create_pool(
        TEST_POSTGRES_URL,
        min_size=1,
        max_size=3,
        command_timeout=60,
        statement_cache_size=0
    )
    print(f"   Pool created: {pool}")
    
    # Step 3: Create tables
    print("\n3. Creating tables...")
    async with pool.acquire() as conn:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                algorithm VARCHAR,
                symbol VARCHAR,
                target_col VARCHAR DEFAULT 'close',
                feature_cols JSONB,
                hyperparameters JSONB,
                metrics JSONB,
                status VARCHAR,
                created_at TIMESTAMP DEFAULT NOW(),
                artifact_path VARCHAR,
                error_message TEXT,
                data_options JSONB,
                timeframe VARCHAR DEFAULT '1m',
                train_window INTEGER,
                test_window INTEGER,
                parent_model_id VARCHAR,
                group_id VARCHAR,
                target_transform VARCHAR DEFAULT 'none',
                columns_initial INTEGER,
                columns_remaining INTEGER,
                fingerprint VARCHAR(64),
                alpha_grid JSONB,
                l1_ratio_grid JSONB,
                regime_configs JSONB,
                context_symbols JSONB,
                cv_folds INTEGER DEFAULT 5,
                cv_strategy VARCHAR DEFAULT 'time_series_split',
                is_grid_member BOOLEAN DEFAULT FALSE
            )
        """)
        print("   Tables created")
    
    # Step 4: Set pg_db._pool (simulating db_tables fixture)
    print("\n4. Setting pg_db._pool...")
    import training_service.pg_db as pg_db
    
    print(f"   Before: pg_db._pool = {pg_db._pool}")
    print(f"   Before: pg_db.POSTGRES_URL = {pg_db.POSTGRES_URL}")
    
    pg_db._pool = pool
    pg_db.POSTGRES_URL = TEST_POSTGRES_URL
    
    print(f"   After: pg_db._pool = {pg_db._pool}")
    print(f"   After: pg_db.POSTGRES_URL = {pg_db.POSTGRES_URL}")
    
    # Step 5: Create TrainingDB (simulating training_db fixture)
    print("\n5. Creating TrainingDB...")
    db = pg_db.TrainingDB()
    print(f"   TrainingDB created: {db}")
    
    # Step 6: Verify get_pool() returns our pool
    print("\n6. Verifying get_pool()...")
    returned_pool = await pg_db.get_pool()
    print(f"   get_pool() returned: {returned_pool}")
    print(f"   Same as our pool? {returned_pool is pool}")
    
    # Step 7: Create model (the actual test)
    print("\n7. Creating model record...")
    model_id = str(uuid.uuid4())
    sample_model_data = {
        "id": model_id,
        "name": "test_model",
        "algorithm": "RandomForest",
        "symbol": "RDDT",
        "target_col": "close",
        "feature_cols": json.dumps(["sma_20", "rsi_14", "ema_10"]),
        "hyperparameters": json.dumps({"n_estimators": 100, "max_depth": 10}),
        "metrics": json.dumps({"accuracy": 0.85, "f1": 0.82}),
        "status": "completed",
        "data_options": json.dumps({"train_size": 0.8}),
        "timeframe": "1m",
        "train_window": 30,
        "test_window": 7,
        "target_transform": "log_return",
        "fingerprint": "test_fingerprint_123",
        "cv_folds": 5,
        "cv_strategy": "time_series_split"
    }
    
    try:
        await db.create_model_record(sample_model_data)
        print(f"   ✓ Model created: {model_id}")
    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        await pool.close()
        return
    
    # Step 8: Get model back
    print("\n8. Retrieving model...")
    try:
        model = await db.get_model(model_id)
        if model:
            print(f"   ✓ Model retrieved!")
            print(f"     id: {model['id']}")
            print(f"     algorithm: {model['algorithm']}")
            print(f"     symbol: {model['symbol']}")
            print(f"     status: {model['status']}")
        else:
            print(f"   ✗ Model not found!")
    except Exception as e:
        print(f"   ✗ Error retrieving model: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
