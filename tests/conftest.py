"""
Shared pytest fixtures for all tests.
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test database URL (use separate test database)
TEST_POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL",
    "postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory_test"
)

# Set test environment variables before importing services
os.environ["POSTGRES_URL"] = TEST_POSTGRES_URL
os.environ["TEST_MODE"] = "true"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_db_pool():
    """
    Create test database connection pool.
    Creates test database if it doesn't exist, drops all tables before tests.
    """
    import asyncpg
    
    # Parse connection URL to get database name
    # postgresql://user:pass@host:port/dbname
    parts = TEST_POSTGRES_URL.split("/")
    base_url = "/".join(parts[:-1])
    test_db_name = parts[-1].split("?")[0]
    
    # Connect to default postgres database to create test database
    try:
        conn = await asyncpg.connect(base_url + "/postgres")
        
        # Check if test database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", test_db_name
        )
        
        if not exists:
            # Create test database
            await conn.execute(f'CREATE DATABASE "{test_db_name}"')
            print(f"\n✓ Created test database: {test_db_name}")
        
        await conn.close()
    except Exception as e:
        print(f"\n⚠ Could not create test database: {e}")
        print(f"  Assuming it already exists or using main database")
    
    # Create connection pool to test database
    pool = await asyncpg.create_pool(
        TEST_POSTGRES_URL,
        min_size=2,
        max_size=10,
        command_timeout=60
    )
    
    # Clean up any existing tables
    async with pool.acquire() as conn:
        # Drop all tables in public schema
        await conn.execute("""
            DROP SCHEMA IF EXISTS public CASCADE;
            CREATE SCHEMA public;
        """)
        print(f"✓ Cleaned test database schema")
    
    yield pool
    
    # Cleanup after all tests
    await pool.close()
    print("\n✓ Closed test database pool")


@pytest.fixture
async def db_tables(test_db_pool):
    """
    Create fresh tables for each test.
    Drops and recreates tables before each test.
    """
    from training_service.pg_db import ensure_tables
    
    # Monkey-patch the pool to use test pool
    import training_service.pg_db as pg_db
    original_get_pool = pg_db.get_pool
    
    async def mock_get_pool():
        return test_db_pool
    
    pg_db.get_pool = mock_get_pool
    
    # Create tables
    await ensure_tables()
    
    yield test_db_pool
    
    # Cleanup tables after test
    async with test_db_pool.acquire() as conn:
        await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    
    # Restore original function
    pg_db.get_pool = original_get_pool


@pytest.fixture
def sample_model_data():
    """Generate sample model data for tests."""
    import uuid
    import json
    
    return {
        "id": str(uuid.uuid4()),
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


@pytest.fixture
def sample_simulation_data():
    """Generate sample simulation data for tests."""
    import uuid
    import json
    
    return {
        "id": str(uuid.uuid4()),
        "model_id": str(uuid.uuid4()),
        "ticker": "RDDT",
        "return_pct": 15.5,
        "trades_count": 42,
        "hit_rate": 0.65,
        "sqn": 2.3,
        "params": json.dumps({
            "initial_cash": 10000,
            "min_prediction_threshold": 0.6
        })
    }


@pytest.fixture
async def training_db():
    """
    Get TrainingDB instance configured for testing.
    """
    from training_service.pg_db import TrainingDB, get_pool, close_pool
    import training_service.pg_db as pg_db
    
    # Set test URL
    original_url = pg_db.POSTGRES_URL
    pg_db.POSTGRES_URL = TEST_POSTGRES_URL
    
    # Create tables
    from training_service.pg_db import ensure_tables
    await ensure_tables()
    
    db = TrainingDB()
    
    yield db
    
    # Cleanup
    await close_pool()
    pg_db.POSTGRES_URL = original_url


@pytest.fixture
async def simulation_db():
    """
    Get SimulationDB instance configured for testing.
    """
    from simulation_service.pg_db import SimulationDB, get_pool, close_pool, ensure_tables
    import simulation_service.pg_db as pg_db
    
    # Set test URL
    original_url = pg_db.POSTGRES_URL
    pg_db.POSTGRES_URL = TEST_POSTGRES_URL
    
    # Create tables
    await ensure_tables()
    
    db = SimulationDB()
    
    yield db
    
    # Cleanup
    await close_pool()
    pg_db.POSTGRES_URL = original_url


@pytest.fixture
async def api_client():
    """
    Create async HTTP client for API testing.
    Returns httpx.AsyncClient instance.
    """
    from httpx import AsyncClient
    
    client = AsyncClient(timeout=30.0)
    yield client
    await client.aclose()


@pytest.fixture
def mock_data_available():
    """
    Mock fixture to simulate data being available for testing.
    Prevents tests from failing due to missing parquet files.
    """
    with patch("training_service.data.get_data_options") as mock_options, \
         patch("simulation_service.core.get_available_models") as mock_models, \
         patch("simulation_service.core.get_available_tickers") as mock_tickers:
        
        # Mock data options
        mock_options.return_value = ["option1", "option2"]
        
        # Mock available models
        mock_models.return_value = ["model-1", "model-2"]
        
        # Mock available tickers
        mock_tickers.return_value = ["AAPL", "GOOGL", "MSFT"]
        
        yield {
            "options": mock_options,
            "models": mock_models,
            "tickers": mock_tickers
        }


# Mark all async tests
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
