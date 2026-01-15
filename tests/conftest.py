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

# Test database URL - use postgres hostname for Docker, localhost for local
# Docker test_runner uses postgres service, local dev uses localhost
TEST_POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL",
    "postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test"
)

# Set test environment variables before importing services
os.environ["POSTGRES_URL"] = TEST_POSTGRES_URL
os.environ["TEST_MODE"] = "true"


# Use pytest-asyncio's built-in event loop management
pytest_plugins = ['pytest_asyncio']


@pytest.fixture(scope="function")
async def test_db_pool():
    """
    Create test database connection pool for each test.
    Creates test database if it doesn't exist, cleans tables before tests.
    """
    import asyncpg
    
    # Parse connection URL to get database name
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
        print(f"  Assuming it already exists")
    
    # Define JSON codec initialization function
    async def init_connection(conn):
        """Initialize connection with JSON codecs for JSONB support."""
        import json
        await conn.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
        await conn.set_type_codec(
            'json',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
    
    # Create connection pool to test database with statement cache disabled
    # This prevents "prepared statement already exists" errors
    pool = await asyncpg.create_pool(
        TEST_POSTGRES_URL,
        min_size=1,
        max_size=3,
        command_timeout=60,
        statement_cache_size=0,  # Disable statement caching to prevent collisions
        init=init_connection  # Initialize JSON codecs on each connection
    )
    
    # Clean up any existing tables
    async with pool.acquire() as conn:
        await conn.execute("""
            DROP SCHEMA IF EXISTS public CASCADE;
            CREATE SCHEMA public;
        """)
    
    yield pool
    
    # Cleanup after test
    await pool.close()


@pytest.fixture
async def db_tables(test_db_pool):
    """
    Create fresh tables for each test.
    Schema must match training_service/pg_db.py exactly.
    """
    import training_service.pg_db as pg_db
    
    # Save original pool (for cleanup)
    original_pool = pg_db._pool
    original_url = pg_db.POSTGRES_URL
    
    # Set global pool for any code that still uses get_pool() directly
    pg_db._pool = test_db_pool
    pg_db.POSTGRES_URL = TEST_POSTGRES_URL
    
    # Create tables using test pool - MUST match pg_db.py schema exactly
    async with test_db_pool.acquire() as conn:
        # Create models table - matches training_service/pg_db.py
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
                
                -- Data configuration for fingerprint
                data_options JSONB,
                timeframe VARCHAR DEFAULT '1m',
                train_window INTEGER,
                test_window INTEGER,
                
                -- Model lineage
                parent_model_id VARCHAR,
                group_id VARCHAR,
                
                -- Target configuration for fingerprint
                target_transform VARCHAR DEFAULT 'none',
                
                -- Feature evolution tracking
                columns_initial INTEGER,
                columns_remaining INTEGER,
                
                -- Fingerprint for deduplication
                fingerprint VARCHAR(64),
                
                -- Grid search configuration
                alpha_grid JSONB,
                l1_ratio_grid JSONB,
                regime_configs JSONB,
                
                -- Context models used during training
                context_symbols JSONB,
                
                -- Cross-validation folds
                cv_folds INTEGER DEFAULT 5,
                cv_strategy VARCHAR DEFAULT 'time_series_split',
                
                -- Grid member flag
                is_grid_member BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create features_log table - matches pg_db.py
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS features_log (
                id SERIAL PRIMARY KEY,
                model_id VARCHAR REFERENCES models(id) ON DELETE CASCADE,
                feature_name VARCHAR,
                importance DOUBLE PRECISION,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Create simulation_history table - matches pg_db.py structure
        # NOTE: FK constraint removed for testing to allow isolated simulation tests
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS simulation_history (
                id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT NOW(),
                model_id VARCHAR,
                ticker VARCHAR,
                return_pct DOUBLE PRECISION,
                trades_count INTEGER,
                hit_rate DOUBLE PRECISION,
                sqn DOUBLE PRECISION,
                params JSONB
            )
        """)
        
        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_models_fingerprint ON models(fingerprint)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_models_symbol ON models(symbol)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_features_log_model ON features_log(model_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_simulation_history_model ON simulation_history(model_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_simulation_history_sqn ON simulation_history(sqn DESC)")
    
    yield test_db_pool
    
    # Restore original pool
    pg_db._pool = original_pool
    pg_db.POSTGRES_URL = original_url


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
async def training_db(db_tables):
    """
    Get TrainingDB instance configured for testing.
    Uses db_tables to ensure tables exist and passes the test pool directly.
    """
    import training_service.pg_db as pg_db
    
    # db_tables is the test_db_pool that was yielded
    # Pass it directly to TrainingDB so it doesn't need to use global _pool
    db = pg_db.TrainingDB(pool=db_tables)
    yield db


# Alias for test_sync_wrapper.py tests that use training_db_fixture
@pytest.fixture
def training_db_fixture():
    """
    Sync fixture wrapper for TrainingDB.
    Used by sync tests that need database access.
    Creates its own connection setup without depending on async fixtures.
    Schema must match training_service/pg_db.py exactly.
    """
    import asyncio
    import asyncpg
    import training_service.pg_db as pg_db
    
    # Save original state
    original_pool = pg_db._pool
    original_url = pg_db.POSTGRES_URL
    
    # Set test URL
    pg_db.POSTGRES_URL = TEST_POSTGRES_URL
    
    pool = None  # Will be set in setup
    
    async def init_connection(conn):
        """Initialize connection with JSON codecs for JSONB support."""
        import json
        await conn.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
        await conn.set_type_codec(
            'json',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
    
    async def setup_tables():
        """Set up test database tables and create pool."""
        nonlocal pool
        
        # First, connect to create/clean the test database
        try:
            conn = await asyncpg.connect(TEST_POSTGRES_URL.replace('/strategy_factory_test', '/postgres'))
            # Check if test database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", 'strategy_factory_test'
            )
            if not exists:
                await conn.execute('CREATE DATABASE "strategy_factory_test"')
            await conn.close()
        except Exception:
            pass  # Database might already exist
        
        # Create the pool with statement caching disabled and JSON codecs
        pool = await asyncpg.create_pool(
            TEST_POSTGRES_URL,
            min_size=1,
            max_size=3,
            command_timeout=60,
            statement_cache_size=0,
            init=init_connection  # Initialize JSON codecs
        )
        
        # Set the global pool so get_pool() uses it
        pg_db._pool = pool
        
        # Clean and create tables
        async with pool.acquire() as conn:
            await conn.execute("DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;")
            
            # Create models table - matches training_service/pg_db.py
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
            
            # Create features_log table - matches pg_db.py
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS features_log (
                    id SERIAL PRIMARY KEY,
                    model_id VARCHAR REFERENCES models(id) ON DELETE CASCADE,
                    feature_name VARCHAR,
                    importance DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
    
    # Run setup
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(setup_tables())
    finally:
        pass  # Don't close the loop yet
    
    # Create TrainingDB using the same module
    db = pg_db.TrainingDB()
    
    yield db
    
    # Cleanup - close pool and restore state
    async def cleanup():
        nonlocal pool
        if pool is not None:
            await pool.close()
        pg_db._pool = None
    
    try:
        loop.run_until_complete(cleanup())
    finally:
        loop.close()
    
    pg_db._pool = original_pool
    pg_db.POSTGRES_URL = original_url


@pytest.fixture
async def simulation_db(db_tables):
    """
    Get SimulationDB instance configured for testing.
    Uses db_tables to ensure tables exist and passes the test pool directly.
    """
    import simulation_service.pg_db as sim_pg_db
    
    # Save original pool state (for cleanup)
    original_sim_pool = sim_pg_db._pool
    original_sim_url = sim_pg_db.POSTGRES_URL
    
    # Set global pool for any code that still uses get_pool() directly
    sim_pg_db._pool = db_tables
    sim_pg_db.POSTGRES_URL = TEST_POSTGRES_URL
    
    # Pass pool directly to SimulationDB
    db = sim_pg_db.SimulationDB(pool=db_tables)
    yield db
    
    # Restore original pool
    sim_pg_db._pool = original_sim_pool
    sim_pg_db.POSTGRES_URL = original_sim_url


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
