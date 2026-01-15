"""
PostgreSQL database layer for simulation service.
Handles simulation_history table and model metadata queries.
"""
import asyncpg
import json
import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
import os

log = logging.getLogger("simulation.pg_db")

# PostgreSQL connection string from environment
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory"
)

_pool: Optional[asyncpg.Pool] = None


async def _init_connection(conn):
    """Initialize connection with JSON codecs for JSONB support."""
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


async def get_pool() -> asyncpg.Pool:
    """Get or create connection pool."""
    global _pool
    if _pool is None:
        log.info(f"Creating PostgreSQL connection pool for simulation service...")
        _pool = await asyncpg.create_pool(
            POSTGRES_URL,
            min_size=2,
            max_size=10,
            command_timeout=60,
            statement_cache_size=0,  # Disable statement caching to prevent collisions
            init=_init_connection  # Initialize JSON codecs on each connection
        )
        log.info("PostgreSQL pool created for simulation service")
    return _pool


async def close_pool():
    """Close connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("PostgreSQL pool closed")


async def ensure_tables():
    """
    Ensure simulation_history table exists.
    Note: models table is managed by training_service.
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Simulation history (migrated from DuckDB)
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
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_simulation_history_model 
            ON simulation_history(model_id)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_simulation_history_ticker 
            ON simulation_history(ticker)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_simulation_history_sqn 
            ON simulation_history(sqn DESC)
        """)
        
        log.info("✓ Simulation tables ensured")


async def save_simulation_history(
    model_id: str,
    ticker: str,
    stats: Dict[str, Any],
    params: Dict[str, Any]
) -> str:
    """
    Save simulation result to PostgreSQL.
    
    Args:
        model_id: Model UUID
        ticker: Stock symbol
        stats: Simulation statistics (return_pct, trades_count, hit_rate, sqn)
        params: Simulation parameters
        
    Returns:
        Simulation record ID (UUID)
    """
    import uuid
    
    pool = await get_pool()
    record_id = str(uuid.uuid4())
    
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO simulation_history 
            (id, model_id, ticker, return_pct, trades_count, hit_rate, sqn, params)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, 
            record_id,
            model_id,
            ticker,
            stats.get('strategy_return_pct', 0.0),
            stats.get('total_trades', 0),
            stats.get('hit_rate_pct', 0.0),
            stats.get('sqn', 0.0),
            params  # asyncpg accepts Python dicts for JSONB columns
        )
        
    log.info(f"Saved simulation history: {record_id} (model={model_id}, ticker={ticker})")
    return record_id


async def get_simulation_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve recent simulation history.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of simulation records
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, timestamp, model_id, ticker, return_pct, trades_count, 
                   hit_rate, sqn, params
            FROM simulation_history
            ORDER BY timestamp DESC
            LIMIT $1
        """, limit)
        
        history = []
        for r in rows:
            # asyncpg returns JSONB columns as Python objects directly
            params = r['params'] if r['params'] else {}
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = {}
            
            history.append({
                "id": r['id'],
                "timestamp": r['timestamp'].isoformat() if r['timestamp'] else None,
                "model_id": r['model_id'],
                "ticker": r['ticker'],
                "return_pct": r['return_pct'],
                "trades_count": r['trades_count'],
                "hit_rate_pct": r['hit_rate'],
                "sqn": r['sqn'],
                "params": params
            })
        
        return history


async def get_top_strategies(limit: int = 15, offset: int = 0) -> Dict[str, Any]:
    """
    Retrieve top strategies sorted by SQN with pagination.
    
    Args:
        limit: Number of records per page
        offset: Pagination offset
        
    Returns:
        {"items": [...], "total": count}
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Get total count
        total_result = await conn.fetchval("""
            SELECT COUNT(*) FROM simulation_history WHERE trades_count > 5
        """)
        total = total_result or 0
        
        # Get paginated results
        rows = await conn.fetch("""
            SELECT id, timestamp, model_id, ticker, return_pct, trades_count,
                   hit_rate, sqn, params
            FROM simulation_history
            WHERE trades_count > 5
            ORDER BY sqn DESC
            LIMIT $1 OFFSET $2
        """, limit, offset)
        
        items = []
        for r in rows:
            # asyncpg returns JSONB columns as Python objects directly
            params = r['params'] if r['params'] else {}
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = {}
            
            items.append({
                "id": r['id'],
                "timestamp": r['timestamp'].isoformat() if r['timestamp'] else None,
                "model_id": r['model_id'],
                "ticker": r['ticker'],
                "return_pct": r['return_pct'],
                "trades_count": r['trades_count'],
                "hit_rate_pct": r['hit_rate'],
                "sqn": r['sqn'],
                "params": params
            })
        
        return {"items": items, "total": total}


async def delete_all_simulation_history() -> bool:
    """Delete all simulation history records."""
    try:
        pool = await get_pool()
        
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM simulation_history")
        
        log.info("Deleted all simulation history")
        return True
        
    except Exception as e:
        log.error(f"Failed to delete simulation history: {e}")
        return False


async def get_models_metadata() -> List[Dict[str, Any]]:
    """
    Get model metadata from PostgreSQL.
    Used to supplement model files with metadata.
    
    Returns:
        List of model metadata dictionaries
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if models table exists (managed by training_service)
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'models'
            )
        """)
        
        if not table_exists:
            log.warning("models table does not exist yet")
            return []
        
        rows = await conn.fetch("""
            SELECT id, algorithm, symbol, created_at, metrics, 
                   timeframe, data_options
            FROM models
            WHERE status = 'completed'
        """)
        
        metadata = []
        for r in rows:
            metadata.append({
                "id": r['id'],
                "algorithm": r['algorithm'],
                "symbol": r['symbol'],
                "created_at": r['created_at'].isoformat() if r['created_at'] else None,
                "metrics": json.loads(r['metrics']) if isinstance(r['metrics'], str) else r['metrics'],
                "timeframe": r['timeframe'] or "1m",
                "data_options": json.loads(r['data_options']) if isinstance(r['data_options'], str) else r['data_options']
            })
        
        return metadata


class SimulationDB:
    """Async PostgreSQL database interface for simulation service."""
    
    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        """
        Initialize SimulationDB.
        
        Args:
            pool: Optional connection pool. If not provided, uses global pool via get_pool().
        """
        self._injected_pool = pool
    
    async def _get_pool(self) -> asyncpg.Pool:
        """Get connection pool - either injected or global."""
        if self._injected_pool is not None:
            return self._injected_pool
        return await get_pool()
    
    async def save_history(self, model_id: str, ticker: str, stats: Dict, params: Dict) -> str:
        """Save simulation result."""
        pool = await self._get_pool()
        record_id = str(uuid.uuid4())
        
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO simulation_history 
                (id, model_id, ticker, return_pct, trades_count, hit_rate, sqn, params)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, 
                record_id,
                model_id,
                ticker,
                stats.get('strategy_return_pct', 0.0),
                stats.get('total_trades', 0),
                stats.get('hit_rate_pct', 0.0),
                stats.get('sqn', 0.0),
                params  # asyncpg accepts Python dicts for JSONB columns
            )
        
        log.info(f"Saved simulation history: {record_id} (model={model_id}, ticker={ticker})")
        return record_id
    
    async def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent simulation history."""
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, timestamp, model_id, ticker, return_pct, trades_count, 
                       hit_rate, sqn, params
                FROM simulation_history
                ORDER BY timestamp DESC
                LIMIT $1
            """, limit)
            
            return [dict(row) for row in rows]
    
    async def get_top_strategies(self, limit: int = 15, offset: int = 0) -> Dict:
        """Get top strategies by SQN."""
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            # Get total count
            total = await conn.fetchval("SELECT COUNT(*) FROM simulation_history")
            
            # Get paginated results
            rows = await conn.fetch("""
                SELECT id, timestamp, model_id, ticker, return_pct, trades_count, 
                       hit_rate, sqn, params
                FROM simulation_history
                ORDER BY sqn DESC NULLS LAST
                LIMIT $1 OFFSET $2
            """, limit, offset)
            
            return {
                'items': [dict(row) for row in rows],
                'total': total or 0,
                'limit': limit,
                'offset': offset
            }
    
    async def delete_all_history(self) -> bool:
        """Delete all simulation history."""
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM simulation_history")
        
        log.info("Deleted all simulation history")
        return True
    
    async def get_models_metadata(self) -> List[Dict]:
        """Get model metadata."""
        return await get_models_metadata()
