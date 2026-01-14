"""
PostgreSQL database layer for simulation service.
Handles simulation_history table and model metadata queries.
"""
import asyncpg
import json
import logging
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
            statement_cache_size=0  # Disable statement caching to prevent collisions
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
            json.dumps(params)
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
            history.append({
                "id": r['id'],
                "timestamp": r['timestamp'].isoformat() if r['timestamp'] else None,
                "model_id": r['model_id'],
                "ticker": r['ticker'],
                "return_pct": r['return_pct'],
                "trades_count": r['trades_count'],
                "hit_rate_pct": r['hit_rate'],
                "sqn": r['sqn'],
                "params": json.loads(r['params']) if r['params'] else {}
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
            items.append({
                "id": r['id'],
                "timestamp": r['timestamp'].isoformat() if r['timestamp'] else None,
                "model_id": r['model_id'],
                "ticker": r['ticker'],
                "return_pct": r['return_pct'],
                "trades_count": r['trades_count'],
                "hit_rate_pct": r['hit_rate'],
                "sqn": r['sqn'],
                "params": json.loads(r['params']) if r['params'] else {}
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
    
    async def save_history(self, model_id: str, ticker: str, stats: Dict, params: Dict) -> str:
        """Save simulation result."""
        return await save_simulation_history(model_id, ticker, stats, params)
    
    async def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent simulation history."""
        return await get_simulation_history(limit)
    
    async def get_top_strategies(self, limit: int = 15, offset: int = 0) -> Dict:
        """Get top strategies by SQN."""
        return await get_top_strategies(limit, offset)
    
    async def delete_all_history(self) -> bool:
        """Delete all simulation history."""
        return await delete_all_simulation_history()
    
    async def get_models_metadata(self) -> List[Dict]:
        """Get model metadata."""
        return await get_models_metadata()
