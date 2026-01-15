"""
Synchronous wrapper around async PostgreSQL operations.
Used by trainer.py which runs in separate processes.
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
import httpx
import os
import threading
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("training.sync_db")

# PostgreSQL URL from environment (for process workers)
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory"
)


class SyncDBWrapper:
    """
    Synchronous wrapper for PostgreSQL operations.
    Each process worker creates its own connection pool.
    """
    
    def __init__(self, postgres_url: Optional[str] = None):
        self._pool = None
        self._pool_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db_async")
        # Use provided URL, or check env at init time (not module load time)
        self._postgres_url = postgres_url or os.environ.get(
            "POSTGRES_URL",
            POSTGRES_URL  # Fall back to module-level default
        )
    
    async def _init_connection(self, conn):
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
    
    async def _get_or_create_pool(self):
        """Get or create connection pool (async version)."""
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:  # Double-check locking
                    import asyncpg
                    self._pool = await asyncpg.create_pool(
                        self._postgres_url,  # Use instance URL, not module-level
                        min_size=1,
                        max_size=3,
                        command_timeout=30,
                        statement_cache_size=0,  # Disable statement caching to prevent collisions
                        init=self._init_connection  # Initialize JSON codecs
                    )
        return self._pool
    
    def _execute_async(self, coro):
        """
        Execute async operation safely, handling running event loops.
        
        If an event loop is already running in this thread, we run the
        coroutine in a separate thread with its own event loop.
        """
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            # Event loop is running - execute in separate thread
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            future = self._executor.submit(run_in_thread)
            return future.result()
        except RuntimeError:
            # No running event loop - safe to run directly
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(coro)
    
    def update_model_status(
        self,
        model_id: str,
        status: str,
        metrics: Optional[str] = None,
        artifact_path: Optional[str] = None,
        error: Optional[str] = None,
        feature_cols: Optional[str] = None,
        target_transform: Optional[str] = None,
        columns_initial: Optional[int] = None,
        columns_remaining: Optional[int] = None,
        fingerprint: Optional[str] = None
    ):
        """Update model status."""
        async def _update():
            pool = await self._get_or_create_pool()
            updates = ["status = $1"]
            params = [status]
            param_idx = 2
            
            if metrics:
                updates.append(f"metrics = ${param_idx}")
                # Parse JSON string to Python object for JSONB column (codec handles serialization)
                if isinstance(metrics, str):
                    try:
                        params.append(json.loads(metrics))
                    except json.JSONDecodeError:
                        params.append(metrics)
                else:
                    params.append(metrics)
                param_idx += 1
            
            if artifact_path:
                updates.append(f"artifact_path = ${param_idx}")
                params.append(artifact_path)
                param_idx += 1
            
            if error:
                updates.append(f"error_message = ${param_idx}")
                params.append(error)
                param_idx += 1
            
            if feature_cols:
                updates.append(f"feature_cols = ${param_idx}")
                # Parse JSON string to Python object for JSONB column
                if isinstance(feature_cols, str):
                    try:
                        params.append(json.loads(feature_cols))
                    except json.JSONDecodeError:
                        params.append(feature_cols)
                else:
                    params.append(feature_cols)
                param_idx += 1
            
            if target_transform:
                updates.append(f"target_transform = ${param_idx}")
                params.append(target_transform)
                param_idx += 1
            
            if columns_initial is not None:
                updates.append(f"columns_initial = ${param_idx}")
                params.append(columns_initial)
                param_idx += 1
            
            if columns_remaining is not None:
                updates.append(f"columns_remaining = ${param_idx}")
                params.append(columns_remaining)
                param_idx += 1
            
            if fingerprint:
                updates.append(f"fingerprint = ${param_idx}")
                params.append(fingerprint)
                param_idx += 1
            
            params.append(model_id)
            query = f"UPDATE models SET {', '.join(updates)} WHERE id = ${param_idx}"
            
            async with pool.acquire() as conn:
                await conn.execute(query, *params)
        
        return self._execute_async(_update())
    
    def create_model_record(self, data: Dict[str, Any]):
        """Create model record."""
        async def _create():
            pool = await self._get_or_create_pool()
            
            # Parse JSON strings to Python objects for JSONB columns
            # The pool has JSON codecs that handle serialization
            for json_field in ['feature_cols', 'hyperparameters', 'metrics', 'data_options',
                              'alpha_grid', 'l1_ratio_grid', 'regime_configs', 'context_symbols']:
                if json_field in data and data[json_field] is not None:
                    # If it's a JSON string, parse to Python object
                    if isinstance(data[json_field], str):
                        try:
                            data[json_field] = json.loads(data[json_field])
                        except json.JSONDecodeError:
                            pass  # Keep as string if not valid JSON
            
            columns = list(data.keys())
            values = list(data.values())
            placeholders = ', '.join(f'${i+1}' for i in range(len(values)))
            columns_str = ', '.join(columns)
            
            query = f"INSERT INTO models ({columns_str}) VALUES ({placeholders})"
            
            async with pool.acquire() as conn:
                await conn.execute(query, *values)
        
        return self._execute_async(_create())
    
    def save_feature_importance(self, model_id: str, feature_name: str, importance: float):
        """Save feature importance."""
        async def _save():
            pool = await self._get_or_create_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO features_log (model_id, feature_name, importance)
                    VALUES ($1, $2, $3)
                """, model_id, feature_name, importance)
        
        return self._execute_async(_save())
    
    def list_models(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List models with pagination."""
        async def _list():
            pool = await self._get_or_create_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM models ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    limit, offset
                )
                return [dict(row) for row in rows]
        
        return self._execute_async(_list())
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model."""
        async def _get():
            pool = await self._get_or_create_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM models WHERE id = $1",
                    model_id
                )
                return dict(row) if row else None
        
        return self._execute_async(_get())
    
    def close(self):
        """Cleanup connection pool."""
        if self._pool:
            async def _close():
                await self._pool.close()
            
            try:
                self._execute_async(_close())
            except Exception as e:
                log.warning(f"Error closing pool: {e}")
        
        if self._executor:
            try:
                self._executor.shutdown(wait=False)
            except:
                pass
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass


# Global instance for trainer.py to import (created per-process)
db = SyncDBWrapper()
