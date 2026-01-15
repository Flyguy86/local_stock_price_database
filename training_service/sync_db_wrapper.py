"""
Synchronous wrapper around async PostgreSQL operations.
Used by trainer.py which runs in separate processes.

IMPORTANT: Each subprocess must get its own fresh instance.
The global `db` at module level is recreated per-process.
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

# Track the PID to detect when we're in a new process
_init_pid = os.getpid()

# PostgreSQL URL from environment (for process workers)
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory"
)


class SyncDBWrapper:
    """
    Synchronous wrapper for PostgreSQL operations.
    Each process worker creates its own connection pool.
    
    IMPORTANT: The connection pool is tied to an event loop. We maintain
    a persistent event loop per instance to avoid invalidating connections.
    """
    
    def __init__(self, postgres_url: Optional[str] = None):
        self._pool = None
        self._pool_lock = threading.Lock()
        self._executor = None  # Created lazily
        self._loop = None  # Persistent event loop for this instance
        self._init_pid = os.getpid()  # Track which process created this instance
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
    
    def _check_forked_process(self):
        """
        Check if we're in a forked subprocess and reset state if so.
        
        When Python forks a process, the child inherits the parent's memory
        including any connection pools and event loops, which become invalid.
        We detect this by comparing the current PID to the PID at init time.
        """
        current_pid = os.getpid()
        if current_pid != self._init_pid:
            log.debug(f"Detected forked process (init PID {self._init_pid} -> current {current_pid}), resetting state")
            # Reset all state for the new process
            self._pool = None
            self._loop = None  # Must reset - old loop is invalid in new process
            self._pool_lock = threading.Lock()
            self._executor = None
            self._init_pid = current_pid
    
    def _get_executor(self):
        """Get or create the thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db_async")
        return self._executor
    
    async def _get_or_create_pool(self):
        """Get or create connection pool (async version)."""
        # Check for forked process before accessing pool
        self._check_forked_process()
        
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:  # Double-check locking
                    import asyncpg
                    log.debug(f"Creating new connection pool for PID {os.getpid()}")
                    self._pool = await asyncpg.create_pool(
                        self._postgres_url,  # Use instance URL, not module-level
                        min_size=1,
                        max_size=3,
                        command_timeout=30,
                        statement_cache_size=0,  # Disable statement caching to prevent collisions
                        init=self._init_connection  # Initialize JSON codecs
                    )
        return self._pool
    
    def _get_event_loop(self):
        """
        Get or create a persistent event loop for this instance.
        
        The event loop must persist across calls because the connection pool
        is tied to it. Closing the loop would invalidate all pool connections.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            # Don't set as current thread's event loop to avoid conflicts
        return self._loop
    
    def _execute_async(self, coro):
        """
        Execute async operation safely, handling running event loops.
        
        CRITICAL: We use a persistent event loop because the asyncpg pool
        is tied to the loop that created it. Closing the loop between calls
        would invalidate the pool connections.
        """
        # Check for forked process first
        self._check_forked_process()
        
        # Get our persistent event loop (creates one if needed)
        loop = self._get_event_loop()
        
        try:
            # Check if there's already a running event loop in this thread
            asyncio.get_running_loop()
            # Event loop is running - must use thread with OUR loop
            executor = self._get_executor()
            
            def run_in_thread():
                return loop.run_until_complete(coro)
            
            future = executor.submit(run_in_thread)
            return future.result(timeout=60)
        except RuntimeError:
            # No running event loop in this thread - run directly on our loop
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
        """Cleanup connection pool and event loop."""
        # Close pool first (while loop is still running)
        if self._pool:
            async def _close_pool():
                await self._pool.close()
            
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.run_until_complete(_close_pool())
            except Exception as e:
                log.warning(f"Error closing pool: {e}")
            finally:
                self._pool = None
        
        # Close event loop
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.close()
            except:
                pass
            finally:
                self._loop = None
        
        # Shutdown executor
        if self._executor:
            try:
                self._executor.shutdown(wait=False)
            except:
                pass
            finally:
                self._executor = None
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass


def _get_db():
    """
    Get or create a SyncDBWrapper instance for the current process.
    
    This function ensures each process gets its own fresh instance,
    avoiding issues with inherited state from forked processes.
    """
    global _db_instance, _db_instance_pid
    
    current_pid = os.getpid()
    
    # Check if we need a new instance (first call or forked process)
    if '_db_instance' not in globals() or _db_instance_pid != current_pid:
        log.debug(f"Creating new SyncDBWrapper for process {current_pid}")
        _db_instance = SyncDBWrapper()
        _db_instance_pid = current_pid
    
    return _db_instance

# Initialize for module-level access
_db_instance_pid = os.getpid()
_db_instance = SyncDBWrapper()

# Global instance for trainer.py to import
# Note: The instance will auto-reset its state when used in a forked process
db = _db_instance
