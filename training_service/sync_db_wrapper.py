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
    
    def __init__(self):
        self._pool = None
    
    def _get_pool(self):
        """Get or create connection pool for this process."""
        if self._pool is None:
            import asyncpg
            # Create a new event loop for this process
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Create connection pool
            self._pool = loop.run_until_complete(
                asyncpg.create_pool(
                    POSTGRES_URL,
                    min_size=1,
                    max_size=3,
                    command_timeout=30
                )
            )
        return self._pool
    
    def _execute_async(self, coro):
        """Execute async operation in process's event loop."""
        try:
            loop = asyncio.get_event_loop()
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
            pool = self._get_pool()
            updates = ["status = $1"]
            params = [status]
            param_idx = 2
            
            if metrics:
                updates.append(f"metrics = ${param_idx}")
                params.append(json.loads(metrics) if isinstance(metrics, str) else metrics)
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
                params.append(json.loads(feature_cols) if isinstance(feature_cols, str) else feature_cols)
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
            pool = self._get_pool()
            
            # Convert JSON fields to proper format
            for json_field in ['feature_cols', 'hyperparameters', 'metrics', 'data_options',
                              'alpha_grid', 'l1_ratio_grid', 'regime_configs', 'context_symbols']:
                if json_field in data and isinstance(data[json_field], str):
                    data[json_field] = json.loads(data[json_field])
            
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
            pool = self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO features_log (model_id, feature_name, importance)
                    VALUES ($1, $2, $3)
                """, model_id, feature_name, importance)
        
        return self._execute_async(_save())
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model."""
        async def _get():
            pool = self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM models WHERE id = $1",
                    model_id
                )
                return dict(row) if row else None
        
        return self._execute_async(_get())
    
    def __del__(self):
        """Cleanup connection pool when process exits."""
        if self._pool:
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._pool.close())
            except:
                pass
        class DummyContext:
            def __enter__(self):
                raise NotImplementedError("Direct connection access not supported with PostgreSQL")
            def __exit__(self, *args):
                pass
        return DummyContext()


# Global instance for trainer.py to import (created per-process)
db = SyncDBWrapper()
