"""
PostgreSQL database layer for training service.
Replaces DuckDB for model metadata to enable multi-worker access.
"""
import asyncpg
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

log = logging.getLogger("training.pg_db")

# PostgreSQL connection string from environment
import os
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://postgres:postgres@postgres:5432/stock_data"
)

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create connection pool."""
    global _pool
    if _pool is None:
        log.info(f"Creating PostgreSQL connection pool for training service...")
        _pool = await asyncpg.create_pool(
            POSTGRES_URL,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        log.info("PostgreSQL pool created for training service")
    return _pool


async def close_pool():
    """Close connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("PostgreSQL pool closed")


async def ensure_tables():
    """Create tables if they don't exist."""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Models table with comprehensive fingerprinting fields
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
                
                -- Fingerprint for deduplication (computed from config)
                fingerprint VARCHAR(64),
                
                -- Grid search configuration (for fingerprint)
                alpha_grid JSONB,
                l1_ratio_grid JSONB,
                regime_configs JSONB,
                
                -- Context models used during training (for fingerprint)
                context_symbols JSONB,
                
                -- Cross-validation folds (for fingerprint)
                cv_folds INTEGER DEFAULT 5,
                cv_strategy VARCHAR DEFAULT 'time_series_split'
            )
        """)
        
        # Create index on fingerprint for fast lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_models_fingerprint 
            ON models(fingerprint)
        """)
        
        # Create index on symbol for filtering
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_models_symbol 
            ON models(symbol)
        """)
        
        # Features importance log
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS features_log (
                id SERIAL PRIMARY KEY,
                model_id VARCHAR REFERENCES models(id) ON DELETE CASCADE,
                feature_name VARCHAR,
                importance DOUBLE PRECISION,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_log_model 
            ON features_log(model_id)
        """)
        
        # Simulation history (migrated from DuckDB)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS simulation_history (
                id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT NOW(),
                model_id VARCHAR REFERENCES models(id) ON DELETE CASCADE,
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
            CREATE INDEX IF NOT EXISTS idx_simulation_history_sqn 
            ON simulation_history(sqn DESC)
        """)
        
        log.info("Training service tables ensured in PostgreSQL")


class TrainingDB:
    """Async PostgreSQL database interface for training service."""
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models with metadata."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    id, name, algorithm, symbol, status, metrics, created_at,
                    error_message, data_options, timeframe, target_col,
                    parent_model_id, group_id, target_transform,
                    columns_initial, columns_remaining, fingerprint,
                    context_symbols, cv_folds, cv_strategy
                FROM models
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in rows]
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a single model by ID."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM models WHERE id = $1",
                model_id
            )
            return dict(row) if row else None
    
    async def create_model_record(self, data: Dict[str, Any]) -> None:
        """Create a new model record."""
        pool = await get_pool()
        
        # Ensure JSONB fields are kept as JSON strings for asyncpg
        # asyncpg expects JSON strings for JSONB columns, not Python objects
        for json_field in ['feature_cols', 'hyperparameters', 'metrics', 'data_options', 
                          'alpha_grid', 'l1_ratio_grid', 'regime_configs', 'context_symbols']:
            if json_field in data:
                # If it's a Python object (dict/list), convert to JSON string
                if not isinstance(data[json_field], str) and data[json_field] is not None:
                    data[json_field] = json.dumps(data[json_field])
        
        async with pool.acquire() as conn:
            # Build dynamic insert
            columns = list(data.keys())
            values = list(data.values())
            placeholders = ', '.join(f'${i+1}' for i in range(len(values)))
            columns_str = ', '.join(columns)
            
            query = f"INSERT INTO models ({columns_str}) VALUES ({placeholders})"
            await conn.execute(query, *values)
            log.info(f"Created model record: {data.get('id', 'unknown')}")
    
    async def update_model_status(
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
    ) -> None:
        """Update model status and metadata."""
        pool = await get_pool()
        
        updates = ["status = $1"]
        params = [status]
        param_idx = 2
        
        if metrics:
            updates.append(f"metrics = ${param_idx}")
            # Keep as JSON string for asyncpg JSONB column
            params.append(metrics if isinstance(metrics, str) else json.dumps(metrics))
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
            # Keep as JSON string for asyncpg JSONB column
            params.append(feature_cols if isinstance(feature_cols, str) else json.dumps(feature_cols))
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
    
    async def save_feature_importance(
        self,
        model_id: str,
        feature_name: str,
        importance: float
    ) -> None:
        """Save feature importance to features_log."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO features_log (model_id, feature_name, importance)
                VALUES ($1, $2, $3)
            """, model_id, feature_name, importance)
    
    async def get_feature_importance(self, model_id: str) -> List[Dict[str, Any]]:
        """Get feature importance for a model."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT feature_name, importance
                FROM features_log
                WHERE model_id = $1
                ORDER BY ABS(importance) DESC
            """, model_id)
            return [{"feature": row['feature_name'], "importance": row['importance']} for row in rows]
    
    async def get_model_by_fingerprint(self, fingerprint: str) -> Optional[str]:
        """Find model ID with matching fingerprint."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id FROM models
                WHERE fingerprint = $1
                LIMIT 1
            """, fingerprint)
            return row['id'] if row else None
    
    async def delete_model(self, model_id: str) -> None:
        """Delete a model and its associated data."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Foreign key cascade will delete features_log and simulation_history
            await conn.execute("DELETE FROM models WHERE id = $1", model_id)
            log.info(f"Deleted model {model_id} from database")
    
    async def delete_all_models(self) -> None:
        """Delete all models and associated data."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM models")
            log.info("Deleted all models from database")
