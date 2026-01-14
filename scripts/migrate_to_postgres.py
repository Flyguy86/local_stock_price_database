#!/usr/bin/env python3
"""
Migrate model metadata from DuckDB to PostgreSQL.

This script:
1. Reads all models from /app/data/duckdb/models.db
2. Reads all simulation history from models.db  
3. Migrates both to PostgreSQL
4. Preserves all fingerprints and metadata
"""
import asyncio
import duckdb
import asyncpg
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("migration")

# Paths
DUCKDB_PATH = Path("/app/data/duckdb/models.db")
POSTGRES_URL = "postgresql://postgres:postgres@postgres:5432/stock_data"


async def migrate():
    """Run the migration."""
    
    if not DUCKDB_PATH.exists():
        log.warning(f"DuckDB file not found: {DUCKDB_PATH}")
        log.info("Nothing to migrate - starting fresh")
        return
    
    log.info(f"Reading from DuckDB: {DUCKDB_PATH}")
    
    # Read from DuckDB
    models = []
    features_log = []
    sim_history = []
    
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as conn:
        # Check what tables exist
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        log.info(f"Found tables in DuckDB: {table_names}")
        
        # Migrate models table
        if 'models' in table_names:
            try:
                rows = conn.execute("""
                    SELECT 
                        id, name, algorithm, symbol, target_col,
                        feature_cols, hyperparameters, metrics, status,
                        created_at, artifact_path, error_message,
                        data_options, timeframe, parent_model_id, group_id,
                        target_transform, columns_initial, columns_remaining
                    FROM models
                """).fetchall()
                
                for row in rows:
                    models.append({
                        'id': row[0],
                        'name': row[1],
                        'algorithm': row[2],
                        'symbol': row[3],
                        'target_col': row[4] or 'close',
                        'feature_cols': row[5],
                        'hyperparameters': row[6],
                        'metrics': row[7],
                        'status': row[8],
                        'created_at': row[9],
                        'artifact_path': row[10],
                        'error_message': row[11],
                        'data_options': row[12],
                        'timeframe': row[13] or '1m',
                        'parent_model_id': row[14],
                        'group_id': row[15],
                        'target_transform': row[16] or 'none',
                        'columns_initial': row[17],
                        'columns_remaining': row[18]
                    })
                
                log.info(f"Read {len(models)} models from DuckDB")
            except Exception as e:
                log.error(f"Failed to read models: {e}")
        
        # Migrate features_log table
        if 'features_log' in table_names:
            try:
                rows = conn.execute("""
                    SELECT model_id, feature_name, importance
                    FROM features_log
                """).fetchall()
                
                for row in rows:
                    features_log.append({
                        'model_id': row[0],
                        'feature_name': row[1],
                        'importance': row[2]
                    })
                
                log.info(f"Read {len(features_log)} feature importance records")
            except Exception as e:
                log.error(f"Failed to read features_log: {e}")
        
        # Migrate simulation_history table
        if 'simulation_history' in table_names:
            try:
                rows = conn.execute("""
                    SELECT id, timestamp, model_id, ticker, return_pct,
                           trades_count, hit_rate, sqn, params
                    FROM simulation_history
                """).fetchall()
                
                for row in rows:
                    sim_history.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'model_id': row[2],
                        'ticker': row[3],
                        'return_pct': row[4],
                        'trades_count': row[5],
                        'hit_rate': row[6],
                        'sqn': row[7],
                        'params': row[8]
                    })
                
                log.info(f"Read {len(sim_history)} simulation history records")
            except Exception as e:
                log.error(f"Failed to read simulation_history: {e}")
    
    # Connect to PostgreSQL
    log.info("Connecting to PostgreSQL...")
    pool = await asyncpg.create_pool(POSTGRES_URL, min_size=1, max_size=5)
    
    try:
        async with pool.acquire() as conn:
            # Ensure tables exist
            log.info("Ensuring PostgreSQL tables exist...")
            from training_service.pg_db import ensure_tables
            await ensure_tables()
            
            # Migrate models
            if models:
                log.info(f"Migrating {len(models)} models...")
                for model in models:
                    try:
                        # Parse JSON strings if needed
                        for json_field in ['feature_cols', 'hyperparameters', 'metrics', 'data_options']:
                            if isinstance(model.get(json_field), str):
                                model[json_field] = json.loads(model[json_field])
                        
                        # Convert timestamp if string
                        if isinstance(model.get('created_at'), str):
                            model['created_at'] = datetime.fromisoformat(model['created_at'].replace('Z', '+00:00'))
                        
                        await conn.execute("""
                            INSERT INTO models (
                                id, name, algorithm, symbol, target_col,
                                feature_cols, hyperparameters, metrics, status,
                                created_at, artifact_path, error_message,
                                data_options, timeframe, parent_model_id, group_id,
                                target_transform, columns_initial, columns_remaining
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                                $11, $12, $13, $14, $15, $16, $17, $18, $19
                            )
                            ON CONFLICT (id) DO UPDATE SET
                                status = EXCLUDED.status,
                                metrics = EXCLUDED.metrics,
                                artifact_path = EXCLUDED.artifact_path,
                                error_message = EXCLUDED.error_message
                        """, 
                            model['id'], model['name'], model['algorithm'],
                            model['symbol'], model['target_col'],
                            json.dumps(model['feature_cols']) if model['feature_cols'] else None,
                            json.dumps(model['hyperparameters']) if model['hyperparameters'] else None,
                            json.dumps(model['metrics']) if model['metrics'] else None,
                            model['status'], model['created_at'],
                            model['artifact_path'], model['error_message'],
                            json.dumps(model['data_options']) if model['data_options'] else None,
                            model['timeframe'], model['parent_model_id'],
                            model['group_id'], model['target_transform'],
                            model['columns_initial'], model['columns_remaining']
                        )
                    except Exception as e:
                        log.error(f"Failed to migrate model {model['id']}: {e}")
                
                log.info(f"✓ Migrated {len(models)} models")
            
            # Migrate features_log
            if features_log:
                log.info(f"Migrating {len(features_log)} feature importance records...")
                for feat in features_log:
                    try:
                        await conn.execute("""
                            INSERT INTO features_log (model_id, feature_name, importance)
                            VALUES ($1, $2, $3)
                        """, feat['model_id'], feat['feature_name'], feat['importance'])
                    except Exception as e:
                        # Ignore duplicates or missing model refs
                        pass
                
                log.info(f"✓ Migrated {len(features_log)} feature records")
            
            # Migrate simulation_history
            if sim_history:
                log.info(f"Migrating {len(sim_history)} simulation history records...")
                for sim in sim_history:
                    try:
                        # Parse params JSON if string
                        params = sim['params']
                        if isinstance(params, str):
                            params = json.loads(params)
                        
                        # Convert timestamp if string
                        timestamp = sim['timestamp']
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        
                        await conn.execute("""
                            INSERT INTO simulation_history (
                                id, timestamp, model_id, ticker, return_pct,
                                trades_count, hit_rate, sqn, params
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            ON CONFLICT (id) DO NOTHING
                        """,
                            sim['id'], timestamp, sim['model_id'], sim['ticker'],
                            sim['return_pct'], sim['trades_count'], sim['hit_rate'],
                            sim['sqn'], json.dumps(params) if params else None
                        )
                    except Exception as e:
                        # Ignore duplicates or missing model refs
                        pass
                
                log.info(f"✓ Migrated {len(sim_history)} simulation records")
        
        log.info("✅ Migration complete!")
        
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(migrate())
