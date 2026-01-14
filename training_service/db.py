import duckdb
from .config import settings
import logging
import tempfile
import shutil
import time
import os
from pathlib import Path
from contextlib import contextmanager

log = logging.getLogger("training.db")

INIT_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id VARCHAR PRIMARY KEY,
    name VARCHAR,
    algorithm VARCHAR,
    symbol VARCHAR,
    target_col VARCHAR,
    feature_cols JSON,
    hyperparameters JSON,
    metrics JSON,
    status VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    artifact_path VARCHAR,
    error_message VARCHAR,
    data_options VARCHAR,
    timeframe VARCHAR,
    parent_model_id VARCHAR,
    group_id VARCHAR
);

CREATE TABLE IF NOT EXISTS features_log (
    model_id VARCHAR,
    feature_name VARCHAR,
    importance DOUBLE
);
"""

class MetadataDB:
    def __init__(self):
        self.path = str(settings.metadata_db_path)
        self._init_db()

    def _cleanup_stale_locks(self):
        """Remove WAL file to clear stale locks."""
        db_path = Path(self.path)
        wal_path = db_path.with_suffix(db_path.suffix + ".wal")
        if wal_path.exists():
            try:
                os.remove(wal_path)
                log.info(f"Removed stale WAL file: {wal_path}")
            except Exception as e:
                log.warning(f"Could not remove WAL file: {e}")
    
    def _init_db(self):
        # Try to cleanup stale locks first
        self._cleanup_stale_locks()
        
        with duckdb.connect(self.path) as conn:
            conn.execute(INIT_SQL)
            # Migration for existing tables: add data_options if missing
            try:
                conn.execute("ALTER TABLE models ADD COLUMN data_options VARCHAR")
            except:
                pass
            try:
                conn.execute("ALTER TABLE models ADD COLUMN timeframe VARCHAR")
            except:
                pass
            try:
                conn.execute("ALTER TABLE models ADD COLUMN parent_model_id VARCHAR")
            except:
                pass
            try:
                conn.execute("ALTER TABLE models ADD COLUMN group_id VARCHAR")
            except:
                pass
            try:
                conn.execute("ALTER TABLE models ADD COLUMN parent_model_id VARCHAR")
            except:
                pass
            try:
                conn.execute("ALTER TABLE models ADD COLUMN group_id VARCHAR")
            except:
                pass
            try:
                conn.execute("ALTER TABLE models ADD COLUMN target_transform VARCHAR")
            except:
                pass
            try:
                conn.execute("ALTER TABLE models ADD COLUMN columns_initial INTEGER")
            except:
                pass
            try:
                conn.execute("ALTER TABLE models ADD COLUMN columns_remaining INTEGER")
            except:
                pass

    @contextmanager
    def get_connection(self, read_only=False):
        """
        Get a DuckDB connection.
        For read_only connections, copy DB to temp file to avoid locking conflicts.
        For write connections, retry with WAL cleanup on lock conflicts.
        """
        if not read_only:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    conn = duckdb.connect(self.path, read_only=False)
                    try:
                        yield conn
                        return
                    finally:
                        conn.close()
                except duckdb.IOException as e:
                    if "lock" in str(e).lower() and attempt < max_retries - 1:
                        log.warning(f"Lock conflict on attempt {attempt + 1}, cleaning up WAL and retrying...")
                        self._cleanup_stale_locks()
                        time.sleep(0.5)
                    else:
                        raise
        else:
            # For read-only, copy to temp file to avoid lock conflicts
            tmpdir = tempfile.TemporaryDirectory()
            try:
                db_path = Path(self.path)
                tmp_db = Path(tmpdir.name) / db_path.name
                shutil.copy2(db_path, tmp_db)
                
                # Also copy WAL file if it exists
                wal_path = db_path.with_suffix(db_path.suffix + ".wal")
                if wal_path.exists():
                    tmp_wal = tmp_db.with_suffix(tmp_db.suffix + ".wal")
                    shutil.copy2(wal_path, tmp_wal)
                
                # Open and yield read-only connection to temp copy
                conn = duckdb.connect(str(tmp_db), read_only=True)
                try:
                    yield conn
                finally:
                    conn.close()
            finally:
                tmpdir.cleanup()

    def list_models(self):
        with self.get_connection(read_only=True) as conn:
            cols = ["id", "name", "algorithm", "symbol", "status", "metrics", "created_at", "error_message", "data_options", "timeframe", "target_col", "parent_model_id", "group_id", "target_transform", "columns_initial", "columns_remaining"]
            return conn.execute(f"SELECT {', '.join(cols)} FROM models ORDER BY created_at DESC").fetch_df().to_dict(orient="records")

    def get_model(self, model_id: str):
        with self.get_connection(read_only=True) as conn:
            return conn.execute("SELECT * FROM models WHERE id = ?", [model_id]).fetchone()

    def create_model_record(self, data: dict):
        keys = list(data.keys())
        placeholders = ", ".join(["?" for _ in keys])
        columns = ", ".join(keys)
        values = list(data.values())
        
        query = f"INSERT INTO models ({columns}) VALUES ({placeholders})"
        with self.get_connection() as conn:
            conn.execute(query, values)
            
    def update_model_status(self, model_id: str, status: str, metrics: str | None = None, artifact_path: str | None = None, error: str | None = None, feature_cols: str | None = None, target_transform: str | None = None, columns_initial: int | None = None, columns_remaining: int | None = None):
        updates = ["status = ?"]
        params = [status]
        
        if metrics:
            updates.append("metrics = ?")
            params.append(metrics)
        if artifact_path:
            updates.append("artifact_path = ?")
            params.append(artifact_path)
        if error:
            updates.append("error_message = ?")
            params.append(error)
        if feature_cols:
            updates.append("feature_cols = ?")
            params.append(feature_cols)
        if target_transform:
             updates.append("target_transform = ?")
             params.append(target_transform)
        if columns_initial is not None:
            updates.append("columns_initial = ?")
            params.append(columns_initial)
        if columns_remaining is not None:
            updates.append("columns_remaining = ?")
            params.append(columns_remaining)
            
        params.append(model_id)
        query = f"UPDATE models SET {', '.join(updates)} WHERE id = ?"
        
        with self.get_connection() as conn:
            conn.execute(query, params)

    def delete_model(self, model_id: str):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM models WHERE id = ?", [model_id])
            conn.execute("DELETE FROM features_log WHERE model_id = ?", [model_id])

    def delete_all_models(self):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM models")
            conn.execute("DELETE FROM features_log")

db = MetadataDB()
