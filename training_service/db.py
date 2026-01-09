import duckdb
from .config import settings
import logging

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
    timeframe VARCHAR
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

    def _init_db(self):
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

    def get_connection(self):
        return duckdb.connect(self.path)

    def list_models(self):
        with self.get_connection() as conn:
            cols = ["id", "name", "algorithm", "symbol", "status", "metrics", "created_at", "error_message", "data_options", "timeframe", "target_col"]
            return conn.execute(f"SELECT {', '.join(cols)} FROM models ORDER BY created_at DESC").fetch_df().to_dict(orient="records")

    def get_model(self, model_id: str):
        with self.get_connection() as conn:
            return conn.execute("SELECT * FROM models WHERE id = ?", [model_id]).fetchone()

    def create_model_record(self, data: dict):
        keys = list(data.keys())
        placeholders = ", ".join(["?" for _ in keys])
        columns = ", ".join(keys)
        values = list(data.values())
        
        query = f"INSERT INTO models ({columns}) VALUES ({placeholders})"
        with self.get_connection() as conn:
            conn.execute(query, values)
            
    def update_model_status(self, model_id: str, status: str, metrics: str | None = None, artifact_path: str | None = None, error: str | None = None, feature_cols: str | None = None):
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
            
        params.append(model_id)
        query = f"UPDATE models SET {', '.join(updates)} WHERE id = ?"
        
        with self.get_connection() as conn:
            conn.execute(query, params)

    def delete_model(self, model_id: str):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM models WHERE id = ?", [model_id])
            conn.execute("DELETE FROM features_log WHERE model_id = ?", [model_id])

db = MetadataDB()
