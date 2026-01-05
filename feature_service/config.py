import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    source_db: Path = Path(os.getenv("SOURCE_DUCKDB_PATH", "/app/data/duckdb/local.db"))
    dest_db: Path = Path(os.getenv("DEST_DUCKDB_PATH", "/app/data/duckdb/features.db"))
    dest_parquet: Path = Path(os.getenv("DEST_PARQUET_DIR", "/app/data/features_parquet"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def ensure_paths(self) -> None:
        self.dest_db.parent.mkdir(parents=True, exist_ok=True)
        self.dest_parquet.mkdir(parents=True, exist_ok=True)
