import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    features_parquet_dir: Path = Path("/app/data/features_parquet")
    models_dir: Path = Path("/app/data/models")
    metadata_db_path: Path = Path("/app/data/duckdb/models.db")
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

    def ensure_paths(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)

settings = Settings()
