from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    alpaca_key_id: str | None = None
    alpaca_secret_key: str | None = None
    alpaca_base_url: str = "https://data.alpaca.markets"
    alpaca_feed: str = "iex"
    iex_token: str | None = None
    iex_base_url: str = "https://cloud.iexapis.com"
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/duckdb/local.db")
    parquet_dir: Path = Path("data/parquet")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.parquet_dir.mkdir(parents=True, exist_ok=True)
settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
