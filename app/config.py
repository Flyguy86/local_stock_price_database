from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices

class Settings(BaseSettings):
    alpaca_key_id: str | None = Field(
        default="PKEPWSDBRZZOMMZQF9CS",
        validation_alias=AliasChoices("ALPACA_KEY_ID", "ALPACA_API_KEY_ID"),
    )
    alpaca_secret_key: str | None = Field(
        default="5iyJMFLNFhCzsGtedThbhuP69Mg1RFuEA0jh44Zn",
        validation_alias=AliasChoices("ALPACA_SECRET_KEY", "ALPACA_API_SECRET_KEY"),
    )
    alpaca_base_url: str = Field(
        default="https://data.alpaca.markets/",
        validation_alias=AliasChoices("ALPACA_BASE_URL", "ALPACA_API_BASE_URL"),
    )
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
