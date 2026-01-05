from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices

DEFAULT_ALPACA_KEY_ID = "PKEPWSDBRZZOMMZQF9CS"
DEFAULT_ALPACA_SECRET_KEY = "5iyJMFLNFhCzsGtedThbhuP69Mg1RFuEA0jh44Zn"
DEFAULT_ALPACA_BASE_URL = "https://data.alpaca.markets/"
DEFAULT_ALPACA_TRADING_BASE_URL = "https://paper-api.alpaca.markets"


class Settings(BaseSettings):
    alpaca_key_id: str | None = Field(
        default=DEFAULT_ALPACA_KEY_ID,
        validation_alias=AliasChoices("ALPACA_KEY_ID", "ALPACA_API_KEY_ID"),
    )
    alpaca_secret_key: str | None = Field(
        default=DEFAULT_ALPACA_SECRET_KEY,
        validation_alias=AliasChoices("ALPACA_SECRET_KEY", "ALPACA_API_SECRET_KEY"),
    )
    alpaca_base_url: str = Field(
        default=DEFAULT_ALPACA_BASE_URL,
        validation_alias=AliasChoices("ALPACA_BASE_URL", "ALPACA_API_BASE_URL"),
    )
    alpaca_trading_base_url: str = Field(
        default=DEFAULT_ALPACA_TRADING_BASE_URL,
        validation_alias=AliasChoices("ALPACA_TRADING_BASE_URL", "ALPACA_API_TRADING_BASE_URL"),
    )
    alpaca_feed: str = "iex"
    alpaca_debug_raw: bool = False
    iex_token: str | None = None
    iex_base_url: str = "https://cloud.iexapis.com"
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/duckdb/local.db")
    parquet_dir: Path = Path("data/parquet")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    def model_post_init(self, __context: dict) -> None:
        # If compose provides empty strings, fallback to defaults so real-time ingest has credentials.
        if not self.alpaca_key_id:
            self.alpaca_key_id = DEFAULT_ALPACA_KEY_ID
        if not self.alpaca_secret_key:
            self.alpaca_secret_key = DEFAULT_ALPACA_SECRET_KEY
        if not self.alpaca_base_url:
            self.alpaca_base_url = DEFAULT_ALPACA_BASE_URL
        if not self.alpaca_trading_base_url:
            self.alpaca_trading_base_url = DEFAULT_ALPACA_TRADING_BASE_URL


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.parquet_dir.mkdir(parents=True, exist_ok=True)
settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
