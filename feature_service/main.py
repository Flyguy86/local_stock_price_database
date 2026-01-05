import argparse
import logging
import os
from typing import Optional

from .config import Config
from .pipeline import run_pipeline


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(symbols: Optional[list[str]] = None) -> None:
    cfg = Config()
    cfg.ensure_paths()
    setup_logging(cfg.log_level)
    logging.getLogger("feature_service").info(
        "starting feature builder",
        extra={
            "source_db": str(cfg.source_db),
            "dest_db": str(cfg.dest_db),
            "dest_parquet": str(cfg.dest_parquet),
            "symbols": symbols,
        },
    )
    result = run_pipeline(cfg.source_db, cfg.dest_db, cfg.dest_parquet, symbols)
    logging.getLogger("feature_service").info("run complete", extra=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature engineering over stored bars")
    parser.add_argument("symbols", nargs="*", help="Symbols to process (default: all symbols in source DB)")
    args = parser.parse_args()

    symbols_from_cli = args.symbols if args.symbols else None
    if symbols_from_cli:
        symbols = symbols_from_cli
    else:
        symbols_env = os.getenv("SYMBOLS")
        symbols = [s.strip() for s in symbols_env.split(",") if s.strip()] if symbols_env else None

    main(symbols)
