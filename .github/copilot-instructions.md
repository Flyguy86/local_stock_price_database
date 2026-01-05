# Copilot Instructions
## Code accuracy and completeness
- Ensure all code is syntactically correct and complete.
- Include necessary imports, function definitions, and class definitions.
- Ensure spacing and indentation follow Python conventions (4 spaces per indent level).
## Feature_service UI test bench (required)
- The HTML at `/` in [feature_service/web.py](feature_service/web.py) is the manual test harness. For each step, expose a test button, render raw input/output, and show a pass/fail badge beside the button.
- Steps to support now: (1) read DuckDB `local.db` and list unique tickers; (2) allow selecting any/all tickers; (3) trigger feature generation. Pause after these with visible raw data before/after and per-step indicators.
- Preserve checkbox UX (`select all` / `clear`) and status polling via `/status`. New steps should mirror the pattern: button → fetch → render raw data → pass/fail badge.
(3) for each step, we need to have a narrative of what is happening, what to expect, and how to interpret the results. This is critical for non-technical users to understand the process and outcomes.  Also this provides of with stepping stones to the correct final product.


## Core context
- Services: ingestion API in [app/api/main.py](app/api/main.py) (pulls bars from Alpaca/IEX into DuckDB + Parquet) and feature builder in [feature_service](feature_service) (engineers indicators into DuckDB + partitioned Parquet).
- Default paths (container/devcontainer): source DuckDB `/app/data/duckdb/local.db`, feature DB `/app/data/duckdb/features.db`, bars parquet `/app/data/parquet`, features parquet `/app/data/features_parquet`; override via `SOURCE_DUCKDB_PATH`, `DEST_DUCKDB_PATH`, `DEST_PARQUET_DIR`. docker-compose mounts `./data` to `/app/data`.
- DuckDB locking: readers copy the source DB to a temp file and open read-only to avoid locks (see [feature_service/pipeline.py](feature_service/pipeline.py) and `/symbols` handler). Keep this pattern for any new readers of `local.db`.

## Pipelines and patterns
- Ingestion: idempotent insert on `(symbol, ts)` with `ON CONFLICT DO NOTHING`; Parquet partitions mirror DuckDB rows per symbol/date ([app/storage/duckdb_client.py](app/storage/duckdb_client.py)).
- Features: per-symbol recompute; delete overlapping `ts`, insert into `feature_bars`, then write partitioned Parquet per date ([feature_service/pipeline.py](feature_service/pipeline.py)). Update both DDL and Parquet write when adding feature columns.
- Use `_has_table` before querying DuckDB; sort by `ts`, drop duplicates, volumes non-negative, timestamps UTC.
- Logging: favor structured `extra={...}`; ingestion uses `configure_json_logger`, feature_service uses module loggers.


## Key files
- Ingestion flow: [app/ingestion/poller.py](app/ingestion/poller.py), [app/ingestion/alpaca_client.py](app/ingestion/alpaca_client.py), [app/ingestion/iex_client.py](app/ingestion/iex_client.py)
- Storage: [app/storage/duckdb_client.py](app/storage/duckdb_client.py), [app/storage/schema.py](app/storage/schema.py)
- Feature builder: [feature_service/pipeline.py](feature_service/pipeline.py), [feature_service/web.py](feature_service/web.py), [feature_service/main.py](feature_service/main.py)
