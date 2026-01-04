# local_stock_price_database
Stock price database

## Architecture Overview
- Ingestion: Alpaca client pulling 1-min bars; derive higher intervals from 1-min locally.
- Storage: DuckDB + Parquet, partitioned by symbol/date; metadata tables for feature versions.
- Orchestration: three agents (historical backfill, live updater, feature builder) coordinated via lightweight task queue (Celery/Prefect), with idempotent jobs keyed by symbol/date.
- Interfaces: REST API + web UI (FastAPI + Streamlit/React) for ingest queueing, status, browsing bars/features.
- Logging/Monitoring: JSON logs (timestamp, agent, symbol, stage, message) stored centrally (SQLite/OpenTelemetry export), surfacing heartbeats/progress bars in UI.
- Deployment: Dockerfile + docker-compose; also runnable directly in Codespaces/devcontainer.

## Ingestion Requirements
- Always fetch 1-min bars from Alpaca; compute 5-min+ locally.
- Throttle via Alpaca limit headers; retry with exponential backoff.
- Gap scanner to re-request missing sessions before persist.
- Web input for ticker; history agent pulls multi-page Alpaca data, checks existence, creates tables if absent, inserts only missing data.
- Idempotent inserts based on (symbol, timestamp); skip existing rows.
- Detailed logging around API calls and persistence.

## Storage & Schema
- DuckDB with Parquet tables; columnar layout for raw bars + engineered features.
- Blueprint: symbol, timestamp, open, high, low, close, volume, optional corporate actions.
- Companion metadata table: feature_name, version, generation params/hash, created_at.
- Partition Parquet by symbol/date; indexes on symbol, timestamp, feature groups.
- Store preprocessing/feature pipeline code references alongside schema for reproducibility.

## Web/API Layer
- REST endpoints: queue ingest jobs, check agent status/heartbeats, list symbols, fetch bars/features, trigger feature rebuilds.
- Dashboard (Streamlit or React+FastAPI): filters, charts, feature previews; show per-agent logs/status/start/stop controls; progress bars.

## Logging & Monitoring
- JSON logs standardized across agents.
- Central log store (SQLite table) with optional OpenTelemetry exporter.
- UI surfaces agent heartbeat metrics; alerts when heartbeats stall; progress/state/next-step indicators.

## Orchestration & Agents
- Historical backfill: bulk 1-min pull with gap scan.
- Live updater: tailing near-real-time with throttling/backoff.
- Feature builder: generates/updates feature Parquet partitions, records metadata versions.
- Task queue (Celery/Prefect): schedules/idempotent jobs keyed by symbol/date; avoids double inserts.

## Deployment & Local Dev
- Provide Dockerfile + docker-compose for API + worker + optional UI.
- Also support direct run in GitHub Codespaces/devcontainer (Ubuntu 24.04.3).
- Basic database viewing: DuckDB CLI/SQL + simple UI queries.

## Quickstart (devcontainer/local)
- Install deps: `pip install -e .`
- Run API: `uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000`
- Trigger ingest: `curl -X POST http://localhost:8000/ingest/SPY`
- Fetch latest bars: `curl http://localhost:8000/bars/SPY?limit=5`

## Quickstart (docker-compose)
- Build: `docker-compose build`
- Run: `docker-compose up`
- API available at `http://localhost:8000`

## Notes
- Configure Alpaca keys via env: `ALPACA_KEY_ID`, `ALPACA_SECRET_KEY`.
- Optionally set Alpaca feed (defaults to IEX) via env: `ALPACA_FEED=iex`.
- Or configure IEX via env: `IEX_TOKEN` (uses `https://cloud.iexapis.com` by default).
- Data + DuckDB files default under `./data/duckdb`.
- Parquet partitions: `data/parquet/{symbol}/dt={YYYY-MM-DD}/...`

## Next Steps
- Scaffold FastAPI + task queue + worker processes.
- Implement Alpaca client with throttling/backoff + gap scanner.
- Define DuckDB schema creation + Parquet partitioning helpers.
- Build logging middleware (JSON) + heartbeat table.
- Create minimal UI (Streamlit/React) for ingest queueing, status, and data browsing.
- Add Dockerfile/docker-compose and devcontainer wiring.
