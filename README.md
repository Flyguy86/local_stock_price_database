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

## Feature Architecture & Pipeline Design
This project follows a "Manual Pipeline" architecture with two distinct phases of feature engineering.

### Phase 1: Time Series Engineering (Feature Service)
*   **Role**: The "Data Warehouse" layer. Calculates stateful, history-dependent features (SMA, RSI, Bollinger Bands, Lags).
*   **Logic**: Mathematical indicators are calculated on the **entire continuous history** of a ticker *before* any train/test splitting occurs.
    *   This ensures indicators like `SMA_200` are valid immediately at the start of any testing fold (no "warmup" loss).
    *   Since these are "lagged" indicators (using only past data), calculating them globally is not leakage.
*   **Output**: "Wide" Parquet files containing all possible features.

### Phase 2: Model Training (Training Service)
*   **Role**: The "Model" layer. Handles stateless transformations (Imputation, Scaling, Selection).
*   **Multi-Ticker Support**: You can train on multiple tickers (e.g., `GOOGL,VIX,SPY`).
    *   **Primary Ticker (First)**: Source of `target` variable and base features.
    *   **Context Tickers**: Merged via Inner Join on Timestamp. Columns are automatically renamed (e.g., `close_VIX`, `rsi_14_SPY`).
    *   **Alignment**: Strict 1-minute timestamp locking; missing minutes in any context ticker results in dropped rows to preserve data integrity.
*   **Design**: Models should utilize `scikit-learn` Pipelines (`Pipeline([Scaler, Imputer, Model])`) to bundle preprocessing logic into the saved artifact (`.joblib`), ensuring the Simulation Service can ingest raw feature data.

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

## Simulation & Strategy
The system includes a backtesting simulation engine to evaluate the performance of trained models.

### How it Works
1.  **Data Loading**: The simulation loads historical feature data (created by the Feature Service) for a specific ticker.
2.  **Model Prediction**: The selected trained model (from the Training Service) generates predictions for every time step in the simulation period.
3.  **Signal Generation**:
    *   **Classifiers**: 
        *   Prediction `1` (Up) → **Buy Signal**
        *   Prediction `0` (Down) → **Sell Signal**
    *   **Regressors**: 
        *   Prediction `> 0` → **Buy Signal**
        *   Prediction `<= 0` → **Sell Signal**
4.  **Trading Execution (Walk-Forward Loop)**:
    *   The simulation iterates through the data chronologically, maintaining a cash and share balance.
    *   **Buying**:
        *   **Trigger**: A **Buy Signal** is received AND the portfolio currently holds **0 shares**.
        *   **Action**: Buys the maximum number of shares possible with available cash at the current close price.
    *   **Selling**:
        *   **Trigger**: A **Sell Signal** is received AND the portfolio currently holds **shares > 0**.
        *   **Action**: Sells **100%** of the held shares at the current close price.
    *   **Holding**: If the signal matches the current position (e.g., Buy Signal while holding), no action is taken.
5.  **Metrics**: The results (Equity Curve, Trades, Returns) are calculated and compared against a "Buy and Hold" benchmark.

## Feature Builder (terminal)
- Install deps (from repo root): `pip install -e .`
- Optional env to override paths: `SOURCE_DUCKDB_PATH`, `DEST_DUCKDB_PATH`, `DEST_PARQUET_DIR`.
- Run all symbols from the source DB: `python -m feature_service.main`
- Run selected symbols: `python -m feature_service.main SPY QQQ`
- In docker-compose (override command): `docker-compose run --rm feature_builder python -m feature_service.main SPY`

## Environment variables (.env example)
Create a `.env` in the repo root:
```
ALPACA_API_KEY_ID=PKEPWSDBRZZOMMZQF
ALPACA_API_SECRET_KEY=5iyJMFLNFhCzhbhuP69Mg1RFuEA0jh44Zn
ALPACA_API_BASE_URL=https://data.alpaca.markets/
ALPACA_FEED=iex
# Optional IEX direct token
IEX_TOKEN=YOUR_IEX_TOKEN
```
Both `ALPACA_*` and `ALPACA_API_*` names are accepted.

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
