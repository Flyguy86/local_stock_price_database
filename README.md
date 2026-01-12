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
- **Robustness**: Handles symbols with no data (e.g., VIX) by attempting provider fallbacks (Alpaca -> IEX -> Yahoo). Explicitly marks status as "failed" if no data is returned from any provider.

## Storage & Schema
- DuckDB with Parquet tables; columnar layout for raw bars + engineered features.
- Blueprint: symbol, timestamp, open, high, low, close, volume, optional corporate actions.
- Companion metadata table: feature_name, version, generation params/hash, created_at.
- Partition Parquet by symbol/date; indexes on symbol, timestamp, feature groups.
- Store preprocessing/feature pipeline code references alongside schema for reproducibility.
- **Locking & Concurrency**: Feature Service reads a temporary snapshot copy of the DuckDB file to avoid contentions with the ingestion writer process.
- **Missing Data Backfill**: Automated detection and filling of missing 1-minute bars during market hours. Gaps are filled with the mean value of adjacent bars (OHLCV fields). Backfill operations are iterative and can be run repeatedly to handle multiple gaps. Only processes gaps during US market trading hours (9:30 AM - 4:00 PM ET, Mon-Fri).

## Web/API Layer
- REST endpoints: queue ingest jobs, check agent status/heartbeats, list symbols, fetch bars/features, trigger feature rebuilds, trigger backfill operations.
- Dashboard (Streamlit or React+FastAPI): filters, charts, feature previews; show per-agent logs/status/start/stop controls; progress bars.
- **Feature Builder Viewer**: Inspect generated features with ticker filtering.
- **Backfill API**: `POST /backfill/{symbol}?max_iterations=N` to detect and fill missing 1-minute bars during market hours.

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
*   **Segmented Data**: Optionally splits data into "Train" and "Test" episodes (e.g. 30 days train, 5 days test) directly in the feature table, simplifying backtesting logic.
*   **New Features**: Includes specialized financial indicators:
    *   **Volume & Volatility**: `volume_change`, `log_return_1m`, `return_z_score_20` (Vol Spike), `vol_ratio_60` (Rel Vol), `atr_ratio_15` (Range Expansion).
    *   **Universal Alphas** (New):
        *   **Liquidity**: `amihud_illiquidity` (Price impact per unit volume).
        *   **Mean Reversion**: `ibs` (Internal Bar Strength - position of Close within High-Low range).
        *   **Volatility**: `parkinson_vol` (High-Low based estimator), `efficiency_ratio` (Trend quality).
    *   **Advanced**: `dist_vwap` (Mean Reversion), `intraday_intensity` ("Smart Money" Index), `vol_adj_mom_20` (Z-Score Momentum).
    *   **Market Context**: `VIXY`-derived metrics (`vix_log_ret`, `vix_z_score`, `vix_rel_vol`, `vix_atr_ratio`) are automatically calculated and merged onto *every* ticker to provide regime awareness.
    *   **Regime Tagging** (New):
        *   **Global Context**: The pipeline pre-calculates market states using QQQ and VIXY before processing individual tickers.
        *   **VIX Regimes**: 4-Quadrant Logic (Bull/Bear x Quiet/Volatile) based on VIX levels and trends.
        *   **GMM Clusters**: Gaussian Mixture Models cluster market conditions into "Low Volatility/Steady" vs "High Volatility/Crash" regimes.
            *   **Smoothing**: Applied a 30-minute Rolling Mode (Hysteresis) to the GMM output to prevent rapid state-switching noise.
        *   **Trend Filter**: Regimes based on Price vs SMA 200 interaction.
*   **Output**: "Wide" Parquet files containing all possible features.

### Phase 2: Model Training (Training Service)
*   **Clean Model feature identification**:  Phase	Model	Goal	Output
    * Phase 1: Filter	ElasticNet	Feature Selection. Identify which features have a linear relationship with the target.	A reduced list of features where Coeff != 0.
    * Phase 2: Predict	XGBoost	Non-Linear Alpha. Find complex interactions (e.g., "If Volume is high AND RSI is low, then Buy").	High-fidelity price/return predictions.
*   **Role**: The "Model" layer. Handles stateless transformations (Imputation, Scaling, Selection).
*   **Capabilities**:
    *   **Timeframe Selection**: Train models on resampled bars (e.g., `1h`, `4h`, `8h`) derived from the base 1-minute data. Custom aggregation ensures "Test" data never leaks into "Train" buckets during resampling.
    *   **Parent-Child Models**: Support for **Iterative Model Training**.
        *   **Workflow**: Train a "Parent" model (e.g., RandomForest) to identify key features, then train a "Child" model (e.g., LinearRegression) using *only* those selected features.
        *   **Inheritance**: Child models strictly inherit the feature subset from the parent.
        *   **Feature Selection UI**: The dashboard allows you to view parent features, see their importance metrics (SHAP, Permutation, Coeff), filter by name, and manually whitelist/blacklist features for the new training job.
        *   **Smart Selection**: "Auto-Select" button automatically picks the top 2 features per category (Volatility, Momentum, Benchmark, etc.) based on SHAP importance, optimizing for feature diversity. **It automatically excludes features with negative coefficients/importance to ensure stability.**
    *   **Supported Algorithms**:
        *   **Standard**: Linear Regression, RandomForest (Regressor/Classifier), Gradient Boosting (Regressor/Classifier).
        *   **Advanced**: XGBoost (Regressor/Classifier), LightGBM (Regressor/Classifier).
        *   **Regularized**: ElasticNet (combines L1/L2 penalties), Ridge, Lasso.
    *   **Batch Training (Grouped Models - Method B)**: 
        *   **UI Separation**: Interface clearly splits "Single Model" (Method A) and "Group Model" (Method B) workflows.
        *   **"Train Group"**: One-click orchestration to spawn 4 related models simultaneously sharing a `group_id`.
        *   **Config**: Trains [Open (1m), Close (1m), High (1d), Low (1d)] parallel jobs.
        *   **Purpose**: Rapidly build a complete predict set for a ticker.
    *   **ElasticNet Grid Search**:
        *   **Auto-Tuning**: When `elasticnet_regression` is selected without custom parameters, the system triggers a **Multi-threaded Grid Search** (`n_jobs=-1`).
        *   **Optimization**: It tests varying combinations of `alpha` (0.0001 to 1.0) and `l1_ratio` (0.1 to 0.99) to find the perfect balance between Lasso (feature selection) and Ridge (stability), preventing models where all features are zeroed out.
    *   **Stationarity & Target Selection**: 
        *   **Prediction Type**: System defaults to **Log Return** (`ln(Future/Current)`) or **Percent Change** to ensure the target variable is **stationary**. Predicting Raw Prices is flagged with a warning to prevent high-coefficient/low-value models.
        *   **Flexibility**: Users can still select "Raw Price" for specific research needs, but "Log Return" is enforced for Batch jobs.
        *   **Target Column**: Predict any column (Close, Open, High, etc.) `N` steps into the future.
    *   **Leakage Prevention**: 
        *   **Raw Feature Exclusion (Strict)**: The system strictly excludes raw OHLCV columns (`open`, `close`, `high`, `low`, `volume`, `vwap`) from the input feature set (X). Models used to learn from "Levels" (e.g. Price=200), which causes leakage. They are now forced to learn from stationary indicators.
            *   *Enhancement*: The target column is preserved in the dataframe index for validation but dropped from `X`, allowing for accurate Price RMSE calculation without leakage.
        *   **Aggressive Splitting**: When resampling (e.g., 1m to 1h), if a bucket contains *any* "Test" data, the entire bucket is labeled "Test" to ensure no future data leaks into the training set.
        *   **Boundary Protection**: System automatically identifies and drops rows at the Train->Test boundary where a training input's future label would be derived from the test set.
        *   **Timestamp Alignment**: Feature data is explicitly indexed by `ts` to ensure that validation logic matches predictions to the exact historical moment, essential for accurate price reconstruction.
    *   **Model Management**: 
        *   **Metrics & Reality Checks**:
            *   **Reconstructed Price RMSE ($)**: Models trained on Log Returns (Stationary) produce abstract error values (e.g. 0.0012). The system now automatically converts this back to Dollars ($) by applying the predicted return to the base price. This flags "bad" models that look good in abstract math but fail in real dollar terms.
        *   Dashboard to view metrics, feature importance (SHAP, Standardized Coefficients), and report pop-ups (now including Model Name).
        *   **SHAP Support**: Native support for TreeExplainers across XGBoost, LightGBM, and Random Forest, ensuring accurate feature contribution analysis even for gradient-boosted models.
        *   The Registry displays models in a **Tree Structure** to visualize lineage and batches.
        *   **Bulk Deletion**: Users can delete all models via a protected "Delete All" button (double confirmation required).
    *   **Global Data Options**: The UI scans the entire database to find all unique feature configurations (e.g., "Train:30 days, Test:5 days"). Once a selected, the list of available symbols is automatically filtered to match.
    *   **Debug & Observability**: Trainer logs detailed data shapes (`X.shape`, `y.shape`) and feature types to the terminal to aid in diagnosing data issues.
*   **Multi-Ticker / Context Awareness**:
    *   **Primary Ticker**: The target symbol you are trying to predict.
    *   **Context Tickers**: You can select up to 3 additional tickers (e.g., `VIX`, `SPY`, `QQQ`) to feed into the model as features.
    *   **Integration**: The system performs a strict Inner Join on the 1-minute timestamps.
    *   **Naming**: Context features are automatically suffixed (e.g., `close_VIX`, `rsi_14_SPY`) to distinguish them from the primary ticker's features.
    *   **Feature Selection**: Includes a **Top-Down Pruning Step** to automatically select the most significant features.
        *   **Standardization**: All features are standardized (Mean=0, Std=1) to allow for coefficient comparison.
        *   **P-Value Pruning**: Features with a P-value > 0.05 (configurable) are automatically dropped to remove statistical noise.
        *   **Ranking**: Logs the standardized Beta Coefficients of the top features, helping to identify which indicators (e.g., `EMA_9` vs `RSI_14`) are truly driving the model.
    *   **Conditional Preprocessing** (New):
        *   **Context**: Different algorithms need different input formats for Regime features.
        *   **Linear Models (ElasticNet)**: Applying One-Hot Encoding to categorical regimes (`regime_vix`, `regime_gmm`) and using continuous distance metrics (`regime_sma_dist`) for structural breaks.
        *   **Tree Models (XGBoost/RF)**: Keeping regimes as Ordinal Integers (1-4) for efficient tree splitting.
        *   **Heterogeneous Scaling**: The trainer now intelligently scales features based on their type:
            *   **RobustScaler**: For Volume/Count Data (Outlier Protection).
            *   **StandardScaler**: For Returns/Z-Scores (Zero-centering).
            *   **Passthrough**: For Bounded Oscillators (RSI, IBS) to preserve their natural 0-100 scale.
    *   **Interactions**:
        *   **Regime-Conditional Importance**: The trainer automatically partitions feature importance (SHAP) by Market Regime (if available).
        *   **Insight**: This allows you to see if a feature like `dist_vwap` is critical during "Mean Reverting" regimes but useless during "Trending" regimes.
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
    *   **Filtering Logic** (New):
        *   **Target Transform Awareness**: Now correctly handles models trained on "Log Returns" vs "Raw Prices". If your model predicts 0.002 (0.2%), the simulation knows this is a high-conviction value, rather than mistakenly comparing it to a $150 stock price.
        *   **Prediction Threshold**: "Don't trade unless conviction is high." A slider filters out weak predictions (e.g. only trade if pred > 0.005).
        *   **Outlier Filter (Z-Score)**: "Don't trust massive predictions." A checkbox filters out predictions that are statistical outliers (Z > 3), which often indicate data errors or model instability.
        *   **Volatility Normalization**: "Adapt to the market." Scales the prediction threshold dynamically based on recent Rolling Volatility.
    *   **Regime Gating (New)**:
        *   **Concept**: Solves the "0 Trades" problem or "Knife Catching" by only allowing trades in favorable market conditions.
        *   **Logic**: "IF Market Regime == 'Bull Quiet' THEN Allow Trades, ELSE Exit/Halt."
        *   **Controls**: Select a Regime Type (GMM, VIX, or SMA) and whitelist the allowed states (e.g., `0` for Calm GMM). The simulation will force a "Sell/Exit" signal whenever the market enters a forbidden regime.
4.  **Trading Execution (Walk-Forward Loop)**:
    *   The simulation iterates through the data chronologically, maintaining a cash and share balance.
    *   **Buying**:
        *   **Trigger**: A **Buy Signal** is received AND the portfolio currently holds **0 shares**.
        *   **Action**: Buys the maximum number of shares possible with available cash at the current close price.
    *   **Selling**:
        *   **Trigger**: A **Sell Signal** is received AND the portfolio currently holds **shares > 0**.
        *   **Action**: Sells **100%** of the held shares at the current close price.
    *   **Holding**: If the signal matches the current position (e.g., Buy Signal while holding), no action is taken.
5.  **Metrics & History**: 
    *   **History Tracking**: Every simulation run is now saved to the DuckDB `sim_runs` table.
        *   **Batch Run**: Execute simulations across all models for a ticker in one click to find the best performer.
        *   **Comparison**: Review past runs in the "History" tab, comparing sharpe ratios and equity curves.
    *   **Equity Curve**: Visualizes portfolio value over time vs Buy & Hold.
    *   **Directional Hit Rate**:
        *   Calculates the % of time the model correctly predicted the *direction* of the price movement (e.g., predicted Up and price went Up).
        *   **Rolling Average**: The chart displays a 100-period rolling average of the Hit Rate on a secondary Y-axis to visualize model stability over time.
        *   **Threshold**: A Hit Rate > 50% generally indicates predictive power better than random chance (ignoring transaction costs).
    *   **Trading Bot (Meta-Learner)**:
        *   **Purpose**: A secondary "Bot" model (Random Forest) trained to predict when the Base Model is correct vs incorrect.
        *   **Workflow**:
            1.  Train Base Model (e.g., Predict Price Direction).
            2.  Train Bot: Input = Market Features, Target = Did Base Model Hit? (1/0).
            3.  **Run Simulation with Filter**: The Bot acts as a gatekeeper. It approves a trade only if the Base Model signals AND the Bot is >55% confident the Base Model is correct.
        *   **Outcome**: Reduces false positives and improves Sharpe Ratio by staying out of uncertain trades.
    *   **Advanced Features**: The simulation now leverages VIXY-derived market context (Z-Scores, Relative Volume) and novel indicators (VWAP Distance, Intraday Intensity) to filter trades.

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

## Troubleshooting & Extension Pitfalls

### Adding New Features
When adding new indicators or columns to `feature_service/pipeline.py`, keep the following in mind:

1.  **Dependencies**:
    *   If your new feature requires a new library (e.g., `scikit-learn` for GMM), you MUST add it to `feature_service/requirements.txt`.
    *   **Crucial**: You must rebuild the Docker image for the changes to take effect. Run `docker-compose up --build -d feature_builder`.
    *   Example Error: `ModuleNotFoundError: No module named 'sklearn'`.

2.  **Schema Migration**:
    *   DuckDB schema changes (adding columns) can be brittle if not handled carefully during the `ensure_dest_schema` step.
    *   **Common Error**: `Catalog Error: Column with name ... already exists!`.
    *   **Fix**: Avoid generic `IF NOT EXISTS` or blind `ALTER TABLE`. Use `DESCRIBE feature_bars` to fetch the current schema and only `ALTER TABLE` for truly missing columns.

3.  **Service Reloading**:
    *   The `feature_builder` service runs `uvicorn` in production mode (without `--reload`) by default.
    *   Code changes in `pipeline.py` or `web.py` require a container restart: `docker-compose restart feature_builder`.
    *   Only rebuild (`--build`) if you modify `requirements.txt` or the `Dockerfile`.

4.  **Column Types**:
    *   Ensure new columns in the `CREATE TABLE` and `ALTER TABLE` statements match the data types in your Pandas DataFrame (usually `DOUBLE` or `INTEGER`).
    *   DuckDB is strict about types; mismatched types can lead to conversion errors during insertion.

5.  **Context Data Handling & Defaults**:
    *   When using cross-sectional features (like Beta or Sector Relative Strength) that rely on external context (e.g., QQQ data), you **MUST** handle the case where that context is missing (empty DB or missing symbols).
    *   **Fix**: Always initialize the dependent columns (e.g., `qqq_return`, `regime_sma_dist`) with default values (0.0) in the `else` block of your merge logic.
    *   **Reason**: If these columns are missing from the DataFrame, DuckDB insertion will fail with `Binder Error: Referenced update column ... not found`.

## Orchestrator Service - Recursive Strategy Factory

The Orchestrator Service automates the **Train → Prune → Simulate → Evolve** loop to discover optimal trading models without manual intervention.

### Quick Start
```bash
# Start orchestrator + dependencies
docker-compose up -d postgres orchestrator priority_worker

# Access dashboard
open http://localhost:8400
```

### Key Features
- **Automated Evolution**: Train model → Prune features (importance ≤ 0) → Simulate strategies → Repeat
- **Model Fingerprinting**: SHA-256 deduplication prevents retraining identical configurations
- **Priority Queue**: Higher parent SQN → higher child simulation priority
- **Holy Grail Criteria**: Configurable thresholds (SQN 3-5, Profit Factor 2-4, Trades 200-10K)
- **Lineage Tracking**: Full ancestry DAG for every promoted model
- **Distributed Workers**: Scale simulation execution across multiple nodes

### Architecture
- **Port**: 8400
- **Database**: PostgreSQL (shared state, priority queue, lineage)
- **Workers**: Pull from priority queue, run simulations, report results
- **Integration**: HTTP calls to Training (8200), Simulation (8300), Feature (8100) services

### API Endpoints
- `POST /evolve` - Start new evolution run
- `GET /runs` - List evolution runs
- `GET /promoted` - List models meeting Holy Grail criteria
- `POST /jobs/claim` - Worker job claiming (internal)

### Evolution Loop
1. **Train** model on engineered features
2. **Prune** features with importance ≤ 0
3. **Check** fingerprint - reuse existing model if configuration seen before
4. **Queue** simulations with priority = parent_sqn
5. **Evaluate** results against Holy Grail criteria
6. **Promote** if criteria met, else **Recurse** with pruned features

### Stopping Conditions
- ✅ Model promoted (meets Holy Grail criteria)
- ✅ Max generations reached
- ✅ No features left to prune
- ✅ All features would be pruned

See [orchestrator_service/README.md](orchestrator_service/README.md) for detailed documentation.

## Optimization Service (Grid Search)

The optimization service allows running automated grid searches across models, tickers, and parameters to find the best performing configurations. It uses a "Commander-Worker" architecture.

1.  **Start the Commander (C2)**:
    ```bash
    python optimization_service/main.py
    ```
    Open `http://localhost:8002` to view the dashboard and queue jobs.

2.  **Start Workers (Agents)**:
    Run as many agents as you have CPU cores. Each agent pulls jobs from the Commander.
    ```bash
    python optimization_service/worker.py
    ```

## Development

## Recent Updates (2025-01-XX)

### 🚀 New: Optimization Service - Strategy Heatmap Grid Search

A distributed grid search system for automated hyperparameter optimization across models, tickers, and trading parameters.

**Access:** `http://localhost:8002`

#### Features

1. **3D Grid Search Space**
   - **Axis X (Models):** Test multiple trained models simultaneously
   - **Axis Y (Regimes):** Filter trades by market conditions (VIX/GMM regimes)
   - **Axis Z (Thresholds):** Fine-tune prediction confidence levels (0.0001 - 0.002)

2. **Realistic Trading Simulation**
   - **Slippage Modeling:** 4-bar execution delay with midpoint fill pricing
   - **Transaction Costs:** Fixed $0.02 fee per trade (entry + exit)
   - **Regime Gating:** Automatically halt trading during unfavorable market conditions

3. **Command & Control Dashboard**
   - Real-time worker status monitoring
   - Live job queue tracking (Pending/Running/Completed/Failed)
   - Sortable leaderboard with expanded parameter columns
   - Auto-refreshing statistics every 2 seconds

#### Quick Start

```bash
# 1. Start Optimization Service
docker-compose up -d optimization

# 2. Access Dashboard
open http://localhost:8002

# 3. Launch Grid Search
# - Select models and tickers from dropdowns
# - Configure thresholds and regime filters
# - Click "Queue Heatmap Batch"

# 4. (Optional) Scale Workers for Parallel Processing
docker-compose exec optimization python optimization_service/worker.py
```

#### Grid Search Configuration

**Default Parameter Grid:**
```python
{
    "thresholds": [0.0001, 0.0005, 0.0010, 0.0020],
    "z_score_check": [True, False],
    "volatility_normalization": [False, True],
    "use_bot": [False, True],
    "regime_filters": {
        "regime_vix": [0, 1, 2, 3],  # Bear Vol, Bear Quiet, Bull Vol, Bull Quiet
        "regime_gmm": [0, 1, 2, 3]
    }
}
```

**Example:** 2 models × 3 tickers × 4 thresholds × 2 z-score × 2 vol_norm = **96 simulations**

#### Interpreting Results

The leaderboard displays **individual parameter columns** for easy pattern recognition:

| Rank | Ticker | Model | Return % | Hit Rate | Trades | Threshold | Z-Score | Vol Norm | Use Bot | Regime Col | Allowed Regimes |
|------|--------|-------|----------|----------|--------|-----------|---------|----------|---------|------------|-----------------|
| 🏆   | AAPL   | XGB...| 12.5%    | 58.2%    | 45     | 0.0005    | ✓       | ✗        | ✗       | regime_vix | [3]             |
| 🥈   | MSFT   | Elas..| 8.3%     | 53.1%    | 32     | 0.0015    | ✗       | ✓        | ✓       | None       | All             |

**Look for "Clusters of Success":**
- ✅ **High Hit Rate (>55%) + Moderate Trades (20-50)** = Sweet Spot
- ⚠️ **Low Hit Rate (<50%) + Many Trades (>100)** = Overfitting/Noise
- ❌ **High Return + Few Trades (<10)** = Lucky outlier, not robust

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Optimization C2 Server                     │
│  - Job Queue Management (DuckDB: optimization.db)            │
│  - Worker Heartbeat Tracking (30s timeout)                   │
│  - Leaderboard Aggregation & Ranking                         │
│  - REST API (/api/worker/claim, /api/worker/complete)        │
└─────────────┬───────────────────────────────────────────────┘
              │
       ┌──────┴──────┬──────────┬──────────┐
       ▼             ▼          ▼          ▼
   Worker 1      Worker 2   Worker 3   Worker N
   (Thread)      (Process)  (Container) (Remote)
       │             │          │          │
       └─────────────┴──────────┴──────────┘
                     │
              ┌──────▼──────┐
              │ Simulation  │ ← Uses simulation_service.core.run_simulation()
              │   Engine    │ ← Accesses models/ and features_parquet/
              └─────────────┘
```

**Worker Modes:**
1. **Internal Thread** (Default): Auto-starts with C2 server (single-node testing)
2. **External Process**: Manual launch for CPU parallelism
3. **Distributed Containers**: Scale across multiple machines (future)

#### Database Schema

```sql
-- optimization.db
CREATE TABLE jobs (
    id VARCHAR PRIMARY KEY,
    batch_id VARCHAR,
    status VARCHAR,  -- PENDING | RUNNING | COMPLETED | FAILED
    params JSON,     -- Simulation configuration
    result JSON,     -- Strategy metrics
    worker_id VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    progress DOUBLE  -- 0.0 to 1.0
);

CREATE TABLE workers (
    id VARCHAR PRIMARY KEY,
    last_heartbeat TIMESTAMP,
    current_job_id VARCHAR,
    status VARCHAR  -- ACTIVE | IDLE
);
```

### 🐳 Docker Optimization Improvements

**Image Size Reduction: 77% savings (5.4GB → 1.2GB)**

#### Multi-Stage Base Image Architecture

```dockerfile
# Shared base image (built once)
Dockerfile.base (950MB)
    ├── Python 3.10-slim
    ├── All dependencies (requirements.txt)
    └── Pre-compiled wheels

# Service images (inherit from base)
Dockerfile           (50MB) ← API
Dockerfile.feature   (50MB) ← Feature Builder
Dockerfile.training  (50MB) ← Training Service
Dockerfile.simulation(50MB) ← Simulation Service
Dockerfile.optimize  (50MB) ← Optimization C2
```

#### Build Process

```bash
# One-time setup
./build.sh

# Breakdown:
# 1. Build stock_base:latest (950MB) - Takes ~3 minutes
# 2. Build 5 service images   (250MB total) - Takes ~30 seconds
# 3. Total: 1.2GB vs previous 5.4GB
```

#### Benefits

- ✅ **Shared Dependencies:** Install once, use everywhere
- ✅ **Fast Rebuilds:** Only recompile changed services
- ✅ **Live Code Reload:** Source mounted as volume (`-v .:/app`)
- ✅ **Consistent Environment:** Same sklearn/numpy versions across all services

#### Volume Strategy

All services mount source code as volume (development mode):
```yaml
volumes:
  - ./data:/app/data  # Persistent data
  - .:/app            # Live code (no rebuild needed for changes)
```

**Hot Reload:** Python code changes reflect immediately without `docker-compose build`

### 📦 Updated Dependencies

**Added:**
- `jinja2>=3.1.0` - Template rendering for simulation UI
- `httpx>=0.24.0` - Async HTTP client for ingestion poller
- `yfinance>=0.2.0` - Yahoo Finance data source
- `ta>=0.11.0` - Technical analysis library (replaced unstable pandas-ta)
- `scikit-learn==1.8.0` - Pinned for model compatibility

**Removed:**
- `pandas-ta` (compatibility issues with Python 3.10)
- `polygon-api-client` (unused in current implementation)

### 🔧 Bug Fixes

1. **DuckDB Connection Conflicts**
   - Fixed read-only/write connection conflicts in `optimization_service/database.py`
   - Reuse single connection across nested function calls

2. **FastAPI Deprecation Warnings**
   - Migrated from `@app.on_event("startup")` to modern `lifespan` context manager
   - Future-proof for FastAPI 1.0+

3. **Sklearn Version Mismatch**
   - Pinned `scikit-learn==1.8.0` to match training environment
   - Eliminates `InconsistentVersionWarning` when loading models

4. **Slippage Calculation Edge Cases**
   - Handle NaN in `exec_price` at end of data (fallback to close)
   - Prevent division by zero in PnL percentage calculations

### 📊 Performance Metrics

**Grid Search Benchmark (96 jobs):**
- Single Worker (Internal Thread): ~12 minutes
- 4 Workers (Parallel Processes): ~3.5 minutes
- Expected scaling: Linear up to CPU core count

**Slippage Impact Analysis (4-bar delay):**
- Average fill price difference: 0.02% - 0.08% (realistic)
- Transaction cost drag: ~$0.04 per round-trip trade
- Net effect on returns: -2% to -5% vs instant fills

### 🚦 Known Limitations

1. **Worker Scalability:** Current implementation uses HTTP polling (2s interval). For >10 workers, consider websockets or message queue (Redis/RabbitMQ).

2. **Slippage Model:** Fixed 4-bar delay doesn't account for volatility-dependent slippage. Future enhancement: adaptive delay based on ATR.

3. **Transaction Fees:** $0.02 flat fee is simplified. Real costs include:
   - SEC fees (variable)
   - Exchange fees (tier-based)
   - Clearing fees
   - Slippage (market impact)

4. **Regime Detection Lag:** GMM/VIX regimes calculated on historical data. Real-time trading would need streaming regime classification.

### 📖 Related Documentation

- [Training Service](training_service/README.md) - Model training workflows
- [Simulation Service](simulation_service/README.md) - Backtesting engine
- [Feature Engineering](feature_service/README.md) - Technical indicator pipeline

### 🛠️ Troubleshooting

**"No active workers" warning:**
```bash
# Start manual worker
docker-compose exec optimization python optimization_service/worker.py
```

**Slow grid search:**
```bash
# Check worker logs
docker-compose logs -f optimization

# Scale workers (run in separate terminals)
for i in {1..4}; do
  docker-compose exec optimization python optimization_service/worker.py &
done
```

**Model unpickle errors:**
```bash
# Verify sklearn version matches training env
docker-compose exec optimization python -c "import sklearn; print(sklearn.__version__)"
# Should output: 1.8.0

# Rebuild if mismatch
docker build -t stock_base:latest -f Dockerfile.base .
docker-compose up -d --build optimization
```

---

## Contributing

When modifying the optimization service, follow these patterns:

1. **New Parameters:** Add to `optimization_service/main.py` grid config AND `simulation_service/core.py` function signature
2. **Database Changes:** Update both `optimization_service/database.py` schema AND add migration logic
3. **Worker Protocol:** Maintain backward compatibility with `/api/worker/claim` and `/api/worker/complete` endpoints

**Testing Checklist:**
- [ ] Grid search completes without errors
- [ ] Leaderboard displays all parameter columns
- [ ] Worker heartbeats appear in dashboard
- [ ] Simulation results match manual run
- [ ] Transaction fees correctly deducted from equity

## Orchestrator Service - Recursive Strategy Factory

The orchestrator service is an evolutionary ML system that automatically discovers, trains, and validates trading strategies through multi-generation optimization.

**Access:** `http://localhost:8003`

### 🧬 Core Concept: Evolutionary Model Development

Traditional ML workflow requires manual hyperparameter tuning and feature selection. The orchestrator automates this through a **recursive feedback loop**:

```
Generation 1: Train baseline models
    ↓ (simulate + rank by SQN/PF)
Generation 2: Train on best Gen1 features + new hyperparams
    ↓ (simulate + rank)
Generation 3: Train on best Gen2 features + new hyperparams
    ↓ (converge to Holy Grail criteria)
Promoted Model: Export winner to production
```

### 🎯 Features

#### 1. Multi-Generation Evolution Runs

- **Seed Options:**
  - **From Scratch:** Auto-fetch features from feature_service, start with empty lineage
  - **From Parent Model:** Inherit feature subset from a promoted model

- **Per-Generation Workflow:**
  1. **Training Phase:** Spawns `N` parallel training jobs with mutated hyperparameters
  2. **Simulation Phase:** Tests each model across grid of thresholds/regime filters
  3. **Selection Phase:** Ranks by Holy Grail criteria (SQN, Profit Factor, Trade Count)
  4. **Inheritance Phase:** Best model's features → seed for next generation

- **Automatic Termination:** Stops early if Holy Grail criteria met before `max_generations`

#### 2. Data Fold Workflow (Critical for Train/Test Integrity)

**UI Flow: Options → Symbols → Features**

**Step 1: Load Data Folds**
```
📁 Available Folds (from feature_service):
  ○ {"train_days":30, "test_days":5, "regime_col":"regime_vix"}
  ○ {"train_days":60, "test_days":10, "regime_col":"regime_gmm"}
  ○ Legacy / No Config
```

**Step 2: Select Fold → Auto-load Symbols**
```
Fold Selected: {"train_days":30, "test_days":5}
✅ 8 symbols available: AAPL, GOOGL, MSFT, NVDA, QQQ, RDDT, VIXY, XOM
```

**Step 3: Configure Evolution**
- **Target Symbol:** AAPL (what you're predicting)
- **Reference Symbols:** ☑ SPY ☑ QQQ (relational context features)
- **Data Options:** Passed as JSON to training_service (fold configuration)

**Why This Matters:**
- Prevents using folds/symbols that don't have computed features
- Ensures train/test splits are consistent across all training jobs
- Enables multi-ticker TS-aligned training (merge on timestamp index)

#### 3. Multi-Ticker TS-Aligned Training

**Problem:** Predicting AAPL in isolation ignores market-wide trends (SPY, QQQ movements)

**Solution:** Load multiple tickers, align on timestamp, merge features

**Implementation:**
```python
# Orchestrator sends:
{
  "symbol": "AAPL",
  "reference_symbols": ["SPY", "QQQ"],
  "data_options": "{\"train_days\":30}"
}

# Training service loads:
df_aapl = load_features("AAPL")  # columns: ts, rsi, macd, ...
df_spy = load_features("SPY")    # columns: ts, rsi, macd, ...
df_qqq = load_features("QQQ")    # columns: ts, rsi, macd, ...

# Rename to avoid collisions:
df_spy.columns = ["ts", "rsi_SPY", "macd_SPY", ...]
df_qqq.columns = ["ts", "rsi_QQQ", "macd_QQQ", ...]

# Inner join on ts (drops misaligned rows):
df_merged = df_aapl.merge(df_spy, on="ts").merge(df_qqq, on="ts")

# Result: Features from 3 tickers, TS-aligned
# X: [rsi_AAPL, macd_AAPL, rsi_SPY, rsi_QQQ, ...]
# y: AAPL_close (future target)
```

**Benefits:**
- Captures regime-dependent behavior (AAPL behaves differently when SPY is rallying)
- Improves feature selection (model learns which cross-ticker signals matter)
- Prevents data leakage (inner join ensures exact timestamp matches)

#### 4. Holy Grail Criteria Filtering

**Purpose:** Eliminate overfit/lucky models, promote only robust strategies

**Default Thresholds:**
```python
{
  "sqn_min": 3.0,      # System Quality Number (Van Tharp metric)
  "sqn_max": 10.0,     # Upper bound (too high = overfitting)
  "profit_factor_min": 1.5,
  "profit_factor_max": 5.0,
  "trade_count_min": 20,
  "trade_count_max": 200
}
```

**Validation Logic:**
```python
def is_holy_grail(sim_result):
    return (
        sqn_min <= sim_result['sqn'] <= sqn_max and
        pf_min <= sim_result['profit_factor'] <= pf_max and
        tc_min <= sim_result['trade_count'] <= tc_max
    )
```

**Outcome:**
- ✅ **Promoted:** Model meets all criteria → saved to `promoted_models` table
- ⏭️ **Next Gen:** Best model (even if not Holy Grail) seeds next generation
- 🛑 **Early Stop:** If Holy Grail found, skip remaining generations

#### 5. Simulation Grid Search (Per Model)

For each trained model, orchestrator runs simulations across:

**Parameters:**
- **Thresholds:** `[0.0001, 0.0003, 0.0005, 0.0007]` (prediction confidence levels)
- **Regime Configs:** 
  ```python
  [
    {"regime_gmm": [0]},      # Only trade in calm GMM cluster
    {"regime_gmm": [1]},      # Only trade in volatile GMM cluster
    {"regime_vix": [0, 1]}    # Trade in bear/quiet VIX regimes
  ]
  ```

**Example Scale:**
- 1 model × 4 thresholds × 3 regime configs = **12 simulations**
- 10 models/generation × 12 sims = **120 simulations/generation**
- 4 generations × 120 sims = **480 total simulations/run**

#### 6. Web Dashboard Features

**Modern Architecture (Separated HTML/CSS/JS):**
```
orchestrator_service/
  ├── templates/dashboard_new.html    (Clean HTML structure)
  ├── static/css/dashboard.css        (280 lines, all styling)
  └── static/js/dashboard.js          (664 lines, all logic)
```

**Benefits:**
- Easier maintenance (edit CSS without touching Python)
- Browser caching (CSS/JS served as static files)
- Cleaner git diffs (changes isolated to relevant files)

**Dashboard Sections:**

**1. Evolution Setup**
- Load data folds → Select fold → Choose symbols
- Configure algorithm, target column, max generations
- Select reference symbols (multi-ticker training)
- Set Holy Grail thresholds

**2. Active Runs**
- Real-time status (PENDING / RUNNING / COMPLETED)
- Current generation progress
- Live job counts (training/simulation)
- Expandable details (per-generation results)

**3. Promoted Models**
- Filter by Holy Grail criteria met
- View full model config (features, hyperparams, lineage)
- Download model artifacts
- See simulation performance that earned promotion

**4. System Stats**
- Total runs / active runs / completed runs
- Job queue depth (pending / running)
- Promoted model count

#### 7. API Endpoints

```http
POST /api/evolve
  Body: {
    "symbol": "AAPL",
    "reference_symbols": ["SPY", "QQQ"],
    "data_options": "{\"train_days\":30}",
    "algorithm": "XGBoost",
    "max_generations": 4,
    "thresholds": [0.0001, 0.0005],
    "sqn_min": 3.0, ...
  }
  Response: {"run_id": "abc-123", "status": "queued"}

GET /api/runs
  Response: {
    "runs": [
      {"run_id": "abc-123", "symbol": "AAPL", "status": "running", "generation": 2, ...}
    ],
    "count": 1
  }

GET /api/runs/{run_id}
  Response: {
    "run_id": "abc-123",
    "symbol": "AAPL",
    "current_generation": 2,
    "max_generations": 4,
    "lineage": ["model-gen1-id", "model-gen2-id"],
    "jobs": [...],
    "promoted_models": [...]
  }

GET /api/runs/{run_id}/generations
  Response: {
    "generations": [
      {
        "generation": 1,
        "models_trained": 10,
        "simulations_run": 120,
        "best_model": {"sqn": 2.8, "pf": 1.9, ...},
        "top_3": [...]
      },
      ...
    ]
  }

GET /api/promoted
  Response: {
    "models": [
      {
        "model_id": "xyz-789",
        "run_id": "abc-123",
        "symbol": "AAPL",
        "generation": 3,
        "sqn": 3.2,
        "profit_factor": 2.1,
        "feature_count": 45,
        "lineage_depth": 2,
        ...
      }
    ],
    "count": 1
  }

GET /api/promoted/{model_id}
  Response: {
    "model_id": "xyz-789",
    "features": ["rsi_14", "macd", "rsi_SPY", ...],
    "hyperparameters": {"n_estimators": 200, "max_depth": 6, ...},
    "lineage": ["parent-id", "grandparent-id"],
    "simulation_config": {"threshold": 0.0005, "regime": "gmm_0"},
    "performance": {"sqn": 3.2, "sharpe": 1.8, ...}
  }

GET /api/features/options
  Response: [
    "{\"train_days\":30, \"test_days\":5}",
    "Legacy / No Config"
  ]

GET /api/features/symbols?options={...}
  Response: ["AAPL", "GOOGL", "MSFT", ...]

GET /api/features/columns?symbol=AAPL
  Response: {
    "columns": ["ts", "close", "rsi_14", "macd", ...],
    "sample_data": [...]
  }
```

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│             Orchestrator Service (FastAPI)              │
│  - Evolution logic (evolution.py)                       │
│  - Job queue management (database.py)                   │
│  - Dashboard UI (templates/static)                      │
│  - Feature service integration                          │
└────────────┬────────────────────────────────────────────┘
             │
      ┌──────┴──────┬───────────┬───────────────┐
      ▼             ▼           ▼               ▼
 Training API  Simulation  Feature API    Job Workers
 (port 8001)   (port 8004) (port 8100)    (async tasks)
      │             │           │               │
      └─────────────┴───────────┴───────────────┘
                    │
            ┌───────▼───────┐
            │  DuckDB Jobs  │
            │  orchestrator.db
            │   - runs
            │   - jobs
            │   - promoted_models
            └───────────────┘
```

### 🔄 Evolution Lifecycle Example

**Run Configuration:**
```json
{
  "symbol": "AAPL",
  "reference_symbols": ["SPY", "QQQ"],
  "data_options": "{\"train_days\":30, \"test_days\":5}",
  "algorithm": "xgboost_regressor",
  "max_generations": 3,
  "thresholds": [0.0001, 0.0005],
  "regime_configs": [{"regime_gmm": [0]}, {"regime_gmm": [1]}]
}
```

**Generation 1:**
1. Auto-fetch 50 features from feature_service (exclude OHLCV)
2. Train 5 XGBoost models with random hyperparams
3. Run 5 models × 2 thresholds × 2 regimes = **20 simulations**
4. Best model: SQN=2.5 (below 3.0 threshold)
5. Extract 40 features used by best model → seed for Gen 2

**Generation 2:**
1. Inherit 40 features from Gen 1 best model
2. Train 5 XGBoost models (mutate learning_rate, max_depth)
3. Run **20 simulations**
4. Best model: SQN=3.1, PF=2.0, Trades=45 ✅ **Holy Grail!**
5. **Promote model** → `promoted_models` table
6. Early termination (Holy Grail reached)

**Result:**
- Total jobs: 10 training + 40 simulation = 50 jobs
- Best model found: Generation 2, Model 3
- Features evolved: 50 → 40 (10 pruned by selection)
- Ready for production deployment

### 🧪 Usage Examples

**Start Evolution (API):**
```bash
curl -X POST http://localhost:8003/api/evolve \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "reference_symbols": ["SPY", "QQQ"],
    "algorithm": "xgboost_regressor",
    "max_generations": 4,
    "sqn_min": 3.0,
    "profit_factor_min": 1.5
  }'
```

**Check Run Status:**
```bash
curl http://localhost:8003/api/runs/{run_id}
```

**List Promoted Models:**
```bash
curl http://localhost:8003/api/promoted
```

**Web UI Workflow:**
1. Navigate to `http://localhost:8003`
2. Click "Load Data Folds" → Select fold
3. Select target symbol: AAPL
4. Check reference symbols: ☑ SPY ☑ QQQ
5. Configure Holy Grail thresholds
6. Click "Start Evolution"
7. Monitor Active Runs section (auto-refresh 10s)
8. View promoted models in Promoted Models section

### 🎛️ Hyperparameter Mutation Strategy

Each generation mutates hyperparameters to explore optimization space:

**XGBoost Mutations:**
```python
{
  "n_estimators": random.choice([100, 200, 300, 500]),
  "max_depth": random.choice([3, 4, 5, 6, 8]),
  "learning_rate": random.choice([0.01, 0.05, 0.1, 0.2]),
  "subsample": random.uniform(0.7, 1.0),
  "colsample_bytree": random.uniform(0.7, 1.0),
  "reg_alpha": random.choice([0, 0.1, 1, 10]),
  "reg_lambda": random.choice([1, 10, 50, 100])
}
```

**ElasticNet Grid Search (Automatic):**
```python
{
  "alpha": [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],  # 6 values
  "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]  # 7 values
}
# 6 × 7 × 5 CV folds = 210 models tested per job
```

### 📊 Database Schema

```sql
-- orchestrator.db

CREATE TABLE runs (
    id VARCHAR PRIMARY KEY,
    symbol VARCHAR,
    algorithm VARCHAR,
    max_generations INTEGER,
    current_generation INTEGER,
    status VARCHAR,  -- PENDING | RUNNING | COMPLETED | FAILED
    config JSON,     -- Full evolution config
    lineage JSON,    -- ["gen1_model_id", "gen2_model_id", ...]
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE jobs (
    id VARCHAR PRIMARY KEY,
    run_id VARCHAR,
    generation INTEGER,
    job_type VARCHAR,  -- TRAINING | SIMULATION
    status VARCHAR,
    params JSON,
    result JSON,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE promoted_models (
    id VARCHAR PRIMARY KEY,
    run_id VARCHAR,
    model_id VARCHAR,      -- training_service model ID
    generation INTEGER,
    symbol VARCHAR,
    sqn DOUBLE,
    profit_factor DOUBLE,
    sharpe_ratio DOUBLE,
    trade_count INTEGER,
    total_return DOUBLE,
    features JSON,         -- ["rsi_14", "macd", ...]
    hyperparameters JSON,
    simulation_config JSON,
    promoted_at TIMESTAMP
);
```

### 🚨 Known Limitations & Future Enhancements

**Current:**
- Single-objective optimization (SQN primary, PF secondary)
- Fixed mutation strategy (random sampling)
- No genetic crossover (features don't blend between models)
- Manual Holy Grail threshold tuning

**Planned:**
- **Multi-objective Pareto optimization** (SQN vs Sharpe vs Drawdown)
- **Bayesian hyperparameter optimization** (smarter than random)
- **Genetic operators:** Crossover (blend top 2 models' features), mutation (add/remove features)
- **Adaptive thresholds:** Learn Holy Grail criteria from historical promoted models
- **Ensemble promotion:** Auto-create voting ensembles from top 3 models
- **Live trading integration:** Direct deployment to paper trading accounts

### 🔗 Service Integration

**Orchestrator depends on:**
- **Feature Service (8100):** Fetches available folds, symbols, feature columns
- **Training Service (8001):** Spawns model training jobs
- **Simulation Service (8004):** Runs backtests on trained models

**Flow:**
```
User → Orchestrator UI
  ↓
Orchestrator → Feature Service (/options, /symbols)
  ↓
Orchestrator → Training Service (/train)
  ↓
Training Service → Feature Service (loads data)
  ↓
Orchestrator → Simulation Service (/run)
  ↓
Simulation Service → Training Service (loads model)
  ↓
Orchestrator → Database (save results)
  ↓
User ← Dashboard (view promoted models)
```

### 🛠️ Troubleshooting

**"No symbols found in feature service"**
- **Cause:** Trying to load symbols before selecting a data fold
- **Fix:** Click "Load Data Folds" first, then select a fold

**"Must provide either seed_model_id or seed_features"**
- **Cause:** Feature service integration disabled or unreachable
- **Fix:** Check `FEATURE_URL` env var, ensure feature_builder service running

**Evolution stuck in "RUNNING" status**
- **Cause:** Training or simulation jobs failed silently
- **Fix:** Check logs: `docker-compose logs training_service simulation`

**High memory usage during evolution**
- **Cause:** Loading full feature datasets for multi-ticker training
- **Fix:** Reduce `max_generations` or limit `reference_symbols` to 1-2 tickers

**Promoted models not appearing**
- **Cause:** No models meeting Holy Grail criteria
- **Fix:** Lower `sqn_min` / `profit_factor_min` thresholds or increase `max_generations`

---

## Feature Engineering

### Technical Indicators
- **Trend**: SMA(20), EMA(12), EMA(26), MACD
- **Momentum**: RSI(14), Stochastic Oscillator
- **Volatility**: Bollinger Bands, ATR(14)
- **Volume**: On-Balance Volume (OBV)

### Time Features (Cyclical Encoding)

**Problem**: Linear models treat time as a linear value, where 23:59 is numerically far from 00:01, even though they're only 2 minutes apart in real time.

**Solution**: Sin/Cos encoding creates circular features that properly represent the cyclical nature of time.

**Implementation**:
```python
# Convert time to minutes since midnight (0-1439)
minutes_of_day = hour * 60 + minute

# Encode as circular features
time_sin = sin(2π × minutes_of_day / 1440)
time_cos = cos(2π × minutes_of_day / 1440)
```

**Features Generated**:
- `time_sin`, `time_cos` - Minutes of day (0-1439) encoded circularly
- `day_of_week_sin`, `day_of_week_cos` - Day of week (Mon=0, Sun=6) encoded circularly

**Why This Matters**:
1. **Market Open/Close Proximity**: Pre-market (9:29) and after-hours (16:01) are adjacent in time but far apart numerically
2. **Intraday Patterns**: Morning volatility (9:30-10:00) vs lunch lull (12:00-13:00) vs power hour (15:00-16:00)
3. **Week Cycles**: Friday close behavior vs Monday open behavior are adjacent in the weekly cycle

**Visualization**:
```
Traditional Time (Linear):
0:00 -------- 12:00 -------- 23:59
[Far]                        [Far from 0:00]

Circular Time (Sin/Cos):
     12:00 (π, 0°)
         |
9:00 ---|--- 15:00
    \   |   /
     \  |  /
      \ | /
   0:00 (0, 360°) ≈ 23:59
```
