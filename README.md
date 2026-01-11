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

## Web/API Layer
- REST endpoints: queue ingest jobs, check agent status/heartbeats, list symbols, fetch bars/features, trigger feature rebuilds.
- Dashboard (Streamlit or React+FastAPI): filters, charts, feature previews; show per-agent logs/status/start/stop controls; progress bars.
- **Feature Builder Viewer**: Inspect generated features with ticker filtering.

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
