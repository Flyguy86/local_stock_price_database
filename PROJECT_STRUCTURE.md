# Project Architecture & System Breakdown

This document provides a hierarchical explanation of the `local_stock_price_database` project, moving from high-level concepts to specific file responsibilities.

---

## 1. High-Level Explanation (The Concept)
**"The Automated Quant Factory"**

Think of this project as a manufacturing pipeline for trading strategies:

1.  **Raw Materials (Ingestion):** We fetch raw stock market data (prices, volume) from external providers (Alpaca/IEX) and store it in a raw formats.
2.  **Refinement (Feature Engineering):** We take that raw data and calculate mathematical indicators (RSI, SMA, MACD, etc.). This "refined" data is optimized for speed.
3.  **Product Design (Training):** We use Machine Learning (Random Forest, XGBoost, etc.) to look at the refined data and learn patterns to predict future price movements.
4.  **Quality Assurance (Simulation):** We take the trained models and run them through historical simulations ("Backtesting") to see if they would have made money.
5.  **Mass Production/Optimization (Grid Search):** A command center that automates the running of thousands of variations of these models to find the absolute best configurations.

---

## 2. Medium-Level Explanation (The Architecture)

The system is built as a **Microservices Architecture** running inside Docker containers. Each service handles one specific stage of the pipeline using a shared filesystem for data.

### The 5 Core Services (Docker Containers)

1.  **API / Ingestion (`api`)**
    *   **Role:** The Gateway. Connects to Alpaca/IEX APIs.
    *   **Action:** Runs background pollers to fetch OHLCV (Open, High, Low, Close, Volume) bars.
    *   **Output:** Stores data in DuckDB (`local.db`).

2.  **Feature Builder (`feature_builder`)**
    *   **Role:** The Calculator.
    *   **Action:** Reads raw data, applies technical analysis libraries (Pandas-TA, Talib), and calculates "Regimes" (market conditions).
    *   **Output:** Writes highly optimized Parquet files (`/data/features_parquet`) partitioned by symbol and date.

3.  **Training Service (`training_service`)**
    *   **Role:** The Scientist.
    *   **Action:** Loads Parquet data, splits it into Train/Test sets, and trains Scikit-Learn/XGBoost models.
    *   **Output:** Saves serializes model files (`.joblib`) and records metadata in `models.db`.

4.  **Simulation Service (`simulation_service`)**
    *   **Role:** The Tester.
    *   **Action:** Loads a saved Model and historical data. It walks through time, generating buy/sell signals based on the model, and calculates PnL (Profit and Loss).
    *   **Output:** Generates equity curves and trade logs.

5.  **Optimization Service (`optimization`)**
    *   **Role:** The Commander.
    *   **Action:** A "Command & Control" (C2) server that issues orders to "Worker" agents to run thousands of training/simulation jobs with different settings to find the best strategies.

---

## 3. Specific & Detailed Explanation (File by File)

### Root Directory
*   `docker-compose.yml`: The orchestrator. Defines how all 5 services launch, how they talk to each other, and maps the `/data` folder so they all see the same files.
*   `Dockerfile.*`: Individual build instructions for each service (installing specific Python dependencies).
*   `requirements.txt`: The master list of Python libraries used (Safe version pinning).

### Folder: `app/` (Ingestion Service)
*   **`app/api/main.py`**: The entry point for the Ingestion API. Explicitly exposes endpoints to trigger data downloads manually.
*   **`app/ingestion/poller.py`**: The "Heartbeat". A loop that runs endlessly, checking if market hours are open, and triggering data fetches.
*   **`app/ingestion/alpaca_client.py`**: The driver that actually talks to Alpaca. It handles authentication and JSON parsing.
*   **`app/storage/duckdb_client.py`**: A wrapper around DuckDB. It handles thread-safety (using read-only connections where possible) and executing SQL queries.

### Folder: `feature_service/` (Feature Engineering)
*   **`feature_service/main.py` & `web.py`**: The UI for the Feature Builder. Allows you to select tickers and click "Generate".
*   **`feature_service/pipeline.py`**: The heavy lifter. It:
    1.  Reads from DuckDB.
    2.  Cleans data (deduplication).
    3.  Calls the calculator functions.
    4.  Writes to Parquet files (partitioned directory structure).
*   **`feature_service/features/`**: Contains the actual math logic (e.g., `generators.py` for standard indicators, `regime.py` for GMM/Volatility clusters).

### Folder: `training_service/` (Machine Learning)
*   **`training_service/main.py`**: The API/UI server. Displays available features and models.
*   **`training_service/trainer.py`**: The Core ML Logic.
    *   `train_model_task`: The master function.
    *   Handles "Pruning" (removing useless features).
    *   Handles "cv" (Cross Validation).
    *   Wraps models in Pipelines (Scalers -> Model).
    *   Using `joblib` to save the final brain to disk.
*   **`training_service/data.py`**: Specialized loader that reads the Parquet feature files efficiently using DuckDB queries (`SELECT * FROM '.../*.parquet'`).

### Folder: `simulation_service/` (Backtesting)
*   **`simulation_service/main.py`**: The Dashboard. Shows charts of your equity curve.
*   **`simulation_service/core.py`**: The Simulation Engine.
    *   `run_simulation`: Loops through historical rows.
    *   `_prepare_simulation_inputs`: Aligns Model expectations with Data reality.
    *   `save_simulation_history`: Logs results to DuckDB for the leaderboard.
    *   **Key Logic:** Implements the "Bot" logic (checking if a secondary model confirms the trade) and "Regime Filter" (checking if we are allowed to trade in this volatility).

### Folder: `optimization_service/` (Grid Search)
*   **`optimization_service/main.py`**: The Server.
    *   Hosts a queue of jobs (Pending/Running/Completed).
    *   Provides a dashboard to see which parameters are winning.
*   **`optimization_service/worker.py`**: The Agent.
    *   A headless script that wakes up, asks the Server "Do you have work?", runs a Simulation using `simulation_service.core`, reports the result, and sleeps.
*   **`optimization_service/database.py`**: A dedicated DuckDB interface for tracking the thousands of job results.

### Folder: `data/` (The Shared Brain)
*   **`duckdb/local.db`**: Raw price data.
*   **`duckdb/features.db`**: Metadata about features.
*   **`features_parquet/`**: The massive, optimized dataset used for training.
*   **`models/`**: The saved `.joblib` AI brains.

---
