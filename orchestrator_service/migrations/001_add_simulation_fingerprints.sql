-- Migration: Add simulation_fingerprints table
-- This table tracks simulation results by fingerprint to avoid duplicate simulations

CREATE TABLE IF NOT EXISTS simulation_fingerprints (
    fingerprint VARCHAR(64) PRIMARY KEY,         -- SHA-256 hash of simulation config
    model_fingerprint VARCHAR(64) NOT NULL,      -- SHA-256 hash of model config (links to model_fingerprints.fingerprint)
    model_id VARCHAR(64),                        -- Optional: actual model_id used (for tracking)
    target_ticker VARCHAR(16) NOT NULL,          -- Symbol model was trained on
    simulation_ticker VARCHAR(16) NOT NULL,      -- Symbol being simulated
    threshold DOUBLE PRECISION NOT NULL,         -- Trading signal threshold
    z_score_threshold DOUBLE PRECISION NOT NULL, -- Outlier filter cutoff
    regime_config JSONB,                         -- Market regime filter
    train_window INTEGER NOT NULL,               -- Training fold size
    test_window INTEGER NOT NULL,                -- Test fold size
    result_sqn DOUBLE PRECISION,                 -- SQN from simulation
    result_profit_factor DOUBLE PRECISION,       -- Profit factor
    result_total_trades INTEGER,                 -- Trade count
    full_result JSONB,                           -- Complete simulation output
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sim_fingerprints_model_fp ON simulation_fingerprints(model_fingerprint);
CREATE INDEX IF NOT EXISTS idx_sim_fingerprints_model_id ON simulation_fingerprints(model_id);
CREATE INDEX IF NOT EXISTS idx_sim_fingerprints_ticker ON simulation_fingerprints(target_ticker, simulation_ticker);
CREATE INDEX IF NOT EXISTS idx_sim_fingerprints_sqn ON simulation_fingerprints(result_sqn DESC);
