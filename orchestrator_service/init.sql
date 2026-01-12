-- Orchestrator Service PostgreSQL Schema
-- Run via docker-entrypoint-initdb.d or manually

-- ============================================
-- Model Fingerprints (Deduplication)
-- ============================================
CREATE TABLE IF NOT EXISTS model_fingerprints (
    fingerprint VARCHAR(64) PRIMARY KEY,       -- SHA-256 hash
    model_id VARCHAR(64) NOT NULL,              -- References training_service models
    features_json JSONB NOT NULL,               -- Sorted feature list
    hyperparameters_json JSONB NOT NULL,
    target_transform VARCHAR(32),
    symbol VARCHAR(16) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fingerprints_model_id ON model_fingerprints(model_id);
CREATE INDEX IF NOT EXISTS idx_fingerprints_symbol ON model_fingerprints(symbol);

-- ============================================
-- Evolution Lineage (DAG Tracking)
-- ============================================
CREATE TABLE IF NOT EXISTS evolution_log (
    id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,                -- Groups a full evolution run
    parent_model_id VARCHAR(64),
    child_model_id VARCHAR(64) NOT NULL,
    generation INTEGER NOT NULL,                -- 0 = seed, 1 = first prune, etc.
    parent_sqn DOUBLE PRECISION,                -- For priority calculation
    pruned_features JSONB,                      -- Array of removed features
    remaining_features JSONB,                   -- Features kept after pruning
    pruning_reason VARCHAR(64),                 -- "importance_zero", "permutation_negative"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_evolution_run_id ON evolution_log(run_id);
CREATE INDEX IF NOT EXISTS idx_evolution_parent ON evolution_log(parent_model_id);
CREATE INDEX IF NOT EXISTS idx_evolution_child ON evolution_log(child_model_id);
CREATE INDEX IF NOT EXISTS idx_evolution_generation ON evolution_log(generation);

-- ============================================
-- Evolution Runs (Top-level tracking)
-- ============================================
CREATE TABLE IF NOT EXISTS evolution_runs (
    id VARCHAR(64) PRIMARY KEY,
    seed_model_id VARCHAR(64),
    symbol VARCHAR(16) NOT NULL,
    max_generations INTEGER NOT NULL DEFAULT 4,
    current_generation INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(16) NOT NULL DEFAULT 'PENDING',  -- PENDING/RUNNING/COMPLETED/STOPPED/FAILED
    config JSONB NOT NULL,                          -- Full EvolutionConfig
    best_sqn DOUBLE PRECISION,
    best_model_id VARCHAR(64),
    promoted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON evolution_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_symbol ON evolution_runs(symbol);

-- ============================================
-- Priority Job Queue
-- ============================================
CREATE TABLE IF NOT EXISTS priority_jobs (
    id VARCHAR(64) PRIMARY KEY,
    batch_id VARCHAR(64),
    run_id VARCHAR(64) NOT NULL,                -- Links to evolution run
    model_id VARCHAR(64) NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
    parent_sqn DOUBLE PRECISION DEFAULT 0,      -- Higher = higher priority
    status VARCHAR(16) DEFAULT 'PENDING',       -- PENDING/RUNNING/COMPLETED/FAILED
    params JSONB NOT NULL,
    result JSONB,
    worker_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_priority_jobs_status ON priority_jobs(status);
CREATE INDEX IF NOT EXISTS idx_priority_jobs_run_id ON priority_jobs(run_id);
CREATE INDEX IF NOT EXISTS idx_priority_jobs_model ON priority_jobs(model_id);
-- Priority index: higher parent_sqn first, then older jobs first
CREATE INDEX IF NOT EXISTS idx_priority_jobs_priority ON priority_jobs(parent_sqn DESC, created_at ASC);

-- Function to claim highest-priority job atomically
CREATE OR REPLACE FUNCTION claim_priority_job(p_worker_id VARCHAR)
RETURNS TABLE(job_id VARCHAR, job_params JSONB, job_model_id VARCHAR, job_run_id VARCHAR) AS $$
DECLARE
    claimed_id VARCHAR;
    claimed_params JSONB;
    claimed_model_id VARCHAR;
    claimed_run_id VARCHAR;
BEGIN
    UPDATE priority_jobs
    SET status = 'RUNNING',
        worker_id = p_worker_id,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = (
        SELECT id FROM priority_jobs
        WHERE status = 'PENDING'
        ORDER BY parent_sqn DESC, created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    RETURNING id, params, model_id, run_id 
    INTO claimed_id, claimed_params, claimed_model_id, claimed_run_id;
    
    RETURN QUERY SELECT claimed_id, claimed_params, claimed_model_id, claimed_run_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- Promoted Models (Holy Grail successes)
-- ============================================
CREATE TABLE IF NOT EXISTS promoted_models (
    id VARCHAR(64) PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL,
    run_id VARCHAR(64) NOT NULL,
    job_id VARCHAR(64),
    generation INTEGER NOT NULL,
    sqn DOUBLE PRECISION NOT NULL,
    profit_factor DOUBLE PRECISION NOT NULL,
    trade_count INTEGER NOT NULL,
    weekly_consistency DOUBLE PRECISION,        -- StdDev/Mean ratio
    ticker VARCHAR(16),
    regime_config JSONB,
    threshold DOUBLE PRECISION,
    full_result JSONB,                          -- Store complete simulation result
    promoted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_promoted_sqn ON promoted_models(sqn DESC);
CREATE INDEX IF NOT EXISTS idx_promoted_run_id ON promoted_models(run_id);
CREATE INDEX IF NOT EXISTS idx_promoted_model ON promoted_models(model_id);

-- ============================================
-- Worker Registry
-- ============================================
CREATE TABLE IF NOT EXISTS workers (
    id VARCHAR(64) PRIMARY KEY,
    last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    current_job_id VARCHAR(64),
    status VARCHAR(16) DEFAULT 'IDLE',          -- IDLE/BUSY/OFFLINE
    jobs_completed INTEGER DEFAULT 0,
    jobs_failed INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status);
