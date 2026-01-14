-- Migration: Add progress tracking columns to evolution_runs

-- Add models_trained column
DO $$ BEGIN
    ALTER TABLE evolution_runs ADD COLUMN IF NOT EXISTS models_trained INTEGER DEFAULT 0;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;

-- Add models_total column
DO $$ BEGIN
    ALTER TABLE evolution_runs ADD COLUMN IF NOT EXISTS models_total INTEGER DEFAULT 0;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;

-- Add simulations_completed column
DO $$ BEGIN
    ALTER TABLE evolution_runs ADD COLUMN IF NOT EXISTS simulations_completed INTEGER DEFAULT 0;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;

-- Add simulations_total column
DO $$ BEGIN
    ALTER TABLE evolution_runs ADD COLUMN IF NOT EXISTS simulations_total INTEGER DEFAULT 0;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
