#!/bin/bash
# Apply database migration for simulation_fingerprints table

echo "Applying migration: 001_add_simulation_fingerprints.sql"
docker-compose exec -T postgres psql -U postgres -d orchestrator -f /docker-entrypoint-initdb.d/migrations/001_add_simulation_fingerprints.sql

echo "Migration complete. Checking table exists..."
docker-compose exec -T postgres psql -U postgres -d orchestrator -c "\d simulation_fingerprints"
