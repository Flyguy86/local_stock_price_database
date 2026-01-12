"""
PostgreSQL Database Connection Pool for Orchestrator Service.
Uses asyncpg for async operations.
"""
import os
import asyncio
import asyncpg
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

log = logging.getLogger("orchestrator.db")

# Configuration from environment
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory"
)


class Database:
    """Async PostgreSQL connection pool manager."""
    
    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Initialize the connection pool."""
        if self._pool is None:
            log.info(f"Connecting to PostgreSQL...")
            self._pool = await asyncpg.create_pool(
                POSTGRES_URL,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            log.info("PostgreSQL connection pool established")
    
    async def disconnect(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            log.info("PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if self._pool is None:
            await self.connect()
        async with self._pool.acquire() as conn:
            yield conn

    # ==========================================
    # Fingerprint Operations
    # ==========================================
    
    async def get_model_by_fingerprint(self, fingerprint: str) -> Optional[str]:
        """Check if fingerprint exists, return model_id if found."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT model_id FROM model_fingerprints WHERE fingerprint = $1",
                fingerprint
            )
            return row["model_id"] if row else None
    
    async def insert_fingerprint(
        self,
        fingerprint: str,
        model_id: str,
        features: List[str],
        hyperparams: Dict[str, Any],
        target_transform: str,
        symbol: str
    ) -> None:
        """Insert a new fingerprint record."""
        import json
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_fingerprints 
                (fingerprint, model_id, features_json, hyperparameters_json, target_transform, symbol)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (fingerprint) DO NOTHING
                """,
                fingerprint,
                model_id,
                json.dumps(sorted(features)),
                json.dumps(hyperparams),
                target_transform,
                symbol
            )

    # ==========================================
    # Evolution Run Operations
    # ==========================================
    
    async def create_evolution_run(
        self,
        run_id: str,
        seed_model_id: Optional[str],
        symbol: str,
        max_generations: int,
        config: Dict[str, Any]
    ) -> None:
        """Create a new evolution run."""
        import json
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO evolution_runs 
                (id, seed_model_id, symbol, max_generations, status, config)
                VALUES ($1, $2, $3, $4, 'PENDING', $5)
                """,
                run_id,
                seed_model_id,
                symbol,
                max_generations,
                json.dumps(config)
            )
    
    async def update_evolution_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        step_status: Optional[str] = None,
        current_generation: Optional[int] = None,
        best_sqn: Optional[float] = None,
        best_model_id: Optional[str] = None,
        promoted: Optional[bool] = None
    ) -> None:
        """Update evolution run status."""
        updates = ["updated_at = CURRENT_TIMESTAMP"]
        params = []
        param_idx = 1
        
        if status:
            updates.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1
        if step_status is not None:
            updates.append(f"step_status = ${param_idx}")
            params.append(step_status)
            param_idx += 1
        if current_generation is not None:
            updates.append(f"current_generation = ${param_idx}")
            params.append(current_generation)
            param_idx += 1
        if best_sqn is not None:
            updates.append(f"best_sqn = ${param_idx}")
            params.append(best_sqn)
            param_idx += 1
        if best_model_id:
            updates.append(f"best_model_id = ${param_idx}")
            params.append(best_model_id)
            param_idx += 1
        if promoted is not None:
            updates.append(f"promoted = ${param_idx}")
            params.append(promoted)
            param_idx += 1
        
        params.append(run_id)
        query = f"UPDATE evolution_runs SET {', '.join(updates)} WHERE id = ${param_idx}"
        
        async with self.acquire() as conn:
            await conn.execute(query, *params)
    
    async def get_evolution_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get evolution run by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM evolution_runs WHERE id = $1", run_id
            )
            return dict(row) if row else None
    
    async def list_evolution_runs(
        self, 
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List evolution runs."""
        async with self.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT * FROM evolution_runs 
                    WHERE status = $1 
                    ORDER BY created_at DESC LIMIT $2
                    """,
                    status, limit
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM evolution_runs ORDER BY created_at DESC LIMIT $1",
                    limit
                )
            return [dict(r) for r in rows]

    # ==========================================
    # Evolution Log Operations
    # ==========================================
    
    async def insert_evolution_log(
        self,
        log_id: str,
        run_id: str,
        parent_model_id: Optional[str],
        child_model_id: str,
        generation: int,
        parent_sqn: Optional[float],
        pruned_features: List[str],
        remaining_features: List[str],
        pruning_reason: str
    ) -> None:
        """Insert evolution lineage record."""
        import json
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO evolution_log 
                (id, run_id, parent_model_id, child_model_id, generation, 
                 parent_sqn, pruned_features, remaining_features, pruning_reason)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                log_id,
                run_id,
                parent_model_id,
                child_model_id,
                generation,
                parent_sqn,
                json.dumps(pruned_features),
                json.dumps(remaining_features),
                pruning_reason
            )
    
    async def get_lineage(self, run_id: str) -> List[Dict[str, Any]]:
        """Get full lineage for a run."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM evolution_log 
                WHERE run_id = $1 
                ORDER BY generation ASC
                """,
                run_id
            )
            return [dict(r) for r in rows]

    # ==========================================
    # Priority Job Queue Operations
    # ==========================================
    
    async def enqueue_jobs(
        self,
        jobs: List[Dict[str, Any]]
    ) -> int:
        """Bulk insert priority jobs. Returns count inserted."""
        import json
        async with self.acquire() as conn:
            # Use copy for bulk insert
            values = [
                (
                    job["id"],
                    job.get("batch_id"),
                    job["run_id"],
                    job["model_id"],
                    job.get("generation", 0),
                    job.get("parent_sqn", 0),
                    json.dumps(job["params"])
                )
                for job in jobs
            ]
            await conn.executemany(
                """
                INSERT INTO priority_jobs 
                (id, batch_id, run_id, model_id, generation, parent_sqn, params)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                values
            )
            return len(values)
    
    async def claim_job(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Claim highest priority pending job atomically."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM claim_priority_job($1)",
                worker_id
            )
            if row and row["job_id"]:
                return {
                    "id": row["job_id"],
                    "params": row["job_params"],
                    "model_id": row["job_model_id"],
                    "run_id": row["job_run_id"]
                }
            return None
    
    async def complete_job(
        self,
        job_id: str,
        result: Dict[str, Any],
        success: bool = True
    ) -> None:
        """Mark job as completed with result."""
        import json
        status = "COMPLETED" if success else "FAILED"
        async with self.acquire() as conn:
            await conn.execute(
                """
                UPDATE priority_jobs 
                SET status = $1, result = $2, updated_at = CURRENT_TIMESTAMP
                WHERE id = $3
                """,
                status,
                json.dumps(result),
                job_id
            )
    
    async def get_pending_job_count(self, run_id: str) -> int:
        """Get count of pending jobs for a run."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) as cnt FROM priority_jobs 
                WHERE run_id = $1 AND status IN ('PENDING', 'RUNNING')
                """,
                run_id
            )
            return row["cnt"] if row else 0
    
    async def get_completed_jobs(
        self, 
        run_id: str,
        generation: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get completed jobs for a run, optionally filtered by generation."""
        async with self.acquire() as conn:
            if generation is not None:
                rows = await conn.fetch(
                    """
                    SELECT * FROM priority_jobs 
                    WHERE run_id = $1 AND generation = $2 AND status = 'COMPLETED'
                    ORDER BY (result->>'sqn')::float DESC NULLS LAST
                    """,
                    run_id, generation
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM priority_jobs 
                    WHERE run_id = $1 AND status = 'COMPLETED'
                    ORDER BY (result->>'sqn')::float DESC NULLS LAST
                    """,
                    run_id
                )
            return [dict(r) for r in rows]

    async def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all jobs with optional limit."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, run_id, model_id, generation, parent_sqn, status, 
                       created_at, updated_at
                FROM priority_jobs 
                ORDER BY created_at DESC LIMIT $1
                """,
                limit
            )
            return [dict(r) for r in rows]

    # ==========================================
    # Promoted Models Operations
    # ==========================================
    
    async def insert_promoted_model(
        self,
        promoted_id: str,
        model_id: str,
        run_id: str,
        job_id: str,
        generation: int,
        sqn: float,
        profit_factor: float,
        trade_count: int,
        weekly_consistency: Optional[float],
        ticker: str,
        regime_config: Dict[str, Any],
        threshold: float,
        full_result: Dict[str, Any]
    ) -> None:
        """Insert a promoted model record."""
        import json
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO promoted_models 
                (id, model_id, run_id, job_id, generation, sqn, profit_factor,
                 trade_count, weekly_consistency, ticker, regime_config, threshold, full_result)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                promoted_id,
                model_id,
                run_id,
                job_id,
                generation,
                sqn,
                profit_factor,
                trade_count,
                weekly_consistency,
                ticker,
                json.dumps(regime_config),
                threshold,
                json.dumps(full_result)
            )
    
    async def list_promoted_models(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List promoted models by SQN descending."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM promoted_models 
                ORDER BY sqn DESC LIMIT $1
                """,
                limit
            )
            return [dict(r) for r in rows]

    # ==========================================
    # Worker Operations
    # ==========================================
    
    async def register_worker(self, worker_id: str) -> None:
        """Register or update worker heartbeat."""
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO workers (id, last_heartbeat, status)
                VALUES ($1, CURRENT_TIMESTAMP, 'IDLE')
                ON CONFLICT (id) DO UPDATE 
                SET last_heartbeat = CURRENT_TIMESTAMP
                """,
                worker_id
            )
    
    async def update_worker_status(
        self, 
        worker_id: str, 
        status: str,
        current_job_id: Optional[str] = None
    ) -> None:
        """Update worker status."""
        async with self.acquire() as conn:
            await conn.execute(
                """
                UPDATE workers 
                SET status = $1, current_job_id = $2, last_heartbeat = CURRENT_TIMESTAMP
                WHERE id = $3
                """,
                status, current_job_id, worker_id
            )


# Singleton instance
db = Database()
