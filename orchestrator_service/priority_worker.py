"""
Priority Worker - Pulls jobs from PostgreSQL priority queue and runs simulations.

This worker:
1. Connects to PostgreSQL priority_jobs table
2. Claims highest priority pending job (based on parent_sqn)
3. Runs simulation via simulation_service HTTP API
4. Reports results back to PostgreSQL
5. Repeats

Run with: python -m orchestrator_service.priority_worker
"""
import os
import sys
import time
import asyncio
import logging
import httpx
from typing import Optional, Dict, Any

# Add parent to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator_service.db import Database

# Configuration
WORKER_ID = os.getenv("WORKER_ID", f"worker_{os.getpid()}")
SIMULATION_URL = os.getenv("SIMULATION_URL", "http://simulation:8300")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format=f"%(asctime)s %(levelname)s [{WORKER_ID}] %(message)s"
)
log = logging.getLogger("priority_worker")


class PriorityWorker:
    """Worker that pulls from priority queue and runs simulations."""
    
    def __init__(self):
        self.db = Database()
        self.http_client: Optional[httpx.AsyncClient] = None
        self.running = True
        self.jobs_completed = 0
        self.jobs_failed = 0
    
    async def start(self):
        """Initialize connections."""
        log.info(f"Starting Priority Worker: {WORKER_ID}")
        log.info(f"Simulation URL: {SIMULATION_URL}")
        
        await self.db.connect()
        self.http_client = httpx.AsyncClient(timeout=300.0)  # 5 min timeout for sims
        
        await self.db.register_worker(WORKER_ID)
        
        # Clean up orphaned jobs from stopped/cancelled runs
        await self._cleanup_orphaned_jobs()
        
        log.info("Worker registered and ready")
    
    async def _cleanup_orphaned_jobs(self):
        """Cancel pending/running jobs that belong to stopped evolution runs."""
        try:
            async with self.db.acquire() as conn:
                result = await conn.execute("""
                    UPDATE priority_jobs 
                    SET status = 'CANCELLED', 
                        result = '{"error": "Evolution run no longer active", "cancelled": true}'
                    WHERE status IN ('PENDING', 'RUNNING')
                    AND run_id IN (
                        SELECT id FROM evolution_runs 
                        WHERE status NOT IN ('RUNNING', 'PENDING')
                    )
                """)
                
                # Get count of cancelled jobs
                cancelled = await conn.fetch("""
                    SELECT run_id, COUNT(*) as cnt 
                    FROM priority_jobs 
                    WHERE status = 'CANCELLED' 
                    AND result::text LIKE '%Evolution run no longer active%'
                    GROUP BY run_id
                """)
                
                if cancelled:
                    for row in cancelled:
                        log.warning(f"Cancelled {row['cnt']} orphaned jobs from run {row['run_id']}")
                else:
                    log.info("No orphaned jobs to clean up")
        except Exception as e:
            log.error(f"Failed to clean up orphaned jobs: {e}")
    
    async def stop(self):
        """Clean shutdown."""
        self.running = False
        if self.http_client:
            await self.http_client.aclose()
        await self.db.disconnect()
        log.info(f"Worker stopped. Completed: {self.jobs_completed}, Failed: {self.jobs_failed}")
    
    async def run(self):
        """Main worker loop."""
        await self.start()
        
        # Initial health check
        log.info("Performing initial simulation service health check...")
        is_healthy = await self._check_simulation_health()
        if not is_healthy:
            log.warning("Simulation service is not healthy at startup - will retry on each job")
        
        health_check_counter = 0
        
        try:
            while self.running:
                try:
                    # Periodic health check every 10 iterations (when idle)
                    health_check_counter += 1
                    if health_check_counter >= 10:
                        log.debug("Performing periodic health check...")
                        await self._check_simulation_health()
                        health_check_counter = 0
                    
                    # Try to claim a job
                    job = await self.db.claim_job(WORKER_ID)
                    
                    if job:
                        health_check_counter = 0  # Reset counter when processing
                        await self._process_job(job)
                    else:
                        # No jobs available, wait and retry
                        log.debug(f"No jobs available, waiting {POLL_INTERVAL}s...")
                        await asyncio.sleep(POLL_INTERVAL)
                        
                except asyncio.CancelledError:
                    log.info("Worker cancelled")
                    break
                except Exception as e:
                    log.error(f"Error in worker loop: {e}", exc_info=True)
                    await asyncio.sleep(POLL_INTERVAL)
                    
        finally:
            await self.stop()
    
    async def _check_simulation_health(self) -> bool:
        """Check if simulation service is healthy."""
        try:
            resp = await self.http_client.get(f"{SIMULATION_URL}/health", timeout=5.0)
            if resp.status_code == 200:
                health = resp.json()
                log.info(f"Simulation service health: {health}")
                return health.get("status") == "healthy"
            else:
                log.error(f"Simulation health check returned {resp.status_code}")
                return False
        except Exception as e:
            log.error(f"Simulation health check failed: {e}")
            return False
    
    async def _process_job(self, job: Dict[str, Any]):
        """Process a single job."""
        job_id = job["id"]
        run_id = job.get("run_id")
        params = job["params"]
        
        # Parse params if it's a JSON string
        if isinstance(params, str):
            import json
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                log.error(f"Failed to parse params as JSON: {params[:100]}")
                params = {}
        
        log.info(f"Processing job {job_id}")
        log.info(f"  Run: {run_id}")
        log.info(f"  Model: {job.get('model_id')}")
        log.info(f"  Params: threshold={params.get('threshold')}, regime={params.get('regime_config')}")
        
        # Health check before processing
        is_healthy = await self._check_simulation_health()
        if not is_healthy:
            log.error(f"Simulation service is unhealthy - aborting job {job_id}")
            await self.db.complete_job(
                job_id,
                result={"error": "Simulation service is unhealthy or unreachable"},
                status="FAILED"
            )
            return
        
        # CRITICAL: Check if evolution run is still active before processing
        if run_id:
            run_status = await self.db.get_evolution_run_status(run_id)
            if run_status not in ["RUNNING", "PENDING"]:
                log.warning(f"Job {job_id} belongs to inactive run {run_id} (status: {run_status}) - cancelling")
                await self.db.complete_job(
                    job_id,
                    {"error": f"Evolution run {run_id} is {run_status}, not running", "cancelled": True},
                    success=False
                )
                await self.db.update_worker_status(WORKER_ID, "IDLE", None)
                return
        
        try:
            # Update worker status
            await self.db.update_worker_status(WORKER_ID, "BUSY", job_id)
            
            # Run simulation
            result = await self._run_simulation(params)
            
            # Save simulation fingerprint if provided
            if "simulation_fingerprint" in params and "model_fingerprint" in params:
                try:
                    await self.db.insert_simulation_fingerprint(
                        fingerprint=params["simulation_fingerprint"],
                        model_fingerprint=params["model_fingerprint"],
                        model_id=params["model_id"],
                        target_ticker=params.get("target_ticker", params.get("ticker")),
                        simulation_ticker=params.get("ticker"),
                        threshold=params.get("threshold", 0.0),
                        z_score_threshold=params.get("z_score_threshold", 0.0),
                        regime_config=params.get("regime_config", {}),
                        train_window=params.get("train_window", 20000),
                        test_window=params.get("test_window", 1000),
                        result=result
                    )
                    log.info(f"Saved simulation fingerprint {params['simulation_fingerprint'][:16]}... (model: {params['model_fingerprint'][:16]}...)")
                except Exception as e:
                    log.warning(f"Failed to save simulation fingerprint: {e}")
            
            # Report success
            await self.db.complete_job(job_id, result, success=True)
            await self.db.update_worker_status(WORKER_ID, "IDLE", None)
            
            self.jobs_completed += 1
            log.info(f"Job {job_id} completed. SQN: {result.get('sqn', 'N/A')}")
            
        except Exception as e:
            log.error(f"Job {job_id} failed: {e}")
            
            # Report failure
            await self.db.complete_job(
                job_id, 
                {"error": str(e)}, 
                success=False
            )
            await self.db.update_worker_status(WORKER_ID, "IDLE", None)
            
            self.jobs_failed += 1
    
    async def _check_simulation_health(self) -> bool:
        """Check if simulation service is healthy."""
        try:
            resp = await self.http_client.get(f"{SIMULATION_URL}/health", timeout=5.0)
            if resp.status_code == 200:
                health = resp.json()
                log.info(f"Simulation service health: {health}")
                return health.get("status") == "healthy"
            else:
                log.error(f"Simulation health check returned {resp.status_code}")
                return False
        except Exception as e:
            log.error(f"Simulation health check failed: {e}")
            return False
    
    async def _run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run simulation via HTTP API.
        
        Expected params:
            - model_id: UUID of the model
            - ticker: Trading symbol
            - threshold: Prediction threshold
            - regime_config: Dict like {"regime_gmm": [1]} or {"regime_vix": [3]}
            - z_score_threshold: Optional
            - use_trading_bot: Optional
            - use_volume_normalization: Optional
        """
        # Build simulation request
        sim_request = {
            "model_id": params["model_id"],
            "ticker": params["ticker"],
            "min_prediction_threshold": params.get("threshold", 0.0003),
            "enable_z_score_check": params.get("z_score_threshold") is not None,
            "volatility_normalization": params.get("use_volume_normalization", True),
            "use_bot": params.get("use_trading_bot", False)
        }
        
        # Add regime filter if specified
        # Format: {"regime_gmm": [1]} -> regime_col="regime_gmm", allowed_regimes=[1]
        # Format: {"regime_vix": [3]} -> regime_col="regime_vix", allowed_regimes=[3]
        regime_config = params.get("regime_config", {})
        if regime_config:
            for regime_col, values in regime_config.items():
                sim_request["regime_col"] = regime_col
                sim_request["allowed_regimes"] = values
                break  # Only use first regime config
        
        # Call simulation service
        url = f"{SIMULATION_URL}/api/simulate"
        log.info(f"Calling simulation service: POST {url}")
        log.info(f"Payload: model_id={sim_request['model_id'][:12]}..., ticker={sim_request['ticker']}")
        log.debug(f"Full simulation request: {sim_request}")
        
        try:
            resp = await self.http_client.post(url, json=sim_request)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            log.error(f"Simulation TIMEOUT: {e}")
            log.error(f"  URL: {url}")
            log.error(f"  Model: {sim_request['model_id']}")
            log.error(f"  Ticker: {sim_request['ticker']}")
            raise
        except httpx.HTTPStatusError as e:
            log.error(f"Simulation HTTP {e.response.status_code} error")
            log.error(f"  URL: {url}")
            log.error(f"  Response: {e.response.text[:500]}")
            log.error(f"  Model: {sim_request['model_id']}")
            log.error(f"  Ticker: {sim_request['ticker']}")
            raise
        except Exception as e:
            log.error(f"Unexpected simulation error: {e}", exc_info=True)
            log.error(f"  URL: {url}")
            log.error(f"  Request: {sim_request}")
            raise
        
        result = resp.json()
        
        # Add params to result for traceability
        result["params"] = params
        
        return result


async def main():
    """Entry point."""
    worker = PriorityWorker()
    
    # Handle signals for graceful shutdown
    import signal
    
    def handle_signal(sig, frame):
        log.info(f"Received signal {sig}, shutting down...")
        worker.running = False
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
