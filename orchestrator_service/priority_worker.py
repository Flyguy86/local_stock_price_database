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
        log.info("Worker registered and ready")
    
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
        
        try:
            while self.running:
                try:
                    # Try to claim a job
                    job = await self.db.claim_job(WORKER_ID)
                    
                    if job:
                        await self._process_job(job)
                    else:
                        # No jobs available, wait and retry
                        await asyncio.sleep(POLL_INTERVAL)
                        
                except asyncio.CancelledError:
                    log.info("Worker cancelled")
                    break
                except Exception as e:
                    log.error(f"Error in worker loop: {e}")
                    await asyncio.sleep(POLL_INTERVAL)
                    
        finally:
            await self.stop()
    
    async def _process_job(self, job: Dict[str, Any]):
        """Process a single job."""
        job_id = job["id"]
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
        log.info(f"  Model: {job.get('model_id')}")
        log.info(f"  Params: threshold={params.get('threshold')}, regime={params.get('regime_config')}")
        
        try:
            # Update worker status
            await self.db.update_worker_status(WORKER_ID, "BUSY", job_id)
            
            # Run simulation
            result = await self._run_simulation(params)
            
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
        log.debug(f"POST {url} with {sim_request}")
        
        resp = await self.http_client.post(url, json=sim_request)
        resp.raise_for_status()
        
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
