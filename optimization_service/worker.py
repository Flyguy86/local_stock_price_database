import time
import requests
import uuid
import sys
import logging
from pathlib import Path
import traceback

# Add shared modules to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from simulation_service.core import run_simulation

log = logging.getLogger("worker")
logging.basicConfig(level=logging.INFO)

API_URL = "http://localhost:8002/api" # Assume local for now, can be env var
WORKER_ID = f"worker-{str(uuid.uuid4())[:8]}"

def run_worker():
    log.info(f"Worker {WORKER_ID} started. Connecting to {API_URL}")
    
    while True:
        try:
            # 1. Claim Job
            try:
                resp = requests.post(f"{API_URL}/worker/claim", json={"worker_id": WORKER_ID}, timeout=5)
                job = resp.json()
            except Exception as e:
                log.error(f"Failed to connect to C2: {e}")
                time.sleep(10)
                continue
                
            if not job:
                # No jobs, sleep
                time.sleep(2)
                continue
            
            job_id = job["id"]
            params = job["params"]
            log.info(f"Processing Job {job_id} | {params.get('ticker')} | {params.get('model_id')}")
            
            # 2. Run Simulation
            try:
                # Map params to function args
                sim_result = run_simulation(
                    model_id=params["model_id"],
                    ticker=params["ticker"],
                    initial_cash=params.get("initial_cash", 10000),
                    use_bot=params.get("use_bot", False),
                    min_prediction_threshold=params.get("min_prediction_threshold", 0.0),
                    enable_z_score_check=params.get("enable_z_score_check", False),
                    volatility_normalization=params.get("volatility_normalization", False),
                    save_to_history=False # Important: Don't flood manual history
                )
                
                # Extract simple stats for the grid result
                summary = {
                    "strategy_return_pct": sim_result["stats"]["strategy_return_pct"],
                    "total_trades": sim_result["stats"]["total_trades"],
                    "hit_rate_pct": sim_result["stats"]["hit_rate_pct"],
                    "final_value": sim_result["stats"]["final_strategy_value"]
                }
                
                # 3. Submit Result
                requests.post(f"{API_URL}/worker/complete", json={
                    "job_id": job_id,
                    "result": summary,
                    "status": "COMPLETED"
                })
                log.info(f"Job {job_id} Completed. Return: {summary['strategy_return_pct']:.2f}%")
                
            except Exception as e:
                log.error(f"Job {job_id} Failed: {e}")
                traceback.print_exc()
                requests.post(f"{API_URL}/worker/complete", json={
                    "job_id": job_id,
                    "result": {"error": str(e)},
                    "status": "FAILED"
                })
                
        except KeyboardInterrupt:
            log.info("Worker stopped.")
            break
        except Exception as e:
            log.error(f"Unexpected worker error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_worker()
