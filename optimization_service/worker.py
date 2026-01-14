import time
import requests
import uuid
import sys
import logging
import os
from pathlib import Path
import traceback

# Add shared modules to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from simulation_service.core import run_simulation

log = logging.getLogger("worker")
logging.basicConfig(level=logging.INFO)

# Use localhost when running as internal thread, container name when distributed
API_URL = os.getenv("C2_API_URL", "http://localhost:8002/api")
WORKER_ID = f"worker-{str(uuid.uuid4())[:8]}"

def run_worker():
    log.info(f"Worker {WORKER_ID} started. Connecting to {API_URL}")
    
    # Initial startup delay to ensure server is ready
    import time
    time.sleep(2)
    
    log.info(f"Worker {WORKER_ID} entering main loop...")
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        try:
            # 0. Heartbeat (even when idle)
            try:
                resp = requests.post(f"{API_URL}/worker/heartbeat", json={"worker_id": WORKER_ID}, timeout=5)
                resp.raise_for_status()
                consecutive_errors = 0  # Reset error counter on success
                log.debug(f"Worker {WORKER_ID} heartbeat sent successfully")
            except Exception as e:
                consecutive_errors += 1
                log.warning(f"Heartbeat failed (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    log.error(f"Worker {WORKER_ID} failed {max_consecutive_errors} consecutive heartbeats. Exiting.")
                    break
                
                time.sleep(5)
                continue
            
            # 1. Claim Job
            try:
                resp = requests.post(f"{API_URL}/worker/claim", json={"worker_id": WORKER_ID}, timeout=5)
                resp.raise_for_status()
                job = resp.json()
            except Exception as e:
                log.error(f"Failed to connect to C2: {e}")
                time.sleep(10)
                continue
                
            if not job:
                log.debug(f"Worker {WORKER_ID} idle - no pending jobs")
                time.sleep(2)
                continue
            
            job_id = job["id"]
            params = job["params"]
            log.info(f"Worker {WORKER_ID} claimed job {job_id}: {params.get('ticker')} | {params.get('model_id')[:12]}...")
            
            # Heartbeat with job ID
            try:
                requests.post(f"{API_URL}/worker/heartbeat", json={"worker_id": WORKER_ID, "job_id": job_id}, timeout=5)
            except:
                pass
            
            # 2. Run Simulation with Timeout Protection
            import signal
            from contextlib import contextmanager
            
            @contextmanager
            def timeout(seconds):
                """Context manager for timeout on blocking operations."""
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Simulation exceeded {seconds}s timeout")
                
                # Set the signal handler
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            
            try:
                # Apply 5-minute timeout for simulation
                with timeout(300):  # 5 minutes
                    sim_result = run_simulation(
                        model_id=params["model_id"],
                        ticker=params["ticker"],
                        initial_cash=params.get("initial_cash", 10000),
                        use_bot=params.get("use_bot", False),
                        min_prediction_threshold=params.get("min_prediction_threshold", 0.0),
                        enable_z_score_check=params.get("enable_z_score_check", False),
                        volatility_normalization=params.get("volatility_normalization", False),
                        regime_col=params.get("regime_col"),
                        allowed_regimes=params.get("allowed_regimes"),
                        enable_slippage=params.get("enable_slippage", True),
                        slippage_bars=params.get("slippage_bars", 4),
                        transaction_fee=params.get("transaction_fee", 0.02),
                        save_to_history=False
                    )
                
                # Validate result structure
                if not sim_result or "stats" not in sim_result:
                    raise ValueError("Simulation returned invalid result (missing stats)")
                
                summary = {
                    "strategy_return_pct": sim_result["stats"]["strategy_return_pct"],
                    "total_trades": sim_result["stats"]["total_trades"],
                    "hit_rate_pct": sim_result["stats"]["hit_rate_pct"],
                    "final_value": sim_result["stats"]["final_strategy_value"],
                    "total_fees": sim_result["stats"].get("total_fees", 0.0),
                    "sqn": sim_result["stats"].get("sqn", 0.0),
                    "expectancy": sim_result["stats"].get("expectancy", 0.0),
                    "profit_factor": sim_result["stats"].get("profit_factor", 0.0)
                }
                
                # 3. Submit Result
                requests.post(f"{API_URL}/worker/complete", json={
                    "job_id": job_id,
                    "result": summary,
                    "status": "COMPLETED"
                }, timeout=10)
                log.info(f"Worker {WORKER_ID} completed job {job_id}. Return: {summary['strategy_return_pct']:.2f}%, SQN: {summary['sqn']:.2f}")
                
            except TimeoutError as e:
                log.error(f"Worker {WORKER_ID} job {job_id} TIMEOUT: {e}")
                try:
                    requests.post(f"{API_URL}/worker/complete", json={
                        "job_id": job_id,
                        "result": {"error": f"Simulation timeout (>300s): {str(e)}"},
                        "status": "FAILED"
                    }, timeout=10)
                except Exception as report_err:
                    log.error(f"Failed to report timeout to C2: {report_err}")
                    
            except Exception as e:
                log.error(f"Worker {WORKER_ID} job {job_id} failed: {e}", exc_info=True)
                traceback.print_exc()
                
                try:
                    requests.post(f"{API_URL}/worker/complete", json={
                        "job_id": job_id,
                        "result": {"error": str(e)},
                        "status": "FAILED"
                    }, timeout=10)
                except:
                    log.error(f"Failed to report job failure to C2")
                
        except KeyboardInterrupt:
            log.info(f"Worker {WORKER_ID} stopped by user.")
            break
        except Exception as e:
            log.error(f"Unexpected worker error: {e}", exc_info=True)
            consecutive_errors += 1
            
            if consecutive_errors >= max_consecutive_errors:
                log.error(f"Worker {WORKER_ID} encountered {max_consecutive_errors} consecutive errors. Exiting.")
                break
                
            time.sleep(5)
    
    log.info(f"Worker {WORKER_ID} shutdown complete.")

if __name__ == "__main__":
    run_worker()
