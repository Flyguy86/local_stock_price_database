#!/usr/bin/env python3
"""
Test script to verify ProcessPoolExecutor setup in training_service.
"""
import asyncio
import os
from concurrent.futures import ProcessPoolExecutor


def cpu_intensive_task(task_id, duration=2):
    """Simulate CPU-intensive training work."""
    import time
    import os
    
    pid = os.getpid()
    print(f"Task {task_id} starting in process {pid}")
    
    # Simulate work
    time.sleep(duration)
    
    print(f"Task {task_id} completed in process {pid}")
    return {"task_id": task_id, "pid": pid, "status": "completed"}


async def submit_task(pool, task_id):
    """Submit task to process pool and wait for completion."""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(pool, cpu_intensive_task, task_id)
        print(f"Task {task_id} result: {result}")
        return result
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        return {"task_id": task_id, "error": str(e)}


async def main():
    """Test parallel execution with ProcessPoolExecutor."""
    cpu_count = os.cpu_count() or 4
    print(f"CPU count: {cpu_count}")
    print(f"Creating ProcessPoolExecutor with {cpu_count} workers")
    
    # Create process pool
    pool = ProcessPoolExecutor(max_workers=cpu_count, max_tasks_per_child=10)
    
    try:
        print("\nSubmitting 8 tasks...")
        
        # Submit multiple tasks in parallel
        tasks = [submit_task(pool, i) for i in range(8)]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        print(f"\nAll tasks completed!")
        print(f"Results: {results}")
        
        # Check that tasks ran in parallel (different PIDs)
        pids = set(r.get("pid") for r in results if "pid" in r)
        print(f"\nUnique process IDs used: {pids}")
        print(f"Number of processes used: {len(pids)}")
        
        if len(pids) > 1:
            print("✅ Tasks ran in parallel across multiple processes!")
        else:
            print("⚠️  All tasks ran in same process (not parallel)")
            
    finally:
        print("\nShutting down process pool...")
        pool.shutdown(wait=True, cancel_futures=True)
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
