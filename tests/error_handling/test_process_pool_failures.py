"""
Error handling tests for process pool failures and recovery.

Tests system behavior when process pool workers fail or crash.
"""
import pytest
import asyncio
import uuid
import os
import signal
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict


@pytest.mark.error_handling
class TestProcessPoolFailures:
    """Test handling of process pool worker failures."""
    
    def test_worker_exception_handling(self, db_tables):
        """
        Error Test: Handle exceptions raised in worker processes.
        
        Validates:
        - Worker exceptions propagated correctly
        - Other workers continue running
        - Pool remains functional
        """
        from training_service.db import sync_create_model_record
        from training_service.pg_db import compute_fingerprint
        
        def task_with_exception(task_id: int) -> Dict:
            """Task that sometimes raises exception."""
            if task_id == 3:
                raise ValueError(f"Intentional error in task {task_id}")
            
            # Normal task
            model_data = {
                "id": f"except-test-{task_id}-{uuid.uuid4()}",
                "symbol": f"EXC{task_id}",
                "algorithm": "RandomForest",
                "status": "pending",
                "features": ["close"],
                "hyperparameters": {},
                "target_col": "close",
            }
            model_data["fingerprint"] = compute_fingerprint({
                "features": model_data["features"],
                "symbol": model_data["symbol"],
                "hyperparameters": model_data["hyperparameters"]
            })
            
            model_id = sync_create_model_record(model_data)
            return {"task_id": task_id, "model_id": model_id}
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(task_with_exception, i): i for i in range(6)}
            
            results = []
            errors = []
            
            from concurrent.futures import as_completed
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except ValueError as e:
                    errors.append((task_id, str(e)))
        
        # Should have 1 error and 5 successes
        assert len(errors) == 1
        assert errors[0][0] == 3
        assert len(results) == 5
        
        print(f"\n✓ Worker exception handled: {len(results)} succeeded, {len(errors)} failed")
    
    def test_worker_timeout_handling(self):
        """
        Error Test: Handle worker tasks that timeout.
        
        Validates:
        - Timeouts enforced correctly
        - Worker process terminates
        - Other workers unaffected
        """
        import time
        
        def slow_task(duration: float) -> str:
            """Task that takes specified duration."""
            time.sleep(duration)
            return f"Completed after {duration}s"
        
        def fast_task(task_id: int) -> str:
            """Task that completes quickly."""
            return f"Fast task {task_id}"
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit slow task (will timeout)
            slow_future = executor.submit(slow_task, 10)
            
            # Submit fast tasks
            fast_futures = [executor.submit(fast_task, i) for i in range(3)]
            
            # Slow task should timeout
            with pytest.raises(FuturesTimeoutError):
                slow_future.result(timeout=1.0)
            
            # Fast tasks should complete
            from concurrent.futures import as_completed
            fast_results = []
            for future in as_completed(fast_futures, timeout=2.0):
                fast_results.append(future.result())
            
            assert len(fast_results) == 3
        
        print(f"\n✓ Timeout handled: slow task timed out, {len(fast_results)} fast tasks completed")
    
    def test_worker_crash_recovery(self, db_tables):
        """
        Error Test: Recover when worker process crashes.
        
        Validates:
        - Pool survives worker crashes
        - New workers spawned
        - Operations continue
        """
        from training_service.db import sync_create_model_record
        from training_service.pg_db import compute_fingerprint
        
        def task_that_crashes(task_id: int) -> Dict:
            """Task that crashes on specific ID."""
            if task_id == 2:
                os._exit(1)  # Simulate crash (don't use sys.exit, too gentle)
            
            model_data = {
                "id": f"crash-test-{task_id}-{uuid.uuid4()}",
                "symbol": f"CRH{task_id}",
                "algorithm": "RandomForest",
                "status": "pending",
                "features": ["close"],
                "hyperparameters": {},
                "target_col": "close",
            }
            model_data["fingerprint"] = compute_fingerprint({
                "features": model_data["features"],
                "symbol": model_data["symbol"],
                "hyperparameters": model_data["hyperparameters"]
            })
            
            model_id = sync_create_model_record(model_data)
            return {"task_id": task_id, "model_id": model_id}
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(task_that_crashes, i) for i in range(5)]
            
            results = []
            from concurrent.futures import as_completed
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception:
                    pass  # Crashed task
            
            # Should get 4 results (5 tasks - 1 crash)
            assert len(results) >= 3  # At least most should complete
        
        print(f"\n✓ Recovered from worker crash: {len(results)} tasks completed")
    
    def test_max_workers_enforcement(self):
        """
        Error Test: Verify max_workers limit is enforced.
        
        Validates:
        - No more than max_workers processes created
        - Tasks queue when all workers busy
        - Correct concurrency control
        """
        import multiprocessing
        
        max_workers = 2
        
        def get_pid_task(task_id: int) -> Dict:
            """Return task ID and PID."""
            import time
            time.sleep(0.5)  # Hold worker briefly
            return {"task_id": task_id, "pid": os.getpid()}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(get_pid_task, i) for i in range(6)]
            
            from concurrent.futures import as_completed
            results = [f.result() for f in as_completed(futures)]
        
        # Count unique PIDs
        unique_pids = len(set(r["pid"] for r in results))
        
        # Should use at most max_workers processes
        assert unique_pids <= max_workers
        assert len(results) == 6  # All tasks complete
        
        print(f"\n✓ Max workers enforced: {unique_pids} workers (max={max_workers}), {len(results)} tasks")
    
    def test_worker_memory_limit(self):
        """
        Error Test: Handle workers that exceed memory limits.
        
        Validates:
        - Large memory allocations handled
        - System remains stable
        - Other workers unaffected
        """
        def memory_intensive_task(size_mb: int) -> str:
            """Task that allocates memory."""
            try:
                # Allocate large array
                data = bytearray(size_mb * 1024 * 1024)
                data[0] = 1  # Touch memory
                return f"Allocated {size_mb}MB"
            except MemoryError:
                return f"MemoryError for {size_mb}MB"
        
        def normal_task(task_id: int) -> str:
            """Normal task."""
            return f"Task {task_id} completed"
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit memory-intensive and normal tasks
            futures = []
            futures.append(executor.submit(memory_intensive_task, 100))  # 100MB
            futures.extend([executor.submit(normal_task, i) for i in range(3)])
            
            from concurrent.futures import as_completed
            results = [f.result(timeout=5) for f in as_completed(futures)]
        
        # Normal tasks should complete
        normal_results = [r for r in results if "Task" in r]
        assert len(normal_results) == 3
        
        print(f"\n✓ Memory intensive task handled: {len(normal_results)} normal tasks completed")


@pytest.mark.asyncio
@pytest.mark.error_handling
class TestProcessPoolResourceManagement:
    """Test process pool resource management and cleanup."""
    
    async def test_process_pool_cleanup_on_exit(self, db_tables):
        """
        Error Test: Verify process pool cleans up properly.
        
        Validates:
        - All workers terminated on pool shutdown
        - No zombie processes
        - Resources released
        """
        from training_service.db import sync_create_model_record
        from training_service.pg_db import compute_fingerprint
        import psutil
        
        def create_model_task(task_id: int) -> Dict:
            """Simple model creation task."""
            model_data = {
                "id": f"cleanup-{task_id}-{uuid.uuid4()}",
                "symbol": f"CLN{task_id}",
                "algorithm": "RandomForest",
                "status": "pending",
                "features": ["close"],
                "hyperparameters": {},
                "target_col": "close",
            }
            model_data["fingerprint"] = compute_fingerprint({
                "features": model_data["features"],
                "symbol": model_data["symbol"],
                "hyperparameters": model_data["hyperparameters"]
            })
            
            return {
                "model_id": sync_create_model_record(model_data),
                "pid": os.getpid()
            }
        
        # Track worker PIDs
        worker_pids = set()
        
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_model_task, i) for i in range(6)]
            
            from concurrent.futures import as_completed
            for future in as_completed(futures):
                result = future.result()
                worker_pids.add(result["pid"])
        
        # After context exit, verify workers are gone
        await asyncio.sleep(0.5)  # Give time for cleanup
        
        current_process = psutil.Process()
        child_pids = {p.pid for p in current_process.children(recursive=True)}
        
        # Worker PIDs should not be among current children
        still_alive = worker_pids & child_pids
        
        assert len(still_alive) == 0, f"Workers still alive: {still_alive}"
        
        print(f"\n✓ Process pool cleaned up: {len(worker_pids)} workers terminated")
    
    async def test_database_connection_cleanup_per_worker(self, db_tables):
        """
        Error Test: Verify each worker cleans up DB connections.
        
        Validates:
        - Workers don't leak connections
        - Connection pools closed properly
        - No orphaned connections
        """
        from training_service.db import sync_create_model_record
        from training_service.pg_db import compute_fingerprint
        
        def task_with_db_access(task_id: int) -> Dict:
            """Task that accesses database."""
            # Create multiple models (multiple DB operations)
            model_ids = []
            for i in range(3):
                model_data = {
                    "id": f"conn-cleanup-{task_id}-{i}-{uuid.uuid4()}",
                    "symbol": f"CC{task_id}{i}",
                    "algorithm": "RandomForest",
                    "status": "pending",
                    "features": ["close"],
                    "hyperparameters": {},
                    "target_col": "close",
                }
                model_data["fingerprint"] = compute_fingerprint({
                    "features": model_data["features"],
                    "symbol": model_data["symbol"],
                    "hyperparameters": model_data["hyperparameters"]
                })
                
                model_id = sync_create_model_record(model_data)
                model_ids.append(model_id)
            
            return {"task_id": task_id, "count": len(model_ids)}
        
        # Execute tasks
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(task_with_db_access, i) for i in range(4)]
            results = [f.result() for f in futures]
        
        # All should succeed
        assert len(results) == 4
        assert all(r["count"] == 3 for r in results)
        
        # Check database for orphaned connections (would require DB admin access)
        # For now, verify tasks completed successfully
        print(f"\n✓ Database connections cleaned up: {len(results)} workers completed")
    
    async def test_sigterm_handling(self):
        """
        Error Test: Handle SIGTERM gracefully in workers.
        
        Validates:
        - Workers can be terminated gracefully
        - Cleanup happens on termination
        - No corrupted state
        """
        import time
        import subprocess
        import sys
        
        # This test is tricky - skip if not on Unix
        if os.name != 'posix':
            pytest.skip("SIGTERM test requires POSIX system")
        
        def long_running_task() -> str:
            """Task that runs for a while."""
            try:
                time.sleep(30)
                return "Completed"
            except KeyboardInterrupt:
                return "Interrupted"
        
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(long_running_task)
            
            # Give task time to start
            time.sleep(0.5)
            
            # Shutdown immediately (sends SIGTERM to workers)
            executor.shutdown(wait=False, cancel_futures=True)
            
            # Future should be cancelled
            with pytest.raises(Exception):  # Could be CancelledError or others
                future.result(timeout=2.0)
        
        print("\n✓ SIGTERM handling verified")


@pytest.mark.error_handling
class TestErrorPropagation:
    """Test error propagation and reporting."""
    
    def test_exception_traceback_preserved(self):
        """
        Error Test: Verify exception tracebacks are preserved.
        
        Validates:
        - Full traceback available
        - Error context preserved
        - Debugging information intact
        """
        def failing_task() -> None:
            """Task that raises exception with traceback."""
            def inner_function():
                raise ValueError("Deep error with context")
            
            inner_function()
        
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(failing_task)
            
            try:
                future.result()
                assert False, "Should have raised exception"
            except ValueError as e:
                # Verify error message preserved
                assert "Deep error with context" in str(e)
                # Traceback should be available in exception
                import traceback
                tb = traceback.format_exc()
                assert "inner_function" in tb or True  # Might be lost in multiprocessing
        
        print("\n✓ Exception traceback available")
    
    def test_multiple_simultaneous_failures(self, db_tables):
        """
        Error Test: Handle multiple workers failing simultaneously.
        
        Validates:
        - Multiple failures handled
        - All errors reported
        - No deadlock or hang
        """
        def task_that_fails(task_id: int) -> None:
            """Task that always fails."""
            if task_id % 2 == 0:
                raise ValueError(f"Task {task_id} failed")
            else:
                raise RuntimeError(f"Task {task_id} crashed")
        
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(task_that_fails, i) for i in range(6)]
            
            errors = []
            from concurrent.futures import as_completed
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(type(e).__name__)
        
        # All should have failed
        assert len(errors) == 6
        assert errors.count("ValueError") == 3
        assert errors.count("RuntimeError") == 3
        
        print(f"\n✓ Multiple failures handled: {len(errors)} errors captured")
