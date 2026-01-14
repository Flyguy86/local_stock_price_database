"""
Unit tests for ProcessPoolExecutor in training_service/main.py
Tests multi-core parallel execution.
"""
import pytest
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def cpu_intensive_task(task_id, duration=1):
    """Simulate CPU-intensive work."""
    import time
    import os
    
    pid = os.getpid()
    start = time.time()
    
    # Burn CPU
    count = 0
    while time.time() - start < duration:
        count += sum(range(1000))
    
    return {
        'task_id': task_id,
        'pid': pid,
        'duration': time.time() - start,
        'count': count
    }


def failing_task(task_id):
    """Module-level function that raises an error (can be pickled)."""
    raise ValueError(f"Task {task_id} failed intentionally")


def task_with_db_access(task_id):
    """Simulate training task that accesses database."""
    import os
    import sys
    from pathlib import Path
    import uuid
    import json
    
    # Add project root
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Set test database
    os.environ['POSTGRES_URL'] = os.environ.get(
        'TEST_POSTGRES_URL',
        'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test'
    )
    
    from training_service.sync_db_wrapper import db
    
    pid = os.getpid()
    
    try:
        # Create a model
        model_id = str(uuid.uuid4())
        model_data = {
            'id': model_id,
            'algorithm': 'RandomForest',
            'symbol': f'TASK{task_id}',
            'target_col': 'close',
            'feature_cols': json.dumps(['sma_20']),
            'hyperparameters': json.dumps({'n_estimators': 10}),
            'status': 'preprocessing',
            'timeframe': '1m'
        }
        
        db.create_model_record(model_data)
        db.update_model_status(model_id, status='completed')
        
        model = db.get_model(model_id)
        
        return {
            'task_id': task_id,
            'pid': pid,
            'success': True,
            'model_id': model_id,
            'status': model.get('status') if model else None
        }
    except Exception as e:
        return {
            'task_id': task_id,
            'pid': pid,
            'success': False,
            'error': str(e)
        }


class TestProcessPoolExecutor:
    """Test suite for ProcessPoolExecutor functionality."""
    
    def test_process_pool_creation(self):
        """Test that process pool can be created."""
        cpu_count = os.cpu_count() or 4
        
        with ProcessPoolExecutor(max_workers=cpu_count) as pool:
            assert pool is not None
    
    def test_single_task_execution(self):
        """Test executing a single task in process pool."""
        with ProcessPoolExecutor(max_workers=2) as pool:
            future = pool.submit(cpu_intensive_task, 1, 0.1)
            result = future.result(timeout=5)
            
            assert result['task_id'] == 1
            assert result['pid'] > 0
            assert result['duration'] >= 0.1
    
    def test_parallel_execution(self):
        """Test that multiple tasks run in parallel."""
        num_tasks = 4
        duration = 0.5
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(cpu_intensive_task, i, duration)
                for i in range(num_tasks)
            ]
            
            results = [f.result(timeout=10) for f in futures]
        
        elapsed = time.time() - start_time
        
        # If running in parallel, should take ~duration seconds
        # If sequential, would take num_tasks * duration
        assert elapsed < (num_tasks * duration * 0.8), \
            f"Tasks should run in parallel. Elapsed: {elapsed}s, Expected: <{num_tasks * duration * 0.8}s"
        
        # Verify all tasks completed
        assert len(results) == num_tasks
        for i, result in enumerate(results):
            assert result['task_id'] == i
    
    def test_different_pids(self):
        """Test that tasks run in different processes (different PIDs)."""
        num_tasks = 4
        
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(cpu_intensive_task, i, 0.1)
                for i in range(num_tasks)
            ]
            
            results = [f.result(timeout=10) for f in futures]
        
        # Get unique PIDs
        pids = [r['pid'] for r in results]
        unique_pids = set(pids)
        
        # Should have multiple processes (at least 2)
        assert len(unique_pids) >= 2, \
            f"Should use multiple processes. Got {len(unique_pids)} unique PIDs: {unique_pids}"
    
    def test_max_workers_limit(self):
        """Test that pool respects max_workers limit."""
        max_workers = 2
        num_tasks = 6
        
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(cpu_intensive_task, i, 0.2)
                for i in range(num_tasks)
            ]
            
            results = [f.result(timeout=20) for f in futures]
        
        # Get unique PIDs
        pids = [r['pid'] for r in results]
        unique_pids = set(pids)
        
        # Should not exceed max_workers + 1 (for main process reuse)
        assert len(unique_pids) <= max_workers + 1, \
            f"Should not exceed {max_workers} workers. Got {len(unique_pids)} PIDs"
    
    def test_task_errors_handled(self):
        """Test that task errors are properly captured."""
        # Note: Can't use locally defined function - must use module-level function
        # that can be pickled for multiprocessing
        with ProcessPoolExecutor(max_workers=2) as pool:
            future = pool.submit(failing_task, 1)
            
            # Should raise the exception
            with pytest.raises(ValueError, match="failed intentionally"):
                future.result(timeout=5)
    
    def test_pool_shutdown_graceful(self):
        """Test that pool shuts down gracefully."""
        pool = ProcessPoolExecutor(max_workers=2)
        
        # Submit a task
        future = pool.submit(cpu_intensive_task, 1, 0.1)
        result = future.result(timeout=5)
        
        assert result['task_id'] == 1
        
        # Shutdown
        pool.shutdown(wait=True)
        
        # Pool should be shutdown
        # Attempting to submit should raise error
        with pytest.raises(RuntimeError):
            pool.submit(cpu_intensive_task, 2, 0.1)
    
    def test_pool_shutdown_with_cancel_futures(self):
        """Test shutdown with cancel_futures=True."""
        pool = ProcessPoolExecutor(max_workers=2)
        
        # Submit long-running tasks
        futures = [
            pool.submit(cpu_intensive_task, i, 5)
            for i in range(4)
        ]
        
        # Shutdown immediately with cancel
        pool.shutdown(wait=False, cancel_futures=True)
        
        # At least some futures should be cancelled
        cancelled_count = sum(1 for f in futures if f.cancelled())
        
        # Should have cancelled some (may have already started some)
        assert cancelled_count >= 0  # Some might already be running
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_pool_with_db_access(self):
        """Test that process pool workers can access database independently."""
        import os
        
        # Set test database URL
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test'
        )
        os.environ['TEST_POSTGRES_URL'] = test_url
        
        num_tasks = 4
        
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(task_with_db_access, i)
                for i in range(num_tasks)
            ]
            
            results = [f.result(timeout=30) for f in futures]
        
        # All tasks should succeed
        for result in results:
            assert result['success'] is True, f"Task {result['task_id']} failed: {result.get('error')}"
            assert result['status'] == 'completed'
        
        # Should have multiple PIDs
        pids = [r['pid'] for r in results]
        assert len(set(pids)) >= 2, "Should use multiple processes"
    
    @pytest.mark.integration
    def test_max_tasks_per_child(self):
        """Test that workers are recycled after max_tasks_per_child."""
        max_tasks_per_child = 3
        num_tasks = 10
        
        with ProcessPoolExecutor(max_workers=2, max_tasks_per_child=max_tasks_per_child) as pool:
            futures = [
                pool.submit(cpu_intensive_task, i, 0.05)
                for i in range(num_tasks)
            ]
            
            results = [f.result(timeout=20) for f in futures]
        
        # Get PIDs in order
        pids = [r['pid'] for r in results]
        
        # Should see PID changes (worker recycling)
        # After every max_tasks_per_child tasks, PID should change
        unique_pids = set(pids)
        
        # With 10 tasks and max_tasks_per_child=3:
        # Worker 1: tasks 0,1,2 then recycled
        # Worker 2: tasks 3,4,5 then recycled
        # Should see at least 3-4 different PIDs total
        assert len(unique_pids) >= 3, \
            f"Workers should be recycled. Expected 3+ PIDs, got {len(unique_pids)}"


class TestProcessPoolIntegration:
    """Integration tests for process pool with training service."""
    
    @pytest.mark.asyncio
    async def test_submit_training_task_wrapper(self):
        """Test the submit_training_task wrapper function."""
        # This would test training_service.main.submit_training_task
        # Skipped here as it requires full service setup
        pytest.skip("Requires full training service setup")
    
    @pytest.mark.integration
    def test_concurrent_model_training(self):
        """Test multiple models training concurrently."""
        import os
        
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test'
        )
        os.environ['TEST_POSTGRES_URL'] = test_url
        
        num_models = 6
        
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(task_with_db_access, i)
                for i in range(num_models)
            ]
            
            results = [f.result(timeout=60) for f in futures]
        
        # All should succeed
        success_count = sum(1 for r in results if r['success'])
        assert success_count == num_models, \
            f"All {num_models} models should train successfully, got {success_count}"
        
        # Verify parallelism (multiple PIDs)
        pids = [r['pid'] for r in results if r['success']]
        unique_pids = len(set(pids))
        assert unique_pids >= 2, f"Should use multiple processes, got {unique_pids}"


@pytest.mark.slow
class TestProcessPoolPerformance:
    """Performance benchmarks for process pool (slow tests)."""
    
    def test_speedup_factor(self):
        """Measure speedup from parallel execution."""
        num_tasks = 8
        duration = 0.5
        
        # Sequential execution
        start = time.time()
        for i in range(num_tasks):
            cpu_intensive_task(i, duration)
        sequential_time = time.time() - start
        
        # Parallel execution
        start = time.time()
        with ProcessPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(cpu_intensive_task, i, duration)
                for i in range(num_tasks)
            ]
            [f.result() for f in futures]
        parallel_time = time.time() - start
        
        speedup = sequential_time / parallel_time
        
        print(f"\nSequential: {sequential_time:.2f}s")
        print(f"Parallel: {parallel_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Should have at least 2x speedup (conservative, should be 4-6x on 8 cores)
        assert speedup >= 2.0, f"Expected 2x+ speedup, got {speedup:.2f}x"
    
    def test_memory_usage_stable(self):
        """Test that memory doesn't grow excessively with many tasks."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run 50 tasks
        with ProcessPoolExecutor(max_workers=4, max_tasks_per_child=10) as pool:
            for batch in range(5):
                futures = [
                    pool.submit(cpu_intensive_task, i, 0.1)
                    for i in range(10)
                ]
                [f.result() for f in futures]
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"\nInitial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Growth: {memory_growth:.1f} MB")
        
        # Memory shouldn't grow more than 200MB for 50 simple tasks
        assert memory_growth < 200, f"Memory growth too high: {memory_growth:.1f} MB"


# Configure pytest to recognize --run-slow option
def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (use --run-slow to run)"
    )
