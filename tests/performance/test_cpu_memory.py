"""
Performance tests for CPU and memory utilization.

Monitors system resources during heavy workloads to validate:
- CPU utilization is efficient
- Memory usage stays within bounds
- No memory leaks
- Process pool CPU distribution
"""
import pytest
import asyncio
import time
import uuid
import psutil
import os
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.slow
class TestCPUUtilization:
    """Test CPU utilization during training and simulation."""
    
    async def test_parallel_training_cpu_usage(self, db_tables, sample_model_data):
        """
        Performance Test: Monitor CPU during parallel training.
        
        Validates:
        - CPU utilization increases with parallel jobs
        - Multi-core utilization
        - No single-threaded bottleneck
        """
        from training_service.db import sync_create_model_record
        from training_service.pg_db import compute_fingerprint
        import multiprocessing
        
        process = psutil.Process()
        
        # Get baseline CPU
        baseline_cpu = process.cpu_percent(interval=0.1)
        
        num_workers = min(4, multiprocessing.cpu_count())
        num_tasks = num_workers * 4
        
        def cpu_intensive_task(task_id: int) -> Dict:
            """Task that uses CPU."""
            model_data = {
                "id": f"cpu-test-{task_id}-{uuid.uuid4()}",
                "symbol": f"CPU{task_id}",
                "algorithm": "RandomForest",
                "status": "pending",
                "features": ["close", "volume", "high", "low"],
                "hyperparameters": {"n_estimators": 100, "max_depth": 10},
                "target_col": "returns",
            }
            model_data["fingerprint"] = compute_fingerprint({
                "features": model_data["features"],
                "symbol": model_data["symbol"],
                "hyperparameters": model_data["hyperparameters"]
            })
            
            # Simulate some CPU work
            result = sum(i * i for i in range(10000))
            
            model_id = sync_create_model_record(model_data)
            return {"model_id": model_id, "task_id": task_id, "result": result}
        
        # Monitor CPU during execution
        cpu_samples = []
        
        async def monitor_cpu():
            """Sample CPU usage."""
            while True:
                cpu_samples.append(process.cpu_percent(interval=0.1))
                await asyncio.sleep(0.2)
        
        # Start monitoring
        monitor_task = asyncio.create_task(monitor_cpu())
        
        # Execute workload
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(cpu_intensive_task, range(num_tasks)))
        
        elapsed_time = time.time() - start_time
        
        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Analyze CPU usage
        if cpu_samples:
            max_cpu = max(cpu_samples)
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            
            print(f"\n✓ CPU Utilization:")
            print(f"  Baseline: {baseline_cpu:.1f}%")
            print(f"  Peak: {max_cpu:.1f}%")
            print(f"  Average: {avg_cpu:.1f}%")
            print(f"  Workers: {num_workers}")
            print(f"  Tasks: {num_tasks}")
            print(f"  Duration: {elapsed_time:.2f}s")
            
            # CPU should increase during parallel work
            assert avg_cpu > baseline_cpu or num_workers == 1
        
        # Verify all tasks completed
        assert len(results) == num_tasks
    
    async def test_database_query_cpu_efficiency(self, db_tables, sample_model_data):
        """
        Performance Test: CPU usage during database operations.
        
        Validates:
        - Database queries are not CPU-bound
        - Efficient query execution
        - Low CPU for I/O operations
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        process = psutil.Process()
        db = TrainingDB()
        
        # Create test data
        model_ids = []
        for i in range(20):
            config = sample_model_data.copy()
            config["id"] = f"cpu-query-{i}-{uuid.uuid4()}"
            config["symbol"] = f"CQ{i}"
            config["fingerprint"] = compute_fingerprint({
                "features": [f"f{i}"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            model_id = await db.create_model_record(config)
            model_ids.append(model_id)
        
        # Monitor CPU during query-heavy workload
        cpu_samples = []
        
        async def query_workload():
            """Execute many queries."""
            for _ in range(100):
                model_id = model_ids[_ % len(model_ids)]
                await db.get_model(model_id)
                await db.list_models(limit=10)
        
        # Measure CPU
        start_cpu = process.cpu_percent(interval=0.1)
        start_time = time.time()
        
        await query_workload()
        
        end_time = time.time()
        end_cpu = process.cpu_percent(interval=0.5)
        
        elapsed_time = end_time - start_time
        
        print(f"\n✓ Query CPU Efficiency:")
        print(f"  CPU before: {start_cpu:.1f}%")
        print(f"  CPU during: {end_cpu:.1f}%")
        print(f"  Queries: 200 (100 get + 100 list)")
        print(f"  Duration: {elapsed_time:.2f}s")
        print(f"  QPS: {200/elapsed_time:.1f} queries/second")
        
        # Database queries should be I/O bound, not CPU bound
        # CPU usage should be moderate
        assert end_cpu < 80  # Should not max out CPU
    
    async def test_process_pool_cpu_distribution(self, db_tables):
        """
        Performance Test: Verify CPU usage distributed across workers.
        
        Validates:
        - Work distributed to multiple cores
        - No single-process bottleneck
        - Efficient parallelization
        """
        from training_service.db import sync_create_model_record
        from training_service.pg_db import compute_fingerprint
        import multiprocessing
        
        num_workers = min(4, multiprocessing.cpu_count())
        
        # Get system-wide CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent_per_core_before = psutil.cpu_percent(interval=0.5, percpu=True)
        
        def worker_task(task_id: int) -> Dict:
            """CPU-intensive task."""
            # Get worker's PID and CPU affinity
            pid = os.getpid()
            proc = psutil.Process(pid)
            
            # Do some CPU work
            result = 0
            for i in range(500000):
                result += i * i
            
            # Create model
            model_data = {
                "id": f"dist-{task_id}-{uuid.uuid4()}",
                "symbol": f"DST{task_id}",
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
            
            return {
                "task_id": task_id,
                "model_id": model_id,
                "pid": pid,
                "result": result
            }
        
        # Execute parallel workload
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker_task, range(num_workers * 2)))
        
        elapsed_time = time.time() - start_time
        
        # Get CPU after
        cpu_percent_per_core_after = psutil.cpu_percent(interval=0.5, percpu=True)
        
        # Analyze results
        unique_pids = len(set(r["pid"] for r in results))
        
        print(f"\n✓ CPU Distribution:")
        print(f"  CPU cores: {cpu_count}")
        print(f"  Workers: {num_workers}")
        print(f"  Unique PIDs: {unique_pids}")
        print(f"  Tasks: {len(results)}")
        print(f"  Duration: {elapsed_time:.2f}s")
        
        # Should use multiple processes
        assert unique_pids > 1 or num_workers == 1


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.slow
class TestMemoryUsage:
    """Test memory usage and leak detection."""
    
    async def test_memory_usage_during_bulk_insert(self, db_tables, sample_model_data):
        """
        Performance Test: Monitor memory during bulk insert.
        
        Validates:
        - Memory usage stays reasonable
        - No excessive memory growth
        - Efficient data handling
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        process = psutil.Process()
        db = TrainingDB()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        num_models = 100
        memory_samples = []
        
        # Insert models and track memory
        for i in range(num_models):
            config = sample_model_data.copy()
            config["id"] = f"mem-test-{i}-{uuid.uuid4()}"
            config["symbol"] = f"MEM{i:03d}"
            config["features"] = [f"f{j}" for j in range(10)]  # Some data
            config["fingerprint"] = compute_fingerprint({
                "features": config["features"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            
            await db.create_model_record(config)
            
            # Sample memory every 10 inserts
            if i % 10 == 0:
                mem_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(mem_mb)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - baseline_memory
        
        print(f"\n✓ Memory Usage:")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")
        print(f"  Records inserted: {num_models}")
        print(f"  Per record: {memory_growth/num_models*1000:.1f} KB")
        
        # Memory growth should be reasonable (< 100MB for 100 records)
        assert memory_growth < 100
    
    async def test_memory_leak_detection(self, db_tables, sample_model_data):
        """
        Performance Test: Detect memory leaks in repeated operations.
        
        Validates:
        - No memory leaks in create/delete cycle
        - Memory returns to baseline
        - Stable memory usage over time
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        import gc
        
        process = psutil.Process()
        db = TrainingDB()
        
        # Force garbage collection
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        memory_samples = []
        num_cycles = 20
        
        for cycle in range(num_cycles):
            # Create 5 models
            model_ids = []
            for i in range(5):
                config = sample_model_data.copy()
                config["id"] = f"leak-test-{cycle}-{i}-{uuid.uuid4()}"
                config["symbol"] = f"LK{cycle}{i}"
                config["fingerprint"] = compute_fingerprint({
                    "features": ["test"],
                    "symbol": config["symbol"],
                    "hyperparameters": {}
                })
                model_id = await db.create_model_record(config)
                model_ids.append(model_id)
            
            # Delete them
            for model_id in model_ids:
                await db.delete_model(model_id)
            
            # Force GC and sample memory
            gc.collect()
            mem_mb = process.memory_info().rss / 1024 / 1024
            memory_samples.append(mem_mb)
        
        final_memory = memory_samples[-1]
        memory_growth = final_memory - baseline_memory
        
        # Check for leak (memory should be stable)
        first_half_avg = sum(memory_samples[:10]) / 10
        second_half_avg = sum(memory_samples[10:]) / 10
        growth_rate = second_half_avg - first_half_avg
        
        print(f"\n✓ Memory Leak Detection:")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Total growth: {memory_growth:.1f} MB")
        print(f"  Growth rate: {growth_rate:.1f} MB")
        print(f"  Cycles: {num_cycles} (100 records created/deleted)")
        
        # Growth rate should be small (< 10MB over cycles)
        assert abs(growth_rate) < 10
    
    async def test_connection_pool_memory_usage(self, db_tables):
        """
        Performance Test: Monitor connection pool memory.
        
        Validates:
        - Connection pool memory is bounded
        - Connections don't leak memory
        - Pool size impacts memory reasonably
        """
        from training_service.pg_db import get_pool
        import gc
        
        process = psutil.Process()
        pool = await get_pool()
        
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Acquire and release connections many times
        num_cycles = 50
        
        for _ in range(num_cycles):
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - baseline_memory
        
        print(f"\n✓ Connection Pool Memory:")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  After {num_cycles} cycles: {final_memory:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")
        print(f"  Pool size: {pool.get_size()}/{pool.get_max_size()}")
        
        # Connection cycling shouldn't cause significant memory growth
        assert memory_growth < 20
    
    async def test_large_result_set_memory(self, db_tables, sample_model_data):
        """
        Performance Test: Memory usage with large result sets.
        
        Validates:
        - Large queries don't cause memory issues
        - Pagination helps memory usage
        - Results released after use
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        import gc
        
        process = psutil.Process()
        db = TrainingDB()
        
        # Create many models
        num_models = 100
        for i in range(num_models):
            config = sample_model_data.copy()
            config["id"] = f"large-result-{i}-{uuid.uuid4()}"
            config["symbol"] = f"LR{i:03d}"
            config["fingerprint"] = compute_fingerprint({
                "features": [f"f{i}"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            await db.create_model_record(config)
        
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Query all at once
        all_models = await db.list_models(limit=1000)
        
        mem_after_query = process.memory_info().rss / 1024 / 1024
        query_memory = mem_after_query - baseline_memory
        
        # Release results
        del all_models
        gc.collect()
        
        mem_after_release = process.memory_info().rss / 1024 / 1024
        
        print(f"\n✓ Large Result Set Memory:")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  After query: {mem_after_query:.1f} MB (+{query_memory:.1f} MB)")
        print(f"  After release: {mem_after_release:.1f} MB")
        print(f"  Records: {num_models}")
        
        # Memory should be released after results deleted
        memory_retained = mem_after_release - baseline_memory
        assert memory_retained < query_memory * 0.5  # At least 50% released


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.slow
class TestQueryPerformance:
    """Test database query performance."""
    
    async def test_query_response_time(self, db_tables, sample_model_data):
        """
        Performance Test: Measure query response times.
        
        Validates:
        - Queries complete quickly
        - Response times are consistent
        - No performance degradation
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create test data
        model_ids = []
        for i in range(50):
            config = sample_model_data.copy()
            config["id"] = f"perf-{i}-{uuid.uuid4()}"
            config["symbol"] = f"PRF{i}"
            config["fingerprint"] = compute_fingerprint({
                "features": [f"f{i}"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            model_id = await db.create_model_record(config)
            model_ids.append(model_id)
        
        # Measure query times
        response_times = []
        
        for model_id in model_ids:
            start = time.time()
            await db.get_model(model_id)
            elapsed = time.time() - start
            response_times.append(elapsed * 1000)  # Convert to ms
        
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        p95_time = sorted(response_times)[int(len(response_times) * 0.95)]
        
        print(f"\n✓ Query Response Times:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Min: {min_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")
        print(f"  P95: {p95_time:.2f}ms")
        print(f"  Queries: {len(response_times)}")
        
        # Queries should be fast
        assert avg_time < 50  # Average < 50ms
        assert p95_time < 100  # 95th percentile < 100ms
    
    async def test_bulk_query_throughput(self, db_tables, sample_model_data):
        """
        Performance Test: Measure bulk query throughput.
        
        Validates:
        - High queries per second
        - Sustained performance
        - No bottlenecks
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create test data
        for i in range(30):
            config = sample_model_data.copy()
            config["id"] = f"throughput-{i}-{uuid.uuid4()}"
            config["symbol"] = f"TP{i}"
            config["fingerprint"] = compute_fingerprint({
                "features": ["close"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            await db.create_model_record(config)
        
        # Execute many queries
        num_queries = 200
        
        start_time = time.time()
        
        for _ in range(num_queries):
            await db.list_models(limit=10)
        
        elapsed_time = time.time() - start_time
        
        qps = num_queries / elapsed_time
        
        print(f"\n✓ Query Throughput:")
        print(f"  Queries: {num_queries}")
        print(f"  Duration: {elapsed_time:.2f}s")
        print(f"  QPS: {qps:.1f} queries/second")
        print(f"  Avg latency: {1000/qps:.2f}ms")
        
        # Should handle at least 50 QPS
        assert qps > 50
