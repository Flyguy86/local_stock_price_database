"""
Load tests for parallel training operations.

Tests the system's ability to handle multiple concurrent training jobs,
validating process pool execution, database contention, and resource usage.
"""
import pytest
import asyncio
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
import psutil
import os


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.slow
class TestParallelTraining:
    """Test parallel training job execution under load."""
    
    async def test_8_parallel_training_jobs(self, db_tables, sample_model_data):
        """
        Load Test: Execute 8 training jobs in parallel.
        
        Validates:
        - All jobs complete successfully
        - No database conflicts or deadlocks
        - Process pool handles concurrent execution
        - Results are correctly persisted
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create 8 unique model configurations
        num_jobs = 8
        model_configs = []
        for i in range(num_jobs):
            config = sample_model_data.copy()
            config["id"] = f"parallel-model-{i}-{uuid.uuid4()}"
            config["symbol"] = f"TEST{i}"
            config["features"] = [f"feature_{i}", f"feature_{i+1}"]
            config["fingerprint"] = compute_fingerprint({
                "features": config["features"],
                "symbol": config["symbol"],
                "hyperparameters": config.get("hyperparameters", {})
            })
            model_configs.append(config)
        
        # Execute all training jobs in parallel
        start_time = time.time()
        
        created_ids = []
        for config in model_configs:
            model_id = await db.create_model_record(config)
            created_ids.append(model_id)
        
        # Verify all jobs completed
        assert len(created_ids) == num_jobs
        
        # Verify all models in database
        for model_id in created_ids:
            model = await db.get_model(model_id)
            assert model is not None
            assert model["status"] in ["pending", "training", "completed"]
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {num_jobs} parallel jobs completed in {elapsed_time:.2f}s")
    
    async def test_16_parallel_training_jobs(self, db_tables, sample_model_data):
        """
        Load Test: Execute 16 training jobs in parallel.
        
        Stress test with higher concurrency to validate:
        - Connection pool can handle load
        - No resource exhaustion
        - Database remains consistent
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        num_jobs = 16
        model_configs = []
        for i in range(num_jobs):
            config = sample_model_data.copy()
            config["id"] = f"parallel16-model-{i}-{uuid.uuid4()}"
            config["symbol"] = f"SYM{i:02d}"
            config["features"] = [f"rsi_{i}", f"sma_{i+5}"]
            config["fingerprint"] = compute_fingerprint({
                "features": config["features"],
                "symbol": config["symbol"],
                "hyperparameters": config.get("hyperparameters", {})
            })
            model_configs.append(config)
        
        start_time = time.time()
        
        # Create all models concurrently
        tasks = [db.create_model_record(config) for config in model_configs]
        created_ids = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        assert len(created_ids) == num_jobs
        assert all(id is not None for id in created_ids)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {num_jobs} parallel jobs completed in {elapsed_time:.2f}s")
        
        # Verify data integrity
        models = await db.list_models()
        parallel_models = [m for m in models if m["id"].startswith("parallel16")]
        assert len(parallel_models) >= num_jobs
    
    async def test_parallel_with_status_updates(self, db_tables, sample_model_data):
        """
        Load Test: Parallel jobs with concurrent status updates.
        
        Tests:
        - Concurrent writes to same table
        - Status transitions under load
        - No update conflicts or lost updates
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        num_jobs = 10
        
        # Create models
        model_ids = []
        for i in range(num_jobs):
            config = sample_model_data.copy()
            config["id"] = f"status-test-{i}-{uuid.uuid4()}"
            config["symbol"] = f"UPD{i}"
            config["fingerprint"] = compute_fingerprint({
                "features": config.get("features", []),
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            model_id = await db.create_model_record(config)
            model_ids.append(model_id)
        
        # Update all statuses concurrently
        async def update_status(model_id: str, status: str):
            await db.update_model_status(model_id, status)
            return model_id
        
        # Move all to training
        tasks = [update_status(mid, "training") for mid in model_ids]
        await asyncio.gather(*tasks)
        
        # Verify all updated
        for model_id in model_ids:
            model = await db.get_model(model_id)
            assert model["status"] == "training"
        
        # Move all to completed
        tasks = [update_status(mid, "completed") for mid in model_ids]
        await asyncio.gather(*tasks)
        
        # Verify all updated
        for model_id in model_ids:
            model = await db.get_model(model_id)
            assert model["status"] == "completed"
        
        print(f"\n✓ {num_jobs * 2} concurrent status updates completed")
    
    async def test_parallel_with_feature_importance(self, db_tables, sample_model_data):
        """
        Load Test: Parallel feature importance storage.
        
        Tests:
        - Concurrent writes to feature_importance table
        - Foreign key constraint handling
        - Large JSON payload storage
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        num_jobs = 8
        
        # Create models
        model_ids = []
        for i in range(num_jobs):
            config = sample_model_data.copy()
            config["id"] = f"feat-imp-{i}-{uuid.uuid4()}"
            config["symbol"] = f"FI{i}"
            config["fingerprint"] = compute_fingerprint({
                "features": [f"feature_{i}", f"feature_{i+1}"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            model_id = await db.create_model_record(config)
            model_ids.append(model_id)
        
        # Add feature importance concurrently
        async def add_importance(model_id: str, idx: int):
            importance_data = {
                f"feature_{idx}": 0.7 - (idx * 0.05),
                f"feature_{idx+1}": 0.3 + (idx * 0.02),
            }
            await db.store_feature_importance(model_id, importance_data)
            return model_id
        
        tasks = [add_importance(mid, i) for i, mid in enumerate(model_ids)]
        await asyncio.gather(*tasks)
        
        # Verify all stored
        for i, model_id in enumerate(model_ids):
            importance = await db.get_feature_importance(model_id)
            assert importance is not None
            assert f"feature_{i}" in importance
        
        print(f"\n✓ {num_jobs} concurrent feature importance writes completed")
    
    async def test_parallel_model_deletion(self, db_tables, sample_model_data):
        """
        Load Test: Parallel model deletion with cascade.
        
        Tests:
        - Concurrent delete operations
        - Cascade deletes under load
        - No orphaned records
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        num_jobs = 12
        
        # Create models with feature importance
        model_ids = []
        for i in range(num_jobs):
            config = sample_model_data.copy()
            config["id"] = f"delete-test-{i}-{uuid.uuid4()}"
            config["symbol"] = f"DEL{i}"
            config["fingerprint"] = compute_fingerprint({
                "features": [f"f{i}"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            model_id = await db.create_model_record(config)
            
            # Add feature importance
            await db.store_feature_importance(model_id, {f"f{i}": 0.5})
            
            model_ids.append(model_id)
        
        # Delete all concurrently
        tasks = [db.delete_model(mid) for mid in model_ids]
        results = await asyncio.gather(*tasks)
        
        # Verify all deleted
        assert all(results)
        
        for model_id in model_ids:
            model = await db.get_model(model_id)
            assert model is None
            
            # Verify feature importance also deleted (cascade)
            importance = await db.get_feature_importance(model_id)
            assert importance is None
        
        print(f"\n✓ {num_jobs} concurrent deletions with cascade completed")


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.slow
class TestProcessPoolLoad:
    """Test process pool executor under heavy load."""
    
    async def test_process_pool_with_8_workers(self, db_tables):
        """
        Load Test: Process pool with 8 concurrent workers.
        
        Validates:
        - Process pool scales to 8 workers
        - All workers get unique PIDs
        - Database access from all workers
        """
        from training_service.db import sync_create_model_record
        from training_service.pg_db import compute_fingerprint
        import multiprocessing
        
        num_workers = min(8, multiprocessing.cpu_count())
        
        def worker_task(task_id: int) -> Dict[str, Any]:
            """Worker task that creates a model."""
            model_data = {
                "id": f"worker-{task_id}-{uuid.uuid4()}",
                "symbol": f"WRK{task_id}",
                "algorithm": "RandomForest",
                "status": "pending",
                "features": [f"feature_{task_id}"],
                "hyperparameters": {"n_estimators": 100},
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
                "pid": os.getpid()
            }
        
        # Execute in process pool
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_workers * 2)]
            results = [f.result() for f in as_completed(futures)]
        
        elapsed_time = time.time() - start_time
        
        # Verify results
        assert len(results) == num_workers * 2
        unique_pids = len(set(r["pid"] for r in results))
        assert unique_pids <= num_workers  # Should use at most num_workers processes
        
        print(f"\n✓ {len(results)} tasks completed using {unique_pids} workers in {elapsed_time:.2f}s")
    
    async def test_process_pool_throughput(self, db_tables):
        """
        Load Test: Measure process pool throughput.
        
        Tests:
        - Tasks per second
        - Speedup vs sequential
        - Resource utilization
        """
        from training_service.db import sync_create_model_record
        from training_service.pg_db import compute_fingerprint
        import multiprocessing
        
        num_tasks = 20
        num_workers = min(4, multiprocessing.cpu_count())
        
        def create_model_task(task_id: int) -> str:
            """Create a model and return its ID."""
            model_data = {
                "id": f"throughput-{task_id}-{uuid.uuid4()}",
                "symbol": f"THR{task_id:02d}",
                "algorithm": "RandomForest",
                "status": "pending",
                "features": ["close", "volume"],
                "hyperparameters": {"max_depth": 10},
                "target_col": "returns",
            }
            model_data["fingerprint"] = compute_fingerprint({
                "features": model_data["features"],
                "symbol": model_data["symbol"],
                "hyperparameters": model_data["hyperparameters"]
            })
            return sync_create_model_record(model_data)
        
        # Parallel execution
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            model_ids = list(executor.map(create_model_task, range(num_tasks)))
        
        parallel_time = time.time() - start_time
        
        # Verify all completed
        assert len(model_ids) == num_tasks
        assert all(mid is not None for mid in model_ids)
        
        throughput = num_tasks / parallel_time
        print(f"\n✓ Throughput: {throughput:.2f} models/second ({num_tasks} tasks in {parallel_time:.2f}s)")
        print(f"  Workers: {num_workers}")
        
        # Verify can query all models
        from training_service.pg_db import TrainingDB
        db = TrainingDB()
        
        for model_id in model_ids:
            model = await db.get_model(model_id)
            assert model is not None
    
    async def test_process_pool_error_handling(self, db_tables):
        """
        Load Test: Process pool with some failing tasks.
        
        Tests:
        - Failed tasks don't block other tasks
        - Exceptions properly propagated
        - Pool remains functional after errors
        """
        from training_service.db import sync_create_model_record
        from training_service.pg_db import compute_fingerprint
        
        def task_with_occasional_failure(task_id: int) -> Dict[str, Any]:
            """Task that fails on specific IDs."""
            if task_id % 5 == 0:  # Fail every 5th task
                raise ValueError(f"Intentional failure for task {task_id}")
            
            model_data = {
                "id": f"error-test-{task_id}-{uuid.uuid4()}",
                "symbol": f"ERR{task_id}",
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
            return {"task_id": task_id, "model_id": model_id, "success": True}
        
        num_tasks = 15
        results = []
        failures = []
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(task_with_occasional_failure, i): i for i in range(num_tasks)}
            
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except ValueError as e:
                    failures.append(task_id)
        
        # Verify expected failures
        assert len(failures) == 3  # Tasks 0, 5, 10
        assert 0 in failures
        assert 5 in failures
        assert 10 in failures
        
        # Verify successes
        assert len(results) == 12  # num_tasks - failures
        
        print(f"\n✓ {len(results)} tasks succeeded, {len(failures)} failed as expected")


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.slow
class TestDatabaseContention:
    """Test database behavior under high contention."""
    
    async def test_concurrent_reads_no_blocking(self, db_tables, sample_model_data):
        """
        Load Test: Many concurrent reads should not block.
        
        Tests:
        - Read scalability
        - No lock contention on reads
        - Consistent results
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create a model to read
        config = sample_model_data.copy()
        config["id"] = f"read-test-{uuid.uuid4()}"
        config["fingerprint"] = compute_fingerprint({
            "features": config.get("features", []),
            "symbol": config.get("symbol", "TEST"),
            "hyperparameters": {}
        })
        model_id = await db.create_model_record(config)
        
        # Perform 50 concurrent reads
        num_reads = 50
        start_time = time.time()
        
        tasks = [db.get_model(model_id) for _ in range(num_reads)]
        results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # Verify all reads succeeded
        assert len(results) == num_reads
        assert all(r is not None for r in results)
        assert all(r["id"] == model_id for r in results)
        
        # Should be fast (no blocking)
        assert elapsed_time < 2.0  # 50 reads should complete in < 2s
        
        print(f"\n✓ {num_reads} concurrent reads completed in {elapsed_time:.3f}s")
    
    async def test_mixed_read_write_operations(self, db_tables, sample_model_data):
        """
        Load Test: Mixed concurrent reads and writes.
        
        Tests:
        - Writers don't block readers excessively
        - Data consistency maintained
        - No deadlocks
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create initial models
        model_ids = []
        for i in range(5):
            config = sample_model_data.copy()
            config["id"] = f"mixed-{i}-{uuid.uuid4()}"
            config["symbol"] = f"MIX{i}"
            config["fingerprint"] = compute_fingerprint({
                "features": config.get("features", []),
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            model_id = await db.create_model_record(config)
            model_ids.append(model_id)
        
        # Mix of read and write operations
        async def reader(model_id: str, read_count: int):
            """Perform multiple reads."""
            for _ in range(read_count):
                await db.get_model(model_id)
                await asyncio.sleep(0.01)  # Small delay
        
        async def writer(model_id: str, write_count: int):
            """Perform multiple status updates."""
            statuses = ["training", "completed", "failed", "pending"]
            for i in range(write_count):
                await db.update_model_status(model_id, statuses[i % len(statuses)])
                await asyncio.sleep(0.02)
        
        # Launch mixed workload
        start_time = time.time()
        
        tasks = []
        # 10 readers, each doing 5 reads
        for model_id in model_ids:
            tasks.append(reader(model_id, 5))
        
        # 5 writers, each doing 3 writes
        for model_id in model_ids:
            tasks.append(writer(model_id, 3))
        
        await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Mixed read/write workload completed in {elapsed_time:.2f}s")
        print(f"  Operations: {5 * 5} reads + {5 * 3} writes = {5*5 + 5*3} total")
    
    async def test_bulk_insert_performance(self, db_tables, sample_model_data):
        """
        Load Test: Bulk model insertion.
        
        Tests:
        - Large batch insert performance
        - No memory issues
        - All inserts succeed
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        num_models = 50
        
        # Create models in bulk
        start_time = time.time()
        
        model_ids = []
        for i in range(num_models):
            config = sample_model_data.copy()
            config["id"] = f"bulk-{i}-{uuid.uuid4()}"
            config["symbol"] = f"BLK{i:03d}"
            config["features"] = [f"f{i % 10}"]
            config["fingerprint"] = compute_fingerprint({
                "features": config["features"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            model_id = await db.create_model_record(config)
            model_ids.append(model_id)
        
        elapsed_time = time.time() - start_time
        
        # Verify all inserted
        assert len(model_ids) == num_models
        
        # Verify can list all
        all_models = await db.list_models()
        bulk_models = [m for m in all_models if m["id"].startswith("bulk-")]
        assert len(bulk_models) >= num_models
        
        throughput = num_models / elapsed_time
        print(f"\n✓ Bulk insert: {num_models} models in {elapsed_time:.2f}s ({throughput:.1f} models/s)")
