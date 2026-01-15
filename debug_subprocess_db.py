#!/usr/bin/env python3
"""
Debug script to reproduce the exact production subprocess + SyncDBWrapper scenario.
This mimics what happens when training_service/main.py submits a task to ProcessPoolExecutor.
"""
import os
import sys
import uuid
import json
from concurrent.futures import ProcessPoolExecutor

# Set the database URL (same as production)
POSTGRES_URL = os.environ.get(
    'POSTGRES_URL',
    'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory'
)

def worker_task(task_id: int):
    """
    This simulates what train_model_task does in trainer.py.
    Runs in a subprocess created by ProcessPoolExecutor.
    """
    import os
    import sys
    import uuid
    import json
    import traceback
    
    pid = os.getpid()
    result = {
        'task_id': task_id,
        'pid': pid,
        'steps': [],
        'success': False,
        'error': None
    }
    
    try:
        # Step 1: Import the module (same as trainer.py does)
        result['steps'].append('importing sync_db_wrapper...')
        from training_service.sync_db_wrapper import db
        result['steps'].append(f'imported db: {db}')
        result['steps'].append(f'db._postgres_url: {getattr(db, "_postgres_url", "N/A")}')
        
        # Step 2: Try to get a model (read operation)
        result['steps'].append('attempting db.list_models()...')
        try:
            models = db.list_models(limit=5)
            result['steps'].append(f'list_models returned {len(models)} models')
        except Exception as e:
            result['steps'].append(f'list_models FAILED: {e}')
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            return result
        
        # Step 3: Create a model (write operation)
        model_id = str(uuid.uuid4())
        result['steps'].append(f'creating model {model_id}...')
        
        model_data = {
            'id': model_id,
            'algorithm': 'RandomForest',
            'symbol': f'TEST{task_id}',
            'target_col': 'close',
            'feature_cols': json.dumps(['sma_20']),
            'hyperparameters': json.dumps({'n_estimators': 10}),
            'status': 'preprocessing',
            'timeframe': '1m'
        }
        
        try:
            db.create_model_record(model_data)
            result['steps'].append('create_model_record succeeded')
        except Exception as e:
            result['steps'].append(f'create_model_record FAILED: {e}')
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            return result
        
        # Step 4: Update model status (simulates training completion)
        result['steps'].append('updating model status to completed...')
        try:
            db.update_model_status(model_id, status='completed')
            result['steps'].append('update_model_status succeeded')
        except Exception as e:
            result['steps'].append(f'update_model_status FAILED: {e}')
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            return result
        
        # Step 5: Verify the model
        result['steps'].append('verifying model...')
        try:
            model = db.get_model(model_id)
            if model:
                result['steps'].append(f'model status: {model.get("status")}')
            else:
                result['steps'].append('model not found!')
        except Exception as e:
            result['steps'].append(f'get_model FAILED: {e}')
        
        result['success'] = True
        result['model_id'] = model_id
        
    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
    
    return result


def main():
    print("=" * 70)
    print("DEBUG: Subprocess + SyncDBWrapper (Production Scenario)")
    print("=" * 70)
    print(f"\nPOSTGRES_URL: {POSTGRES_URL}")
    print(f"Parent PID: {os.getpid()}")
    
    # Test 1: Single subprocess
    print("\n" + "-" * 50)
    print("Test 1: Single subprocess")
    print("-" * 50)
    
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker_task, 1)
        result = future.result(timeout=30)
    
    print(f"\nPID: {result['pid']}")
    print(f"Success: {result['success']}")
    print("\nSteps:")
    for step in result['steps']:
        print(f"  - {step}")
    
    if result.get('error'):
        print(f"\nError: {result['error']}")
        if result.get('traceback'):
            print(f"\nTraceback:\n{result['traceback']}")
    
    # Test 2: Multiple subprocesses (like production)
    if result['success']:
        print("\n" + "-" * 50)
        print("Test 2: Multiple subprocesses (parallel)")
        print("-" * 50)
        
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker_task, i) for i in range(4)]
            results = [f.result(timeout=60) for f in futures]
        
        for r in results:
            status = "✓" if r['success'] else "✗"
            print(f"  {status} Task {r['task_id']} (PID {r['pid']}): {r['steps'][-1] if r['steps'] else 'no steps'}")
            if r.get('error'):
                print(f"    Error: {r['error']}")
        
        success_count = sum(1 for r in results if r['success'])
        print(f"\nResults: {success_count}/4 succeeded")
    
    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
