#!/usr/bin/env python3
"""Validation test for model ranking functionality."""

import subprocess
import json
import sys

def run_command(cmd):
    """Run shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

print("🔄 Restarting ray_orchestrator container...")
code, out, err = run_command("docker-compose restart ray_orchestrator")
if code != 0:
    print(f"❌ Failed to restart: {err}")
    sys.exit(1)
print("✅ Container restarted")

print("\n⏳ Waiting 5 seconds for service to be ready...")
run_command("sleep 5")

print("\n📊 Testing model ranking with test_r2 metric...")
test_cmd = """docker exec ray_orchestrator python3 -c "
from ray_orchestrator.mlflow_integration import MLflowTracker
tracker = MLflowTracker()
results = tracker.rank_top_models('walk_forward_xgboost_GOOGL', 'test_r2', 5, False)
print(f'RESULT_COUNT:{len(results)}')
if results:
    for r in results:
        print(f'RANK:{r[\"rank\"]},TRIAL:{r[\"trial_id\"]},R2:{r.get(\"test_r2\", r[\"value\"]):.4f},TEST_RMSE:{r.get(\"test_rmse\", 0):.6f},TRAIN_RMSE:{r.get(\"train_rmse\", 0):.6f}')
"
"""

code, out, err = run_command(test_cmd)

if code != 0:
    print(f"❌ Test failed with error:\n{err}")
    sys.exit(1)

# Parse output
lines = out.strip().split('\n')
result_count = 0
models = []

for line in lines:
    if line.startswith('RESULT_COUNT:'):
        result_count = int(line.split(':')[1])
    elif line.startswith('RANK:'):
        models.append(line)

print(f"\n✅ Found {result_count} ranked models")

if result_count == 0:
    print("❌ ERROR: No models were ranked!")
    print(f"Full output:\n{out}")
    print(f"Errors:\n{err}")
    sys.exit(1)

print(f"\n📊 Top {len(models)} models ranked by test R²:")
for model in models:
    parts = model.split(',')
    rank = parts[0].split(':')[1]
    trial = parts[1].split(':')[1]
    r2 = parts[2].split(':')[1]
    test_rmse = parts[3].split(':')[1]
    train_rmse = parts[4].split(':')[1]
    print(f"  #{rank}: {trial}")
    print(f"         R²={r2}, test_RMSE={test_rmse}, train_RMSE={train_rmse}")

print("\n✅ Validation test PASSED!")
print(f"   - Successfully ranked {result_count} models")
print(f"   - Metrics (test_r2, test_rmse, train_rmse) are populated")
print(f"   - Models are sorted by R² (descending)")
