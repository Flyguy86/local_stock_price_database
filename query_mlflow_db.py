#!/usr/bin/env python3
"""
Direct SQLite query of MLflow database to see what's actually stored.
"""
import sqlite3
from pathlib import Path

db_path = Path("./data/mlflow/backend/mlflow.db")

if not db_path.exists():
    print(f"❌ Database not found: {db_path}")
    exit(1)

print(f"📊 Querying MLflow database: {db_path}")
print(f"   Size: {db_path.stat().st_size / 1024:.1f} KB")
print()

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Check tables
print("📋 Tables in database:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = cursor.fetchall()
for table in tables:
    print(f"   - {table[0]}")
print()

# Check registered models
print("🏷️  Registered Models:")
cursor.execute("SELECT name, creation_time, last_updated_time, description FROM registered_models;")
models = cursor.fetchall()
print(f"   Count: {len(models)}")
for name, created, updated, desc in models[:5]:
    print(f"   - {name} (created: {created})")
if len(models) > 5:
    print(f"   ... and {len(models) - 5} more")
print()

# Check model versions
print("📦 Model Versions:")
cursor.execute("SELECT name, version, run_id, current_stage FROM model_versions ORDER BY name, version;")
versions = cursor.fetchall()
print(f"   Count: {len(versions)}")
for name, version, run_id, stage in versions[:10]:
    print(f"   - {name} v{version} (run: {run_id[:8]}..., stage: {stage})")
if len(versions) > 10:
    print(f"   ... and {len(versions) - 10} more")
print()

# Check experiments
print("🧪 Experiments:")
cursor.execute("SELECT experiment_id, name FROM experiments;")
experiments = cursor.fetchall()
print(f"   Count: {len(experiments)}")
for exp_id, name in experiments[:5]:
    print(f"   - [{exp_id}] {name}")
print()

# Check runs
print("🏃 Runs:")
cursor.execute("SELECT run_uuid, experiment_id, name, status FROM runs LIMIT 10;")
runs = cursor.fetchall()
print(f"   Showing first 10:")
for run_id, exp_id, name, status in runs:
    print(f"   - {run_id[:8]}... exp={exp_id} name={name} status={status}")

cursor.execute("SELECT COUNT(*) FROM runs;")
total_runs = cursor.fetchone()[0]
print(f"   Total runs: {total_runs}")

conn.close()

print()
print("✅ Database query complete")
