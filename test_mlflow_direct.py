#!/usr/bin/env python3
"""
Direct test of MLflow API to verify models are registered.
"""
import requests
import json

mlflow_url = "http://localhost:5000"

print("Testing MLflow API directly...")
print(f"MLflow URL: {mlflow_url}\n")

# Test 1: Check if MLflow is running
try:
    response = requests.get(f"{mlflow_url}/health")
    print(f"✅ MLflow health check: {response.status_code}")
except Exception as e:
    print(f"❌ MLflow health check failed: {e}")
    exit(1)

# Test 2: List registered models via REST API
try:
    response = requests.get(f"{mlflow_url}/api/2.0/mlflow/registered-models/list")
    data = response.json()
    print(f"\n📋 Registered Models (REST API):")
    print(f"   Status: {response.status_code}")
    
    if "registered_models" in data:
        models = data["registered_models"]
        print(f"   Count: {len(models)}")
        for model in models:
            print(f"   - {model.get('name')}")
    else:
        print(f"   Response: {json.dumps(data, indent=2)}")
except Exception as e:
    print(f"❌ Failed to list models: {e}")

# Test 3: Use MLflow Python client
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    
    mlflow.set_tracking_uri(mlflow_url)
    client = MlflowClient(mlflow_url)
    
    print(f"\n🐍 Using MLflow Python Client:")
    registered = list(client.search_registered_models())
    print(f"   Registered model names: {len(registered)}")
    
    for rm in registered:
        print(f"\n   Model: {rm.name}")
        versions = list(client.search_model_versions(f"name='{rm.name}'"))
        print(f"   Versions: {len(versions)}")
        for v in versions[:3]:  # Show first 3
            print(f"      v{v.version} - stage: {v.current_stage}, run: {v.run_id[:8]}")
        if len(versions) > 3:
            print(f"      ... and {len(versions) - 3} more")
            
except Exception as e:
    print(f"❌ Python client failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test complete")
