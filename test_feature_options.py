#!/usr/bin/env python3
"""Test script to check feature_service /options endpoint directly."""
import requests

# Test feature service directly
print("Testing feature_builder:8100/options...")
try:
    response = requests.get("http://localhost:8100/options", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60 + "\n")

# Test through orchestrator proxy
print("Testing orchestrator:8003/api/features/options...")
try:
    response = requests.get("http://localhost:8003/api/features/options", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
