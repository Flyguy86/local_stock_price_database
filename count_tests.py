#!/usr/bin/env python3
"""
Count test functions in the test suite.
Provides accurate test counts for reporting.
"""
import ast
from pathlib import Path


def count_tests(file_path):
    """Count test functions in a Python file."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if node.name.startswith('test_'):
                    count += 1
        return count
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def main():
    """Count all tests and print summary."""
    project_root = Path(__file__).parent
    
    # Count tests in each file
    unit_tests = {
        'test_pg_db.py': count_tests(project_root / 'tests/unit/test_pg_db.py'),
        'test_sync_wrapper.py': count_tests(project_root / 'tests/unit/test_sync_wrapper.py'),
        'test_process_pool.py': count_tests(project_root / 'tests/unit/test_process_pool.py'),
    }
    
    api_tests = {
        'test_training_api.py': count_tests(project_root / 'tests/api/test_training_api.py'),
        'test_simulation_api.py': count_tests(project_root / 'tests/api/test_simulation_api.py'),
    }
    
    # Print summary
    print()
    print('📊 TEST COUNT SUMMARY')
    print('=' * 60)
    print()
    print('UNIT TESTS:')
    for name, count in unit_tests.items():
        print(f'  {name:30} {count:3} tests')
    unit_total = sum(unit_tests.values())
    print(f'  {"─" * 30} {"─" * 9}')
    print(f'  {"UNIT TOTAL":30} {unit_total:3} tests')
    print()
    
    print('API TESTS:')
    for name, count in api_tests.items():
        print(f'  {name:30} {count:3} tests')
    api_total = sum(api_tests.values())
    print(f'  {"─" * 30} {"─" * 9}')
    print(f'  {"API TOTAL":30} {api_total:3} tests')
    print()
    
    grand_total = unit_total + api_total
    print('=' * 60)
    print(f'  {"GRAND TOTAL":30} {grand_total:3} tests')
    print('=' * 60)
    print()
    
    # Breakdown by category
    print('TEST CATEGORIES:')
    print(f'  PostgreSQL Database Layer      {unit_tests["test_pg_db.py"]:3} tests')
    print(f'  Sync DB Wrapper                {unit_tests["test_sync_wrapper.py"]:3} tests')
    print(f'  Process Pool Executor          {unit_tests["test_process_pool.py"]:3} tests')
    print(f'  Training Service API           {api_tests["test_training_api.py"]:3} tests')
    print(f'  Simulation Service API         {api_tests["test_simulation_api.py"]:3} tests')
    print()
    
    # Return code (0 if tests found, 1 if none)
    return 0 if grand_total > 0 else 1


if __name__ == "__main__":
    exit(main())
