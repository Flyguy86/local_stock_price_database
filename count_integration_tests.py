#!/usr/bin/env python3
"""
Count integration test functions.
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
    """Count integration tests."""
    project_root = Path(__file__).parent
    
    integration_tests = {
        'test_training_workflow.py': count_tests(project_root / 'tests/integration/test_training_workflow.py'),
        'test_simulation_workflow.py': count_tests(project_root / 'tests/integration/test_simulation_workflow.py'),
    }
    
    print()
    print('📊 INTEGRATION TEST COUNT')
    print('=' * 60)
    print()
    for name, count in integration_tests.items():
        print(f'  {name:35} {count:3} tests')
    
    total = sum(integration_tests.values())
    print(f'  {"─" * 35} {"─" * 9}')
    print(f'  {"TOTAL":35} {total:3} tests')
    print('=' * 60)
    print()
    
    return 0 if total > 0 else 1


if __name__ == "__main__":
    exit(main())
