#!/usr/bin/env python3
"""Count validation tests in the test suite."""
import ast
from pathlib import Path


def count_test_methods(file_path):
    """Count test methods in a file."""
    with open(file_path) as f:
        tree = ast.parse(f.read())
    
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith('test_'):
                count += 1
    return count


def main():
    """Count all validation tests."""
    test_dir = Path(__file__).parent / "validation"
    
    print("=" * 70)
    print("VALIDATION TEST COUNT")
    print("=" * 70)
    print()
    
    total = 0
    
    # Fingerprint deduplication tests
    fingerprint_file = test_dir / "test_fingerprint_deduplication.py"
    if fingerprint_file.exists():
        fingerprint_count = count_test_methods(fingerprint_file)
        print(f"Fingerprint Deduplication Tests:    {fingerprint_count:3d} tests")
        total += fingerprint_count
    
    # Connection pool limit tests
    pool_file = test_dir / "test_connection_pool_limits.py"
    if pool_file.exists():
        pool_count = count_test_methods(pool_file)
        print(f"Connection Pool Limit Tests:        {pool_count:3d} tests")
        total += pool_count
    
    print()
    print("=" * 70)
    print(f"TOTAL VALIDATION TESTS:             {total:3d} tests")
    print("=" * 70)
    print()
    
    # Test categories
    print("Test Categories:")
    print("  • Fingerprint Generation (6 tests)")
    print("  • Fingerprint Deduplication (5 tests)")
    print("  • Fingerprint Edge Cases (3 tests)")
    print("  • Fingerprint Integration (2 tests)")
    print("  • Connection Pool Limits (5 tests)")
    print("  • Connection Pool Isolation (2 tests)")
    print("  • Connection Pool Cleanup (2 tests)")
    print("  • Connection Pool Errors (2 tests)")
    print("  • Connection Pool Performance (2 tests, slow)")
    print("  • Process Pool Connections (1 test)")
    print("  • Connection Pool Recovery (1 test)")
    print()
    
    # Quick run commands
    print("Quick Run Commands:")
    print("  ./run_unit_tests.sh validation              # Run all validation tests")
    print("  pytest tests/validation/ -v                 # Verbose output")
    print("  pytest tests/validation/ -m 'not slow'      # Skip slow tests")
    print("  pytest tests/validation/test_fingerprint_deduplication.py -v")
    print("  pytest tests/validation/test_connection_pool_limits.py -v")
    print()


if __name__ == "__main__":
    main()
