#!/usr/bin/env python3
"""
Quick test runner for unit tests.
Validates that all unit tests can be imported and basic setup works.
"""
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_imports():
    """Verify all test modules can be imported."""
    print("=" * 60)
    print("Checking Test Imports")
    print("=" * 60)
    print()
    
    try:
        print("✓ Importing conftest...")
        from tests import conftest
        
        print("✓ Importing test_pg_db...")
        from tests.unit import test_pg_db
        
        print("✓ Importing test_sync_wrapper...")
        from tests.unit import test_sync_wrapper
        
        print("✓ Importing test_process_pool...")
        from tests.unit import test_process_pool
        
        print()
        print("✓ All test modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        return False


def check_services():
    """Check if required services can be imported."""
    print()
    print("=" * 60)
    print("Checking Service Modules")
    print("=" * 60)
    print()
    
    try:
        print("✓ Importing training_service.pg_db...")
        from training_service import pg_db
        
        print("✓ Importing training_service.sync_db_wrapper...")
        from training_service import sync_db_wrapper
        
        print("✓ Importing simulation_service.pg_db...")
        from simulation_service import pg_db as sim_pg_db
        
        print()
        print("✓ All service modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        print("\nMake sure you're running from the project root directory.")
        return False


def check_dependencies():
    """Check if test dependencies are installed."""
    print()
    print("=" * 60)
    print("Checking Test Dependencies")
    print("=" * 60)
    print()
    
    dependencies = {
        'pytest': 'pytest',
        'pytest-asyncio': 'pytest_asyncio',
        'pytest-cov': 'pytest_cov',
        'httpx': 'httpx',
        'aiohttp': 'aiohttp',
        'psutil': 'psutil',
        'asyncpg': 'asyncpg'
    }
    
    missing = []
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (not installed)")
            missing.append(name)
    
    if missing:
        print()
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print()
    print("✓ All dependencies installed!")
    return True


def main():
    """Run all checks."""
    print()
    print("=" * 60)
    print("Unit Test Validation")
    print("=" * 60)
    print()
    
    results = {}
    
    # Check dependencies first
    results['dependencies'] = check_dependencies()
    
    # Check service imports
    results['services'] = check_services()
    
    # Check test imports
    results['tests'] = check_imports()
    
    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("✅ All checks passed! Ready to run tests.")
        print()
        print("Run tests with:")
        print("  pytest tests/unit/ -v")
        print()
        print("Or use the test runner:")
        print("  bash run_unit_tests.sh")
        return 0
    else:
        print("⚠️  Some checks failed. Fix issues above before running tests.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
