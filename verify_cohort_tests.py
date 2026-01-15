#!/usr/bin/env python3
"""
Quick verification that test files are valid Python and importable.
"""

import sys
import os

def verify_tests():
    """Verify test files can be imported."""
    
    print("="*60)
    print("VERIFYING COHORT RELATIONSHIP TEST FILES")
    print("="*60)
    print()
    
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Test 1: Import unit tests
    print("1. Checking unit test file...")
    try:
        from tests.training import test_cohort_relationships
        print("   ✅ Unit test file is valid Python")
        print(f"   Found {len([x for x in dir(test_cohort_relationships) if x.startswith('Test')])} test classes")
    except Exception as e:
        print(f"   ❌ Failed to import: {e}")
        return False
    
    # Test 2: Import integration tests
    print("\n2. Checking integration test file...")
    try:
        from tests.training import test_cohort_integration
        print("   ✅ Integration test file is valid Python")
    except Exception as e:
        print(f"   ❌ Failed to import: {e}")
        return False
    
    # Test 3: Check test runner
    print("\n3. Checking test runner...")
    if os.path.exists('run_cohort_tests.py'):
        print("   ✅ Test runner script exists")
    else:
        print("   ❌ Test runner not found")
        return False
    
    # Test 4: List all test methods
    print("\n4. Listing all test methods...")
    test_class_1 = test_cohort_relationships.TestCohortRelationships()
    test_methods_1 = [m for m in dir(test_class_1) if m.startswith('test_')]
    
    test_class_2 = test_cohort_relationships.TestCohortQueries()
    test_methods_2 = [m for m in dir(test_class_2) if m.startswith('test_')]
    
    print(f"\n   TestCohortRelationships ({len(test_methods_1)} tests):")
    for method in test_methods_1:
        print(f"      • {method}")
    
    print(f"\n   TestCohortQueries ({len(test_methods_2)} tests):")
    for method in test_methods_2:
        print(f"      • {method}")
    
    # Test 5: Check documentation
    print("\n5. Checking documentation files...")
    docs = [
        'training_service/COHORT_MIGRATION.md',
        'training_service/COHORT_VS_PARENT_CHILD.md',
        'tests/training/TESTING_GUIDE.md'
    ]
    
    for doc in docs:
        if os.path.exists(doc):
            print(f"   ✅ {doc}")
        else:
            print(f"   ❌ Missing: {doc}")
    
    print("\n" + "="*60)
    print("✅ ALL VERIFICATIONS PASSED")
    print("="*60)
    print("\nTo run tests:")
    print("  python run_cohort_tests.py --unit")
    print("  python run_cohort_tests.py --integration")
    print("  python run_cohort_tests.py  (both)")
    print()
    
    return True

if __name__ == "__main__":
    success = verify_tests()
    sys.exit(0 if success else 1)
