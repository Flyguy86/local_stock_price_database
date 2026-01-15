#!/usr/bin/env python3
"""
Run all cohort relationship tests.

Usage:
    python run_cohort_tests.py              # Run all tests
    python run_cohort_tests.py --unit       # Unit tests only
    python run_cohort_tests.py --integration # Integration tests only
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run cohort relationship tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    args = parser.parse_args()
    
    # Default: run both
    run_unit = args.unit or (not args.integration)
    run_integration = args.integration or (not args.unit)
    
    exit_code = 0
    
    if run_unit:
        print("\n" + "="*70)
        print("RUNNING UNIT TESTS")
        print("="*70)
        try:
            from tests.training.test_cohort_relationships import run_all_tests
            run_all_tests()
        except Exception as e:
            print(f"\n❌ Unit tests failed: {e}")
            import traceback
            traceback.print_exc()
            exit_code = 1
    
    if run_integration:
        print("\n" + "="*70)
        print("RUNNING INTEGRATION TESTS")
        print("="*70)
        try:
            from tests.training.test_cohort_integration import run_integration_tests
            run_integration_tests()
        except ImportError as e:
            print(f"\n⚠️  Skipping integration tests: {e}")
        except Exception as e:
            print(f"\n❌ Integration tests failed: {e}")
            import traceback
            traceback.print_exc()
            exit_code = 1
    
    if exit_code == 0:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("="*70 + "\n")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
