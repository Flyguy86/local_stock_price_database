#!/bin/bash
# Quick test runner for cohort relationship tests

echo "=============================================================="
echo "COHORT RELATIONSHIP TESTS"
echo "=============================================================="
echo ""

# Unit tests
echo "Running unit tests..."
python3 run_cohort_tests.py --unit

# Integration tests (if database available)
echo ""
echo "Running integration tests..."
python3 run_cohort_tests.py --integration

echo ""
echo "=============================================================="
echo "Test run complete!"
echo "=============================================================="
