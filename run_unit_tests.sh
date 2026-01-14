#!/bin/bash
# Run unit and API tests for PostgreSQL migration

set -e

echo "======================================"
echo "Test Runner"
echo "======================================"
echo ""

# Parse test type argument
TEST_TYPE="${1:-all}"
if [[ "$TEST_TYPE" == "--coverage" ]] || [[ "$TEST_TYPE" == "-c" ]]; then
    TEST_TYPE="all"
    RUN_COVERAGE=true
elif [[ "$TEST_TYPE" == "--verbose" ]] || [[ "$TEST_TYPE" == "-v" ]]; then
    TEST_TYPE="all"
    VERBOSE=true
fi

# Check if PostgreSQL is running
echo "Checking PostgreSQL..."
if ! docker compose ps postgres | grep -q "Up"; then
    echo "⚠️  PostgreSQL not running. Starting..."
    docker compose up postgres -d
    sleep 3
fi
echo "✓ PostgreSQL is running"
echo ""

# Install test dependencies
echo "Installing test dependencies..."
pip install -q pytest pytest-asyncio pytest-cov httpx aiohttp psutil pytest-mock faker 2>/dev/null || true
echo "✓ Dependencies installed"
echo ""

# Set test database URL
export TEST_POSTGRES_URL="postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory_test"

# Determine which tests to run
case "$TEST_TYPE" in
    unit)
        TEST_PATH="tests/unit/"
        echo "Running UNIT tests only"
        ;;
    api)
        TEST_PATH="tests/api/"
        echo "Running API tests only"
        ;;
    integration)
        TEST_PATH="tests/integration/"
        echo "Running INTEGRATION tests only"
        ;;
    validation)
        TEST_PATH="tests/validation/"
        echo "Running VALIDATION tests only"
        ;;
    fast)
        TEST_PATH="tests/"
        EXTRA_ARGS="-m 'not slow'"
        echo "Running FAST tests (excluding slow tests)"
        ;;
    all|*)
        TEST_PATH="tests/"
        echo "Running ALL tests (unit + api + integration + validation)"
        ;;
esac

# Run tests
echo "======================================"
echo "Running Tests"
echo "======================================"
echo ""

if [[ "$RUN_COVERAGE" == "true" ]]; then
    pytest "$TEST_PATH" $EXTRA_ARGS -v --cov=training_service --cov=simulation_service --cov-report=term --cov-report=html
    echo ""
    echo "✓ Coverage report generated in htmlcov/index.html"
elif [[ "$VERBOSE" == "true" ]]; then
    pytest "$TEST_PATH" $EXTRA_ARGS -vv
else
    pytest "$TEST_PATH" $EXTRA_ARGS -v --tb=short
fi

echo ""
echo "======================================"
echo "Test Summary"
echo "======================================"
echo ""
echo "✓ Tests completed!"
echo ""
echo "Usage:"
echo "  ./run_unit_tests.sh                # Run all tests"
echo "  ./run_unit_tests.sh unit           # Run unit tests only"
echo "  ./run_unit_tests.sh api            # Run API tests only"
echo "  ./run_unit_tests.sh integration    # Run integration tests only"
echo "  ./run_unit_tests.sh validation     # Run validation tests only"
echo "  ./run_unit_tests.sh fast           # Run fast tests (skip slow)"
echo "  ./run_unit_tests.sh --coverage     # Run with coverage report"
echo "  ./run_unit_tests.sh --verbose      # Run with extra verbosity"
