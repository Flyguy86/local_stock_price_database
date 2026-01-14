#!/usr/bin/env python3
"""
List all available tests with descriptions.
Helps developers understand test coverage and find specific tests.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title):
    """Print formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def list_unit_tests():
    """List all unit tests."""
    print_header("UNIT TESTS (53 tests)")
    
    print("📦 PostgreSQL Database Layer (30 tests)")
    print("   File: tests/unit/test_pg_db.py")
    print("   Tests:")
    print("   • Table creation and schema validation")
    print("   • Model CRUD operations")
    print("   • Fingerprint-based deduplication")
    print("   • Status transitions")
    print("   • Feature importance storage")
    print("   • Cascade deletes")
    print("   • Simulation history operations")
    print("   • Pagination for top strategies")
    print()
    print("   Run: pytest tests/unit/test_pg_db.py -v")
    print()
    
    print("🔄 Sync DB Wrapper (8 tests)")
    print("   File: tests/unit/test_sync_wrapper.py")
    print("   Tests:")
    print("   • Multi-process model creation")
    print("   • Concurrent updates from different processes")
    print("   • Process isolation (separate connection pools)")
    print("   • Performance under sequential load")
    print()
    print("   Run: pytest tests/unit/test_sync_wrapper.py -v")
    print()
    
    print("⚡ Process Pool Executor (15 tests)")
    print("   File: tests/unit/test_process_pool.py")
    print("   Tests:")
    print("   • Parallel task execution")
    print("   • Different PIDs verification")
    print("   • Max workers limit enforcement")
    print("   • Worker recycling (max_tasks_per_child)")
    print("   • Database access from worker processes")
    print("   • Speedup measurement benchmarks")
    print()
    print("   Run: pytest tests/unit/test_process_pool.py -v")
    print()


def list_integration_tests():
    """List all integration tests."""
    print_header("INTEGRATION TESTS (22 tests)")
    
    print("🔄 Training Service Workflows (11 tests)")
    print("   File: tests/integration/test_training_workflow.py")
    print("   Workflows:")
    print("   • Train → Save → Query complete flow")
    print("   • Batch training (multiple models)")
    print("   • Parent-child training (transfer learning)")
    print("   • Model retraining workflow")
    print("   • Feature importance storage and retrieval")
    print("   • Model deletion with cascade")
    print("   • Concurrent training jobs (multi-process)")
    print("   • Sync DB wrapper integration")
    print()
    print("   Run: pytest tests/integration/test_training_workflow.py -v")
    print()
    
    print("🎲 Simulation Service Workflows (11 tests)")
    print("   File: tests/integration/test_simulation_workflow.py")
    print("   Workflows:")
    print("   • Simulate → Save → History complete flow")
    print("   • Batch simulations with ranking")
    print("   • Pagination for large result sets")
    print("   • History deletion workflow")
    print("   • Performance metrics validation")
    print("   • Configuration and health checks")
    print("   • Multiple strategies comparison")
    print("   • Cross-symbol testing")
    print("   • Full simulation lifecycle")
    print()
    print("   Run: pytest tests/integration/test_simulation_workflow.py -v")
    print()


def list_validation_tests():
    """List validation tests organized by category."""
    print_header("VALIDATION TESTS (31 tests)")
    
    print("🔍 Fingerprint Deduplication (16 tests)")
    print("   File: tests/validation/test_fingerprint_deduplication.py")
    print("   Categories:")
    print("   • Fingerprint generation consistency")
    print("   • Identical configs → same fingerprint")
    print("   • Different features/hyperparams → different fingerprints")
    print("   • Order-independent hashing (sorted features)")
    print("   • Database lookup by fingerprint")
    print("   • Duplicate prevention detection")
    print("   • Edge cases (null fields, JSON serialization)")
    print("   • Integration with training workflow")
    print()
    print("   Run: pytest tests/validation/test_fingerprint_deduplication.py -v")
    print()
    
    print("🔌 Connection Pool Limits (15 tests)")
    print("   File: tests/validation/test_connection_pool_limits.py")
    print("   Categories:")
    print("   • Pool respects max_size configuration")
    print("   • Concurrent connection handling")
    print("   • Connection reuse after release")
    print("   • Pool isolation per service (training/simulation)")
    print("   • Clean shutdown and cleanup")
    print("   • Error handling (query errors, timeouts)")
    print("   • Performance benchmarks (acquisition speed, throughput)")
    print("   • Multi-process pool creation")
    print("   • Recovery from failed connections")
    print()
    print("   Run: pytest tests/validation/test_connection_pool_limits.py -v")
    print()
    print("   ⚠️  Note: Some tests marked 'slow' - skip with -m 'not slow'")
    print()


def list_api_tests():
    """List all API tests."""
    print_header("API TESTS (39 tests)")
    
    print("🎯 Training Service API (21 tests)")
    print("   File: tests/api/test_training_api.py")
    print("   Endpoints:")
    print("   • GET  /health - Health check")
    print("   • GET  /logs - Log buffer")
    print("   • GET  /data/options - Available data configurations")
    print("   • GET  /data/options/{symbol} - Symbol-specific data")
    print("   • GET  /data/map - Feature mapping")
    print("   • GET  /algorithms - Available algorithms")
    print("   • GET  /models - List all models")
    print("   • GET  /models/{model_id} - Get model details")
    print("   • DELETE /models/{model_id} - Delete model")
    print("   • DELETE /models/all - Delete all models")
    print("   • POST /train - Submit training job")
    print("   • POST /train/batch - Batch training")
    print("   • POST /retrain/{model_id} - Retrain model")
    print("   • GET  /api/model/{model_id}/importance - Feature importance")
    print("   • GET  /api/model/{model_id}/config - Model configuration")
    print("   • POST /api/train_with_parent - Transfer learning")
    print()
    print("   Run: pytest tests/api/test_training_api.py -v")
    print()
    
    print("🎲 Simulation Service API (18 tests)")
    print("   File: tests/api/test_simulation_api.py")
    print("   Endpoints:")
    print("   • GET  /health - Health check")
    print("   • GET  /logs - Log buffer")
    print("   • GET  /api/config - Available models and tickers")
    print("   • GET  /api/history - Simulation history")
    print("   • GET  /history/top - Top performing strategies")
    print("   • DELETE /history/all - Delete all history")
    print("   • POST /api/simulate - Run simulation")
    print("   • POST /api/batch_simulate - Batch simulations")
    print("   • POST /api/train_bot - Train trading bot")
    print()
    print("   Run: pytest tests/api/test_simulation_api.py -v")
    print()


def list_test_commands():
    """List common test commands."""
    print_header("COMMON COMMANDS")
    
    commands = [
        ("Run all tests", "./run_unit_tests.sh"),
        ("Run with coverage", "./run_unit_tests.sh --coverage"),
        ("Run unit tests only", "./run_unit_tests.sh unit"),
        ("Run API tests only", "./run_unit_tests.sh api"),
        ("Run integration tests only", "./run_unit_tests.sh integration"),
        ("Run validation tests only", "./run_unit_tests.sh validation"),
        ("Run fast tests (skip slow)", "./run_unit_tests.sh fast"),
        ("Run specific file", "pytest tests/unit/test_pg_db.py -v"),
        ("Run specific class", "pytest tests/api/test_training_api.py::TestModelManagement -v"),
        ("Run tests matching pattern", "pytest tests/ -k 'health' -v"),
        ("Stop on first failure", "pytest tests/ -x"),
        ("Rerun last failures", "pytest tests/ --lf"),
        ("Show test coverage", "pytest tests/ --cov --cov-report=term"),
        ("List all tests without running", "pytest tests/ --collect-only"),
        ("Validate test setup", "python check_unit_tests.py"),
    ]
    
    max_desc_len = max(len(desc) for desc, _ in commands)
    
    for description, command in commands:
        padding = " " * (max_desc_len - len(description))
        print(f"  {description}{padding}  →  {command}")
    print()


def list_test_fixtures():
    """List available test fixtures."""
    print_header("AVAILABLE FIXTURES")
    
    fixtures = [
        ("test_db_pool", "session", "PostgreSQL connection pool for tests"),
        ("db_tables", "function", "Fresh database tables per test (auto-cleanup)"),
        ("sample_model_data", "function", "Generate mock model records"),
        ("training_db", "function", "TrainingDB instance for tests"),
        ("simulation_db", "function", "SimulationDB instance for tests"),
        ("api_client", "function", "AsyncClient for HTTP testing"),
        ("mock_data_available", "function", "Mock data availability"),
    ]
    
    for name, scope, description in fixtures:
        print(f"  • {name:25} (scope={scope:8})  {description}")
    print()


def list_test_markers():
    """List pytest markers."""
    print_header("TEST MARKERS")
    
    markers = [
        ("asyncio", "Async test (uses pytest-asyncio)"),
        ("slow", "Slow running test (skip with -m 'not slow')"),
        ("integration", "Integration test (requires services running)"),
        ("unit", "Unit test (isolated, no external dependencies)"),
        ("api", "API test (HTTP endpoint testing)"),
        ("load", "Load test (stress testing, performance)"),
    ]
    
    for marker, description in markers:
        print(f"  @pytest.mark.{marker:15}  {description}")
    print()
    
    print("  Examples:")
    print("    pytest tests/ -m 'unit'           # Run only unit tests")
    print("    pytest tests/ -m 'not slow'       # Skip slow tests")
    print("    pytest tests/ -m 'api and asyncio' # API async tests")
    print()


def print_summary():
    """Print test suite summary."""
    print_header("TEST SUITE SUMMARY")
    
    print(f"  Total Tests:        145 tests")
    print(f"  Unit Tests:         53 tests (PostgreSQL, Sync, ProcessPool)")
    print(f"  API Tests:          39 tests (Training, Simulation)")
    print(f"  Integration Tests:  22 tests (End-to-end workflows)")
    print(f"  Validation Tests:   31 tests (Fingerprint, Connection pools)")
    print(f"  Coverage Target:    85%+ overall, 90%+ core modules")
    print(f"  Expected Runtime:   25-50 seconds (full suite)")
    print()
    print("  Documentation:")
    print("    • TEST_SUITE_SUMMARY.md    - Complete overview")
    print("    • UNIT_TESTS.md            - Unit test guide")
    print("    • API_TESTS.md             - API test guide")
    print("    • INTEGRATION_TESTS.md     - Integration test guide")
    print("    • VALIDATION_TESTS.md      - Validation test guide")
    print()


def main():
    """Main entry point."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "TEST SUITE CATALOG" + " " * 32 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print_summary()
    list_unit_tests()
    list_api_tests()
    list_integration_tests()
    list_validation_tests()
    list_test_commands()
    list_test_fixtures()
    list_test_markers()
    
    print_header("QUICK START")
    print("  1. Ensure PostgreSQL is running:")
    print("     docker compose up postgres -d")
    print()
    print("  2. Install test dependencies:")
    print("     pip install -r tests/requirements.txt")
    print()
    print("  3. Run the test suite:")
    print("     ./run_unit_tests.sh")
    print()
    print("  4. View coverage report:")
    print("     ./run_unit_tests.sh --coverage")
    print()
    print("  For detailed documentation, see:")
    print("    tests/TEST_SUITE_SUMMARY.md")
    print()


if __name__ == "__main__":
    main()
