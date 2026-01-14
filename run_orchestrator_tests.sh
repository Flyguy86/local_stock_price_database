#!/bin/bash
# Test runner for orchestrator workflows

echo "================================================"
echo "Running Orchestrator Unit Tests (UI Logic)"
echo "================================================"
python -m pytest tests/unit/test_orchestrator_ui.py -v --tb=short

echo ""
echo "================================================"
echo "Running Orchestrator Integration Tests"
echo "================================================"
echo "NOTE: Integration tests require services running"
echo "Run: docker-compose up -d first"
echo ""
python -m pytest tests/integration/test_orchestrator_workflows.py -v --tb=short -m integration

echo ""
echo "================================================"
echo "Test Summary"
echo "================================================"
echo "Unit tests: Pure Python logic for UI interactions"
echo "Integration tests: API endpoints with live services"
