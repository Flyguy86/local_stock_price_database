#!/bin/bash
# Run this to see the complete proof

echo "Running comprehensive proof of feature selection workflow..."
echo ""

# Make scripts executable
chmod +x PROOF_OF_CONCEPT.sh check_status.sh quick_test_submit.sh

echo "Step 1: Check system status"
bash check_status.sh

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Step 2: Run proof of concept"
bash PROOF_OF_CONCEPT.sh
