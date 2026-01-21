#!/bin/bash
echo "📊 Checking Ray cluster status and current jobs..."
echo ""

echo "1️⃣ Ray cluster health:"
curl -s http://localhost:8265/api/v0/cluster_status 2>/dev/null | python3 -m json.tool | head -20 || echo "Ray cluster not responding"

echo ""
echo "2️⃣ Recent Ray jobs:"
curl -s http://localhost:8265/api/v0/jobs 2>/dev/null | python3 -m json.tool | head -50 || echo "Ray jobs endpoint not responding"

echo ""
echo "3️⃣ Orchestrator health:"
curl -s http://localhost:8100/health 2>/dev/null | python3 -m json.tool || echo "Orchestrator not responding"

echo ""
echo "4️⃣ Available experiments (from checkpoints):"
curl -s http://localhost:8100/experiments 2>/dev/null | python3 -m json.tool || echo "Experiments endpoint not responding"
