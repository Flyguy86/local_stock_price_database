#!/bin/bash
JOB_ID="raysubmit_iJY7mRpEdPmsrmFE"

echo "Monitoring Ray job: $JOB_ID"
echo "Waiting for job to start executing..."
echo ""

for i in {1..20}; do
    echo "Check #$i ($(date +%H:%M:%S)):"
    
    RESPONSE=$(curl -s "http://localhost:8100/train/job/$JOB_ID")
    STATUS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'UNKNOWN'))" 2>/dev/null)
    LOGS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('logs', ''))" 2>/dev/null)
    
    echo "  Status: $STATUS"
    
    # Check for key indicators
    if echo "$LOGS" | grep -q "symbols=\['GOOGL'\]"; then
        echo "  ✅ GOOGL parameter confirmed!"
    fi
    
    if echo "$LOGS" | grep -q "Generating walk-forward folds"; then
        echo "  ✅ Fold generation started"
    fi
    
    if echo "$LOGS" | grep -q "Starting walk-forward"; then
        echo "  ✅ Training started"
    fi
    
    if echo "$LOGS" | grep -q "No data found"; then
        echo "  ❌ ERROR: No data found"
        echo ""
        echo "Last 500 chars of logs:"
        echo "$LOGS" | tail -c 500
        exit 1
    fi
    
    if echo "$LOGS" | grep -q "ValueError"; then
        echo "  ❌ ERROR: ValueError occurred"
        echo ""
        echo "Last 500 chars of logs:"
        echo "$LOGS" | tail -c 500
        exit 1
    fi
    
    # Show recent log snippet
    RECENT=$(echo "$LOGS" | tail -c 200)
    if [ ! -z "$RECENT" ]; then
        echo "  Recent: $(echo $RECENT | head -c 100)..."
    fi
    
    # If we see training started, we're good!
    if echo "$LOGS" | grep -q "symbols=\['GOOGL'\]"; then
        echo ""
        echo "✅ SUCCESS! Parameters confirmed in logs:"
        echo "$LOGS" | grep "symbols=\['GOOGL'\]"
        echo ""
        echo "Full job details:"
        echo "$RESPONSE" | python3 -m json.tool | head -50
        exit 0
    fi
    
    # Check if job completed or failed
    if [ "$STATUS" = "SUCCEEDED" ] || [ "$STATUS" = "FAILED" ]; then
        echo ""
        echo "Job finished with status: $STATUS"
        echo "Full logs:"
        echo "$LOGS"
        exit 0
    fi
    
    sleep 3
    echo ""
done

echo "Timeout after 60 seconds. Final status: $STATUS"
echo "Last logs:"
echo "$LOGS" | tail -c 1000
