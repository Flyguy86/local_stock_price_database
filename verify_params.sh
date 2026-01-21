#!/bin/bash
JOB_ID="raysubmit_iJY7mRpEdPmsrmFE"

echo "Checking Ray job logs for job: $JOB_ID"
echo "Looking for evidence that GOOGL ticker and parameters were passed..."
echo ""

sleep 3

curl -s "http://localhost:8100/train/job/$JOB_ID" | python3 -c "
import sys, json
data = json.load(sys.stdin)
logs = data.get('logs', '')

print('Job Status:', data.get('status'))
print('')
print('Log Analysis:')
print('=' * 60)

if 'GOOGL' in logs:
    print('✅ GOOGL ticker found in logs')
else:
    print('❌ GOOGL ticker NOT found')

if \"symbols=['GOOGL']\" in logs:
    print('✅ symbols=[\"GOOGL\"] parameter confirmed')
else:
    print('⚠️  symbols parameter not yet visible')

if 'No data found' in logs:
    print('❌ ERROR: No data found (wrong ticker or dates)')
elif 'ValueError' in logs:
    print('❌ ERROR: ValueError in execution')
elif 'Runtime env is setting up' in logs:
    print('⏳ Job initializing - runtime environment setting up')
elif 'Generating walk-forward folds' in logs:
    print('✅ Job is processing - generating folds')
elif 'Starting walk-forward' in logs:
    print('✅ Job is running - training started')

print('')
print('Full logs (first 1000 chars):')
print('=' * 60)
print(logs[:1000] if logs else 'No logs yet...')
"
