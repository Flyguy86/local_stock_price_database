#!/bin/bash
JOB_ID="raysubmit_zpYthCiU9GWQpnyi"

echo "Getting detailed logs for job: $JOB_ID"
echo ""

curl -s "http://localhost:8100/train/job/$JOB_ID" | python3 -c "
import sys, json
data = json.load(sys.stdin)
logs = data.get('logs', '')

print('Job Status:', data.get('status'))
print('')
print('Key Evidence in Logs:')
print('=' * 70)

# Look for the training command
for line in logs.split('\n'):
    if 'symbols=' in line and 'GOOGL' in line:
        print('✅ TICKER CONFIRMED:', line.strip())
    if 'Generating walk-forward folds' in line:
        print('✅ FOLD GENERATION:', line.strip())
    if 'excluded' in line.lower() and 'features' in line.lower():
        print('✅ FEATURE EXCLUSION:', line.strip())
    if 'Starting walk-forward' in line:
        print('✅ TRAINING STARTED:', line.strip())

print('')
print('Last 1000 characters of logs:')
print('=' * 70)
print(logs[-1000:] if logs else 'No logs')
"
