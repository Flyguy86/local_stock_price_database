#!/bin/bash

# Start TensorBoard for viewing training metrics
# Usage: ./start_tensorboard.sh [logdir]

# Accept logdir as argument or auto-detect
if [ -z "$1" ]; then
    echo "🔍 Auto-detecting most recent training run..."
    
    # Find the most recent Ray session artifacts
    LATEST=$(find /tmp/ray -type d -name "driver_artifacts" 2>/dev/null | sort -r | head -1)
    
    if [ -z "$LATEST" ]; then
        echo "⚠️  No Ray training runs found. Using checkpoint directory..."
        LOG_DIR="/app/data/ray_checkpoints"
    else
        LOG_DIR="$LATEST"
    fi
else
    LOG_DIR="$1"
fi

PORT=6006

echo "📊 Starting TensorBoard..."
echo "  Log Directory: $LOG_DIR"
echo "  Port: $PORT"
echo "  URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

# Start TensorBoard (bind to 0.0.0.0 so it's accessible from host)
tensorboard --logdir="$LOG_DIR" --host=0.0.0.0 --port="$PORT"
