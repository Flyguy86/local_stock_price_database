#!/bin/bash

# Start TensorBoard for viewing training metrics
# Usage: ./start_tensorboard.sh [port]

PORT=${1:-6006}
LOG_DIR="/app/data/tensorboard_logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "Starting TensorBoard..."
echo "  Log Directory: $LOG_DIR"
echo "  Port: $PORT"
echo "  URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

# Start TensorBoard
tensorboard --logdir="$LOG_DIR" --host=0.0.0.0 --port="$PORT"
