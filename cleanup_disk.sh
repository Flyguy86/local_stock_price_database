#!/bin/bash
# Comprehensive disk cleanup script for local_stock_price_database

set -e

AUTO_MODE=false
if [[ "$1" == "--auto" ]]; then
    AUTO_MODE=true
fi

echo "╔════════════════════════════════════════════════════════╗"
echo "║      Disk Cleanup Script for Trading Bot Project      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Function to ask for confirmation
confirm() {
    if [ "$AUTO_MODE" = true ]; then
        return 0
    fi
    read -p "$1 [y/N]: " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# Show current disk usage
echo "📊 Current Disk Usage:"
df -h /workspaces | grep -v tmpfs
echo ""

# 1. Clean Python cache files
echo "🐍 Python Cache Files..."
PYSIZE=$(find /workspaces/local_stock_price_database -type d -name __pycache__ -exec du -sh {} + 2>/dev/null | awk '{sum+=$1} END {print sum}' || echo "0")
echo "   Found: ~${PYSIZE}MB in __pycache__ directories"
if confirm "   Clean Python cache files?"; then
    find /workspaces/local_stock_price_database -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find /workspaces/local_stock_price_database -type f -name '*.pyc' -delete 2>/dev/null || true
    echo "   ✅ Python cache cleaned"
fi
echo ""

# 2. Clean Docker
echo "🐳 Docker Resources..."
docker system df
echo ""
if confirm "   Clean Docker (unused images, containers, volumes)?"; then
    echo "   Stopping containers first..."
    docker compose down 2>/dev/null || true
    echo "   Pruning Docker system..."
    docker system prune -a --volumes -f
    echo "   ✅ Docker cleaned"
    echo "   Restart services with: docker compose up -d"
fi
echo ""

# 3. Clean old Ray checkpoints (keep only last 5 experiments)
echo "🎯 Ray Checkpoints..."
CHECKPOINT_DIR="/workspaces/local_stock_price_database/data/ray_checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "   Current checkpoints:"
    du -sh "$CHECKPOINT_DIR"/* 2>/dev/null | sort -hr | head -10 || echo "   (empty)"
    echo ""
    if confirm "   Delete old Ray checkpoints? (keeps newest 5 experiments)"; then
        cd "$CHECKPOINT_DIR"
        # Keep newest 5 directories, delete the rest
        ls -dt */ 2>/dev/null | tail -n +6 | xargs -I {} rm -rf {} 2>/dev/null || true
        echo "   ✅ Old checkpoints cleaned (kept newest 5)"
    fi
fi
echo ""

# 4. Clean old walk-forward folds
echo "📁 Walk-Forward Folds..."
FOLDS_DIR="/workspaces/local_stock_price_database/data/walk_forward_folds"
if [ -d "$FOLDS_DIR" ]; then
    du -sh "$FOLDS_DIR" 2>/dev/null || echo "   (empty)"
    if confirm "   Delete all walk-forward folds? (will regenerate on next training)"; then
        rm -rf "$FOLDS_DIR"/*
        echo "   ✅ Folds cleaned"
    fi
fi
echo ""

# 5. Clean old parquet files
echo "📦 Parquet Data Files..."
PARQUET_DIR="/workspaces/local_stock_price_database/data/parquet"
if [ -d "$PARQUET_DIR" ]; then
    du -sh "$PARQUET_DIR" 2>/dev/null || echo "   (empty)"
    echo "   Symbols: $(ls -1 $PARQUET_DIR 2>/dev/null | wc -l)"
    if confirm "   Delete old parquet data? (keeps DuckDB, can re-fetch)"; then
        find "$PARQUET_DIR" -type f -name '*.parquet' -mtime +30 -delete 2>/dev/null || true
        echo "   ✅ Old parquet files cleaned (deleted files older than 30 days)"
    fi
fi
echo ""

# 6. Clean features parquet
echo "🔧 Feature Parquet Files..."
FEATURES_DIR="/workspaces/local_stock_price_database/data/features_parquet"
if [ -d "$FEATURES_DIR" ]; then
    du -sh "$FEATURES_DIR" 2>/dev/null || echo "   (empty)"
    if confirm "   Delete feature parquet files? (can regenerate from DuckDB)"; then
        rm -rf "$FEATURES_DIR"/*
        echo "   ✅ Feature parquet cleaned"
    fi
fi
echo ""

# 7. Clean DuckDB temp files
echo "🦆 DuckDB Temporary Files..."
find /workspaces/local_stock_price_database/data -name "*.tmp" -o -name "*.wal" -o -name "*.tmp.*" 2>/dev/null | head -5
if confirm "   Delete DuckDB temp files?"; then
    find /workspaces/local_stock_price_database/data -name "*.tmp" -delete 2>/dev/null || true
    find /workspaces/local_stock_price_database/data -name "*.wal" -delete 2>/dev/null || true
    find /workspaces/local_stock_price_database/data -name "*.tmp.*" -delete 2>/dev/null || true
    echo "   ✅ DuckDB temp files cleaned"
fi
echo ""

# 8. Clean logs
echo "📝 Log Files..."
LOG_SIZE=$(find /workspaces/local_stock_price_database -name "*.log" -exec du -ch {} + 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
echo "   Found: $LOG_SIZE in log files"
if confirm "   Delete old log files (>7 days)?"; then
    find /workspaces/local_stock_price_database -name "*.log" -mtime +7 -delete 2>/dev/null || true
    echo "   ✅ Old logs cleaned"
fi
echo ""

# Final report
echo "═════════════════════════════════════════════════════════"
echo "✅ Cleanup Complete!"
echo ""
echo "📊 New Disk Usage:"
df -h /workspaces | grep -v tmpfs
echo ""
echo "💡 Tips:"
echo "   • Run 'docker compose up -d' to restart services"
echo "   • Folds will regenerate automatically on next training"
echo "   • Features can be regenerated from feature_service"
echo "═════════════════════════════════════════════════════════"
