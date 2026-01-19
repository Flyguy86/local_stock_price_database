#!/bin/bash
# Quick disk space analysis

echo "=== Disk Space Analysis ==="
echo ""
echo "Overall /workspaces usage:"
df -h /workspaces
echo ""

echo "Top 15 largest directories in project:"
du -h /workspaces/local_stock_price_database --max-depth=3 2>/dev/null | sort -rh | head -15
echo ""

echo "Breakdown by data type:"
echo "  Ray Checkpoints: $(du -sh /workspaces/local_stock_price_database/data/ray_checkpoints 2>/dev/null | awk '{print $1}' || echo '0')"
echo "  Parquet Files:   $(du -sh /workspaces/local_stock_price_database/data/parquet 2>/dev/null | awk '{print $1}' || echo '0')"
echo "  Features:        $(du -sh /workspaces/local_stock_price_database/data/features_parquet 2>/dev/null | awk '{print $1}' || echo '0')"
echo "  DuckDB:          $(du -sh /workspaces/local_stock_price_database/data/duckdb 2>/dev/null | awk '{print $1}' || echo '0')"
echo "  Walk-Fwd Folds:  $(du -sh /workspaces/local_stock_price_database/data/walk_forward_folds 2>/dev/null | awk '{print $1}' || echo '0')"
echo "  Models:          $(du -sh /workspaces/local_stock_price_database/data/models 2>/dev/null | awk '{print $1}' || echo '0')"
echo ""

echo "Docker usage:"
docker system df 2>/dev/null || echo "  Docker not available"
echo ""

echo "Python cache:"
PYCACHE_SIZE=$(find /workspaces/local_stock_price_database -type d -name __pycache__ 2>/dev/null | wc -l)
echo "  __pycache__ dirs: $PYCACHE_SIZE"
echo ""

echo "💡 To clean up, run:"
echo "   bash /workspaces/local_stock_price_database/cleanup_disk.sh"
