#!/bin/bash
# Script to analyze and clean up disk space in /workspaces

echo "=== Current Disk Usage ==="
df -h /workspaces

echo ""
echo "=== Top 10 Largest Directories ==="
du -h /workspaces/local_stock_price_database --max-depth=2 2>/dev/null | sort -hr | head -20

echo ""
echo "=== Docker Usage ==="
docker system df

echo ""
echo "=== Safe Cleanup Options ==="
echo ""
echo "1. Clean Docker (removes unused images, containers, networks):"
echo "   docker system prune -a --volumes"
echo ""
echo "2. Clean Python cache:"
echo "   find /workspaces/local_stock_price_database -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null"
echo "   find /workspaces/local_stock_price_database -type f -name '*.pyc' -delete"
echo ""
echo "3. Clean Ray checkpoints (keeps only recent models):"
echo "   # Review first: ls -lh /workspaces/local_stock_price_database/data/ray_checkpoints/"
echo "   # Delete old experiments manually"
echo ""
echo "4. Clean old parquet files:"
echo "   # Review first: du -sh /workspaces/local_stock_price_database/data/parquet/*"
echo "   # Delete old data if needed"
echo ""
echo "5. Clean logs:"
echo "   docker compose logs --tail=0 > /dev/null"
echo "   find /workspaces/local_stock_price_database -name '*.log' -mtime +7 -delete"
echo ""
echo "=== To run all safe cleanups automatically ==="
echo "Run: bash /workspaces/local_stock_price_database/cleanup_disk.sh --auto"
