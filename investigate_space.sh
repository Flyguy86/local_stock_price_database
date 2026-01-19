#!/bin/bash
# Detailed disk space investigation

echo "=== DETAILED DISK SPACE ANALYSIS ==="
echo ""

echo "Total /workspaces usage:"
df -h /workspaces
echo ""

echo "=== TOP SPACE CONSUMERS ==="
echo ""

echo "1. Project root breakdown:"
du -sh /workspaces/local_stock_price_database/* 2>/dev/null | sort -rh
echo ""

echo "2. Data directory breakdown:"
du -sh /workspaces/local_stock_price_database/data/* 2>/dev/null | sort -rh
echo ""

echo "3. Ray checkpoints detail:"
if [ -d "/workspaces/local_stock_price_database/data/ray_checkpoints/walk_forward_elasticnet_GOOGL" ]; then
    echo "   walk_forward_elasticnet_GOOGL subdirs:"
    du -sh /workspaces/local_stock_price_database/data/ray_checkpoints/walk_forward_elasticnet_GOOGL/* 2>/dev/null | sort -rh | head -20
    echo ""
    CHECKPOINT_COUNT=$(find /workspaces/local_stock_price_database/data/ray_checkpoints/walk_forward_elasticnet_GOOGL -type d -name "train_on_folds_*" | wc -l)
    echo "   Total checkpoint directories: $CHECKPOINT_COUNT"
fi
echo ""

echo "4. Walk-forward folds detail:"
for symbol in AAPL GOOGL MSFT; do
    if [ -d "/workspaces/local_stock_price_database/data/walk_forward_folds/$symbol" ]; then
        FOLD_SIZE=$(du -sh /workspaces/local_stock_price_database/data/walk_forward_folds/$symbol 2>/dev/null | awk '{print $1}')
        FOLD_COUNT=$(ls -1 /workspaces/local_stock_price_database/data/walk_forward_folds/$symbol 2>/dev/null | wc -l)
        echo "   $symbol: $FOLD_SIZE ($FOLD_COUNT folds)"
    fi
done
echo ""

echo "5. Parquet files:"
if [ -d "/workspaces/local_stock_price_database/data/parquet" ]; then
    PARQUET_SYMBOLS=$(ls -1 /workspaces/local_stock_price_database/data/parquet 2>/dev/null | wc -l)
    PARQUET_SIZE=$(du -sh /workspaces/local_stock_price_database/data/parquet 2>/dev/null | awk '{print $1}')
    echo "   Size: $PARQUET_SIZE"
    echo "   Symbols: $PARQUET_SYMBOLS"
fi
echo ""

echo "6. Features parquet:"
if [ -d "/workspaces/local_stock_price_database/data/features_parquet" ]; then
    du -sh /workspaces/local_stock_price_database/data/features_parquet/* 2>/dev/null | sort -rh | head -10
fi
echo ""

echo "7. DuckDB databases:"
find /workspaces/local_stock_price_database/data/duckdb -type f -exec ls -lh {} \; 2>/dev/null
echo ""

echo "8. MLflow artifacts:"
if [ -d "/workspaces/local_stock_price_database/data/mlflow" ]; then
    du -sh /workspaces/local_stock_price_database/data/mlflow 2>/dev/null
fi
echo ""

echo "9. Docker space:"
docker system df 2>/dev/null || echo "   Docker not available"
echo ""

echo "=== RECOMMENDATIONS ==="
echo ""
echo "🔴 LARGEST ITEMS TO CLEAN:"

# Check Ray checkpoints
RAY_SIZE=$(du -sh /workspaces/local_stock_price_database/data/ray_checkpoints 2>/dev/null | awk '{print $1}')
if [ ! -z "$RAY_SIZE" ]; then
    echo "   • Ray Checkpoints: $RAY_SIZE"
    echo "     → Keep only recent experiments, delete old ones"
fi

# Check walk-forward folds
FOLDS_SIZE=$(du -sh /workspaces/local_stock_price_database/data/walk_forward_folds 2>/dev/null | awk '{print $1}')
if [ ! -z "$FOLDS_SIZE" ]; then
    echo "   • Walk-Forward Folds: $FOLDS_SIZE"
    echo "     → Can regenerate these, safe to delete"
fi

# Check features
FEATURES_SIZE=$(du -sh /workspaces/local_stock_price_database/data/features_parquet 2>/dev/null | awk '{print $1}')
if [ ! -z "$FEATURES_SIZE" ]; then
    echo "   • Feature Parquet: $FEATURES_SIZE"
    echo "     → Can regenerate from DuckDB, safe to delete"
fi

echo ""
echo "💡 Quick cleanup commands:"
echo ""
echo "  # Clean all walk-forward folds (can regenerate):"
echo "  rm -rf /workspaces/local_stock_price_database/data/walk_forward_folds/*"
echo ""
echo "  # Clean feature parquet (can regenerate):"
echo "  rm -rf /workspaces/local_stock_price_database/data/features_parquet/*"
echo ""
echo "  # Keep only 3 newest Ray experiments:"
echo "  cd /workspaces/local_stock_price_database/data/ray_checkpoints/walk_forward_elasticnet_GOOGL"
echo "  ls -dt train_on_folds_* | tail -n +4 | xargs rm -rf"
echo ""
echo "  # OR run the automated cleanup:"
echo "  bash /workspaces/local_stock_price_database/cleanup_disk.sh"
