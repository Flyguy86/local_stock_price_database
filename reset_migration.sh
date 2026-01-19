#!/bin/bash
# Reset migration markers to allow re-migration with verification

echo "Removing migration markers..."
find /workspaces/local_stock_price_database/data/ray_checkpoints -name ".mlflow_migrated" -delete

count=$(find /workspaces/local_stock_price_database/data/ray_checkpoints -type d -name "checkpoint_*" | wc -l)
echo "Found $count checkpoints ready for re-migration"

echo ""
echo "To re-run migration with verification:"
echo "  docker-compose restart mlflow"
echo "  docker-compose logs -f mlflow"
