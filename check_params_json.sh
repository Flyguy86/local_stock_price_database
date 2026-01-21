#!/bin/bash
echo "Checking what params.json actually contains..."
echo ""

CHECKPOINT_PATH="/app/data/ray_checkpoints/walk_forward_xgboost_GOOGL/train_on_folds_0b701_00044_44_colsample_bytree=0.8582,learning_rate=0.0226,max_depth=6,n_estimators=300,subsample=0.9261_2026-01-21_01-55-39"

if [ -f "$CHECKPOINT_PATH/params.json" ]; then
    echo "✅ params.json exists"
    echo "Contents:"
    cat "$CHECKPOINT_PATH/params.json" | python3 -m json.tool
else
    echo "❌ params.json not found"
    echo "Checking what files exist:"
    ls -la "$CHECKPOINT_PATH/"
fi

echo ""
echo "This is what the /models/trial-details endpoint returns for config"
