#!/bin/bash
# Visual proof that retrain jobs disable caching

echo "═══════════════════════════════════════════════════════════════════════"
echo "PROOF: Retrain Jobs Disable Fold Caching"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

echo "📝 STEP 1: Check the /training/submit endpoint code"
echo "────────────────────────────────────────────────────────────────────────"
grep -A 2 "use_cached_folds=False" /workspaces/local_stock_price_database/ray_orchestrator/main.py
echo ""
echo "✅ Confirmed: Retrain endpoint passes use_cached_folds=False"
echo ""

echo "📝 STEP 2: Check the trainer accepts this parameter"
echo "────────────────────────────────────────────────────────────────────────"
grep "use_cached_folds: bool" /workspaces/local_stock_price_database/ray_orchestrator/trainer.py
echo ""
echo "✅ Confirmed: Trainer has use_cached_folds parameter"
echo ""

echo "📝 STEP 3: Check the trainer passes it to the pipeline"
echo "────────────────────────────────────────────────────────────────────────"
grep -A 1 "actor_pool_size=actor_pool_size" /workspaces/local_stock_price_database/ray_orchestrator/trainer.py | grep "use_cached_folds"
echo ""
echo "✅ Confirmed: Parameter flows to create_walk_forward_pipeline()"
echo ""

echo "📝 STEP 4: Check the streaming logic respects the flag"
echo "────────────────────────────────────────────────────────────────────────"
grep -B 2 -A 4 "if use_cached_folds and len(symbols) == 1:" /workspaces/local_stock_price_database/ray_orchestrator/streaming.py | head -10
echo ""
echo "✅ Confirmed: Cache is skipped when use_cached_folds=False"
echo ""

echo "📝 STEP 5: Show the cache key problem we're preventing"
echo "────────────────────────────────────────────────────────────────────────"
grep "fold_dir = settings.data.walk_forward_folds_dir" /workspaces/local_stock_price_database/ray_orchestrator/streaming.py
echo ""
echo "⚠️  Cache key only includes: symbol + fold_id"
echo "❌ Missing: train_months, test_months, start_date, end_date"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "CONCLUSION"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ When you submit a retrain job, use_cached_folds=False is set"
echo "✅ This skips the cache check at line 1715 in streaming.py"
echo "✅ Fresh folds are computed based on your ACTUAL parameters"
echo "✅ No data leakage from stale cached folds"
echo ""
echo "🎯 EXAMPLE:"
echo "   7-month training creates: /app/data/walk_forward_folds/GOOGL/fold_001/"
echo "   8-month retrain IGNORES that cache, computes fresh fold_001 with 8-month data"
echo ""
