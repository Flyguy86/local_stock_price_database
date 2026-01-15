#!/usr/bin/env python3
"""Verify that comprehensive metrics are being stored in the database."""
import asyncio
import asyncpg
import json
import os

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory"
)

async def main():
    print("Connecting to PostgreSQL...")
    conn = await asyncpg.connect(POSTGRES_URL)
    
    try:
        # Get recent completed models
        models = await conn.fetch("""
            SELECT id, name, algorithm, status, metrics, fingerprint, created_at
            FROM models 
            WHERE status = 'completed'
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        
        print(f"\n{'='*80}")
        print(f"Found {len(models)} completed models")
        print(f"{'='*80}\n")
        
        for model in models:
            print(f"Model: {model['name']}")
            print(f"  ID: {model['id']}")
            print(f"  Algorithm: {model['algorithm']}")
            print(f"  Fingerprint: {model['fingerprint'][:16] if model['fingerprint'] else 'None'}...")
            print(f"  Created: {model['created_at']}")
            
            # Parse and display metrics
            if model['metrics']:
                metrics = model['metrics']  # Already parsed by asyncpg
                
                is_regression = 'regressor' in model['algorithm'] or 'regression' in model['algorithm']
                
                print(f"\n  📊 Stored Metrics:")
                
                if is_regression:
                    print(f"    REGRESSION METRICS:")
                    if 'r2' in metrics:
                        print(f"      ✓ R² Score: {metrics['r2']:.4f}")
                    if 'mae' in metrics:
                        print(f"      ✓ MAE: {metrics['mae']:.6f}")
                    if 'mse' in metrics:
                        print(f"      ✓ MSE: {metrics['mse']:.6f}")
                    if 'rmse' in metrics:
                        print(f"      ✓ RMSE: {metrics['rmse']:.6f}")
                    if 'rmse_price' in metrics:
                        print(f"      ✓ RMSE (Price): ${metrics['rmse_price']:.2f}")
                else:
                    print(f"    CLASSIFICATION METRICS:")
                    if 'accuracy' in metrics:
                        print(f"      ✓ Accuracy: {metrics['accuracy']*100:.2f}%")
                    if 'precision' in metrics:
                        print(f"      ✓ Precision: {metrics['precision']*100:.2f}%")
                    if 'recall' in metrics:
                        print(f"      ✓ Recall: {metrics['recall']*100:.2f}%")
                    if 'f1_score' in metrics:
                        print(f"      ✓ F1-Score: {metrics['f1_score']*100:.2f}%")
                    if 'confusion_matrix' in metrics:
                        print(f"      ✓ Confusion Matrix: {len(metrics['confusion_matrix'])}x{len(metrics['confusion_matrix'][0])} matrix stored")
                
                # Check for feature importance
                if 'feature_importance' in metrics:
                    print(f"    ✓ Feature Importance: {len(metrics['feature_importance'])} features")
                
                if 'cv_folds' in metrics:
                    print(f"    ✓ Cross-Validation: {metrics['cv_folds']} folds")
                
                print(f"\n    Total metric keys stored: {len(metrics)}")
            else:
                print(f"  ⚠ No metrics stored")
            
            print(f"\n{'-'*80}\n")
        
        # Summary
        print(f"{'='*80}")
        print(f"VERIFICATION SUMMARY")
        print(f"{'='*80}")
        
        total = await conn.fetchval("SELECT COUNT(*) FROM models")
        with_metrics = await conn.fetchval("SELECT COUNT(*) FROM models WHERE metrics IS NOT NULL")
        with_fingerprint = await conn.fetchval("SELECT COUNT(*) FROM models WHERE fingerprint IS NOT NULL")
        
        print(f"  Total models: {total}")
        print(f"  Models with metrics: {with_metrics}")
        print(f"  Models with fingerprint: {with_fingerprint}")
        print(f"\n  ✓ Metrics storage: {'WORKING' if with_metrics > 0 else 'NO DATA'}")
        print(f"  ✓ Fingerprint storage: {'WORKING' if with_fingerprint > 0 else 'NO DATA'}")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
