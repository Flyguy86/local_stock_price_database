#!/usr/bin/env python3
"""Debug script to check if hyperparameters are being saved for grid children."""
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
        # Check grid children
        rows = await conn.fetch("""
            SELECT id, parent_model_id, hyperparameters, is_grid_member, algorithm
            FROM models 
            WHERE is_grid_member = true 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        
        print(f"\nFound {len(rows)} grid children:")
        for row in rows:
            print(f"\n{'='*60}")
            print(f"ID: {row['id'][:16]}...")
            print(f"Parent: {row['parent_model_id'][:16] if row['parent_model_id'] else 'None'}...")
            print(f"Algorithm: {row['algorithm']}")
            print(f"is_grid_member: {row['is_grid_member']}")
            print(f"Hyperparameters type: {type(row['hyperparameters'])}")
            print(f"Hyperparameters: {row['hyperparameters']}")
            
            if row['hyperparameters']:
                try:
                    # Already parsed by asyncpg
                    hp = row['hyperparameters'] if isinstance(row['hyperparameters'], dict) else json.loads(row['hyperparameters'])
                    print(f"  alpha: {hp.get('alpha', 'NOT FOUND')}")
                    print(f"  l1_ratio: {hp.get('l1_ratio', 'NOT FOUND')}")
                except Exception as e:
                    print(f"  ERROR parsing: {e}")
        
        # Check if parent has grid_children_count
        print(f"\n{'='*60}")
        print("Checking parent models with children...")
        parents = await conn.fetch("""
            SELECT m.id, m.algorithm, 
                   (SELECT COUNT(*) FROM models c WHERE c.parent_model_id = m.id) as child_count
            FROM models m
            WHERE (SELECT COUNT(*) FROM models c WHERE c.parent_model_id = m.id) > 0
            LIMIT 3
        """)
        
        for p in parents:
            print(f"  Parent {p['id'][:16]}... ({p['algorithm']}): {p['child_count']} children")
            
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
