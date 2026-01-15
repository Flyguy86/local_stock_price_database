#!/usr/bin/env python3
"""
Migration script to add cohort_id column and migrate grid search data.

This script:
1. Adds the cohort_id column to the models table
2. Migrates existing grid search models from parent_model_id → cohort_id
3. Clears parent_model_id for grid members (unless they have actual feature evolution)
"""

import os
import psycopg2
from psycopg2 import sql

def get_db_connection():
    """Get database connection from environment variables."""
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "postgres"),
        port=int(os.getenv("PG_PORT", "5432")),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "postgres"),
        database=os.getenv("PG_DATABASE", "training")
    )

def migrate():
    """Run the migration."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Step 1: Add cohort_id column if it doesn't exist
        print("Step 1: Adding cohort_id column...")
        cur.execute("""
            ALTER TABLE models 
            ADD COLUMN IF NOT EXISTS cohort_id VARCHAR;
        """)
        conn.commit()
        print("✅ cohort_id column added")
        
        # Step 2: Find all grid search models (is_grid_member = true)
        print("\nStep 2: Finding grid search models...")
        cur.execute("""
            SELECT id, parent_model_id, is_grid_member 
            FROM models 
            WHERE is_grid_member = true;
        """)
        grid_models = cur.fetchall()
        print(f"Found {len(grid_models)} grid search models")
        
        # Step 3: Migrate parent_model_id → cohort_id for grid members
        print("\nStep 3: Migrating grid search models...")
        migrated = 0
        for model_id, parent_id, is_grid in grid_models:
            if parent_id:
                # Set cohort_id = parent_model_id
                cur.execute("""
                    UPDATE models 
                    SET cohort_id = %s 
                    WHERE id = %s;
                """, (parent_id, model_id))
                migrated += 1
        
        conn.commit()
        print(f"✅ Migrated {migrated} models to use cohort_id")
        
        # Step 4: Find "parent" models (those referenced by grid members)
        print("\nStep 4: Finding cohort leader models...")
        cur.execute("""
            SELECT DISTINCT parent_model_id 
            FROM models 
            WHERE is_grid_member = true AND parent_model_id IS NOT NULL;
        """)
        cohort_leaders = [row[0] for row in cur.fetchall()]
        print(f"Found {len(cohort_leaders)} cohort leader models")
        
        # Set cohort_id for the leader models themselves
        leader_updated = 0
        for leader_id in cohort_leaders:
            cur.execute("""
                UPDATE models 
                SET cohort_id = %s, is_grid_member = true
                WHERE id = %s;
            """, (leader_id, leader_id))
            leader_updated += 1
        
        conn.commit()
        print(f"✅ Updated {leader_updated} cohort leaders to include themselves in cohort")
        
        # Step 5: Clear parent_model_id for grid members
        # (parent_model_id should only be used for feature evolution, not grid search)
        print("\nStep 5: Clearing parent_model_id for grid members...")
        cur.execute("""
            UPDATE models 
            SET parent_model_id = NULL 
            WHERE is_grid_member = true;
        """)
        cleared = cur.rowcount
        conn.commit()
        print(f"✅ Cleared parent_model_id for {cleared} grid members")
        
        # Step 6: Show summary
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)
        
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(cohort_id) as with_cohort,
                COUNT(CASE WHEN is_grid_member THEN 1 END) as grid_members,
                COUNT(parent_model_id) as with_parent
            FROM models;
        """)
        stats = cur.fetchone()
        print(f"Total models: {stats[0]}")
        print(f"Models with cohort_id: {stats[1]}")
        print(f"Grid search members: {stats[2]}")
        print(f"Models with parent_model_id: {stats[3]}")
        
        # Show cohort breakdown
        print("\nCohort breakdown:")
        cur.execute("""
            SELECT cohort_id, COUNT(*) as size
            FROM models
            WHERE cohort_id IS NOT NULL
            GROUP BY cohort_id
            ORDER BY size DESC
            LIMIT 10;
        """)
        cohorts = cur.fetchall()
        for cohort_id, size in cohorts:
            print(f"  {cohort_id[:12]}... : {size} models")
        
        print("\n✅ Migration completed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    migrate()
