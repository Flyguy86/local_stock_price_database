"""Database migration utilities for DuckDB."""
import duckdb
from pathlib import Path
import logging

log = logging.getLogger(__name__)

def migrate_add_is_backfilled(db_path: Path) -> None:
    """Add is_backfilled column to bars table if it doesn't exist."""
    conn = duckdb.connect(str(db_path))
    
    try:
        # Check if column exists
        result = conn.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'bars' AND column_name = 'is_backfilled'
        """).fetchall()
        
        if not result:
            log.info("Adding is_backfilled column to bars table")
            conn.execute("""
                ALTER TABLE bars ADD COLUMN is_backfilled BOOLEAN DEFAULT FALSE
            """)
            log.info("Migration complete: is_backfilled column added")
        else:
            log.info("Migration skipped: is_backfilled column already exists")
    except Exception as e:
        log.error(f"Migration failed: {e}")
        raise
    finally:
        conn.close()

def run_all_migrations(db_path: Path) -> None:
    """Run all pending migrations."""
    log.info(f"Running migrations on {db_path}")
    migrate_add_is_backfilled(db_path)
    log.info("All migrations complete")
