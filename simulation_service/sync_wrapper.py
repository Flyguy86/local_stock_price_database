"""
Synchronous wrapper for async PostgreSQL operations.
Used by core.py which is synchronous but needs to save to PostgreSQL.
"""
import asyncio
import logging
from typing import Dict, Any

log = logging.getLogger("simulation.sync_wrapper")

# Global reference to the async db instance
_async_db = None


def set_db_instance(db):
    """Set the async database instance to use."""
    global _async_db
    _async_db = db


def _run_async(coro):
    """
    Run an async coroutine from sync context.
    Tries to use existing loop if available, otherwise creates new one.
    """
    try:
        # Try to get existing loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, can't use run_until_complete
            # Create a new task instead
            return asyncio.create_task(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def save_simulation_history_sync(model_id: str, ticker: str, stats: Dict[str, Any], params: Dict[str, Any]) -> str:
    """
    Synchronous wrapper for saving simulation history to PostgreSQL.
    
    Args:
        model_id: Model UUID
        ticker: Stock symbol
        stats: Simulation statistics
        params: Simulation parameters
        
    Returns:
        Simulation record ID
    """
    from .pg_db import save_simulation_history
    
    try:
        # Run the async function in sync context
        result = _run_async(save_simulation_history(model_id, ticker, stats, params))
        return result
    except Exception as e:
        log.error(f"Failed to save simulation history: {e}", exc_info=True)
        return None
