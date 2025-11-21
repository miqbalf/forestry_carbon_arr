"""
Utility functions for Planet API integration.
"""

import asyncio
from datetime import datetime
from typing import Any

# Try to import dateutil, fallback to datetime parsing
try:
    from dateutil import parser as date_parser
    
    def parse_date(date_str):
        """Parse ISO format date string to datetime object."""
        return date_parser.parse(date_str) if isinstance(date_str, str) else date_str
except ImportError:
    # Fallback: use datetime.fromisoformat (Python 3.7+)
    def parse_date(date_str):
        """Parse ISO format date string to datetime object."""
        if isinstance(date_str, str):
            # Remove 'Z' and replace with '+00:00' if needed
            if date_str.endswith('Z'):
                date_str = date_str[:-1] + '+00:00'
            return datetime.fromisoformat(date_str)
        return date_str


def run_async(coro):
    """
    Run async coroutine, handling both sync and async contexts (e.g., Jupyter notebooks).
    
    Args:
        coro: Async coroutine to run
        
    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context (like Jupyter), we need to use nest_asyncio or create_task
        # For Jupyter, we'll use nest_asyncio if available, otherwise create a new task
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(coro)
        except ImportError:
            # If nest_asyncio not available, create a task
            import concurrent.futures
            import threading
            
            # Run in a new thread with a new event loop
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)

