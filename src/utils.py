"""
Utility helpers shared across the project.

This module is intentionally minimal and focused on small,
reusable, side-effect-free helper functions.
"""

from datetime import date


def today_iso() -> str:
    """
    Return today's date in ISO 8601 format (YYYY-MM-DD).

    - Default end dates in data downloads
    - Logging
    - API responses
    - Reproducible CLI commands

    Returns
    -------
    str
        Current date formatted as ISO string.
    """
    return date.today().isoformat()
