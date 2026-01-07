"""Utility functions for data module."""

from __future__ import annotations

from pathlib import Path


def get_data_root() -> Path:
    """Get the root data directory (.data at project root)."""
    # Find project root by looking for common marker files
    current = Path(__file__).resolve()
    markers = ["pyproject.toml", "README.md"]
    
    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return current / ".data"
        current = current.parent
    
    # Fallback: assume we're in src/expforge/data, go up 4 levels
    return Path(__file__).resolve().parent.parent.parent.parent / ".data"
