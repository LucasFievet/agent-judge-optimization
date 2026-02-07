"""Shared paths for experiment data. Results are never stored under src/."""

import os
from pathlib import Path

# Env override for experiments base directory
EXPFORGE_EXPERIMENTS_DIR_ENV = "EXPFORGE_EXPERIMENTS_DIR"


def get_project_root() -> Path:
    """Directory containing pyproject.toml or .git, walking up from cwd; else cwd."""
    cwd = Path.cwd().resolve()
    current = cwd
    for _ in range(20):
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return cwd


def get_experiments_base_dir() -> Path:
    """Base directory for experiment outputs (persona, goals, samples).
    Prefer env EXPFORGE_EXPERIMENTS_DIR; else <project_root>/.data."""
    if os.getenv(EXPFORGE_EXPERIMENTS_DIR_ENV):
        return Path(os.environ[EXPFORGE_EXPERIMENTS_DIR_ENV]).expanduser().resolve()
    return get_project_root() / ".data"


def experiment_dir(base_dir: Path, experiment_id: str) -> Path:
    """Path to a single experiment under base_dir: base_dir/experiment/<experiment_id>/."""
    return base_dir / "experiment" / experiment_id


# For backward compatibility: default base dir when no override is passed
DEFAULT_EXPERIMENTS_DIR = get_experiments_base_dir()
