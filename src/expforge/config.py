"""Configuration management for Vertex AI resources."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class ExpforgeConfig:
    """Experiment Forge configuration settings."""
    
    project_id: str
    location: str
    bucket_name: str
    experiment_name: str
    tensorboard_name: str
    
    # Custom Training Job defaults
    machine_type: str
    accelerator_type: Optional[str]
    accelerator_count: int
    container_uri: str
    serving_container_image_uri: str
    
    # Data paths
    data_root_gcs: str = "data"  # Relative path in GCS bucket for datasets
    data_root_local: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / ".data")  # Local data root directory
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> ExpforgeConfig:
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def load(cls, path: Path) -> ExpforgeConfig:
        """Load configuration from JSON file."""
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {path}\n"
                f"Copy config.json.example to config.json and update with your values."
            )
        
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_config_path() -> Path:
    """Get the default configuration file path."""
    # Look for config in project root or user home
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config.json"
    
    # Also check for .expforge directory in home
    home_config = Path.home() / ".expforge" / "config.json"
    
    # Prefer project root, fall back to home
    if config_path.exists():
        return config_path
    elif home_config.exists():
        return home_config
    else:
        # Default to project root
        return config_path


def load_config() -> ExpforgeConfig:
    """Load configuration from file."""
    config_path = get_config_path()
    return ExpforgeConfig.load(config_path)


def main():
    """CLI entry point for config module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View Vertex AI configuration")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    args = parser.parse_args()
    
    if args.show:
        config = load_config()
        print(json.dumps(config.to_dict(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
