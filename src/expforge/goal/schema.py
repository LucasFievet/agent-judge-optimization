"""Schema and config for goal generation."""

from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """Configuration for goal set generation."""

    experiment_id: str
    seed: int | None = None
