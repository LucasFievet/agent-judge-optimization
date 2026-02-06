"""Schema and config for persona generation."""

from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """Configuration for persona set generation."""

    experiment_id: str
    n_personas: int
    seed: int | None = None
