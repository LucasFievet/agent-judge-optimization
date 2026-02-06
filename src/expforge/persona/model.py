"""Data structures for persona configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PersonaSpec:
    """Specification for a single persona (user type)."""

    id: str
    name: str
    weight: float
    technical: float
    determined: float
    swearing: float
    baseline_sentiment: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonaSet:
    """Configuration for a set of personas (finite user distribution)."""

    experiment_id: str
    personas: list[PersonaSpec]

    def get_weights(self) -> list[float]:
        """Return list of persona weights (must sum to 1)."""
        return [p.weight for p in self.personas]

    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(p.weight for p in self.personas)
        if total <= 0:
            raise ValueError("Total persona weight must be positive")
        for p in self.personas:
            p.weight /= total
