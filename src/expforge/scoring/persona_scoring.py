"""Score trajectory with function f(trajectory) -> logits for persona membership."""

from pathlib import Path
from typing import Any

# Stub: in practice this would call an LLM or trained model
# f(trajectory) produces logits for P(user in persona k) for each k


def score_persona_logits(trajectory: dict[str, Any]) -> list[float]:
    """
    Return logits (or probabilities) for the user belonging to each persona.

    Default: uniform over K personas (stub). Replace with LLM or classifier
    that takes full message sequence and tool-use pattern.
    """
    # Stub: assume 5 personas, uniform
    return [0.2] * 5


def score_persona_logits_from_path(trajectory_path: Path) -> list[float]:
    """Load trajectory from YAML and return persona logits."""
    import yaml

    with trajectory_path.open() as f:
        data = yaml.safe_load(f)
    return score_persona_logits(data)
