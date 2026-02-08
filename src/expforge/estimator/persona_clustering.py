"""Cluster trajectories by inferred persona (assign each trajectory to a persona)."""

from pathlib import Path
from typing import Any

from expforge.estimator.persona_scoring import score_persona_logits_from_path


def assign_persona_soft(trajectory_path: Path, n_personas: int = 5) -> list[float]:
    """Return soft assignment (probabilities over personas) for one trajectory."""
    logits = score_persona_logits_from_path(trajectory_path)
    # Normalize to probabilities
    import math
    max_l = max(logits)
    exp_l = [math.exp(l - max_l) for l in logits[:n_personas]]
    total = sum(exp_l)
    return [e / total for e in exp_l]


def assign_persona_hard(trajectory_path: Path, n_personas: int = 5) -> int:
    """Return hard assignment (persona index 0..K-1) for one trajectory."""
    probs = assign_persona_soft(trajectory_path, n_personas=n_personas)
    return int(max(range(len(probs)), key=lambda i: probs[i]))
