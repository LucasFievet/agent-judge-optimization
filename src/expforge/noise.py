"""
Configurable label noise: corrupt persona and/or goal-phase labels to simulate
a labeller with small unknown error rate (paper §3.8).
"""

import logging
import random

logger = logging.getLogger(__name__)
from copy import deepcopy
from pathlib import Path
from typing import Any

from expforge.trajectory.io import load_trajectory, save_trajectory
from expforge.trajectory.steps import Trajectory

PHASES = ("succeeded", "continue", "failed")


def _flip_phase(phase: str, error_rate: float, rng: random.Random) -> str:
    """With probability error_rate replace phase with a different one (uniform)."""
    if phase not in PHASES or error_rate <= 0 or rng.random() >= error_rate:
        return phase
    others = [p for p in PHASES if p != phase]
    return rng.choice(others)


def _flip_persona(
    persona_id: str,
    persona_ids: list[str],
    error_rate: float,
    rng: random.Random,
) -> str:
    """With probability error_rate replace persona with another (uniform)."""
    if not persona_ids or error_rate <= 0 or rng.random() >= error_rate:
        return persona_id
    others = [p for p in persona_ids if p != persona_id]
    if not others:
        return persona_id
    return rng.choice(others)


def add_label_noise(
    trajectory: Trajectory,
    *,
    persona_ids: list[str] | None = None,
    phase_error_rate: float = 0.0,
    persona_error_rate: float = 0.0,
    goal_error_rate: float = 0.0,
    rng: random.Random | None = None,
) -> Trajectory:
    """
    Return a copy of the trajectory with a corrupted `label` (and original in `label_true`).
    phase_error_rate: per-step probability of flipping phase (succeeded/continue/failed).
    persona_error_rate: probability of flipping persona_id.
    goal_error_rate: per-step probability of flipping goal_id to another goal from the trajectory (if any).
    persona_ids: list of all persona ids (for persona flip); if None, no persona flip.
    """
    rng = rng or random.Random()
    if not trajectory.label:
        return trajectory
    true_label = deepcopy(trajectory.label)
    persona_id = true_label.get("persona_id", trajectory.persona_id)
    goal_phase = true_label.get("goal_phase", [])
    goal_ids_in_traj = list(dict.fromkeys(g.get("goal_id", "") for g in goal_phase if g.get("goal_id")))
    if persona_error_rate > 0 and persona_ids:
        persona_id = _flip_persona(persona_id, persona_ids, persona_error_rate, rng)
    noisy_gp = []
    for g in goal_phase:
        goal_id = g.get("goal_id", "")
        phase = g.get("phase", "continue")
        if goal_error_rate > 0 and goal_ids_in_traj and rng.random() < goal_error_rate:
            goal_id = rng.choice(goal_ids_in_traj) if goal_ids_in_traj else goal_id
        phase = _flip_phase(phase, phase_error_rate, rng)
        noisy_gp.append({"goal_id": goal_id, "phase": phase})
    noisy_label = {"persona_id": persona_id, "goal_phase": noisy_gp, "label_true": true_label}
    out = deepcopy(trajectory)
    out.label = noisy_label
    return out


def add_label_noise_to_file(
    path: Path | str,
    output_path: Path | str | None = None,
    *,
    persona_ids: list[str] | None = None,
    phase_error_rate: float = 0.05,
    persona_error_rate: float = 0.0,
    goal_error_rate: float = 0.0,
    seed: int | None = None,
) -> Path:
    """
    Load trajectory from path, add label noise, save to output_path (default: overwrite).
    Returns path where saved.
    """
    path = Path(path)
    output_path = Path(output_path) if output_path else path
    traj = load_trajectory(path)
    rng = random.Random(seed)
    noisy = add_label_noise(
        traj,
        persona_ids=persona_ids,
        phase_error_rate=phase_error_rate,
        persona_error_rate=persona_error_rate,
        goal_error_rate=goal_error_rate,
        rng=rng,
    )
    save_trajectory(noisy, output_path)
    return output_path


def add_label_noise_to_experiment(
    experiment_dir: Path | str,
    persona_ids: list[str],
    *,
    phase_error_rate: float = 0.05,
    persona_error_rate: float = 0.02,
    goal_error_rate: float = 0.0,
    seed: int | None = None,
    in_place: bool = True,
    max_samples: int | None = None,
) -> list[Path]:
    """
    Add label noise to sample_*.yaml in experiment_dir/samples/.
    If max_samples is set, only the first max_samples (after sorting) are processed.
    If in_place=True, overwrites files; else writes to samples_noisy/.
    Returns list of paths written.
    """
    exp_dir = Path(experiment_dir)
    samples_dir = exp_dir / "samples"
    if not samples_dir.is_dir():
        return []
    out_dir = samples_dir if in_place else (exp_dir / "samples_noisy")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        samples_dir.glob("sample_*.yaml"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
    )
    if max_samples is not None:
        paths = paths[:max_samples]
    written = []
    for i, p in enumerate(paths):
        if (i + 1) % 500 == 0 or i == 0:
            logger.info("[noise] Processing sample %d/%d", i + 1, len(paths))
        rng = random.Random(seed + i if seed is not None else None)
        traj = load_trajectory(p)
        noisy = add_label_noise(
            traj,
            persona_ids=persona_ids,
            phase_error_rate=phase_error_rate,
            persona_error_rate=persona_error_rate,
            goal_error_rate=goal_error_rate,
            rng=rng,
        )
        out_path = out_dir / p.name
        save_trajectory(noisy, out_path)
        written.append(out_path)
    return written
