"""Load/save experiment config and copy experiments."""

import shutil
from pathlib import Path

from expforge.persona import PersonaSet, load_persona_set
from expforge.goal import GoalSet, load_goal_set


def _bootstrap_experiment(base_dir: Path, experiment_id: str, *, seed: int | None = None) -> None:
    """Create persona.yaml, goals.yaml, transitions.yaml by running simulator with 0 samples."""
    from expforge.simulator.experiment_simulator import run_simulator
    run_simulator(
        experiment_id,
        0,
        base_dir=base_dir,
        seed=seed,
        reuse_config=False,
        use_llm=False,
    )


def ensure_experiment_exists(base_dir: Path, experiment_id: str, *, seed: int | None = None) -> None:
    """If experiment dir or persona.yaml is missing, generate config (persona + goals + transitions)."""
    exp_dir = experiment_dir(base_dir, experiment_id)
    if not (exp_dir / "persona.yaml").exists():
        _bootstrap_experiment(base_dir, experiment_id, seed=seed)


def experiment_dir(base_dir: Path, experiment_id: str) -> Path:
    return base_dir / "experiment" / experiment_id


def copy_experiment(
    source_id: str,
    target_id: str,
    base_dir: Path | str,
) -> Path:
    """Copy source experiment (persona, goals, transitions) to target. Returns target exp dir."""
    base_dir = Path(base_dir)
    src = experiment_dir(base_dir, source_id)
    dst = experiment_dir(base_dir, target_id)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("persona.yaml", "goals.yaml", "transitions.yaml"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    (dst / "samples").mkdir(exist_ok=True)
    return dst


def load_experiment(base_dir: Path, experiment_id: str) -> tuple[PersonaSet, GoalSet]:
    exp_dir = experiment_dir(base_dir, experiment_id)
    persona_set = load_persona_set(exp_dir / "persona.yaml")
    goal_set = load_goal_set(exp_dir / "goals.yaml")
    return persona_set, goal_set
