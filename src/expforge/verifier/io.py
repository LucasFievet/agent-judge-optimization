"""Load/save experiment config and copy experiments."""

import re
import shutil
from pathlib import Path

from expforge.persona import PersonaSet, load_persona_set
from expforge.goal import GoalSet, load_goal_set
from expforge.paths import get_experiments_base_dir, experiment_dir as _experiment_dir

# Re-export for backward compatibility; default is <project_root>/.data (set EXPFORGE_EXPERIMENTS_DIR to override)
DEFAULT_EXPERIMENTS_DIR = get_experiments_base_dir()


def experiment_dir(base_dir: Path, experiment_id: str) -> Path:
    """Path to experiment dir: base_dir/experiment/<experiment_id>/."""
    return _experiment_dir(base_dir, experiment_id)


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


def load_existing_sample_paths(base_dir: Path, experiment_id: str) -> list[Path]:
    """Return sorted list of sample_*.yaml paths (by sample number). Empty if no samples dir or no files."""
    exp_dir = experiment_dir(base_dir, experiment_id)
    samples_dir = exp_dir / "samples"
    if not samples_dir.is_dir():
        return []
    paths = list(samples_dir.glob("sample_*.yaml"))
    paths.sort(key=lambda p: int(re.search(r"\d+", p.stem).group(0)) if re.search(r"\d+", p.stem) else 0)
    return paths


def delete_sample_files(base_dir: Path, experiment_id: str) -> None:
    """Remove all sample_*.yaml in the experiment's samples dir (for --override)."""
    exp_dir = experiment_dir(base_dir, experiment_id)
    samples_dir = exp_dir / "samples"
    if not samples_dir.is_dir():
        return
    for p in samples_dir.glob("sample_*.yaml"):
        p.unlink()
