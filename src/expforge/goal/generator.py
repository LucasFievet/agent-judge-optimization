"""Generate a set of goals and tools for an experiment."""

import hashlib
import random

from expforge.goal.model import GoalSet, GoalSpec, Tool
from expforge.goal.schema import GeneratorConfig


def _effective_seed(base_seed: int | None, experiment_id: str) -> int | None:
    """Derive a stable seed per (base_seed, experiment_id) so different experiments get different tools/goals."""
    if base_seed is None:
        return None
    h = hashlib.sha256(f"{base_seed}_{experiment_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# Default tools: id -> (name, base quality). Qualities may be perturbed when seed is set.
DEFAULT_TOOLS = [
    Tool("llm", 0.7, "LLM"),
    Tool("edit", 0.65, "Edit"),
    Tool("table", 0.6, "Table"),
    Tool("fetch_web", 0.6, "Fetch web"),
    Tool("screenshot", 0.5, "Screenshot"),
    Tool("score", 0.55, "Score"),
    Tool("execute", 0.5, "Execute"),
]

# Default goals: (id, name, list of tool ids)
DEFAULT_GOALS = [
    ("write_abstract", "Write abstract", ["llm", "edit"]),
    ("create_table", "Create table", ["llm", "table"]),
    ("make_diagram", "Make diagram", ["fetch_web", "screenshot", "score"]),
    ("research_refs", "Research references", ["fetch_web", "llm"]),
    ("run_script", "Run script", ["execute", "llm"]),
    ("write_conclusion", "Write conclusion", ["llm", "edit"]),
]


class GoalGenerator:
    """Generates goal sets from a config (default or template-based). With a seed, tool qualities are perturbed per experiment."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def generate(self) -> GoalSet:
        """Produce a GoalSet for the experiment (tools + goals). If seed is set, tool qualities are perturbed
        with a wider range and a per-experiment strength multiplier so systems vary more (some 'easy', some 'hard')."""
        effective = _effective_seed(self.config.seed, self.config.experiment_id)
        if effective is not None:
            rng = random.Random(effective)
            # Wide strength range [0.35, 1.5]: some experiments have weak tools (low p_success), others strong
            strength = rng.uniform(0.35, 1.5)
            tools = []
            for t in DEFAULT_TOOLS:
                base = rng.uniform(0.15, 0.95)
                q = max(0.12, min(0.98, base * strength))
                tools.append(Tool(t.id, round(q, 2), t.name, {}))
        else:
            tools = [Tool(t.id, t.quality, t.name, {}) for t in DEFAULT_TOOLS]
        goals = [
            GoalSpec(id=gid, name=name, tools=list(tool_ids), meta={})
            for gid, name, tool_ids in DEFAULT_GOALS
        ]
        return GoalSet(experiment_id=self.config.experiment_id, goals=goals, tools=tools)


def generate_goal_set(experiment_id: str, seed: int | None = None) -> GoalSet:
    """Convenience: generate a goal set for an experiment."""
    config = GeneratorConfig(experiment_id=experiment_id, seed=seed)
    return GoalGenerator(config).generate()
