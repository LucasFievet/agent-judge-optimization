"""Generate a set of goals and tools for an experiment."""

from expforge.goal.model import GoalSet, GoalSpec, Tool
from expforge.goal.schema import GeneratorConfig


# Default tools: id -> (name, quality)
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
    """Generates goal sets from a config (default or template-based)."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def generate(self) -> GoalSet:
        """Produce a GoalSet for the experiment (tools + goals)."""
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
