"""State representation for the nested Markov trajectory."""

from dataclasses import dataclass
from enum import Enum


class NestedState(str, Enum):
    """Per-goal sub-state."""

    SUCCEEDED = "succeeded"
    CONTINUE = "continue"
    FAILED = "failed"


class TopLevelState(str, Enum):
    """Top-level chain state."""

    START = "start"
    FINISHED = "finished"
    ABANDONED = "abandoned"
    PUBLISH = "publish"
    SUBSCRIBE = "subscribe"
    # Goal states are dynamic (goal_id)


@dataclass
class TrajectoryState:
    """Current latent state of a trajectory."""

    top_level: str  # start | goal_id | publish | subscribe | finished | abandoned
    nested: NestedState | None  # None when not in a goal
    quality: float
