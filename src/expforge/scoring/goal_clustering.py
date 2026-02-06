"""Cluster message segments into goals (goal identity from tools or embeddings)."""

from pathlib import Path
from typing import Any

from expforge.scoring.goal_scoring import score_goal_phases_for_trajectory


def segment_by_phases(phase_labels: list[str]) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) for each segment (start..success/fail)."""
    segments = []
    i = 0
    while i < len(phase_labels):
        if phase_labels[i] in ("start", "continue"):
            j = i + 1
            while j < len(phase_labels) and phase_labels[j] not in ("success", "fail"):
                j += 1
            if j < len(phase_labels):
                segments.append((i, j))
            i = j + 1
        else:
            i += 1
    return segments


def segment_trajectory(trajectory_path: Path) -> list[tuple[int, int, list[str]]]:
    """
    Return list of (start, end, tools_used) for each goal segment in the trajectory.
    Goal identity (cluster) is not assigned here; use cluster_goal_segments.
    """
    import yaml

    phases = score_goal_phases_for_trajectory(trajectory_path)
    segs = segment_by_phases(phases)
    with trajectory_path.open() as f:
        data = yaml.safe_load(f)
    steps = data.get("steps", [])
    out = []
    for start, end in segs:
        tools = []
        for s in steps[start : end + 1]:
            tools.extend(s.get("tools_used", []))
        out.append((start, end, list(set(tools))))
    return out


def cluster_goal_segments(
    all_segments: list[tuple[int, int, list[str]]], n_clusters: int = 6
) -> list[int]:
    """
    Assign each segment to a goal cluster (0..n_clusters-1) by tool-set similarity.
    Returns cluster index per segment.
    """
    from collections import Counter

    # Stub: cluster by frozenset of tools (or use sklearn KMeans on one-hot tools)
    tool_sets = [frozenset(tools) for (_, _, tools) in all_segments]
    unique = list(dict.fromkeys(tool_sets))
    cluster_map = {s: i % n_clusters for i, s in enumerate(unique)}
    return [cluster_map[s] for s in tool_sets]
