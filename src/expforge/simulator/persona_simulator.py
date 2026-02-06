"""
Simulated user: overarching sample goal (LLM) and per-turn message + next action (LLM).
Persona sees full conversation, sample_goal, current state, possible actions; decides user message and next transition.
"""

import re
from expforge.persona import PersonaSpec
from expforge.goal import GoalSet

from expforge.simulator.llm_utils import generate_text


def generate_sample_goal(goal_set: GoalSet, persona: PersonaSpec | None = None) -> str:
    """
    Generate one realistic overarching goal/plan for this sample (LLM).
    Constrained to academic or semi-academic papers (research, technical reports, thesis), not marketing or blogs.
    """
    goal_names = [g.name for g in goal_set.goals]
    prompt = (
        "Generate a realistic, specific goal for writing an academic or semi-academic paper in one sentence (20-40 words). "
        "Context: research papers, technical reports, thesis chapters, or similar — NOT marketing, blog posts, or emails. "
        "Include: (1) a concrete academic topic (e.g. 'paper on Bayesian inference in climate models', 'technical report on RAG systems'), "
        "and (2) at least two specific deliverables from this list: %s. "
        "Example: 'Write a short paper on coral reef conservation with an abstract, one results section with a table of species counts, and a conclusion.' "
        "Reply with only the goal sentence, no quotes or labels."
    ) % (", ".join(goal_names),)
    out = generate_text(prompt, max_tokens=80)
    if out:
        return out.strip().strip('"\'')
    return "Write an academic paper with an abstract, one section with a table, and a conclusion."


def generate_user_message_and_next_action(
    persona: PersonaSpec,
    goal_set: GoalSet,
    conversation_so_far: list[tuple[str, str]],
    sample_goal: str,
    top_level_state: str,
    nested_state: str | None,
    nested_outcome: str | None,
    allowed_next: list[str],
) -> tuple[str, str]:
    """
    Persona sees full conversation, sample_goal, current state, and possible actions.
    Returns (user_message, next_action) where next_action is one of allowed_next.
    """
    goal_name = ""
    if top_level_state not in ("start", "publish", "subscribe", "finished", "abandoned"):
        g = next((x for x in goal_set.goals if x.id == top_level_state), None)
        goal_name = g.name if g else top_level_state
    conv_str = "\n".join(
        f"User: {u}\nAgent: {a}" for u, a in conversation_so_far[-10:]
    )  # last 10 turns
    if not conv_str:
        conv_str = "(No conversation yet.)"
    actions_str = ", ".join(allowed_next)
    n_turns = len(conversation_so_far)
    length_hint = (
        " Prefer choosing 'finished' or 'abandoned' if they are allowed and the conversation has already had several exchanges."
        if n_turns >= 4 and ("finished" in allowed_next or "abandoned" in allowed_next)
        else ""
    )
    prompt = (
        "You simulate a realistic user in a writing-assistant session.\n"
        "Overarching goal: %s\n"
        "User traits (0-1): technical=%.2f, determined=%.2f, swearing=%.2f, baseline_sentiment=%.2f.\n"
        "The user describes context and gives concrete details about what they want; they do not dwell on their feelings or ask the AI for guidance. "
        "They look at what the agent actually produced (the last Agent message) and react to it. "
        "They may get frustrated or correct the agent if it did not properly interpret their intent or if the outcome was failure.\n"
        "Conversation so far:\n%s\n"
        "Current context: %s, nested=%s. This step's outcome: %s.\n"
        "Possible next actions (choose exactly one): %s.%s\n"
        "Output exactly two lines:\nLine 1: The user's message (describe what they want / react to the agent's output; be concrete, not meta).\n"
        "Line 2: NEXT: <action> where <action> is one of the possible actions, exactly as written."
    ) % (
        sample_goal,
        persona.technical,
        persona.determined,
        persona.swearing,
        persona.baseline_sentiment,
        conv_str,
        goal_name or top_level_state,
        nested_state or "n/a",
        nested_outcome or "n/a",
        actions_str,
        length_hint,
    )
    out = generate_text(prompt, max_tokens=200)
    if out:
        user_msg, next_act = _parse_user_and_next(out, allowed_next)
        if user_msg is not None and next_act is not None:
            return (user_msg, next_act)
    return (
        _fallback_user_message(top_level_state, nested_state, goal_name),
        allowed_next[0] if allowed_next else "finished",
    )


def _parse_user_and_next(text: str, allowed_next: list[str]) -> tuple[str | None, str | None]:
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return (None, None)
    user_msg = lines[0].strip().strip('"\'')
    next_line = lines[-1].upper()
    # "NEXT: write_abstract" or "NEXT: finished"
    m = re.search(r"NEXT:\s*(\S+)", next_line, re.I)
    if m:
        chosen = m.group(1).strip()
        for a in allowed_next:
            if a.upper() == chosen.upper():
                return (user_msg, a)
    return (user_msg if user_msg else None, None)


def _fallback_user_message(top_level: str, nested: str | None, goal_name: str) -> str:
    if top_level == "start":
        return "I need help with my document."
    if top_level in ("publish", "subscribe", "finished", "abandoned"):
        return "Thanks."
    if nested == "succeeded":
        return f"Got it, the {goal_name or top_level} part worked."
    if nested == "failed":
        return f"The {goal_name or top_level} step didn't work. What now?"
    return f"Working on {goal_name or top_level}."
