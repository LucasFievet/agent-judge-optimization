"""
Simulated agent response: produces actual artifacts as if it executed the task
(abstract text, table, diagram description, references, conclusion, etc.).
The user will then react to what the agent actually produced.
"""

from expforge.simulator.llm_utils import generate_text


# Goal id -> short description of the artifact to produce (for the prompt)
ARTIFACT_BRIEF = {
    "write_abstract": "a short abstract (2-4 sentences) for the document",
    "write_section": "a short body section (2-5 sentences) on the topic",
    "create_table": "a small markdown or text table (3-5 rows) relevant to the request",
    "make_diagram": "a brief description of a diagram or figure (what it shows and labels)",
    "research_refs": "2-3 reference citations or URLs as if you looked them up",
    "run_script": "one line stating what the script did (e.g. 'Ran the script; output: 3 files generated')",
    "write_conclusion": "a short conclusion paragraph (2-4 sentences)",
}


def generate_agent_message(
    user_message: str,
    tools_used: list[str],
    nested_outcome: str | None,
    goal_name: str = "",
    goal_id: str = "",
) -> str:
    """
    Generate an agent reply that contains an actual artifact (abstract, table, diagram description, etc.)
    as if the agent truly executed the task. The user will look at this output and may get frustrated
    if it does not match their intent or the outcome was failure.
    """
    tools_str = ", ".join(tools_used) if tools_used else "none"
    outcome_str = nested_outcome or "n/a"
    artifact_hint = ARTIFACT_BRIEF.get(goal_id, "the requested content")
    prompt = (
        "You are simulating the agent's reply in a writing-assistant session. "
        "The user said: \"%s\"\n"
        "The agent used tools: [%s]. Goal: %s. Outcome: %s.\n"
        "Produce the actual artifact the agent would return: %s. "
        "Do NOT add meta-commentary like 'Here is the abstract' or 'I have created...' — output the artifact itself "
        "(e.g. the abstract text, the table, the diagram description, the references, or the conclusion). "
        "If outcome is failed, output a short note on what went wrong or a partial/broken artifact. "
        "Keep it concise (under 150 words for the artifact). Output only the agent reply, no quotes."
    ) % (user_message[:300], tools_str, goal_name or goal_id, outcome_str, artifact_hint)
    out = generate_text(prompt, max_tokens=350)
    if out:
        return out.strip().strip('"\'')
    return _fallback_agent_message(tools_used, nested_outcome, goal_name)


def _fallback_agent_message(tools_used: list[str], nested_outcome: str | None, goal_name: str = "") -> str:
    t = ", ".join(tools_used) if tools_used else "no tools"
    if nested_outcome == "succeeded":
        return f"[Used {t} for %s.] Done." % (goal_name or "task")
    if nested_outcome == "failed":
        return f"[Tried {t} for %s.] It failed." % (goal_name or "task")
    return f"[Used {t} for %s.] In progress." % (goal_name or "task")
