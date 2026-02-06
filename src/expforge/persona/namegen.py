"""Generate a short persona name from traits using a lightweight LLM (e.g. Gemini 2.0 Flash)."""

from typing import Any


# Prefer Gemini 2.0 Flash or 2.5 Flash-Lite; no Vertex project required if using API key.
DEFAULT_MODEL = "gemini-2.0-flash"


def generate_persona_name(
    technical: float,
    determined: float,
    swearing: float,
    baseline_sentiment: float,
    *,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Return a short (2–4 word) persona name that fits the trait profile.
    Uses Gemini 2.0 Flash (or similar) if available; otherwise returns a fallback.
    """
    prompt = (
        "Generate a very short persona label (2 to 4 words) for a user profile with these traits (0–1 scale): "
        f"technical={technical:.2f}, determined={determined:.2f}, swearing={swearing:.2f}, baseline_sentiment={baseline_sentiment:.2f}. "
        "Reply with only the label, no quotes or explanation."
    )
    try:
        import google.generativeai as genai
        genai.configure(api_key=_get_api_key())
        m = genai.GenerativeModel(model)
        r = m.generate_content(prompt)
        if r and r.text:
            name = r.text.strip().strip('"\'')
            if name and len(name) < 80:
                return name
    except Exception:
        pass
    return ""


def _get_api_key() -> str:
    import os
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def safe_generate_persona_name(
    technical: float,
    determined: float,
    swearing: float,
    baseline_sentiment: float,
    fallback: str,
    *,
    model: str = DEFAULT_MODEL,
) -> str:
    """Like generate_persona_name but always returns a string; uses fallback on failure."""
    name = generate_persona_name(technical, determined, swearing, baseline_sentiment, model=model)
    return name if name else fallback
