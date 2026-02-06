"""Shared LLM call helper for simulator (Gemini 2.0 Flash)."""

import os
import sys
from pathlib import Path

DEFAULT_MODEL = "gemini-2.0-flash"

# Set EXPFORGE_LOG_LLM=1 to log all LLM inputs/outputs to stderr (default: enabled so you can see generation)
_LOG_LLM = os.environ.get("EXPFORGE_LOG_LLM", "1").strip().lower() in ("1", "true", "yes")


_dotenv_loaded = False


def _load_dotenv() -> None:
    """Load .env from cwd or parent dirs so API key is available when direnv didn't run."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return
    cwd = Path.cwd()
    for d in [cwd] + list(cwd.parents):
        env_file = d / ".env"
        if env_file.is_file():
            try:
                with env_file.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, _, value = line.partition("=")
                            key = key.strip()
                            value = value.strip().strip("'\"").strip()
                            if key and key not in os.environ:
                                os.environ[key] = value
            except OSError:
                pass
            break


def _api_key() -> str:
    _load_dotenv()
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def _log_llm_call(prompt: str, response: str, model: str) -> None:
    if not _LOG_LLM:
        return
    sep = "=" * 60
    print(sep, file=sys.stderr)
    print("LLM INPUT (prompt):", file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    print(prompt, file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    print("LLM OUTPUT (response):", file=sys.stderr)
    print(response or "(empty or error)", file=sys.stderr)
    print(sep, file=sys.stderr)
    sys.stderr.flush()


def generate_text(prompt: str, *, model: str = DEFAULT_MODEL, max_tokens: int = 256) -> str:
    """Return LLM-generated text; empty string on failure or missing API key."""
    out = ""
    try:
        import google.generativeai as genai
        key = _api_key()
        if not key:
            _log_llm_call(prompt, "(skipped: no GEMINI_API_KEY / GOOGLE_API_KEY)", model)
            return ""
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model)
        r = m.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens))
        if r and r.text:
            out = r.text.strip()
    except Exception as e:
        out = f"(error: {e!r})"
    _log_llm_call(prompt, out, model)
    return out if out and not out.startswith("(") else ""
