# Makes 'scripts' a proper Python package so absolute imports like
# `from scripts.serve.utils_llm import get_llm_output` work when running
# modules directly (e.g., `python scripts/generate_responses.py basic ...`).
