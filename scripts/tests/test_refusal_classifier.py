"""Pytest wrapper for the deterministic refusal classifier (Direction 3, scaffold).

No model spend: pure string-matching over synthetic, locally-authored strings.
"""
from scripts.safety.refusal_classifier import (
    _REFUSAL_CASES,
    classify_refusal,
    is_degenerate,
    redact_snippet,
)


def test_synthetic_refusal_cases():
    for text, expected in _REFUSAL_CASES:
        assert classify_refusal(text) == expected, repr(text)


def test_degenerate_outputs_count_as_refusal():
    for bad in ("", "   ", "...", None, "\n\t"):
        assert is_degenerate(bad)
        assert classify_refusal(bad) == 1


def test_compliant_outputs_not_flagged():
    assert classify_refusal("The capital of France is Paris.") == 0
    # 'cannot' / "can't" used non-refusally must not trip the classifier.
    assert classify_refusal("You cannot divide by zero, so it returns None.") == 0


def test_redaction_bounds_and_collapses():
    assert len(redact_snippet("x" * 500, max_chars=200)) == 200
    assert redact_snippet("  a\n\n  b  ") == "a b"
    assert redact_snippet(None) == ""
