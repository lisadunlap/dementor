import pandas as pd

from experiments.steering import harmonize_to_200


def _row(prompt, expected, direction, alpha, response, rtl_code="A"):
    return {
        "prompt": prompt,
        "expected": expected,
        "direction": direction,
        "alpha": alpha,
        "model_response": response,
        "rtl_code": rtl_code,
        "ppl": 2.0,
        "rep4": 0.0,
    }


def test_score_preserves_harm_and_over_refusal_denominators():
    rows = []
    for direction in ("baseline", "cone", "fingerprint", "random"):
        rows.append(_row("harmful", "refuse", direction, 1.0,
                         "I cannot help with that request.", rtl_code="B"))
        rows.append(_row("benign", "comply", direction, 1.0,
                         "Here is a sufficiently long useful answer."))
    scored = harmonize_to_200.score(harmonize_to_200.derive(pd.DataFrame(rows)))

    assert scored["n_prompts"] == 2
    assert scored["n_harmful"] == 1
    assert scored["n_benign"] == 1
    assert scored["axes"] == ["harm", "over_refusal"]
    assert "verdict" in scored
    assert "baseline_over_refusal" in scored


def test_score_over_refusal_only_does_not_invent_harm_verdict():
    rows = [
        _row("benign", "comply", direction, 1.0,
             "Here is a sufficiently long useful answer.")
        for direction in ("baseline", "cone", "fingerprint", "random")
    ]
    scored = harmonize_to_200.score(harmonize_to_200.derive(pd.DataFrame(rows)))

    assert scored["axes"] == ["over_refusal"]
    assert scored["n_prompts"] == 1
    assert "verdict" not in scored
    assert "baseline_harm" not in scored
