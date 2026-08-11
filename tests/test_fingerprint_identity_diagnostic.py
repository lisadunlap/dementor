import json

import pytest

from experiments.steering.test_fingerprint_is_identity import summarize


def test_summary_is_json_serializable_and_paired():
    rows = [
        {"fingerprint_delta": 0.2, "random_delta": 0.1},
        {"fingerprint_delta": -0.1, "random_delta": 0.0},
    ]

    summary = summarize(rows)

    assert summary["n_models"] == 2
    assert summary["mean_fingerprint_delta"] == pytest.approx(0.05)
    assert summary["mean_random_delta"] == pytest.approx(0.05)
    assert summary["fingerprint_gt_random"] == 1
    json.dumps(summary)
