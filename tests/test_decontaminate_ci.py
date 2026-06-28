"""CPU unit tests for the de-confound multiseed CI fix in experiments.analysis.decontaminate.

Verifies the two statistical bugs in ``_report_multiseed``'s CI are fixed, via the tiny
``_ci95_halfwidth`` helper factored out for exactly this purpose (no GPU, no data files,
pure synthetic input):

  (i)  the 95% CI multiplier for n=3 adapter seeds is the Student-t t(.975, df=2)=4.303,
       NOT the normal 1.96 (which understated the half-width ~2.2x);
  (ii) a cell with n<2 has an UNDEFINED CI (NaN) and is EXCLUDED from the survivor set,
       instead of being handed a spurious zero-width interval (old ``sd.fillna(0)``) that
       let it falsely clear the ``mean - ci95 > 0.3`` survivor bar.
"""
import math

import numpy as np
import pandas as pd
import pytest

from experiments.analysis.decontaminate import _ci95_halfwidth


def test_t_multiplier_is_4303_for_n3_not_196():
    sd, n = 0.12, 3
    hw = float(_ci95_halfwidth(sd, n))
    se = sd / math.sqrt(n)
    implied_mult = hw / se  # back out the multiplier applied to the standard error

    assert implied_mult == pytest.approx(4.3027, abs=1e-3)  # Student-t, df = n-1 = 2
    assert abs(implied_mult - 1.96) > 1.0                   # decisively NOT the normal 1.96
    # the correct half-width is ~2.2x wider than the buggy normal approximation
    buggy_normal = 1.96 * se
    assert hw / buggy_normal == pytest.approx(4.3027 / 1.96, rel=1e-3)


def test_n_lt_2_cell_excluded_not_zero_width():
    # n<2: no within-cell variance estimate -> CI undefined (NaN), regardless of sd.
    assert math.isnan(float(_ci95_halfwidth(0.0, 1)))
    assert math.isnan(float(_ci95_halfwidth(0.5, 1)))

    # End-to-end on the survivor rule. The OLD bug: sd.fillna(0) gave the n=1 cell ci95=0,
    # so mean-0 > 0.3 made it "survive" on a single seed. Under the fix it is excluded.
    g = pd.DataFrame({
        "base_rung": ["dpo", "dpo"],
        "mean":      [0.90,  0.90],    # both have high point estimates
        "sd":        [np.nan, 0.05],   # row0: single seed (n=1); row1: genuine n=3 spread
        "n":         [1,      3],
    })
    g["ci95"] = _ci95_halfwidth(g["sd"], g["n"])

    assert math.isnan(g.loc[0, "ci95"])   # n=1 -> NaN, NOT a zero-width interval
    assert g.loc[1, "ci95"] > 0           # n=3 -> a real, positive half-width

    surv = g[(g["n"] >= 2) & (g["mean"] - g["ci95"] > 0.3)]
    assert 0 not in surv.index            # the single-seed cell is excluded...
    assert list(surv.index) == [1]        # ...only the genuine n=3 cell survives


def test_vectorized_matches_scalar_and_nan_for_n1():
    sd = np.array([0.10, 0.20, np.nan])
    n = np.array([3, 5, 1])
    hw = _ci95_halfwidth(sd, n)

    assert hw[0] == pytest.approx(float(_ci95_halfwidth(0.10, 3)))
    assert hw[1] == pytest.approx(float(_ci95_halfwidth(0.20, 5)))
    assert math.isnan(hw[2])              # n=1 element -> NaN even vectorized
