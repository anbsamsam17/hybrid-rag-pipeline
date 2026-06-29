"""Tests for the paired percentile bootstrap (ADR-0004).

The bootstrap is the statistic that licenses a "treatment beats baseline by X [CI95]" claim,
so these tests pin the properties that make such a claim defensible:

* **Reproducibility** — same (data, seed, B) ⇒ a bit-identical :class:`BootstrapResult`.
* **Pairing** — swapping baseline ↔ treatment (same seed, same index matrix) negates the diff
  and maps the interval ``(lo, hi)`` to ``(-hi, -lo)``; this only holds for a *paired* design.
* **Significance flag** — a constant positive uplift is significant; identical vectors are not.
* **Validation** — malformed inputs raise ``ValueError`` rather than emitting a bogus interval.

CI bounds are intentionally NOT hard-coded to magic floats (they depend on the numpy version's
summation order, captured in meta.json per ADR-0004); the structural properties above are what
the tests assert.
"""

from __future__ import annotations

import math

import pytest

from rag.eval.bootstrap import paired_bootstrap_ci

_BASELINE = [0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.6, 0.2]
_TREATMENT = [0.3, 0.5, 0.6, 0.9, 0.4, 0.7, 0.5, 0.5]


def test_deterministic_same_seed_same_result() -> None:
    first = paired_bootstrap_ci(_BASELINE, _TREATMENT, n_resamples=2000, seed=7)
    second = paired_bootstrap_ci(_BASELINE, _TREATMENT, n_resamples=2000, seed=7)
    # Frozen Pydantic models compare field-by-field; identical computation ⇒ identical record.
    assert first == second


def test_observed_diff_is_full_sample_not_bootstrap_mean() -> None:
    result = paired_bootstrap_ci(_BASELINE, _TREATMENT, n_resamples=2000, seed=7)
    expected_diff = sum(_TREATMENT) / len(_TREATMENT) - sum(_BASELINE) / len(_BASELINE)
    assert result.diff == pytest.approx(expected_diff)
    assert result.baseline_mean == pytest.approx(sum(_BASELINE) / len(_BASELINE))
    assert result.treatment_mean == pytest.approx(sum(_TREATMENT) / len(_TREATMENT))
    assert result.n_queries == len(_BASELINE)


def test_pairing_swap_negates_diff_and_mirrors_interval() -> None:
    # Same seed ⇒ same resampled index matrix; the paired diff distribution simply negates,
    # so percentiles mirror: (lo, hi) -> (-hi, -lo) and the point diff flips sign.
    forward = paired_bootstrap_ci(_BASELINE, _TREATMENT, n_resamples=3000, seed=99)
    swapped = paired_bootstrap_ci(_TREATMENT, _BASELINE, n_resamples=3000, seed=99)

    assert swapped.diff == pytest.approx(-forward.diff)
    assert swapped.ci_low == pytest.approx(-forward.ci_high)
    assert swapped.ci_high == pytest.approx(-forward.ci_low)
    # Significance is symmetric under the swap.
    assert swapped.significant == forward.significant


def test_constant_uplift_is_significant() -> None:
    baseline = [0.2, 0.5, 0.1, 0.7, 0.4]
    treatment = [v + 0.2 for v in baseline]
    result = paired_bootstrap_ci(baseline, treatment, n_resamples=3000, seed=1)
    # Every paired diff is exactly +0.2, so the whole CI collapses to 0.2 > 0.
    assert result.diff == pytest.approx(0.2)
    assert result.ci_low == pytest.approx(0.2)
    assert result.ci_high == pytest.approx(0.2)
    assert result.ci_low > 0.0
    assert result.significant is True


def test_identical_vectors_diff_zero_not_significant() -> None:
    data = [0.3, 0.6, 0.1, 0.9, 0.5]
    result = paired_bootstrap_ci(data, list(data), n_resamples=3000, seed=1)
    assert result.diff == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.significant is False


def test_records_provenance_fields() -> None:
    result = paired_bootstrap_ci(_BASELINE, _TREATMENT, n_resamples=1500, seed=2024, ci=0.9)
    assert result.n_resamples == 1500
    assert result.seed == 2024
    assert result.ci_level == 0.9
    assert result.n_queries == len(_BASELINE)


def test_unequal_lengths_raise() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_ci([0.1, 0.2], [0.1, 0.2, 0.3])


def test_empty_inputs_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_ci([], [])


def test_single_pair_raises_degenerate_ci() -> None:
    # n=1: every resample draws the only index, so the CI has width 0 and any non-zero diff
    # would be misread as significant. A paired bootstrap needs >= 2 paired observations.
    with pytest.raises(ValueError, match="at least 2 paired observations"):
        paired_bootstrap_ci([0.3], [0.6])


@pytest.mark.parametrize("bad_b", [0, -5, 1, 999])
def test_n_resamples_below_floor_raises(bad_b) -> None:
    # Below the documented floor the percentile CI is a coin-flip that can flip significance.
    with pytest.raises(ValueError, match="n_resamples must be >= 1000"):
        paired_bootstrap_ci(_BASELINE, _TREATMENT, n_resamples=bad_b)


@pytest.mark.parametrize("bad_ci", [0.0, 1.0, -0.1, 1.5])
def test_ci_outside_open_unit_interval_raises(bad_ci) -> None:
    with pytest.raises(ValueError, match="ci must be"):
        paired_bootstrap_ci(_BASELINE, _TREATMENT, ci=bad_ci)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nan_or_inf_inputs_raise(bad_value) -> None:
    bad = list(_BASELINE)
    bad[0] = bad_value
    with pytest.raises(ValueError, match="NaN or inf"):
        paired_bootstrap_ci(bad, _TREATMENT)
    with pytest.raises(ValueError, match="NaN or inf"):
        paired_bootstrap_ci(_BASELINE, bad)
