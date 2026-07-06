"""Paired percentile bootstrap for per-query metric differences (ADR-0004).

The headline deliverable is a comparison of configs (dense / sparse / hybrid / hybrid+rerank)
on a per-query metric with a CI95. Two configs are scored on the **same** golden queries, so
the bootstrap must be **paired**: each resample draws query *indices* with replacement and
applies the *same* indices to both configs' per-query vectors, then takes the per-iteration
difference. Pairing cancels per-query difficulty variance, giving a correct (and typically
tighter) interval on the difference than an unpaired bootstrap would — and it is what licenses
a claim like "treatment improves the metric by X [CI95: a, b]".

Discipline (all fixed in ADR-0004 so every interval is reproducible and defensible):

* **Resample at the query level**, vectorized: ``idx = rng.integers(0, n, size=(B, n))`` then
  index both vectors with the same ``idx``.
* **Point estimate is the observed diff on the full sample** (``mean(treatment) -
  mean(baseline)``), *not* the mean of the bootstrap distribution. Positive ⇒ treatment better.
* **CI is the percentile method**: the 2.5th/97.5th percentiles of the bootstrap diff
  distribution for ``ci=0.95`` (BCa is deferred — see ADR-0004).
* **Seeded** via ``numpy.random.default_rng(seed)`` only; never time-seeded, never the legacy
  global ``numpy.random``. Same ``(data, seed, B)`` ⇒ bit-identical bounds.
* ``significant`` is ``True`` iff 0 lies outside the interval.

Inputs are validated up front (unequal lengths, empty, ``n_resamples <= 0``, ``ci`` outside
``(0, 1)``, or any NaN/inf) and raise :class:`ValueError`, so a malformed comparison fails
loudly rather than emitting a meaningless interval.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag.eval.models import BootstrapResult

# Hard floors (ADR-0004). A paired bootstrap with a single pair has no resampling variance
# (every resample draws the only index), so its CI collapses to width zero and any non-zero
# diff is misread as "significant" — forbid n < 2. Too few resamples make the percentile CI a
# coin-flip that can even contradict the observed estimate and flip the significance flag, so
# require a documented minimum B (the default 10000 stays well above it).
_MIN_PAIRS = 2
_MIN_RESAMPLES = 1000


def paired_bootstrap_ci(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 12345,
    ci: float = 0.95,
) -> BootstrapResult:
    """Paired percentile bootstrap CI on ``mean(treatment) - mean(baseline)``.

    Args:
        baseline: Per-query metric values for the baseline config (best as a list of floats).
        treatment: Per-query metric values for the treatment config, aligned 1:1 with
            ``baseline`` (same query at the same index).
        n_resamples: Number of bootstrap resamples ``B`` (``>= 1000``); default 10000.
        seed: Seed for ``numpy.random.default_rng``; fixed default 12345 for reproducibility.
        ci: Two-sided confidence level in ``(0, 1)``; default 0.95 → 2.5/97.5 percentiles.

    Returns:
        A :class:`~rag.eval.models.BootstrapResult` carrying the observed means and diff, the
        percentile CI bounds, ``significant`` (0 outside the CI), and the
        ``ci_level``/``n_resamples``/``seed``/``n_queries`` provenance needed to reproduce it.

    Raises:
        ValueError: If the inputs have unequal length, either is empty, fewer than 2 paired
            observations (``n < 2``, no resampling variance), ``n_resamples < 1000``, ``ci`` is
            not strictly inside ``(0, 1)``, or any value is NaN/inf.
    """
    baseline_arr = np.asarray(baseline, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=float)

    if baseline_arr.ndim != 1 or treatment_arr.ndim != 1:
        raise ValueError("baseline and treatment must be 1-D sequences")
    if baseline_arr.size == 0 or treatment_arr.size == 0:
        raise ValueError("baseline and treatment must be non-empty")
    if baseline_arr.size != treatment_arr.size:
        raise ValueError(
            f"baseline and treatment must have equal length, "
            f"got {baseline_arr.size} and {treatment_arr.size}"
        )
    if baseline_arr.size < _MIN_PAIRS:
        raise ValueError(
            f"paired bootstrap requires at least {_MIN_PAIRS} paired observations (n >= "
            f"{_MIN_PAIRS}); got n={baseline_arr.size} — a single pair has no resampling "
            "variance, so its CI is degenerate"
        )
    if n_resamples < _MIN_RESAMPLES:
        raise ValueError(
            f"n_resamples must be >= {_MIN_RESAMPLES} for a stable percentile CI, "
            f"got {n_resamples}"
        )
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in the open interval (0, 1), got {ci}")
    if not np.all(np.isfinite(baseline_arr)) or not np.all(np.isfinite(treatment_arr)):
        raise ValueError("baseline and treatment must not contain NaN or inf")

    n = baseline_arr.size
    rng = np.random.default_rng(seed)
    # Paired resampling: ONE index matrix drives both vectors so the per-iteration difference
    # is taken on the same resampled queries (this is what makes the bootstrap paired).
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_diffs = treatment_arr[idx].mean(axis=1) - baseline_arr[idx].mean(axis=1)

    baseline_mean = float(baseline_arr.mean())
    treatment_mean = float(treatment_arr.mean())
    # Point estimate is the OBSERVED diff on the full sample, not the bootstrap mean.
    diff = treatment_mean - baseline_mean

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = (
        float(x) for x in np.percentile(boot_diffs, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    )
    significant = not (ci_low <= 0.0 <= ci_high)

    return BootstrapResult(
        baseline_mean=baseline_mean,
        treatment_mean=treatment_mean,
        diff=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci,
        n_resamples=n_resamples,
        seed=seed,
        n_queries=n,
        significant=significant,
    )
