"""Auto-detects which of a battery of candidate distributions best fits an
empirical baseline array (e.g. CodonPairBiasAnalysis.cpbPerWindow), then
reduces that fit to a normal-scores transform so change_vector.py's/
gpu_change_vector.py's `_zscore(...) ** 2` scoring stays statistically
meaningful even when the true distribution isn't symmetric.

Why this exists: cached_mean_std() (change_vector.py) always reduced a
baseline array to a plain (mean, std), and every term's z-score is
`(x - mean) / std`, then squared. For a Gaussian baseline that squared
z-score is a real, symmetric rarity measure (chi-squared(1)). For a skewed
baseline -- e.g. a Boltzmann/exponential-tailed shape, visually confirmed
against real CodonPairBias data -- it isn't: a given number of "std devs"
above the mean is a very different probability event than the same number
below (which may not even be reachable if the true distribution has a hard
lower bound), and squaring erases that asymmetry, silently conflating
common and genuinely-rare deviations into the same score.

The fix, standard practice for exactly this situation: fit the empirical
array's actual shape, then run every future value through *that* fitted
CDF and back through the inverse-normal CDF (`norm.ppf`) -- the "normal
scores transform." Whatever the real shape is, this produces an exact
standard normal by construction, so squaring it afterward is finally a
fair rarity measure regardless of the original distribution. When `norm`
itself turns out to be the best fit, this is algebraically identical to
the original `(x - mean) / std` -- see FittedDistribution's docstring --
so nothing changes for baselines that really were Gaussian to begin with.

Deliberately heavyweight: this runs the MLE fit + AIC comparison across
every candidate family once per (baseline object, field) pair, then caches
the result via change_vector.cached_stat() for the rest of that process's
life (see cached_normal_transform() in change_vector.py) -- the same
"expensive once, free forever after" shape cached_mean_std() already used,
just with real fitting work behind it instead of a mean/std reduction.
Appropriate here specifically because get baseline arrays are loaded once
per GA run and never mutated (same assumption cached_stat() already
documents), and this module's caller explicitly wants the accurate,
run-once version rather than a cheaper approximation.
"""

import warnings

import numpy as np
from scipy import stats

# Covers both common senses of "Boltzmann-shaped": expon is the classic
# decaying-exponential-tail shape; gamma covers the peaked-then-decaying
# Maxwell-Boltzmann-speed shape (and subsumes expon's shape family more
# flexibly, kept separate anyway since expon's closed-form fit is free).
# lognorm/skewnorm cover milder, one-sided skew that isn't a full Boltzmann
# shape but would still misrepresent rarity under a raw (x-mean)/std z.
_CANDIDATE_DISTS = (
    ('norm', stats.norm),
    ('expon', stats.expon),
    ('gamma', stats.gamma),
    ('lognorm', stats.lognorm),
    ('skewnorm', stats.skewnorm),
)
# norm/expon have closed-form MLE (just moments/shift -- cheap regardless of
# N), so they always fit against the FULL array. gamma/lognorm/skewnorm need
# an iterative optimizer, whose cost scales with sample size -- bounded by
# _MAX_FIT_SAMPLE so a multi-million-point real Standards array still fits
# in reasonable time. AIC is still evaluated against the FULL array for
# every candidate (see fit_normal_transform()), so the subsampling only
# affects fitting cost, not which family wins.
_CLOSED_FORM_FAMILIES = frozenset({'norm', 'expon'})
_MIN_N_FOR_MULTIPARAM = 20
_MAX_FIT_SAMPLE = 50_000
_GRID_POINTS = 2001
# Clamp the fitted CDF away from exactly 0/1 before norm.ppf -- ppf(0)/ppf(1)
# are +-inf, which would poison the interpolation grid. 1e-6 caps the
# representable |z| around 4.75, a reasonable practical ceiling (existing
# code elsewhere in this codebase -- ga.py's _finite_nonneg -- takes the
# same "cap rather than let inf propagate" approach for the same reason).
_CDF_EPS = 1e-6


def _interp(xp, x, grid_x, grid_z):
    """Piecewise-linear interpolation of x against (grid_x, grid_z), written
    against searchsorted/clip/elementwise ops only (no xp.interp) so the
    identical code runs on numpy or cupy -- cupy.interp's availability and
    exact signature aren't guaranteed to match numpy's across versions, and
    this is called from the GPU-batched scoring path (gpu_change_vector.py),
    so it needs to be trustworthy there. grid_x must be sorted ascending.
    Values outside [grid_x[0], grid_x[-1]] clamp to the nearest edge z,
    matching np.interp's default out-of-range behavior -- values more
    extreme than anything seen while fitting are treated as "at least that
    extreme," not extrapolated past it.
    """
    n = grid_x.shape[0]
    idx = xp.clip(xp.searchsorted(grid_x, x, side='left'), 1, n - 1)
    x0 = grid_x[idx - 1]
    x1 = grid_x[idx]
    z0 = grid_z[idx - 1]
    z1 = grid_z[idx]
    t = xp.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return z0 + t * (z1 - z0)


class FittedDistribution:
    """A baseline array's best-fit distribution, reduced to a normal-scores
    lookup table: transform(x) = norm.ppf(fitted_cdf(x)). When `family` is
    'norm', this is exactly the classic (x - mean) / std z-score -- scipy's
    norm.ppf(norm.cdf(x, loc, scale)) reduces to (x - loc) / scale up to
    floating-point precision, and norm.fit()'s MLE loc/scale are the sample
    mean and population (ddof=0) std, matching cached_mean_std()'s old
    (arr.mean(), arr.std()) exactly -- so nothing changes for baselines
    that really were Gaussian. For any other selected family, this is the
    statistically correct generalization (see module docstring).

    Stored as a precomputed (grid_x, grid_z) lookup table rather than
    keeping the scipy distribution object + params around, so scoring only
    ever pays for a cheap interpolation, not a fitted-CDF evaluation --
    the fitting itself is the "heavyweight, run once" part; using the
    result should be as cheap as the (x-mean)/std it replaces.
    """

    __slots__ = ('family', 'params', 'grid_x', 'grid_z')

    def __init__(self, family, params, grid_x, grid_z):
        self.family = family
        self.params = params
        self.grid_x = grid_x
        self.grid_z = grid_z

    def transform(self, value: float) -> float:
        """Scalar path -- change_vector.py's per-individual terms."""
        return float(_interp(np, np.asarray(value, dtype=float), self.grid_x, self.grid_z))

    def transform_array(self, arr, xp=np):
        """Array path -- change_vector.py's own vectorized bucket-window
        scoring (xp=np implicitly) and gpu_change_vector.py's batched terms
        (xp=cupy on GPU runs). Uploads the lookup grid to `xp` on every call
        rather than caching an xp-resident copy, matching how every other
        per-call lookup table in gpu_change_vector.py (cpb_table,
        odds_by_count, etc.) is already re-uploaded via xp.asarray(...) each
        call -- consistent with the existing convention rather than a new
        one-off caching layer for just this table.
        """
        grid_x = self.grid_x if xp is np else xp.asarray(self.grid_x)
        grid_z = self.grid_z if xp is np else xp.asarray(self.grid_z)
        return _interp(xp, arr, grid_x, grid_z)


def _normal_transform(mean: float, std: float) -> FittedDistribution:
    if std == 0:
        return _degenerate_transform(mean)
    grid_x = np.linspace(mean - 8 * std, mean + 8 * std, _GRID_POINTS)
    grid_z = (grid_x - mean) / std  # direct formula, not norm.cdf/ppf -- exact, no round-trip error
    return FittedDistribution('norm', (mean, std), grid_x, grid_z)


def _degenerate_transform(value: float) -> FittedDistribution:
    """Every observation identical (or fewer than 2 total) -- nothing to
    fit. Every value is equally "typical" against a single-point
    distribution, so z is always 0.0, matching _zscore()'s existing
    std == 0 -> 0.0 fallback exactly."""
    grid_x = np.asarray([value - 1.0, value, value + 1.0])
    grid_z = np.asarray([0.0, 0.0, 0.0])
    return FittedDistribution('degenerate', (value,), grid_x, grid_z)


def fit_normal_transform(values) -> FittedDistribution:
    """Auto-detects the best-fitting distribution family for `values` (a
    baseline array like CodonPairBiasAnalysis.cpbPerWindow) and returns a
    FittedDistribution that converts any raw value drawn from that same
    population into a normal-scores z -- see module and class docstrings
    for why, and cached_normal_transform() in change_vector.py for the
    cached entry point every real caller should use instead of calling this
    directly (this recomputes the fit from scratch on every call).

    Selection is by AIC (log-likelihood penalized by parameter count, so a
    more flexible family only wins if it actually explains the data better,
    not merely because it has more parameters to bend with) over a fixed
    battery of candidates: norm, expon, gamma, lognorm, skewnorm. Every
    candidate is scored against the FULL array for a fair, accurate AIC
    comparison; only the iterative-optimizer families' *fitting* step
    (gamma/lognorm/skewnorm -- norm/expon have cheap closed-form fits
    regardless of N) is bounded to _MAX_FIT_SAMPLE observations, so a
    multi-million-point real Standards array doesn't make fitting itself
    the bottleneck. Multi-parameter families are skipped below
    _MIN_N_FOR_MULTIPARAM observations, where their extra flexibility is
    more likely to fit sampling noise than real shape; norm is always a
    candidate.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return _degenerate_transform(0.0)

    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return _degenerate_transform(lo)

    if arr.size > _MAX_FIT_SAMPLE:
        rng = np.random.default_rng(0)  # deterministic: same array -> same fit, run to run
        fit_sample = rng.choice(arr, size=_MAX_FIT_SAMPLE, replace=False)
    else:
        fit_sample = arr

    best = None  # (aic, family_name, dist, params)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # scipy's optimizer-based .fit() can warn on hard cases; failures are caught below either way
        for name, dist in _CANDIDATE_DISTS:
            if name != 'norm' and arr.size < _MIN_N_FOR_MULTIPARAM:
                continue
            sample = arr if name in _CLOSED_FORM_FAMILIES else fit_sample
            try:
                params = dist.fit(sample)
                loglik = float(np.sum(dist.logpdf(arr, *params)))
                if not np.isfinite(loglik):
                    continue
                aic = 2 * len(params) - 2 * loglik
            except Exception:
                continue
            if best is None or aic < best[0]:
                best = (aic, name, dist, params)

    if best is None:
        # Every candidate fit failed outright (pathological data) -- fall
        # back to the plain method-of-moments normal, the same statistic
        # cached_mean_std() always used, rather than raising out of what
        # used to be a simple, non-raising (mean, std) computation.
        return _normal_transform(float(arr.mean()), float(arr.std()))

    _, family, dist, params = best
    if family == 'norm':
        return _normal_transform(*params)

    margin = (hi - lo) * 0.05 or 1.0
    grid_x = np.linspace(lo - margin, hi + margin, _GRID_POINTS)
    grid_cdf = np.clip(dist.cdf(grid_x, *params), _CDF_EPS, 1 - _CDF_EPS)
    grid_z = stats.norm.ppf(grid_cdf)
    return FittedDistribution(family, params, grid_x, grid_z)
