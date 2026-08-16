"""Tests for distribution_fit.py's auto-detected-distribution normal-scores
transform -- see that module's docstring for the motivating problem
(CodonPairBias's real baseline data visually resembles a Boltzmann/
exponential-tailed shape, and the old (x-mean)/std z-score, once squared,
silently misrepresents rarity for anything that isn't actually symmetric).
"""
import numpy as np
import pytest

from genewriter.distribution_fit import fit_normal_transform


def test_normal_data_selects_norm_and_matches_classic_zscore_exactly():
    rng = np.random.default_rng(0)
    data = rng.normal(10.0, 3.0, 5000)
    fd = fit_normal_transform(data)

    assert fd.family == 'norm'
    for value in (16.0, 4.0, 10.0, 22.0):
        classic_z = (value - data.mean()) / data.std()
        assert fd.transform(value) == pytest.approx(classic_z, abs=1e-9)


def test_boltzmann_shaped_data_does_not_select_norm():
    rng = np.random.default_rng(1)
    data = rng.exponential(scale=2.0, size=5000) + 1.0  # hard floor at 1.0, long right tail
    fd = fit_normal_transform(data)

    assert fd.family != 'norm'


def test_boltzmann_shaped_data_treats_deviations_asymmetrically():
    """The whole point of this module: for a skewed baseline, "N std above
    the mean" and "N std below the mean" are NOT equally rare, and the
    transform must reflect that -- unlike the old symmetric (x-mean)/std,
    which scored both as exactly the same magnitude by construction."""
    rng = np.random.default_rng(1)
    data = rng.exponential(scale=2.0, size=5000) + 1.0
    fd = fit_normal_transform(data)

    mean, std = data.mean(), data.std()
    above = fd.transform(mean + 2 * std)
    below = fd.transform(mean - 2 * std)  # near/at the hard floor -- should read as far MORE extreme

    assert abs(above) != pytest.approx(abs(below), rel=0.05)
    assert abs(below) > abs(above)


def test_single_value_is_degenerate_and_always_scores_zero():
    fd = fit_normal_transform([5.0])
    assert fd.family == 'degenerate'
    assert fd.transform(5.0) == 0.0
    assert fd.transform(1000.0) == 0.0


def test_all_identical_values_is_degenerate_and_always_scores_zero():
    fd = fit_normal_transform([3.0] * 50)
    assert fd.family == 'degenerate'
    assert fd.transform(3.0) == 0.0
    assert fd.transform(-99.0) == 0.0


def test_empty_array_is_degenerate_rather_than_raising():
    fd = fit_normal_transform([])
    assert fd.family == 'degenerate'
    assert fd.transform(0.0) == 0.0


def test_tiny_sample_falls_back_to_norm_rather_than_an_unstable_multiparam_fit():
    # N=3 is below _MIN_N_FOR_MULTIPARAM -- gamma/lognorm/skewnorm need more
    # data than this to fit reliably, so norm (which only ever needs 2
    # moments) is the only real candidate.
    fd = fit_normal_transform([1.0, 2.0, 3.0])
    assert fd.family == 'norm'


def test_transform_array_matches_scalar_transform_elementwise():
    rng = np.random.default_rng(2)
    data = rng.gamma(2.0, 3.0, 3000)
    fd = fit_normal_transform(data)

    values = np.asarray([1.0, 3.0, 6.0, 12.0, 30.0])
    array_result = fd.transform_array(values)
    scalar_result = np.asarray([fd.transform(v) for v in values])

    np.testing.assert_allclose(array_result, scalar_result, atol=1e-9)


def test_transform_array_handles_2d_input():
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 1.0, 2000)
    fd = fit_normal_transform(data)
    assert fd.family == 'norm'
    mean, std = fd.params  # sample estimates, not exactly (0.0, 1.0) -- that's fine, compare against these

    values_2d = np.asarray([[0.0, 1.0], [-1.0, 2.0]])
    result = fd.transform_array(values_2d)

    assert result.shape == values_2d.shape
    np.testing.assert_allclose(result, (values_2d - mean) / std, atol=1e-6)


def test_out_of_range_values_clamp_instead_of_extrapolating():
    rng = np.random.default_rng(4)
    data = rng.normal(0.0, 1.0, 5000)
    fd = fit_normal_transform(data)

    far_above = fd.transform(1000.0)
    far_below = fd.transform(-1000.0)

    assert 0 < far_above < 6.0  # capped near norm.ppf(1 - _CDF_EPS), not literally ~1000
    assert -6.0 < far_below < 0
    # Monotonic: further out never produces a LESS extreme score than closer in.
    assert fd.transform(1000.0) >= fd.transform(500.0) >= fd.transform(4.0)
