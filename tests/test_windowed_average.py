"""Regression test for change_vector._windowed_average's numpy-vectorized
rewrite, checked against the original pure-Python loop it replaced (kept
here verbatim as a reference implementation) across many random cases.

This function is private, but it's the one piece of shared infrastructure
every change-vector term depends on -- worth a dedicated, thorough test
given the rewrite specifically had to avoid a subtle correctness trap
(cumsum-based windowing can round differently than summing each window's
own elements directly; see the function's docstring).
"""

import random

from genewriter.change_vector import _windowed_average


def _reference_windowed_average(values, target_len, winsize):
    """The original implementation, verbatim, as the correctness oracle."""
    out = []
    half = max(winsize // 2, 1)
    n = len(values)
    for i in range(target_len):
        if i < winsize:
            relevant = values[0:i]
        elif i > target_len - winsize:
            relevant = values[i:n]
        else:
            relevant = values[max(i - half, 0):min(i + half, n)]
        out.append(sum(relevant) / len(relevant) if relevant else 0.0)
    return out


def test_matches_reference_across_many_random_cases():
    rng = random.Random(0)
    for trial in range(500):
        n = rng.randint(0, 200)
        target_len = rng.randint(0, 200)
        winsize = rng.randint(1, 30)
        values = [rng.uniform(-100, 100) for _ in range(n)]

        expected = _reference_windowed_average(values, target_len, winsize)
        actual = _windowed_average(values, target_len, winsize)

        assert len(actual) == len(expected) == target_len, f"trial {trial}: length mismatch"
        for i, (e, a) in enumerate(zip(expected, actual)):
            assert abs(e - a) < 1e-9, f"trial {trial}, position {i}: expected {e}, got {a} (n={n}, target_len={target_len}, winsize={winsize})"


def test_matches_reference_with_typical_codon_scale_parameters():
    """The random-parameter sweep above covers small/degenerate sizes;
    this specifically covers the shape real calls actually use: target_len
    in the hundreds, winsize ~15, values shorter than target_len by
    roughly winsize (matching how each term builds its windows list)."""
    rng = random.Random(1)
    for trial in range(50):
        target_len = rng.randint(50, 500)
        winsize = 15
        n = max(target_len - winsize, 0)
        values = [rng.uniform(0, 1000) for _ in range(n)]

        expected = _reference_windowed_average(values, target_len, winsize)
        actual = _windowed_average(values, target_len, winsize)
        for i, (e, a) in enumerate(zip(expected, actual)):
            assert abs(e - a) < 1e-9, f"trial {trial}, position {i}: expected {e}, got {a}"


def test_empty_values_returns_all_zeros():
    assert _windowed_average([], 10, 15) == [0.0] * 10


def test_zero_target_len_returns_empty_list():
    assert _windowed_average([1.0, 2.0, 3.0], 0, 15) == []
