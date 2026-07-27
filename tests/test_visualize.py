import json

import matplotlib
matplotlib.use('Agg')  # headless -- no display available in CI/WSL

import numpy as np
import pytest

from genewriter import ga
from genewriter.change_vector import calculate_change_vector, score_changevec
from genewriter.classes import Proposed_Solution
from genewriter.visualize import (
    _sanitize_for_color,
    codon_distance_matrix,
    fitness_color_values,
    load_population_json,
    plot_population_tsne,
    subsample_population,
)


def _population(aa_seq, analysis_objects, n=8):
    pop = []
    for _ in range(n):
        codons = ga.generate_seed(aa_seq)
        pop.append(Proposed_Solution(codons, 1, calculate_change_vector(codons, analysis_objects)))
    return pop


def test_codon_distance_matrix_is_zero_on_diagonal(aa_seq, analysis_objects):
    pop = _population(aa_seq, analysis_objects)
    dist = codon_distance_matrix(pop)
    assert dist.shape == (len(pop), len(pop))
    assert np.allclose(np.diag(dist), 0.0)
    assert np.allclose(dist, dist.T)  # symmetric


def test_codon_distance_matrix_matches_manual_hamming_fraction(aa_seq, analysis_objects):
    codons = ga.generate_seed(aa_seq)
    neighbor = list(codons)
    # aa_seq[0] is Met (single codon) -- mutate position 1 instead (Ala, 4 codons).
    from genewriter.codon_tables import generate_codon_vec
    choices = generate_codon_vec(aa_seq)[1]
    neighbor[1] = next(c for c in choices if c != codons[1])

    pop = [
        Proposed_Solution(codons, 1, calculate_change_vector(codons, analysis_objects)),
        Proposed_Solution(neighbor, 1, calculate_change_vector(neighbor, analysis_objects)),
    ]
    dist = codon_distance_matrix(pop)
    expected = 1 / len(codons)  # exactly one differing position
    assert dist[0, 1] == pytest.approx(expected)
    assert dist[1, 0] == pytest.approx(expected)


def test_codon_distance_matrix_handles_single_individual():
    sol = Proposed_Solution(['ATG'], 1, {})
    dist = codon_distance_matrix([sol])
    assert dist.shape == (1, 1)
    assert dist[0, 0] == 0.0


def test_fitness_color_values_fitness_mode_matches_score_changevec(aa_seq, analysis_objects, weights):
    pop = _population(aa_seq, analysis_objects)
    colors = fitness_color_values(pop, weights, 'fitness')
    expected = [score_changevec(p.change_vecs, weights) for p in pop]
    assert colors.tolist() == pytest.approx(expected)


def test_fitness_color_values_term_mode_is_raw_unweighted_sum(aa_seq, analysis_objects, weights):
    """Deliberately NOT multiplied by weights[term] -- a term at weight 0
    (Uracil defaults to 0.0) must still show its real signal, not an
    all-zero color axis."""
    pop = _population(aa_seq, analysis_objects)
    colors = fitness_color_values(pop, weights, 'GC')
    expected = [sum(p.change_vecs['GC']) for p in pop]
    assert colors.tolist() == pytest.approx(expected)

    uracil_colors = fitness_color_values(pop, weights, 'Uracil')
    expected_uracil = [sum(p.change_vecs['Uracil']) for p in pop]
    assert uracil_colors.tolist() == pytest.approx(expected_uracil)
    # weights['Uracil'] is 0.0 in the shared fixture -- if this were
    # weighted, every value would be zero. Confirm it's not (real signal).
    assert any(v != 0.0 for v in uracil_colors)


def test_fitness_color_values_unknown_mode_raises(aa_seq, analysis_objects, weights):
    pop = _population(aa_seq, analysis_objects)
    with pytest.raises(ValueError, match="Unknown color_by"):
        fitness_color_values(pop, weights, 'NotARealTerm')


def test_sanitize_for_color_clamps_posinf_to_finite_max():
    out = _sanitize_for_color(np.array([1.0, 2.0, np.inf]))
    assert out.tolist() == [1.0, 2.0, 2.0]


def test_sanitize_for_color_clamps_neginf_to_finite_min():
    out = _sanitize_for_color(np.array([1.0, 2.0, -np.inf]))
    assert out.tolist() == [1.0, 2.0, 1.0]


def test_sanitize_for_color_replaces_nan_with_finite_mean():
    out = _sanitize_for_color(np.array([1.0, 3.0, np.nan]))
    assert out.tolist() == [1.0, 3.0, 2.0]


def test_sanitize_for_color_falls_back_to_zeros_when_nothing_is_finite():
    out = _sanitize_for_color(np.array([np.inf, -np.inf, np.nan]))
    assert out.tolist() == [0.0, 0.0, 0.0]


def test_fitness_color_values_never_returns_non_finite(aa_seq, analysis_objects, weights):
    """Real bug, caught by actually rendering a figure against real sample
    gene data (not by the smoke tests alone, whose small synthetic
    baselines never happened to trigger it): change_vector._rare_codon_term
    legitimately returns float('inf') for a window composition never
    observed in the baseline (see that function's own docstring) -- and
    'fitness' inherits it via score_changevec()'s summation across terms.
    An inf-valued matplotlib scatter `c=` array silently drops those
    points instead of erroring, so this must never leak through
    fitness_color_values(). Forces the same all-inf scenario
    test_rare_codon_term_never_produces_nan_from_unobserved_window
    (test_change_vector.py) does."""
    import math

    analysis_objects.rare_codon.rare_codon_windows = {i: 0 for i in range(0, 16)}
    pop = _population(aa_seq, analysis_objects, n=6)
    assert any(math.isinf(sum(p.change_vecs['RareCodons'])) for p in pop), \
        "test setup didn't actually force an inf RareCodons score -- test is meaningless"

    for mode in ('fitness', 'RareCodons'):
        colors = fitness_color_values(pop, weights, mode)
        assert np.isfinite(colors).all(), f"{mode} color values contained a non-finite entry"


def test_subsample_population_returns_unchanged_when_under_cap(aa_seq, analysis_objects):
    pop = _population(aa_seq, analysis_objects, n=5)
    assert subsample_population(pop, 10) == pop


def test_subsample_population_caps_at_max_points(aa_seq, analysis_objects):
    pop = _population(aa_seq, analysis_objects, n=20)
    sampled = subsample_population(pop, 5, rng=__import__('random').Random(0))
    assert len(sampled) == 5
    assert all(p in pop for p in sampled)


def test_load_population_json_round_trips_with_save_gen(tmp_path, aa_seq, analysis_objects):
    pop = _population(aa_seq, analysis_objects, n=4)
    path = ga.save_gen(pop, 0, str(tmp_path), "test_run")
    loaded = load_population_json(path)
    assert len(loaded) == len(pop)
    for original, reloaded in zip(pop, loaded):
        assert reloaded.codons == original.codons
        assert reloaded.number == original.number
        assert reloaded.change_vecs == original.change_vecs


def test_plot_population_tsne_returns_figure_with_one_subplot_per_mode(aa_seq, analysis_objects, weights):
    pop = _population(aa_seq, analysis_objects, n=12)
    fig = plot_population_tsne(pop, weights, color_by=['fitness', 'GC', 'Uracil'], max_points=100)
    assert len(fig.axes) == 3 + 3  # 3 scatter axes + 3 colorbar axes
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_population_tsne_subsamples_large_populations(aa_seq, analysis_objects, weights):
    pop = _population(aa_seq, analysis_objects, n=30)
    fig = plot_population_tsne(pop, weights, color_by=['fitness'], max_points=10)
    assert "10 of 30" in fig._suptitle.get_text()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_population_tsne_rejects_empty_population(weights):
    with pytest.raises(ValueError, match="empty"):
        plot_population_tsne([], weights)
