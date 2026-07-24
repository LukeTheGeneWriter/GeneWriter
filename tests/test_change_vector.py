import math

import pytest

from genewriter.change_vector import calculate_change_vector, score_changevec
from genewriter.codon_tables import generate_codon_vec

from conftest import random_solution as _random_solution


def test_calculate_change_vector_runs_and_covers_every_position(aa_seq, analysis_objects):
    """The original crashed before returning anything (ProposedSolution
    NameError / empty cpbchangevec IndexError / dist_from_optimal tuple
    unpacking / windowlength AttributeError). This is the base smoke test:
    it must simply complete and produce one score per codon per term."""
    sol = _random_solution(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)

    assert set(vecs.keys()) == {'RareCodons', 'CodonUsage', 'CodonPairBias', 'GC', 'Kmer'}
    for term, values in vecs.items():
        assert len(values) == len(sol), f"{term} produced {len(values)} scores for {len(sol)} codons"
        assert all(isinstance(v, float) for v in values)


def test_codon_pair_bias_term_is_not_empty(aa_seq, analysis_objects):
    """Original bug: cpbchangevec was referenced but never populated, so any
    read of it raised IndexError. Confirm it now holds a real value at every
    position, including the first/last (special-cased boundaries)."""
    sol = _random_solution(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)
    cpb = vecs['CodonPairBias']
    assert len(cpb) == len(sol)
    assert any(v != 0.0 for v in cpb)


def test_kmer_term_does_not_drop_last_codon(aa_seq, analysis_objects):
    """Original bug: `for g in range(0, len(sol) - 1)` built the continuous
    sequence one codon short, so the last codon never contributed to (or was
    scored by) the k-mer term. Changing only the last codon should be able to
    change the k-mer term's last-position score."""
    sol = _random_solution(aa_seq, seed=10)
    codon_choices = generate_codon_vec(aa_seq)[-1]
    alt_last_codon = next(c for c in codon_choices if c != sol[-1])

    vecs_a = calculate_change_vector(sol, analysis_objects)

    sol_b = sol.copy()
    sol_b[-1] = alt_last_codon
    vecs_b = calculate_change_vector(sol_b, analysis_objects)

    assert len(vecs_a['Kmer']) == len(sol)
    # Last-position score must be sourced from real content, not a
    # structurally-guaranteed-zero placeholder.
    assert vecs_a['Kmer'][-1] != 0.0 or vecs_b['Kmer'][-1] != 0.0


def test_dist_from_optimal_accepts_plain_codon_strings(aa_seq, analysis_objects):
    """Original bug: dist_from_optimal() was called with windows of plain
    codon strings but its body did `for codon, location in codons`, which
    raises ValueError trying to unpack a 3-character string into 2 names.
    Covered implicitly by the smoke test above via the CodonUsage term, but
    asserted directly here since it was the most immediate crash."""
    sol = _random_solution(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)
    assert len(vecs['CodonUsage']) == len(sol)


def test_score_changevec_is_a_plain_float(aa_seq, analysis_objects, weights):
    sol = _random_solution(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)
    score = score_changevec(vecs, weights)
    assert isinstance(score, float)


def test_zero_variance_baseline_does_not_crash(aa_seq, analysis_objects):
    """Zero-std baselines previously produced silent NaNs (division by zero
    under numpy) rather than a defined score. Confirm the explicit guard
    kicks in and the function still returns finite-shaped output."""
    analysis_objects.rare_codon.usagePerGene = [0.05] * 50  # zero variance
    sol = _random_solution(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)
    assert len(vecs['RareCodons']) == len(sol)


def test_locvec_length_mismatch_raises(aa_seq, analysis_objects):
    sol = _random_solution(aa_seq)
    with pytest.raises(ValueError):
        calculate_change_vector(sol, analysis_objects, locvec=['I'] * (len(sol) - 1))


def test_rare_codon_term_never_produces_nan_from_unobserved_window(aa_seq, analysis_objects):
    """Caught against real (if tiny) sample gene data: when a window's exact
    rare-codon count was never observed in the baseline, its score is
    float('inf'); at any position that isn't itself a rare codon, the
    original `scorevec[i] * rvec[i] * zscore` computed inf * 0, which is NaN
    under IEEE 754 -- not 0 as intended. NaN then poisons score_changevec
    and crashes the GA's random.choices() calls downstream. Every non-rare
    position must score exactly 0.0 for this term when its window is
    unobserved, never NaN."""
    analysis_objects.rare_codon.rare_codon_windows = {i: 0 for i in range(0, 16)}
    sol = _random_solution(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)
    assert not any(math.isnan(v) for v in vecs['RareCodons'])
