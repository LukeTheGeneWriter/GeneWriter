import math

import pytest

from genewriter.change_vector import cached_mean_std, cached_stat, calculate_change_vector, score_changevec
from genewriter.codon_tables import generate_codon_vec

from conftest import random_solution as _random_solution


def test_calculate_change_vector_runs_and_covers_every_position(aa_seq, analysis_objects):
    """The original crashed before returning anything (ProposedSolution
    NameError / empty cpbchangevec IndexError / dist_from_optimal tuple
    unpacking / windowlength AttributeError). This is the base smoke test:
    it must simply complete and produce one score per codon per term."""
    sol = _random_solution(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)

    assert set(vecs.keys()) == {'RareCodons', 'CodonUsage', 'CodonPairBias', 'GC', 'Kmer', 'Uracil'}
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


def test_uracil_term_counts_removable_u_bases(analysis_objects):
    """'CTT'/'CTC'/'CTA'/'CTG'/'TTA'/'TTG' all encode Leu -- every base
    position varies across that synonym set, so a 'T' at base 1 (TTA/TTG)
    counts as removable. 'TGG' (Trp) has only one codon, so nothing about
    it is variable -- its Ts must score 0 regardless of how many there
    are."""
    from genewriter.change_vector import _uracil_term

    sol = ['TTA', 'TGG']  # Leu (variable, 1 removable T), Trp (invariable, 2 Ts but 0 removable)
    scores = _uracil_term(sol, analysis_objects, ['I', 'I'])
    assert scores == [1.0, 0.0]


def test_uracil_term_zero_for_no_t_bases(analysis_objects):
    from genewriter.change_vector import _uracil_term

    sol = ['GCC']  # Ala, no T bases at all
    assert _uracil_term(sol, analysis_objects, ['I']) == [0.0]


def test_uracil_term_counts_multiple_variable_t_bases_in_one_codon(analysis_objects):
    """Ser's 6 synonymous codons (TCT/TCC/TCA/TCG/AGT/AGC) disagree at every
    position, so TCT's base 1 (T) and base 3 (T) are both variable-and-U --
    only base 2 (C) doesn't count, since it isn't U regardless of variability."""
    from genewriter.change_vector import _uracil_term

    assert _uracil_term(['TCT'], analysis_objects, ['I']) == [2.0]


def test_cached_stat_computes_only_once(analysis_objects):
    """The whole point of cached_stat(): a real Colab run against real
    genome-wide Standards baselines showed CodonUsage/GC/CodonPairBias each
    taking a near-constant ~5-13s per batched call *regardless of
    population size*, traced to recomputing baseline mean/std (or a whole
    lookup table) from scratch every call. compute_fn must only ever run
    once per (obj, key)."""
    calls = []

    def compute():
        calls.append(1)
        return 42

    obj = analysis_objects.rare_codon
    assert cached_stat(obj, 'k', compute) == 42
    assert cached_stat(obj, 'k', compute) == 42
    assert cached_stat(obj, 'k', compute) == 42
    assert len(calls) == 1


def test_cached_stat_is_scoped_per_object_and_per_key(analysis_objects):
    """Different objects, and different keys on the same object, must not
    collide in the cache."""
    calls = []

    def compute(tag):
        calls.append(tag)
        return tag

    a, b = analysis_objects.rare_codon, analysis_objects.codon_usage
    assert cached_stat(a, 'x', lambda: compute('a-x')) == 'a-x'
    assert cached_stat(a, 'y', lambda: compute('a-y')) == 'a-y'
    assert cached_stat(b, 'x', lambda: compute('b-x')) == 'b-x'
    assert sorted(calls) == ['a-x', 'a-y', 'b-x']


def test_cached_mean_std_ignores_later_mutation_of_the_source_list(analysis_objects):
    obj = analysis_objects.codon_usage
    values = [1.0, 2.0, 3.0]
    first = cached_mean_std(obj, 'k', values)
    values.append(1000.0)  # mutate in place after first use -- must not affect the cache
    second = cached_mean_std(obj, 'k', values)
    assert first == second


def test_codon_usage_term_baseline_stats_are_cached_across_calls(aa_seq, analysis_objects):
    """Real end-to-end proof the term itself is wired to the cache, not
    just that cached_mean_std() works in isolation: mutating
    ca.windowscores in place *after* the first call must not change the
    second call's result. (Real callers never mutate a loaded baseline in
    place -- see cached_stat()'s docstring -- so this is a deliberate
    contract violation used only to prove the cache is actually active.)"""
    sol = _random_solution(aa_seq)
    first = calculate_change_vector(sol, analysis_objects)['CodonUsage']
    analysis_objects.codon_usage.windowscores = [999.0] * len(analysis_objects.codon_usage.windowscores)
    second = calculate_change_vector(sol, analysis_objects)['CodonUsage']
    assert first == second


def test_gc_term_baseline_stats_are_cached_across_calls(aa_seq, analysis_objects):
    """Mutates the two baseline arrays the GC term actually reads post-
    per-sequence-deviation redesign (gc.windows for the local windowed
    signal, gc.gcPerGene for the whole-candidate deviation) -- taggedGC1-3
    are no longer read by this term at all (see _gc_term's docstring), so
    mutating those wouldn't exercise the cache either way."""
    sol = _random_solution(aa_seq)
    first = calculate_change_vector(sol, analysis_objects)['GC']
    analysis_objects.gc.windows['Exon'] = [999.0] * len(analysis_objects.gc.windows['Exon'])
    analysis_objects.gc.gcPerGene = [999.0] * len(analysis_objects.gc.gcPerGene)
    second = calculate_change_vector(sol, analysis_objects)['GC']
    assert first == second
