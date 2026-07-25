"""Correctness tests for the population-batched change-vector computation.

Strategy: run the numpy backend (xp=numpy) first and compare it,
call-by-call, against calculate_change_vector() -- itself already checked
against the pre-vectorization original in test_change_vector_reference.py.
That numpy-batched pass is the correctness oracle for the GPU (cupy)
backend, checked separately in test_gpu_change_vector_cuda.py (skipped
automatically where no GPU/cupy is available) since the *numeric* result
should be identical regardless of which array backend computed it -- the
whole point of writing this against an injected `xp` module rather than
hardcoding either.
"""
import math

import numpy as np
import pytest

from genewriter.change_vector import calculate_change_vector
from genewriter.classes import KmerAnalysis
from genewriter.gpu_change_vector import (
    _encode_kmer_string,
    batch_calculate_change_vectors,
    batch_codon_pair_bias_term,
    batch_codon_usage_term,
    batch_gc_term,
    batch_kmer_term,
    batch_rare_codon_term,
    encode_population,
    windowed_average_batched,
)

from conftest import random_solution

LONG_AA_SEQ = "MAVLDEFGHIKPQRSTWYCN" * 7 + "MAVLDEFGHI"  # 150 residues


def _population(aa_seq, n, start_seed=0):
    return [random_solution(aa_seq, seed=start_seed + i) for i in range(n)]


def _assert_rows_close(actual_2d, expected_rows, tol=1e-6):
    assert len(actual_2d) == len(expected_rows)
    for p, (row, expected) in enumerate(zip(actual_2d, expected_rows)):
        assert len(row) == len(expected), f"individual {p}: length mismatch"
        for i, (a, e) in enumerate(zip(row, expected)):
            if math.isinf(e) or math.isinf(a):
                assert math.isinf(a) and math.isinf(e) and (a > 0) == (e > 0), f"individual {p}, pos {i}: {a} vs {e}"
                continue
            assert abs(a - e) < tol, f"individual {p}, position {i}: batched={a}, per-individual={e}"


def test_windowed_average_batched_matches_per_individual_for_each_row():
    from genewriter.change_vector import _windowed_average

    rng = np.random.default_rng(0)
    for trial in range(20):
        P = rng.integers(1, 8)
        M = rng.integers(0, 60)
        target_len = rng.integers(0, 60)
        winsize = int(rng.integers(1, 20))
        values = rng.uniform(-50, 50, size=(P, M))

        batched = windowed_average_batched(np, values, target_len, winsize)
        assert batched.shape == (P, target_len)
        for p in range(P):
            expected = _windowed_average(values[p].tolist(), target_len, winsize)
            for i, (a, e) in enumerate(zip(batched[p].tolist(), expected)):
                assert abs(a - e) < 1e-9, f"trial {trial}, row {p}, pos {i}: {a} vs {e}"


def test_encode_population_round_trips_via_decode():
    from genewriter.codon_tables import decode_codons

    pop = _population(LONG_AA_SEQ, 5)
    encoded = encode_population(np, pop)
    assert encoded.shape == (5, len(LONG_AA_SEQ))
    for p, sol in enumerate(pop):
        assert decode_codons(encoded[p]) == sol


@pytest.mark.parametrize("aa_seq_choice", ["short", "long"])
def test_batch_rare_codon_term_matches_per_individual(analysis_objects, aa_seq, aa_seq_choice):
    seq = aa_seq if aa_seq_choice == "short" else LONG_AA_SEQ
    pop = _population(seq, 6)
    codon_idx = encode_population(np, pop)

    actual = batch_rare_codon_term(np, codon_idx, analysis_objects.rare_codon)
    expected = [calculate_change_vector(sol, analysis_objects)['RareCodons'] for sol in pop]
    _assert_rows_close(actual.tolist(), expected)


@pytest.mark.parametrize("aa_seq_choice", ["short", "long"])
def test_batch_codon_usage_term_matches_per_individual(analysis_objects, aa_seq, aa_seq_choice):
    seq = aa_seq if aa_seq_choice == "short" else LONG_AA_SEQ
    pop = _population(seq, 6)
    codon_idx = encode_population(np, pop)

    actual = batch_codon_usage_term(np, codon_idx, analysis_objects.codon_usage)
    expected = [calculate_change_vector(sol, analysis_objects)['CodonUsage'] for sol in pop]
    _assert_rows_close(actual.tolist(), expected)


@pytest.mark.parametrize("aa_seq_choice", ["short", "long"])
def test_batch_codon_pair_bias_term_matches_per_individual(analysis_objects, aa_seq, aa_seq_choice):
    seq = aa_seq if aa_seq_choice == "short" else LONG_AA_SEQ
    pop = _population(seq, 6)
    codon_idx = encode_population(np, pop)

    actual = batch_codon_pair_bias_term(np, codon_idx, analysis_objects.codon_pair_bias)
    expected = [calculate_change_vector(sol, analysis_objects)['CodonPairBias'] for sol in pop]
    _assert_rows_close(actual.tolist(), expected)


@pytest.mark.parametrize("aa_seq_choice", ["short", "long"])
def test_batch_gc_term_matches_per_individual_with_uniform_locvec(analysis_objects, aa_seq, aa_seq_choice):
    seq = aa_seq if aa_seq_choice == "short" else LONG_AA_SEQ
    pop = _population(seq, 6)
    codon_idx = encode_population(np, pop)
    locvec = ['I'] * len(seq)

    actual = batch_gc_term(np, codon_idx, analysis_objects.gc, locvec)
    expected = [calculate_change_vector(sol, analysis_objects, locvec)['GC'] for sol in pop]
    _assert_rows_close(actual.tolist(), expected)


def test_batch_gc_term_matches_per_individual_with_block_structured_locvec(analysis_objects):
    import random as _random

    def block_locvec(n, seed):
        rng = _random.Random(seed)
        locvec = []
        while len(locvec) < n:
            block_len = rng.randint(15, 30)
            edge = min(3, block_len // 4)
            block = ['F'] * edge + ['I'] * (block_len - 2 * edge) + ['T'] * edge
            if block:
                block[rng.randrange(len(block))] = 'S'
            locvec.extend(block)
        return locvec[:n]

    pop = _population(LONG_AA_SEQ, 5)
    codon_idx = encode_population(np, pop)
    locvec = block_locvec(len(LONG_AA_SEQ), seed=42)

    actual = batch_gc_term(np, codon_idx, analysis_objects.gc, locvec)
    expected = [calculate_change_vector(sol, analysis_objects, locvec)['GC'] for sol in pop]
    _assert_rows_close(actual.tolist(), expected)


@pytest.mark.parametrize("aa_seq_choice", ["short", "long"])
def test_batch_kmer_term_matches_per_individual(analysis_objects, aa_seq, aa_seq_choice):
    seq = aa_seq if aa_seq_choice == "short" else LONG_AA_SEQ
    pop = _population(seq, 6)
    codon_idx = encode_population(np, pop)
    locvec = ['I'] * len(seq)

    actual = batch_kmer_term(np, codon_idx, analysis_objects.kmer, locvec)
    expected = [calculate_change_vector(sol, analysis_objects, locvec)['Kmer'] for sol in pop]
    _assert_rows_close(actual.tolist(), expected)


def test_batch_kmer_term_matches_per_individual_with_block_structured_locvec(analysis_objects):
    import random as _random

    def block_locvec(n, seed):
        rng = _random.Random(seed)
        locvec = []
        while len(locvec) < n:
            block_len = rng.randint(15, 30)
            edge = min(3, block_len // 4)
            block = ['F'] * edge + ['I'] * (block_len - 2 * edge) + ['T'] * edge
            if block:
                block[rng.randrange(len(block))] = 'S'
            locvec.extend(block)
        return locvec[:n]

    pop = _population(LONG_AA_SEQ, 5)
    codon_idx = encode_population(np, pop)
    locvec = block_locvec(len(LONG_AA_SEQ), seed=7)

    actual = batch_kmer_term(np, codon_idx, analysis_objects.kmer, locvec)
    expected = [calculate_change_vector(sol, analysis_objects, locvec)['Kmer'] for sol in pop]
    _assert_rows_close(actual.tolist(), expected)


def _multi_k_kmer_analysis(k_values, seed=0):
    """A KmerAnalysis with several k values (the shared conftest fixture
    only has k=2) -- exercises batch_kmer_term's per-k table-building and
    summing-across-k logic, and larger k's wider base-4 encoding range."""
    import random as _random

    rng = _random.Random(seed)
    bases = "ACGT"
    kmer_dict = {}
    l_dict = {}
    for k in k_values:
        seqs = {}
        # Every possible k-mer for small k (2, 3); a random sample for
        # larger k where 4**k is too big to enumerate exhaustively.
        if 4 ** k <= 256:
            import itertools
            all_seqs = [''.join(c) for c in itertools.product(bases, repeat=k)]
        else:
            all_seqs = [''.join(rng.choice(bases) for _ in range(k)) for _ in range(200)]
        for seq in all_seqs:
            seqs[seq] = {
                bucket: {'p': 0.01, 'fold_enrich': rng.uniform(0.3, 3.0)}
                for bucket in ('ExonL50', 'Exon', 'ExonR50')
            }
        kmer_dict[str(k)] = seqs
        l_dict[str(k)] = {'ExonL50': 1000, 'Exon': 1000, 'ExonR50': 1000}

    return KmerAnalysis(organism="test", transcriptome="test", kmer_dict=kmer_dict, l_dict=l_dict, kmer_score_obs={})


def test_batch_kmer_term_handles_multiple_k_values_and_sums_them():
    kmer = _multi_k_kmer_analysis([2, 3, 5], seed=11)
    from genewriter.change_vector import AnalysisObjects
    # Reuse the other 4 terms' baselines from the shared fixture isn't
    # possible here (no fixture access outside a test function's own
    # params) -- Kmer's calculate_change_vector path only touches
    # analysis_objects.kmer, so the other 4 fields are never read; still
    # need *some* AnalysisObjects instance to satisfy the signature.
    import random as _random
    rng = _random.Random(12)

    class _Unused:
        def __getattr__(self, name):
            raise AssertionError(f"Kmer term should never read AnalysisObjects.{name}")

    ao = AnalysisObjects(rare_codon=_Unused(), codon_usage=_Unused(), codon_pair_bias=_Unused(), gc=_Unused(), kmer=kmer)

    pop = _population(LONG_AA_SEQ, 4, start_seed=100)
    codon_idx = encode_population(np, pop)
    locvec = ['I'] * len(LONG_AA_SEQ)

    actual = batch_kmer_term(np, codon_idx, kmer, locvec)
    from genewriter.change_vector import _kmer_term
    expected = [_kmer_term(sol, ao, locvec) for sol in pop]
    _assert_rows_close(actual.tolist(), expected)


def test_batch_kmer_term_when_sequence_shorter_than_largest_k(analysis_objects):
    """A k longer than the whole (nucleotide) sequence contributes zero for
    that k rather than crashing -- num_wins == 0 must be a no-op, not an
    error, matching _kmer_term's identical `max(len(continuous) - k, 0)`
    guard."""
    kmer = _multi_k_kmer_analysis([2, 500], seed=3)
    short_seq = "MAVL"  # 4 residues = 12 nucleotides, well under k=500
    pop = _population(short_seq, 3, start_seed=200)
    codon_idx = encode_population(np, pop)
    locvec = ['I'] * len(short_seq)

    actual = batch_kmer_term(np, codon_idx, kmer, locvec)
    from genewriter.change_vector import AnalysisObjects, _kmer_term

    class _Unused:
        def __getattr__(self, name):
            raise AssertionError(f"Kmer term should never read AnalysisObjects.{name}")

    ao = AnalysisObjects(rare_codon=_Unused(), codon_usage=_Unused(), codon_pair_bias=_Unused(), gc=_Unused(), kmer=kmer)
    expected = [_kmer_term(sol, ao, locvec) for sol in pop]
    _assert_rows_close(actual.tolist(), expected)


def test_encode_kmer_string_matches_manual_base4_computation():
    assert _encode_kmer_string("AA", 2) == 0
    assert _encode_kmer_string("TT", 2) == 4 * 3 + 3  # 15, the max 2-mer code
    assert _encode_kmer_string("AC", 2) == 1  # A=0,C=1 -> 0*4+1
    assert _encode_kmer_string("CA", 2) == 4  # C=1,A=0 -> 1*4+0
    # General case: matches the manual positional-weighted-sum definition.
    from genewriter.codon_tables import NT_TO_BASE4
    seq = "GATTACA"
    k = len(seq)
    expected = sum(NT_TO_BASE4[b] * 4 ** (k - 1 - j) for j, b in enumerate(seq))
    assert _encode_kmer_string(seq, k) == expected


def test_batch_calculate_change_vectors_matches_per_individual_for_all_terms(analysis_objects):
    pop = _population(LONG_AA_SEQ, 5)
    results = batch_calculate_change_vectors(pop, analysis_objects, xp=np)

    assert len(results) == 5
    for p, (sol, result) in enumerate(zip(pop, results)):
        expected = calculate_change_vector(sol, analysis_objects)
        assert set(result.keys()) == set(expected.keys())
        for term in expected:
            _assert_rows_close([result[term]], [expected[term]])


def test_batch_calculate_change_vectors_empty_population_returns_empty_list(analysis_objects):
    assert batch_calculate_change_vectors([], analysis_objects, xp=np) == []
