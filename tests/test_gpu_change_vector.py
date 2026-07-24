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
from genewriter.gpu_change_vector import (
    batch_calculate_change_vectors,
    batch_codon_pair_bias_term,
    batch_codon_usage_term,
    batch_gc_term,
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
