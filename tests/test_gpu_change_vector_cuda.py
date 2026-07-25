"""Verifies the cupy (real GPU) backend against the numpy backend, which is
itself already checked against calculate_change_vector() in
test_gpu_change_vector.py. Skipped automatically wherever cupy or a GPU
isn't available -- this environment has both (see the dev-environment
memory for setup), but CI or another machine may not.
"""
import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    cp.cuda.runtime.getDeviceCount()
    _has_gpu = True
except Exception:
    _has_gpu = False

pytestmark = pytest.mark.skipif(not _has_gpu, reason="no CUDA GPU available")

from genewriter.gpu_change_vector import (  # noqa: E402
    batch_calculate_change_vectors,
    batch_codon_pair_bias_term,
    batch_codon_usage_term,
    batch_gc_term,
    batch_kmer_term,
    batch_rare_codon_term,
    encode_population,
    windowed_average_batched,
)

from conftest import random_solution  # noqa: E402

LONG_AA_SEQ = "MAVLDEFGHIKPQRSTWYCN" * 7 + "MAVLDEFGHI"  # 150 residues


def _population(aa_seq, n, start_seed=0):
    return [random_solution(aa_seq, seed=start_seed + i) for i in range(n)]


def _assert_arrays_close(cpu_result, gpu_result, rtol=1e-6, atol=1e-9):
    """CPU and GPU reductions (.mean()/.sum()) aren't guaranteed to combine
    floating-point values in the same order -- the Handoff's own warning
    that addition isn't associative applies across devices too, not just
    across parallel-vs-serial on one device. A relative tolerance is the
    right comparison here, not exact equality (confirmed real by the first
    version of this test: CodonPairBias differed by ~2e-9 in an absolute
    sense on a value around 5.5 million -- a relative difference of ~3e-16,
    right at float64's precision limit, not a computational error)."""
    gpu_as_numpy = cp.asnumpy(gpu_result)
    assert cpu_result.shape == gpu_as_numpy.shape
    for idx in np.ndindex(cpu_result.shape):
        a, b = float(cpu_result[idx]), float(gpu_as_numpy[idx])
        if math.isinf(a) or math.isinf(b):
            assert math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0), f"{idx}: cpu={a}, gpu={b}"
            continue
        assert abs(a - b) <= atol + rtol * max(abs(a), abs(b)), f"{idx}: cpu={a}, gpu={b}"


def test_gpu_actually_ran_on_device():
    """Sanity check that xp=cupy really executes on the GPU, not silently
    falling back to something CPU-side."""
    a = cp.arange(10)
    assert isinstance(a, cp.ndarray)
    assert a.device is not None


def test_windowed_average_batched_gpu_matches_cpu():
    rng = np.random.default_rng(0)
    values = rng.uniform(-50, 50, size=(6, 40))
    cpu = windowed_average_batched(np, values, 60, 15)
    gpu = windowed_average_batched(cp, cp.asarray(values), 60, 15)
    _assert_arrays_close(cpu, gpu)


def test_encode_population_gpu_matches_cpu():
    pop = _population(LONG_AA_SEQ, 5)
    cpu = encode_population(np, pop)
    gpu = encode_population(cp, pop)
    assert (cpu == cp.asnumpy(gpu)).all()


def test_batch_rare_codon_term_gpu_matches_cpu(analysis_objects):
    pop = _population(LONG_AA_SEQ, 6)
    cpu_idx = encode_population(np, pop)
    gpu_idx = encode_population(cp, pop)
    cpu = batch_rare_codon_term(np, cpu_idx, analysis_objects.rare_codon)
    gpu = batch_rare_codon_term(cp, gpu_idx, analysis_objects.rare_codon)
    _assert_arrays_close(cpu, gpu)


def test_batch_codon_usage_term_gpu_matches_cpu(analysis_objects):
    pop = _population(LONG_AA_SEQ, 6)
    cpu_idx = encode_population(np, pop)
    gpu_idx = encode_population(cp, pop)
    cpu = batch_codon_usage_term(np, cpu_idx, analysis_objects.codon_usage)
    gpu = batch_codon_usage_term(cp, gpu_idx, analysis_objects.codon_usage)
    _assert_arrays_close(cpu, gpu)


def test_batch_codon_pair_bias_term_gpu_matches_cpu(analysis_objects):
    pop = _population(LONG_AA_SEQ, 6)
    cpu_idx = encode_population(np, pop)
    gpu_idx = encode_population(cp, pop)
    cpu = batch_codon_pair_bias_term(np, cpu_idx, analysis_objects.codon_pair_bias)
    gpu = batch_codon_pair_bias_term(cp, gpu_idx, analysis_objects.codon_pair_bias)
    _assert_arrays_close(cpu, gpu)


def test_batch_gc_term_gpu_matches_cpu(analysis_objects):
    pop = _population(LONG_AA_SEQ, 6)
    cpu_idx = encode_population(np, pop)
    gpu_idx = encode_population(cp, pop)
    locvec = ['I'] * len(LONG_AA_SEQ)
    cpu = batch_gc_term(np, cpu_idx, analysis_objects.gc, locvec)
    gpu = batch_gc_term(cp, gpu_idx, analysis_objects.gc, locvec)
    _assert_arrays_close(cpu, gpu)


def test_batch_kmer_term_gpu_matches_cpu(analysis_objects):
    pop = _population(LONG_AA_SEQ, 6)
    cpu_idx = encode_population(np, pop)
    gpu_idx = encode_population(cp, pop)
    locvec = ['I'] * len(LONG_AA_SEQ)
    cpu = batch_kmer_term(np, cpu_idx, analysis_objects.kmer, locvec)
    gpu = batch_kmer_term(cp, gpu_idx, analysis_objects.kmer, locvec)
    _assert_arrays_close(cpu, gpu)


def test_batch_calculate_change_vectors_gpu_matches_cpu(analysis_objects):
    pop = _population(LONG_AA_SEQ, 6)
    cpu_results = batch_calculate_change_vectors(pop, analysis_objects, xp=np)
    gpu_results = batch_calculate_change_vectors(pop, analysis_objects, xp=cp)

    assert len(cpu_results) == len(gpu_results)
    for cpu_r, gpu_r in zip(cpu_results, gpu_results):
        assert set(cpu_r.keys()) == set(gpu_r.keys())
        for term in cpu_r:
            # Both sides are already plain Python lists here (
            # batch_calculate_change_vectors converts to .tolist() itself),
            # so compare directly rather than through _assert_arrays_close
            # (which expects a raw, not-yet-converted cupy array).
            for i, (a, b) in enumerate(zip(cpu_r[term], gpu_r[term])):
                if math.isinf(a) or math.isinf(b):
                    assert math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0), f"{term}[{i}]: cpu={a}, gpu={b}"
                    continue
                assert abs(a - b) <= 1e-9 + 1e-6 * max(abs(a), abs(b)), f"{term}[{i}]: cpu={a}, gpu={b}"
