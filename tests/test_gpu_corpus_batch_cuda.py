"""Verifies gpu_corpus_batch.py's xp-injected primitives on real cupy/GPU
hardware, matching the numpy backend already checked in
test_gpu_corpus_batch.py. Skipped automatically wherever cupy or a GPU
isn't available.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    cp.cuda.runtime.getDeviceCount()
    _has_gpu = True
except Exception:
    _has_gpu = False

pytestmark = pytest.mark.skipif(not _has_gpu, reason="no CUDA GPU available")

from genewriter.gpu_corpus_batch import concat_isoform_batch, segment_mean, select_backend, vram_aware_batch_size  # noqa: E402


def test_select_backend_use_gpu_true_returns_cupy():
    xp = select_backend(True, 'test')
    assert xp is cp


def test_vram_aware_batch_size_cupy_queries_real_free_memory():
    size = vram_aware_batch_size(cp, bytes_per_unit=100)
    assert size > 0
    # sanity: shouldn't wildly exceed what a real GPU could report as free
    free_bytes, _total = cp.cuda.Device().mem_info
    assert size <= free_bytes  # even at vram_fraction=1.0 this must hold


def test_concat_isoform_batch_gpu_matches_cpu():
    a_cpu, a_gpu = np.asarray([1, 2, 3]), cp.asarray([1, 2, 3])
    b_cpu, b_gpu = np.asarray([4, 5]), cp.asarray([4, 5])

    flat_cpu, loc_cpu, starts_cpu, lengths_cpu = concat_isoform_batch(np, [(a_cpu, 'III'), (b_cpu, 'FT')])
    flat_gpu, loc_gpu, starts_gpu, lengths_gpu = concat_isoform_batch(cp, [(a_gpu, 'III'), (b_gpu, 'FT')])

    assert cp.asnumpy(flat_gpu).tolist() == flat_cpu.tolist()
    assert loc_gpu == loc_cpu
    assert starts_gpu == starts_cpu
    assert lengths_gpu == lengths_cpu


def test_segment_mean_gpu_matches_cpu():
    values_cpu = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0])
    values_gpu = cp.asarray(values_cpu)
    segment_ids = np.asarray([0, 0, 0, 1, 1])

    cpu_means = segment_mean(np, values_cpu, segment_ids, n_segments=2)
    gpu_means = segment_mean(cp, values_gpu, segment_ids, n_segments=2)

    assert cp.asnumpy(gpu_means).tolist() == pytest.approx(cpu_means.tolist())
