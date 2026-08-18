"""Verifies the cupy (real GPU) backend of gpu_gc_count.py against the
numpy backend, already checked in test_gpu_gc_count.py. Skipped
automatically wherever cupy or a GPU isn't available.
"""
import dataclasses

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    cp.cuda.runtime.getDeviceCount()
    _has_gpu = True
except Exception:
    _has_gpu = False

pytestmark = pytest.mark.skipif(not _has_gpu, reason="no CUDA GPU available")

from genewriter.gpu_gc_count import count_gc_for_chunk  # noqa: E402

from conftest import make_synthetic_genes  # noqa: E402


def test_count_gc_for_chunk_gpu_matches_cpu():
    genes = make_synthetic_genes(20)
    cpu_result = count_gc_for_chunk(np, genes, winsize=21)
    gpu_result = count_gc_for_chunk(cp, genes, winsize=21)

    cd, gd = dataclasses.asdict(cpu_result), dataclasses.asdict(gpu_result)
    for key in cd:
        if key == 'gcPerGene':
            assert gd[key] == pytest.approx(cd[key], rel=1e-6, abs=1e-6)
        elif key == 'windows':
            for bucket in cd[key]:
                assert gd[key][bucket] == pytest.approx(cd[key][bucket], rel=1e-6, abs=1e-6)
        else:
            assert gd[key] == cd[key]
