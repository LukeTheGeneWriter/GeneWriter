"""Verifies the cupy (real GPU) backend of gpu_codon_usage_count.py against
the numpy backend, which is itself already checked in
test_gpu_codon_usage_count.py. Skipped automatically wherever cupy or a GPU
isn't available.
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

from genewriter.gpu_codon_usage_count import count_codon_usage_for_chunk  # noqa: E402

from conftest import make_synthetic_genes  # noqa: E402

_FLOAT_LIST_FIELDS = ('codonUsageScoreByGene', 'windowscores', 'windowdistancesfromoptimal')


def test_count_codon_usage_for_chunk_gpu_matches_cpu():
    genes = make_synthetic_genes(20)
    cpu_result = count_codon_usage_for_chunk(np, genes, winsize=15)
    gpu_result = count_codon_usage_for_chunk(cp, genes, winsize=15)

    cd, gd = dataclasses.asdict(cpu_result), dataclasses.asdict(gpu_result)
    for key in cd:
        if key in _FLOAT_LIST_FIELDS:
            assert gd[key] == pytest.approx(cd[key], rel=1e-6, abs=1e-6)
        else:
            assert gd[key] == cd[key]
