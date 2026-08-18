"""Verifies the cupy (real GPU) backend of gpu_rare_codon_count.py against
the numpy backend, which is itself already checked in
test_gpu_rare_codon_count.py. Skipped automatically wherever cupy or a GPU
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

from genewriter.gpu_rare_codon_count import count_rare_codons_for_chunk  # noqa: E402

from conftest import make_synthetic_genes  # noqa: E402


def test_count_rare_codons_for_chunk_gpu_matches_cpu():
    genes = make_synthetic_genes(20)
    cpu_result = count_rare_codons_for_chunk(np, genes, winsize=15)
    gpu_result = count_rare_codons_for_chunk(cp, genes, winsize=15)

    cd, gd = dataclasses.asdict(cpu_result), dataclasses.asdict(gpu_result)
    assert cd['usagePerGene'] == pytest.approx(gd['usagePerGene'])
    for key in cd:
        if key == 'usagePerGene':
            continue
        assert cd[key] == gd[key]
