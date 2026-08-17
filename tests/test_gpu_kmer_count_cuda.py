"""Verifies the cupy (real GPU) backend of gpu_kmer_count.py against the
numpy backend, which is itself already checked in test_gpu_kmer_count.py.
Skipped automatically wherever cupy or a GPU isn't available -- mirrors
test_gpu_change_vector_cuda.py's skip-guard shape exactly.
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

from genewriter.gpu_kmer_count import count_kmers_for_chunk, count_kmers_for_isoform, encode_isoform_nt_base4  # noqa: E402

from conftest import make_synthetic_gene, make_synthetic_isoform, random_solution  # noqa: E402

_AA_SEQ = "MAVLDEFGHIKPQRSTWYCN" * 2


def _assert_counts_equal(cpu_counts, gpu_counts):
    assert set(cpu_counts) == set(gpu_counts)
    for name in cpu_counts:
        assert (cpu_counts[name] == cp.asnumpy(gpu_counts[name])).all()


def test_count_kmers_for_isoform_gpu_matches_cpu():
    codons = random_solution(_AA_SEQ, seed=0)
    iso = make_synthetic_isoform(_AA_SEQ, lambda aa, i, c=codons: c[i], ['I'] * len(_AA_SEQ))

    nt_base4_cpu, loc_string = encode_isoform_nt_base4(np, iso)
    nt_base4_gpu, _ = encode_isoform_nt_base4(cp, iso)

    for k in (2, 3, 4):
        cpu_counts, cpu_totals = count_kmers_for_isoform(np, nt_base4_cpu, loc_string, k)
        gpu_counts, gpu_totals = count_kmers_for_isoform(cp, nt_base4_gpu, loc_string, k)
        _assert_counts_equal(cpu_counts, gpu_counts)
        assert cpu_totals == gpu_totals


def test_count_kmers_for_chunk_gpu_matches_cpu():
    genes = []
    for i in range(3):
        codons = random_solution(_AA_SEQ, seed=i)
        iso = make_synthetic_isoform(_AA_SEQ, lambda aa, j, c=codons: c[j], ['I'] * len(_AA_SEQ))
        genes.append(make_synthetic_gene(i, [iso]))

    cpu_raw, cpu_totals = count_kmers_for_chunk(np, genes, k_values=(2, 3))
    gpu_raw, gpu_totals = count_kmers_for_chunk(cp, genes, k_values=(2, 3))

    for k in (2, 3):
        _assert_counts_equal(cpu_raw[k], gpu_raw[k])
        assert cpu_totals[k] == gpu_totals[k]
