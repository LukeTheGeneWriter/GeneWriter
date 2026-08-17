"""baseline_kmer.py's highest-risk property: splitting genes into 2 chunks,
writing raw-count shards, then finalize()-summing-and-normalizing must
produce the same fold_enrich values as baseline.compute_kmer_analysis()
called once on the whole gene list directly -- the two-phase
sum-raw-counts-then-normalize design is only correct if this holds.
"""
import pytest

from genewriter.baseline import compute_kmer_analysis
from genewriter.baseline_kmer import compute_and_write_shard, finalize

from conftest import make_synthetic_gene, make_synthetic_isoform, random_solution

_AA_SEQ = "MAVLDEFGHIKPQRSTWYCN" * 2


def _uniform_tag_genes(n: int) -> list:
    """Every codon tagged 'I' -- sidesteps the majority-vote tie-breaking
    ambiguity an even k can hit exactly at a location-tag boundary (see
    test_gpu_kmer_count.py's matching note); this test is about the
    sum-then-normalize finalize math, not tie-breaking."""
    genes = []
    for i in range(n):
        codons = random_solution(_AA_SEQ, seed=i)
        iso = make_synthetic_isoform(_AA_SEQ, lambda aa, j, c=codons: c[j], ['I'] * len(_AA_SEQ))
        genes.append(make_synthetic_gene(i, [iso]))
    return genes


def test_chunked_finalize_matches_monolithic_fold_enrich(tmp_path):
    genes = _uniform_tag_genes(6)
    chunk_a, chunk_b = genes[:3], genes[3:]
    k_values = (2, 3)

    compute_and_write_shard(chunk_a, str(tmp_path / 'chunk_0000.npz'), k_values=k_values, use_gpu=False)
    compute_and_write_shard(chunk_b, str(tmp_path / 'chunk_0001.npz'), k_values=k_values, use_gpu=False)

    merged = finalize(str(tmp_path), k_values=k_values)
    monolithic = compute_kmer_analysis(genes, k_values=k_values)

    for k in k_values:
        assert merged.l_dict[str(k)] == monolithic.l_dict[str(k)]
        assert set(merged.kmer_dict[str(k)]) == set(monolithic.kmer_dict[str(k)])
        for kmer, buckets in monolithic.kmer_dict[str(k)].items():
            for bucket, entry in buckets.items():
                got = merged.kmer_dict[str(k)][kmer][bucket]['fold_enrich']
                assert got == pytest.approx(entry['fold_enrich'])


def test_finalize_raises_with_no_shards(tmp_path):
    with pytest.raises(RuntimeError):
        finalize(str(tmp_path))
