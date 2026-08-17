import dataclasses

import pytest

from genewriter.baseline import compute_codon_pair_bias_analysis
from genewriter.baseline_codon_pair_bias import compute_and_write_shard, finalize

from conftest import make_synthetic_genes


def test_chunked_finalize_matches_monolithic(tmp_path):
    genes = make_synthetic_genes(6)
    chunk_a, chunk_b = genes[:3], genes[3:]

    compute_and_write_shard(chunk_a, str(tmp_path / 'chunk_0000.json'))
    compute_and_write_shard(chunk_b, str(tmp_path / 'chunk_0001.json'))

    merged = finalize(str(tmp_path))
    monolithic = compute_codon_pair_bias_analysis(genes)

    assert dataclasses.asdict(merged) == dataclasses.asdict(monolithic)


def test_finalize_raises_with_no_shards(tmp_path):
    with pytest.raises(RuntimeError):
        finalize(str(tmp_path))
