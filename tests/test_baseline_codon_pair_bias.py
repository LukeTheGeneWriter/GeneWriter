import dataclasses

import pytest

from genewriter.baseline import compute_codon_pair_bias_analysis
from genewriter.baseline_codon_pair_bias import compute_and_write_shard, finalize

from conftest import make_synthetic_genes

_FLOAT_LIST_FIELDS = ('cpbPerGene', 'cpbPerWindow')


def test_chunked_finalize_matches_monolithic(tmp_path):
    """See test_baseline_codon_usage.py's matching docstring for why float
    list fields need pytest.approx here ("finding 1")."""
    genes = make_synthetic_genes(6)
    chunk_a, chunk_b = genes[:3], genes[3:]

    compute_and_write_shard(chunk_a, str(tmp_path / 'chunk_0000.json'))
    compute_and_write_shard(chunk_b, str(tmp_path / 'chunk_0001.json'))

    merged = finalize(str(tmp_path))
    monolithic = compute_codon_pair_bias_analysis(genes)

    md, cd = dataclasses.asdict(merged), dataclasses.asdict(monolithic)
    assert set(md) == set(cd)
    for key in cd:
        if key in _FLOAT_LIST_FIELDS:
            assert len(md[key]) == len(cd[key]), key
            assert md[key] == pytest.approx(cd[key], rel=1e-9, abs=1e-9), key
        else:
            assert md[key] == cd[key], key


def test_finalize_raises_with_no_shards(tmp_path):
    with pytest.raises(RuntimeError):
        finalize(str(tmp_path))
