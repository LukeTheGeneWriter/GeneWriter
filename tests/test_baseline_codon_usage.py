import dataclasses

import pytest

from genewriter.baseline import compute_codon_usage_analysis
from genewriter.baseline_codon_usage import compute_and_write_shard, finalize

from conftest import make_synthetic_genes

_FLOAT_LIST_FIELDS = ('codonUsageScoreByGene', 'windowscores', 'windowdistancesfromoptimal')


def test_chunked_finalize_matches_monolithic(tmp_path):
    """merged (via compute_and_write_shard()'s xp-injected
    gpu_codon_usage_count.count_codon_usage_for_chunk(), numpy or cupy) vs.
    monolithic (baseline.compute_codon_usage_analysis()'s pure-Python
    sequential sum) -- exact equality on int/string/dict fields, but
    pytest.approx on the float list fields: numpy/cupy reductions aren't
    guaranteed bit-identical to sequential Python summation (pairwise/SIMD
    summation order differs), so these agree numerically but not always to
    the exact ULP. See this session's plan ("finding 1")."""
    genes = make_synthetic_genes(6)
    chunk_a, chunk_b = genes[:3], genes[3:]

    compute_and_write_shard(chunk_a, str(tmp_path / 'chunk_0000.json'))
    compute_and_write_shard(chunk_b, str(tmp_path / 'chunk_0001.json'))

    merged = finalize(str(tmp_path))
    monolithic = compute_codon_usage_analysis(genes)

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
