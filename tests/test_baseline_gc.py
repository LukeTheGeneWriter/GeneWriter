import dataclasses

import pytest

from genewriter.baseline import compute_gc_analysis
from genewriter.baseline_gc import compute_and_write_shard, finalize

from conftest import make_synthetic_gene, make_synthetic_genes, make_synthetic_isoform, random_solution


def _assert_analysis_close(merged, monolithic):
    """See test_baseline_codon_usage.py's matching docstring for why float
    fields need pytest.approx here ("finding 1")."""
    md, cd = dataclasses.asdict(merged), dataclasses.asdict(monolithic)
    assert set(md) == set(cd)
    for key in cd:
        if key == 'gcPerGene':
            assert len(md[key]) == len(cd[key]), key
            assert md[key] == pytest.approx(cd[key], rel=1e-9, abs=1e-9), key
        elif key == 'windows':
            assert set(md[key]) == set(cd[key]), key
            for bucket in cd[key]:
                assert len(md[key][bucket]) == len(cd[key][bucket]), (key, bucket)
                assert md[key][bucket] == pytest.approx(cd[key][bucket], rel=1e-9, abs=1e-9), (key, bucket)
        else:
            assert md[key] == cd[key], key


def test_chunked_finalize_matches_monolithic(tmp_path):
    genes = make_synthetic_genes(6)
    chunk_a, chunk_b = genes[:3], genes[3:]

    compute_and_write_shard(chunk_a, str(tmp_path / 'chunk_0000.json'))
    compute_and_write_shard(chunk_b, str(tmp_path / 'chunk_0001.json'))

    merged = finalize(str(tmp_path))
    monolithic = compute_gc_analysis(genes)

    _assert_analysis_close(merged, monolithic)


def test_chunked_finalize_matches_monolithic_with_s_tags_present(tmp_path):
    # The Finding-2-relevant case, exercised through the real
    # compute_and_write_shard()/finalize() pipeline (not just
    # gpu_gc_count.py directly, see tests/test_gpu_gc_count.py for that) --
    # conftest.make_synthetic_genes() never generates 'S' tags, so this
    # builds one explicitly. Run lengths chosen to avoid an exact raw-tag
    # vote tie (see gpu_corpus_batch.majority_raw_tag_bucket()'s own
    # docstring on why a genuine tie can't be bit-matched).
    aa_seq = "MAVLDEFGHIKPQRSTWYCN" * 2
    codons = random_solution(aa_seq, seed=3)
    loc_tags = (['F'] * 4) + (['S'] * 7) + (['I'] * 22) + (['T'] * 4) + (['S'] * (len(codons) - 37))
    loc_tags = loc_tags[:len(codons)]
    iso = make_synthetic_isoform(aa_seq, lambda aa, i, c=codons: c[i], loc_tags)
    genes = [make_synthetic_gene(1, [iso])] + make_synthetic_genes(5)
    chunk_a, chunk_b = genes[:3], genes[3:]

    compute_and_write_shard(chunk_a, str(tmp_path / 'chunk_0000.json'))
    compute_and_write_shard(chunk_b, str(tmp_path / 'chunk_0001.json'))

    merged = finalize(str(tmp_path))
    monolithic = compute_gc_analysis(genes)

    _assert_analysis_close(merged, monolithic)
    assert all(len(v) > 0 for v in monolithic.windows.values())  # sanity: all 3 buckets actually exercised


def test_finalize_raises_with_no_shards(tmp_path):
    with pytest.raises(RuntimeError):
        finalize(str(tmp_path))
