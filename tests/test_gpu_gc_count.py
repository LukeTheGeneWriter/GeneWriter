"""Numpy-backend correctness oracle for gpu_gc_count.py -- no cupy import,
always runs.
"""
import dataclasses

import numpy as np
import pytest

from genewriter.baseline import compute_gc_analysis
from genewriter.gpu_corpus_batch import concat_isoform_batch
from genewriter.gpu_gc_count import count_gc_for_batch, count_gc_for_chunk, encode_isoform_codons

from conftest import make_synthetic_gene, make_synthetic_genes, make_synthetic_isoform


def _assert_analysis_close(gpu_result, cpu_result):
    """Exact equality on int/string/dict-of-int fields; pytest.approx (not
    raw absolute-diff) on numpy-reduction-derived float fields, whose
    reduction order isn't guaranteed bit-identical to sequential Python
    summation -- see this session's plan ("finding 1")."""
    gd, cd = dataclasses.asdict(gpu_result), dataclasses.asdict(cpu_result)
    assert set(gd) == set(cd)
    for key in cd:
        if key == 'gcPerGene':
            assert len(gd[key]) == len(cd[key]), key
            assert gd[key] == pytest.approx(cd[key], rel=1e-9, abs=1e-9), key
        elif key == 'windows':
            assert set(gd[key]) == set(cd[key]), key
            for bucket in cd[key]:
                assert len(gd[key][bucket]) == len(cd[key][bucket]), (key, bucket)
                assert gd[key][bucket] == pytest.approx(cd[key][bucket], rel=1e-9, abs=1e-9), (key, bucket)
        else:
            assert gd[key] == cd[key], key


def test_count_gc_for_batch_matches_hand_count():
    # ATG (1 GC base: G) AAA (0) GCG (3, all GC) -- 3 codons, 9 nt, 4 GC bases -> 4/9
    iso = make_synthetic_isoform("MKA", lambda aa, i: ['ATG', 'AAA', 'GCG'][i], ['I', 'I', 'I'])
    codon_idx, loc_string = encode_isoform_codons(np, iso)

    result = count_gc_for_batch(np, codon_idx, loc_string, starts=[0], lengths=[3], winsize=9)

    assert result['total_codons'] == 3
    assert result['gc_per_gene'] == pytest.approx([4 / 9])
    assert result['tagged']['Exon'][0] == [0, 0, 1]  # position-1 GC flag per codon: ATG(A)->0, AAA(A)->0, GCG(G)->1
    assert result['tagged']['Exon'][2] == [1, 0, 1]  # position-3: ATG(G)->1, AAA(A)->0, GCG(G)->1
    assert len(result['windows']['Exon']) == 1  # 9nt window, winsize=9 -> exactly 1 valid window


def test_count_gc_for_batch_excludes_windows_spanning_isoform_join():
    iso_a = make_synthetic_isoform("MK", lambda aa, i: ['ATG', 'AAA'][i], ['I', 'I'])  # 6nt
    iso_b = make_synthetic_isoform("KL", lambda aa, i: ['AAA', 'GCG'][i], ['I', 'I'])  # 6nt
    a, b = encode_isoform_codons(np, iso_a), encode_isoform_codons(np, iso_b)
    codon_idx, loc_string, starts, lengths = concat_isoform_batch(np, [a, b])

    result = count_gc_for_batch(np, codon_idx, loc_string, starts, lengths, winsize=6)

    # Each 6nt isoform gets exactly 1 valid window (6-6+1=1); a window
    # straddling the 12nt concatenated join must not appear.
    assert sum(len(v) for v in result['windows'].values()) == 2


def test_count_gc_for_chunk_matches_baseline_compute_gc_analysis():
    genes = make_synthetic_genes(6)
    gpu_result = count_gc_for_chunk(np, genes, winsize=21)
    cpu_result = compute_gc_analysis(genes, winsize=21)
    _assert_analysis_close(gpu_result, cpu_result)


def test_count_gc_for_chunk_matches_with_s_tags_present():
    # The Finding-2-relevant case: S tags mixed with I tags against an F/T
    # plurality. conftest.make_synthetic_genes() never generates S tags, so
    # this test builds one explicitly -- exercises
    # gpu_corpus_batch.majority_raw_tag_bucket()'s divergence from
    # gpu_change_vector.majority_window_bucket() for real.
    #
    # Run lengths chosen (and verified, see this test's own history) to
    # produce ZERO exact ties in any 21nt window's raw-tag vote -- baseline.
    # compute_gc_analysis's own tie-break (`max(set(...), key=...count)`) is
    # itself unspecified (Python set iteration order depends on string
    # hashing, see change_vector.py's matching comment), so a genuine tie
    # can't be reproduced bit-for-bit by any fixed-order re-implementation;
    # this test is about the non-tie S+I-vs-F/T divergence specifically; see
    # test_gpu_corpus_batch.py's isolated
    # test_majority_raw_tag_bucket_diverges_from_majority_window_bucket_with_s_tags
    # for that case in its cleanest, single-window form.
    from conftest import random_solution
    aa_seq = "MAVLDEFGHIKPQRSTWYCN" * 2
    codons = random_solution(aa_seq, seed=3)
    loc_tags = (['F'] * 4) + (['S'] * 7) + (['I'] * 22) + (['T'] * 4) + (['S'] * (len(codons) - 37))
    loc_tags = loc_tags[:len(codons)]
    iso = make_synthetic_isoform(aa_seq, lambda aa, i, c=codons: c[i], loc_tags)
    genes = [make_synthetic_gene(1, [iso])] + make_synthetic_genes(5)

    gpu_result = count_gc_for_chunk(np, genes, winsize=21)
    cpu_result = compute_gc_analysis(genes, winsize=21)
    _assert_analysis_close(gpu_result, cpu_result)
    # sanity: this test setup actually produced windows in all 3 buckets --
    # otherwise it wouldn't actually be exercising the divergence case
    assert all(len(v) > 0 for v in cpu_result.windows.values())


def test_count_gc_for_chunk_matches_across_multiple_forced_sub_batches(monkeypatch):
    import genewriter.gpu_gc_count as mod
    monkeypatch.setattr(mod, 'vram_aware_batch_size', lambda xp, bpu, vram_fraction=0.5, **kw: 1)

    genes = make_synthetic_genes(6)
    gpu_result = count_gc_for_chunk(np, genes, winsize=21)
    cpu_result = compute_gc_analysis(genes, winsize=21)
    _assert_analysis_close(gpu_result, cpu_result)


def test_count_gc_for_chunk_empty_genes_returns_zeroed_analysis():
    result = count_gc_for_chunk(np, [], winsize=21)
    assert result.totalCodons == 0
    assert result.gcPerGene == []
    assert all(v == [] for v in result.windows.values())
