"""Numpy-backend correctness oracle for gpu_codon_usage_count.py -- no cupy
import, always runs.
"""
import dataclasses

import numpy as np
import pytest

from genewriter.baseline import compute_codon_usage_analysis
from genewriter.gpu_codon_usage_count import (
    count_codon_usage_for_batch,
    count_codon_usage_for_chunk,
    encode_isoform_codons,
)
from genewriter.gpu_corpus_batch import concat_isoform_batch

from conftest import make_synthetic_genes, make_synthetic_isoform

_FLOAT_LIST_FIELDS = ('codonUsageScoreByGene', 'windowscores', 'windowdistancesfromoptimal')


def _assert_analysis_close(gpu_result, cpu_result):
    """Exact equality on int/string/dict fields; pytest.approx on the float
    list fields, whose numpy-vectorized reduction order is not guaranteed
    bit-identical to sequential Python summation. See this session's plan
    ("finding 1") for why."""
    gd, cd = dataclasses.asdict(gpu_result), dataclasses.asdict(cpu_result)
    assert set(gd) == set(cd)
    for key in cd:
        if key in _FLOAT_LIST_FIELDS:
            assert len(gd[key]) == len(cd[key]), key
            assert gd[key] == pytest.approx(cd[key], rel=1e-9, abs=1e-9), key
        else:
            assert gd[key] == cd[key], key


def test_count_codon_usage_for_batch_matches_hand_count():
    iso = make_synthetic_isoform("MK", lambda aa, i: ['ATG', 'AAA'][i], ['I', 'I'])
    codon_idx, loc_string = encode_isoform_codons(np, iso)

    result = count_codon_usage_for_batch(np, codon_idx, loc_string, starts=[0], lengths=[2], winsize=2)

    assert result['total_codons'] == 2
    assert len(result['usage_score_by_gene']) == 1
    assert result['codon_freqs_by_location']['ATG']['Exon'] == 1
    assert result['codon_freqs_by_location']['AAA']['Exon'] == 1
    assert len(result['window_scores']) == 1  # 2 - 2 + 1 = 1 valid window
    assert len(result['window_dist_from_optimal']) == 1


def test_count_codon_usage_for_batch_excludes_windows_spanning_isoform_join():
    iso_a = make_synthetic_isoform("MK", lambda aa, i: ['ATG', 'AAA'][i], ['I', 'I'])
    iso_b = make_synthetic_isoform("KL", lambda aa, i: ['AAA', 'CTG'][i], ['I', 'I'])
    a, b = encode_isoform_codons(np, iso_a), encode_isoform_codons(np, iso_b)
    codon_idx, loc_string, starts, lengths = concat_isoform_batch(np, [a, b])

    result = count_codon_usage_for_batch(np, codon_idx, loc_string, starts, lengths, winsize=2)

    # Each 2-codon isoform gets exactly 1 valid window (2-2+1=1); a window
    # straddling the join (codon index 1 and 2, spanning both isoforms)
    # must not be counted -- exactly 2 windows total, not 3.
    assert len(result['window_scores']) == 2


def test_count_codon_usage_for_chunk_matches_baseline_compute_codon_usage_analysis():
    genes = make_synthetic_genes(6)
    gpu_result = count_codon_usage_for_chunk(np, genes, winsize=15)
    cpu_result = compute_codon_usage_analysis(genes, winsize=15)
    _assert_analysis_close(gpu_result, cpu_result)


def test_count_codon_usage_for_chunk_matches_across_multiple_forced_sub_batches(monkeypatch):
    import genewriter.gpu_codon_usage_count as mod
    monkeypatch.setattr(mod, 'vram_aware_batch_size', lambda xp, bpu, vram_fraction=0.5, **kw: 1)

    genes = make_synthetic_genes(6)
    gpu_result = count_codon_usage_for_chunk(np, genes, winsize=15)
    cpu_result = compute_codon_usage_analysis(genes, winsize=15)
    _assert_analysis_close(gpu_result, cpu_result)


def test_count_codon_usage_for_chunk_empty_genes_returns_zeroed_analysis():
    result = count_codon_usage_for_chunk(np, [], winsize=15)
    assert result.totalCodons == 0
    assert result.AAFreqs == {}
    assert result.codonUsageScoreByGene == []
    assert result.windowscores == []
