"""Numpy-backend correctness oracle for gpu_codon_pair_bias_count.py -- no
cupy import, always runs.
"""
import dataclasses

import numpy as np
import pytest

from genewriter.baseline import compute_codon_pair_bias_analysis
from genewriter.gpu_codon_pair_bias_count import (
    _build_cpb_table,
    count_codon_pair_bias_for_batch,
    count_codon_pair_bias_for_chunk,
    encode_isoform_codons,
)
from genewriter.gpu_corpus_batch import concat_isoform_batch

from conftest import make_synthetic_gene, make_synthetic_genes, make_synthetic_isoform

_FLOAT_LIST_FIELDS = ('cpbPerGene', 'cpbPerWindow')


def _assert_analysis_close(gpu_result, cpu_result):
    gd, cd = dataclasses.asdict(gpu_result), dataclasses.asdict(cpu_result)
    assert set(gd) == set(cd)
    for key in cd:
        if key in _FLOAT_LIST_FIELDS:
            assert len(gd[key]) == len(cd[key]), key
            assert gd[key] == pytest.approx(cd[key], rel=1e-9, abs=1e-9), key
        else:
            assert gd[key] == cd[key], key


def test_count_codon_pair_bias_for_batch_matches_hand_count():
    iso = make_synthetic_isoform("MKL", lambda aa, i: ['ATG', 'AAA', 'CTG'][i], ['I', 'I', 'I'])
    codon_idx, _loc = encode_isoform_codons(np, iso)
    cpb_lit = {'ATGAAA': 100.0, 'AAACTG': 200.0}
    cpb_table = _build_cpb_table(cpb_lit)

    result = count_codon_pair_bias_for_batch(np, codon_idx, starts=[0], lengths=[3], winsize=3, cpb_table=cpb_table)

    assert result['total_pairs'] == 2
    assert result['per_gene'] == pytest.approx([150.0])  # mean(100, 200)
    assert len(result['per_window']) == 1  # 3-codon window -> 1 valid window (3-3+1=1)
    assert result['per_window'] == pytest.approx([150.0])  # only window covers both pairs


def test_count_codon_pair_bias_for_batch_excludes_pairs_spanning_isoform_join():
    iso_a = make_synthetic_isoform("MK", lambda aa, i: ['ATG', 'AAA'][i], ['I', 'I'])
    iso_b = make_synthetic_isoform("KL", lambda aa, i: ['AAA', 'CTG'][i], ['I', 'I'])
    a, b = encode_isoform_codons(np, iso_a), encode_isoform_codons(np, iso_b)
    codon_idx, _loc, starts, lengths = concat_isoform_batch(np, [a, b])
    cpb_table = _build_cpb_table({})  # all zeros -- content doesn't matter, only counts do

    result = count_codon_pair_bias_for_batch(np, codon_idx, starts, lengths, winsize=2, cpb_table=cpb_table)

    # Each 2-codon isoform has exactly 1 valid pair (ATG-AAA, AAA-CTG) --
    # the cross-join pair (AAA from iso_a, AAA from iso_b) must not count.
    assert result['total_pairs'] == 2
    assert len(result['per_gene']) == 2


def test_count_codon_pair_bias_for_chunk_skips_isoforms_with_fewer_than_2_codons():
    single_codon_iso = make_synthetic_isoform("M", lambda aa, i: 'ATG', ['I'])
    genes = [make_synthetic_gene(1, [single_codon_iso])]

    gpu_result = count_codon_pair_bias_for_chunk(np, genes, winsize=15)
    cpu_result = compute_codon_pair_bias_analysis(genes, winsize=15)
    _assert_analysis_close(gpu_result, cpu_result)
    assert gpu_result.cpbPerGene == []
    assert gpu_result.totalCodonPairs == 0


def test_count_codon_pair_bias_for_chunk_matches_baseline_compute_codon_pair_bias_analysis():
    single_codon_iso = make_synthetic_isoform("M", lambda aa, i: 'ATG', ['I'])
    genes = make_synthetic_genes(6) + [make_synthetic_gene(99, [single_codon_iso])]

    gpu_result = count_codon_pair_bias_for_chunk(np, genes, winsize=15)
    cpu_result = compute_codon_pair_bias_analysis(genes, winsize=15)
    _assert_analysis_close(gpu_result, cpu_result)


def test_count_codon_pair_bias_for_chunk_matches_across_multiple_forced_sub_batches(monkeypatch):
    import genewriter.gpu_codon_pair_bias_count as mod
    monkeypatch.setattr(mod, 'vram_aware_batch_size', lambda xp, bpu, vram_fraction=0.5, **kw: 1)

    genes = make_synthetic_genes(6)
    gpu_result = count_codon_pair_bias_for_chunk(np, genes, winsize=15)
    cpu_result = compute_codon_pair_bias_analysis(genes, winsize=15)
    _assert_analysis_close(gpu_result, cpu_result)


def test_count_codon_pair_bias_for_chunk_empty_genes_returns_zeroed_analysis():
    result = count_codon_pair_bias_for_chunk(np, [], winsize=15)
    assert result.totalCodonPairs == 0
    assert result.cpbPerGene == []
    assert result.cpbPerWindow == []
