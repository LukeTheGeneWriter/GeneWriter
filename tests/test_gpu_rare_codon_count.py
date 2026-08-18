"""Numpy-backend correctness oracle for gpu_rare_codon_count.py -- no cupy
import, always runs. See test_gpu_rare_codon_count_cuda.py for the
cupy-vs-numpy agreement check.
"""
import dataclasses

import numpy as np

from genewriter.baseline import compute_rare_codon_analysis
from genewriter.gpu_rare_codon_count import (
    count_rare_codons_for_batch,
    count_rare_codons_for_chunk,
    encode_isoform_codons,
)

from conftest import make_synthetic_gene, make_synthetic_genes, make_synthetic_isoform


def test_count_rare_codons_for_batch_matches_hand_count():
    # GCG/CCG/CGT/CGC/TCG/ACG are rare. AAA/GGG are not.
    iso = make_synthetic_isoform("MKLA", lambda aa, i: ['ATG', 'GCG', 'CCG', 'AAA'][i], ['I', 'I', 'I', 'I'])
    codon_idx, loc_string = encode_isoform_codons(np, iso)

    result = count_rare_codons_for_batch(np, codon_idx, loc_string, starts=[0], lengths=[4], winsize=4)

    assert result['usage_per_gene'] == [0.5]  # 2 of 4 codons rare
    assert result['total_codons'] == 4
    # Only one full window (winsize == length -> 1 window), 2 rare codons in it.
    assert result['rare_codon_windows'][2] == 1
    assert sum(result['rare_codon_windows'].values()) == 1
    assert result['rare_by_location']['Exon'] == [2, 4]  # all 4 codons tagged 'I' -> Exon bucket


def test_count_rare_codons_for_batch_excludes_windows_spanning_isoform_join():
    # Isoform A: all rare (GCG x3). Isoform B: none rare (AAA x3).
    # A window straddling the join must not be counted at all.
    iso_a = make_synthetic_isoform("AAA", lambda aa, i: 'GCG', ['I', 'I', 'I'])
    iso_b = make_synthetic_isoform("KKK", lambda aa, i: 'AAA', ['I', 'I', 'I'])
    a = encode_isoform_codons(np, iso_a)
    b = encode_isoform_codons(np, iso_b)

    from genewriter.gpu_corpus_batch import concat_isoform_batch
    codon_idx, loc_string, starts, lengths = concat_isoform_batch(np, [a, b])

    result = count_rare_codons_for_batch(np, codon_idx, loc_string, starts, lengths, winsize=3)

    # Full convention: each 3-codon isoform gets exactly 1 valid window
    # (3 - 3 + 1 = 1) -- isoform A's window has 3 rare codons, isoform B's
    # has 0. No window ever mixes codons from both isoforms.
    assert result['rare_codon_windows'][3] == 1
    assert result['rare_codon_windows'][0] == 1
    assert sum(result['rare_codon_windows'].values()) == 2  # never 3+ -- would mean a cross-join window leaked through


def test_count_rare_codons_for_chunk_matches_baseline_compute_rare_codon_analysis():
    genes = make_synthetic_genes(6)
    gpu_result = count_rare_codons_for_chunk(np, genes, winsize=15)
    cpu_result = compute_rare_codon_analysis(genes, winsize=15)
    assert dataclasses.asdict(gpu_result) == dataclasses.asdict(cpu_result)


def test_count_rare_codons_for_chunk_matches_across_multiple_forced_sub_batches(monkeypatch):
    import genewriter.gpu_rare_codon_count as mod
    monkeypatch.setattr(mod, 'vram_aware_batch_size', lambda xp, bpu, vram_fraction=0.5, **kw: 1)

    genes = make_synthetic_genes(6)
    gpu_result = count_rare_codons_for_chunk(np, genes, winsize=15)
    cpu_result = compute_rare_codon_analysis(genes, winsize=15)
    assert dataclasses.asdict(gpu_result) == dataclasses.asdict(cpu_result)


def test_count_rare_codons_for_chunk_empty_genes_returns_zeroed_analysis():
    result = count_rare_codons_for_chunk(np, [], winsize=15)
    assert result.totalCodons == 0
    assert result.usagePerGene == []
    assert all(n == 0 for n in result.rare_codon_windows.values())
    assert all(pair == [0, 0] for pair in result.rareCodonsByLocation.values())
