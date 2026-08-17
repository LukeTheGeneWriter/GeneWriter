"""Numpy-backend correctness oracle for gpu_kmer_count.py -- no cupy import,
always runs. See test_gpu_kmer_count_cuda.py for the cupy-vs-numpy agreement
check, mirroring test_gpu_change_vector.py/test_gpu_change_vector_cuda.py's
existing split.
"""
import numpy as np

from genewriter.codon_tables import decode_base4_string
from genewriter.gpu_kmer_count import count_kmers_for_chunk, count_kmers_for_isoform, encode_isoform_nt_base4

from conftest import make_synthetic_gene, make_synthetic_isoform


def _decode_counts(counts_array, k) -> dict:
    return {decode_base4_string(code, k): int(n) for code, n in enumerate(counts_array) if n}


def test_count_kmers_for_isoform_matches_hand_count_k2():
    # continuous = 'ATGAAA' (6 nt), all interior -> the 'Exon' bucket
    iso = make_synthetic_isoform("MK", lambda aa, i: ['ATG', 'AAA'][i], ['I', 'I'])
    nt_base4, loc_string = encode_isoform_nt_base4(np, iso)

    counts, totals = count_kmers_for_isoform(np, nt_base4, loc_string, 2)

    # num_wins = max(6-2, 0) = 4, one short of the naive 5 -- see
    # count_kmers_for_isoform's own docstring on this convention (matches
    # baseline.compute_kmer_analysis/gpu_change_vector.batch_kmer_term).
    # Windows: AT, TG, GA, AA.
    assert _decode_counts(counts['Exon'], 2) == {'AT': 1, 'TG': 1, 'GA': 1, 'AA': 1}
    assert totals == {'ExonL50': 0, 'Exon': 4, 'ExonR50': 0}
    assert _decode_counts(counts['ExonL50'], 2) == {}
    assert _decode_counts(counts['ExonR50'], 2) == {}


def test_count_kmers_for_isoform_matches_hand_count_k3():
    iso = make_synthetic_isoform("MK", lambda aa, i: ['ATG', 'AAA'][i], ['I', 'I'])
    nt_base4, loc_string = encode_isoform_nt_base4(np, iso)

    counts, totals = count_kmers_for_isoform(np, nt_base4, loc_string, 3)

    # num_wins = max(6-3, 0) = 3: ATG, TGA, GAA (AAA is the dropped 4th window)
    assert _decode_counts(counts['Exon'], 3) == {'ATG': 1, 'TGA': 1, 'GAA': 1}
    assert totals['Exon'] == 3


def test_count_kmers_for_isoform_returns_zero_for_short_isoform():
    iso = make_synthetic_isoform("M", lambda aa, i: 'ATG', ['I'])  # 3 nt, k=5 has no valid window
    nt_base4, loc_string = encode_isoform_nt_base4(np, iso)

    counts, totals = count_kmers_for_isoform(np, nt_base4, loc_string, 5)

    assert totals == {'ExonL50': 0, 'Exon': 0, 'ExonR50': 0}
    assert int(counts['Exon'].sum()) == 0


def test_count_kmers_for_isoform_respects_location_buckets():
    iso = make_synthetic_isoform("MKL", lambda aa, i: ['ATG', 'AAA', 'CCC'][i], ['F', 'I', 'T'])
    nt_base4, loc_string = encode_isoform_nt_base4(np, iso)

    counts, totals = count_kmers_for_isoform(np, nt_base4, loc_string, 2)

    # No exact assertion on *which* boundary windows land where: a majority
    # vote can tie exactly at an even k (e.g. one F nt + one I nt for k=2),
    # and majority_window_bucket's tie-breaking isn't something this test
    # needs to pin down -- only that both non-interior buckets get exercised.
    assert sum(totals.values()) == max(len(loc_string) - 2, 0)
    assert totals['ExonL50'] > 0
    assert totals['ExonR50'] > 0


def test_count_kmers_for_chunk_accumulates_across_genes():
    iso_a = make_synthetic_isoform("MK", lambda aa, i: ['ATG', 'AAA'][i], ['I', 'I'])  # 'ATGAAA' -> AT,TG,GA,AA
    iso_b = make_synthetic_isoform("MK", lambda aa, i: ['ATG', 'CCC'][i], ['I', 'I'])  # 'ATGCCC' -> AT,TG,GC,CC
    genes = [make_synthetic_gene(1, [iso_a]), make_synthetic_gene(2, [iso_b])]

    raw, loc_totals = count_kmers_for_chunk(np, genes, k_values=(2,))

    assert _decode_counts(raw[2]['Exon'], 2) == {'AT': 2, 'TG': 2, 'GA': 1, 'AA': 1, 'GC': 1, 'CC': 1}
    assert loc_totals[2] == {'ExonL50': 0, 'Exon': 8, 'ExonR50': 0}
