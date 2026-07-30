import pytest

from genewriter.baseline import compute_gc_analysis

from conftest import make_synthetic_gene, make_synthetic_isoform


def test_compute_gc_analysis_gc_per_gene_matches_manual_count():
    aa_seq = "MKL"
    fixed_codons = ['ATG', 'AAA', 'CTG']  # GC bases: 1 (ATG) + 0 (AAA) + 2 (CTG) = 3 of 9
    iso = make_synthetic_isoform(aa_seq, lambda aa, i: fixed_codons[i], ['I'] * len(aa_seq))
    genes = [make_synthetic_gene(1, [iso])]

    gc = compute_gc_analysis(genes)

    assert gc.gcPerGene == pytest.approx([3 / 9])


def test_compute_gc_analysis_gc_per_gene_is_one_value_per_gene():
    aa_seq = "MKL"
    iso_a = make_synthetic_isoform(aa_seq, lambda aa, i: 'GGG', ['I'] * len(aa_seq))
    iso_b = make_synthetic_isoform(aa_seq, lambda aa, i: 'ATA', ['I'] * len(aa_seq))
    genes = [make_synthetic_gene(1, [iso_a]), make_synthetic_gene(2, [iso_b])]

    gc = compute_gc_analysis(genes)

    assert gc.gcPerGene == pytest.approx([1.0, 0.0])


def test_compute_gc_analysis_skips_isoforms_with_no_codons():
    empty_iso = make_synthetic_isoform("", lambda aa, i: '', [])
    genes = [make_synthetic_gene(1, [empty_iso])]

    gc = compute_gc_analysis(genes)

    assert gc.gcPerGene == []
