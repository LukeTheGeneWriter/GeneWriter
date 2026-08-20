import pytest

from genewriter.baseline import accumulate_codon_usage_percent_by_aa, compute_gc_analysis

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


def test_accumulate_codon_usage_percent_by_aa_weights_by_occurrence_count():
    """A gene with 8 occurrences of an amino acid contributes 8 copies of
    its own percentage to the pool -- weighting the pooled distribution
    toward genes with more (hence more reliable) observations, rather than
    counting every gene's percentage equally regardless of sample size."""
    # Glu (E): codons GAA, GAG. 8 occurrences: 6 GAA, 2 GAG -> 75%/25%.
    codons = [('GAA', 'I')] * 6 + [('GAG', 'I')] * 2
    accum: dict = {}
    accumulate_codon_usage_percent_by_aa(codons, accum)
    assert accum['E']['GAA'] == [0.75] * 8
    assert accum['E']['GAG'] == [0.25] * 8


def test_accumulate_codon_usage_percent_by_aa_includes_zero_for_unused_synonyms():
    """A synonym this gene never chose for an amino acid it DOES use is a
    real 0% data point, not something to omit from the pool."""
    codons = [('GAA', 'I')] * 4  # Glu, only GAA, never GAG
    accum: dict = {}
    accumulate_codon_usage_percent_by_aa(codons, accum)
    assert accum['E']['GAA'] == [1.0] * 4
    assert accum['E']['GAG'] == [0.0] * 4


def test_accumulate_codon_usage_percent_by_aa_skips_amino_acids_absent_from_the_gene():
    codons = [('GAA', 'I')] * 4  # only Glu present
    accum: dict = {}
    accumulate_codon_usage_percent_by_aa(codons, accum)
    assert 'W' not in accum  # Trp never occurs -- no entry, no division by zero


def test_accumulate_codon_usage_percent_by_aa_pools_across_multiple_genes():
    accum: dict = {}
    accumulate_codon_usage_percent_by_aa([('GAA', 'I')] * 3 + [('GAG', 'I')] * 1, accum)  # gene 1: 75/25, weight 4
    accumulate_codon_usage_percent_by_aa([('GAG', 'I')] * 2, accum)  # gene 2: 0/100, weight 2
    assert accum['E']['GAA'] == [0.75] * 4 + [0.0] * 2
    assert accum['E']['GAG'] == [0.25] * 4 + [1.0] * 2
