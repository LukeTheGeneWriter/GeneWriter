import pytest

from genewriter.aa_motif_index import (
    DEFAULT_MOTIF_K_VALUES,
    build_aa_motif_index,
    query_motif_count,
    report_motif_occurrences,
)

from conftest import make_synthetic_gene, make_synthetic_isoform


def _first_codon(aa, i):
    from genewriter.codon_tables import codon_choices_for_aa
    if aa not in "ACDEFGHIKLMNPQRSTVWY":
        return "NNN"  # placeholder for a non-standard residue -- aa_motif_index never reads codons
    return codon_choices_for_aa(aa)[0]


def _gene(gene_id, aa_seq, loc_tag='I'):
    iso = make_synthetic_isoform(aa_seq, _first_codon, [loc_tag] * len(aa_seq))
    return make_synthetic_gene(gene_id, [iso])


def test_build_aa_motif_index_counts_exact_occurrences_in_synthetic_proteome():
    # "KLKE" (k=4) occurs once in the first sequence (position 3) and twice
    # (overlapping-free, positions 0 and 4) in the second -- 3 total.
    genes = [
        _gene(1, "AAAKLKEAAA"),
        _gene(2, "KLKEKLKE"),
    ]
    index = build_aa_motif_index(genes, k_values=(3, 4, 5))
    assert query_motif_count(index, "KLKE") == 3


def test_query_motif_count_zero_for_unseen_substring():
    genes = [_gene(1, "AAAKLKEAAA")]
    index = build_aa_motif_index(genes, k_values=(3, 4, 5))
    assert query_motif_count(index, "WWWW") == 0


def test_report_motif_occurrences_output_format_matches_exact_example(capsys):
    genes = [_gene(1, "RTYILEREEAAA")]
    index = build_aa_motif_index(genes, organism="human", k_values=(9,))
    message = report_motif_occurrences(index, "RTYILEREE")
    assert message == 'Found 1 occurrences of substring "RTYILEREE" in the human proteome'
    captured = capsys.readouterr()
    assert message in captured.out


def test_build_rejects_k_values_containing_2():
    with pytest.raises(ValueError, match="k=2"):
        build_aa_motif_index([_gene(1, "AAAKLKEAAA")], k_values=(2, 3))


def test_query_rejects_length_outside_configured_k_values():
    genes = [_gene(1, "AAAKLKEAAA")]
    index = build_aa_motif_index(genes, k_values=(3, 4, 5))
    with pytest.raises(ValueError, match="not in this index's k_values"):
        query_motif_count(index, "AC")  # length 2, both valid amino acids
    with pytest.raises(ValueError, match="not in this index's k_values"):
        query_motif_count(index, "ACDEFGHIKLM")  # length 11, all valid amino acids


def test_motif_index_is_not_restricted_by_codon_location_tag():
    # Same motif, one gene tagged entirely 'F' (exon 5' edge), the other 'S'
    # (splice-spanning) -- both should count fully, since this index never
    # reads location tags at all.
    genes = [
        _gene(1, "AAAKLKEAAA", loc_tag='F'),
        _gene(2, "KLKEKLKE", loc_tag='S'),
    ]
    index = build_aa_motif_index(genes, k_values=(3, 4, 5))
    assert query_motif_count(index, "KLKE") == 3


def test_build_skips_windows_containing_nonstandard_residue():
    # 'X' (unresolved/ambiguous residue in real NCBI data) shouldn't crash
    # and shouldn't be countable as part of any motif -- but the real "KLKE"
    # later in the same sequence should still be counted normally.
    genes = [_gene(1, "AAXAAKLKEAAA")]
    index = build_aa_motif_index(genes, k_values=(3, 4, 5))
    assert query_motif_count(index, "KLKE") == 1
    for motif in index.motif_counts[str(4)]:
        assert 'X' not in motif


def test_default_motif_k_values_excludes_2_and_covers_3_to_10():
    assert 2 not in DEFAULT_MOTIF_K_VALUES
    assert DEFAULT_MOTIF_K_VALUES == tuple(range(3, 11))
