import pytest

from genewriter.special_translation import (
    AA_CODONS_SELENOPROTEIN,
    FRAMESHIFT_GENE_IDS,
    NON_ATG_START_GENE_IDS,
    SELENOPROTEIN_GENE_IDS,
    apply_frameshift,
    apply_frameshift_to_exons,
    check_cds_against_protein_at_position,
    check_cds_against_protein_frameshift,
    check_cds_against_protein_special,
    is_frameshift_gene,
    is_selenoprotein_gene,
)

_A_SELENO_GENE_ID = 2876  # GPX1
_NOT_A_SELENO_GENE_ID = 7157  # TP53


def test_is_selenoprotein_gene():
    assert is_selenoprotein_gene(_A_SELENO_GENE_ID)
    assert is_selenoprotein_gene(str(_A_SELENO_GENE_ID))  # accepts either int or numeric string
    assert not is_selenoprotein_gene(_NOT_A_SELENO_GENE_ID)


def test_selenoprotein_gene_ids_are_all_distinct_and_well_formed():
    assert len(SELENOPROTEIN_GENE_IDS) == 25
    for gene_id, symbol in SELENOPROTEIN_GENE_IDS.items():
        assert isinstance(gene_id, int)
        assert isinstance(symbol, str) and symbol == symbol.upper()


def test_non_atg_start_gene_ids_well_formed_and_disjoint_from_selenoprotein_list():
    assert len(NON_ATG_START_GENE_IDS) >= 1
    for gene_id, symbol in NON_ATG_START_GENE_IDS.items():
        assert isinstance(gene_id, int)
        assert isinstance(symbol, str) and symbol == symbol.upper()
    # The two lists model independent phenomena -- no current member happens
    # to need both, though check_cds_against_protein_at_position supports
    # composing them (see the compose test below) if one ever does.
    assert set(NON_ATG_START_GENE_IDS) & set(SELENOPROTEIN_GENE_IDS) == set()


def test_selenoprotein_codon_table_moves_tga_from_stop_to_sec():
    # TGA must be in exactly one of '*'/'U', never both -- see the module's
    # own comment on why "both" silently favors '*' via dict iteration
    # order and defeats the whole point.
    assert 'TGA' not in AA_CODONS_SELENOPROTEIN['*']
    assert AA_CODONS_SELENOPROTEIN['*'] == ['TAA', 'TAG']
    assert AA_CODONS_SELENOPROTEIN['U'] == ['TGA']


def test_selenoprotein_codon_table_otherwise_matches_standard_table():
    from genewriter.codon_tables import AA_CODONS
    for aa in AA_CODONS:
        if aa == '*':
            continue
        assert AA_CODONS_SELENOPROTEIN[aa] == AA_CODONS[aa]


def test_check_cds_against_protein_special_finds_selenocysteine_match():
    # M(ATG) K(AAA) U(TGA) -- a minimal synthetic selenoprotein-shaped CDS.
    cds = 'ATGAAATGA'
    aa_seq = 'MKU'
    atg_start, codvec = check_cds_against_protein_special(cds, [0], aa_seq, gene_id=_A_SELENO_GENE_ID)
    assert atg_start == 0
    assert codvec == ['ATG', 'AAA', 'TGA']


def test_check_cds_against_protein_special_rejects_without_gene_id():
    # Same CDS/protein, but no gene_id (or a non-selenoprotein gene_id) --
    # TGA must be read as a premature stop, not silently matched.
    cds = 'ATGAAATGA'
    aa_seq = 'MKU'
    assert check_cds_against_protein_special(cds, [0], aa_seq, gene_id=None) == (None, None)
    assert check_cds_against_protein_special(cds, [0], aa_seq, gene_id=_NOT_A_SELENO_GENE_ID) == (None, None)


def test_check_cds_against_protein_special_still_requires_full_match():
    # A selenoprotein gene_id doesn't make the check permissive elsewhere --
    # a genuine mismatch downstream of the Sec codon still fails.
    cds = 'ATGAAATGACCC'  # ...continues with CCC (Pro) after the Sec codon
    aa_seq = 'MKUD'  # but the real protein claims Asp (D), not Pro, there
    assert check_cds_against_protein_special(cds, [0], aa_seq, gene_id=_A_SELENO_GENE_ID) == (None, None)


def test_check_cds_against_protein_special_matches_normal_genes_unaffected():
    # A completely ordinary gene (no selenocysteine involved at all) still
    # matches normally through this function.
    cds = 'ATGAAACTG'
    aa_seq = 'MKL'
    atg_start, codvec = check_cds_against_protein_special(cds, [0], aa_seq, gene_id=_NOT_A_SELENO_GENE_ID)
    assert atg_start == 0
    assert codvec == ['ATG', 'AAA', 'CTG']


def test_check_cds_against_protein_at_position_accepts_non_atg_start():
    # CTG (ordinarily Leu) at position 0, but the real protein says Met --
    # the classic MYC/TXNRD3 pattern found live this session.
    cds = 'CTGAAATTA'  # CTG(would-be L) AAA(K) TTA(L)
    aa_seq = 'MKL'
    assert check_cds_against_protein_at_position(cds, 0, aa_seq) is True


def test_check_cds_against_protein_at_position_still_checks_downstream_residues():
    cds = 'CTGAAATTA'
    aa_seq = 'MKD'  # real protein claims Asp, but the actual codon (TTA) is Leu
    assert check_cds_against_protein_at_position(cds, 0, aa_seq) is False


def test_check_cds_against_protein_at_position_does_not_blindly_skip_position_zero():
    # If the real protein's first residue genuinely isn't 'M', position 0
    # must still be checked normally, not silently waved through.
    cds = 'CTGAAATTA'
    aa_seq = 'LKL'  # first residue really is Leu here, and CTG really is Leu -- should still match
    assert check_cds_against_protein_at_position(cds, 0, aa_seq) is True

    cds_wrong = 'GTGAAATTA'  # GTG is Val, not Leu
    assert check_cds_against_protein_at_position(cds_wrong, 0, aa_seq) is False


def test_check_cds_against_protein_at_position_selenoprotein_and_non_atg_start_compose():
    # Both special cases at once: a non-AUG start AND a downstream Sec
    # codon, in a gene on the selenoprotein list. Not verified against any
    # real gene this session (no known human gene combines both), but the
    # two mechanisms are independent (start-codon identity vs. an internal
    # UGA) and this locks in that composing them doesn't break either one.
    cds = 'CTGAAATGATTA'  # CTG(->M) AAA(K) TGA(->U, seleno) TTA(L)
    aa_seq = 'MKUL'
    assert check_cds_against_protein_at_position(cds, 0, aa_seq, gene_id=_A_SELENO_GENE_ID) is True
    assert check_cds_against_protein_at_position(cds, 0, aa_seq, gene_id=None) is False


def test_is_frameshift_gene():
    assert is_frameshift_gene(4946)  # OAZ1
    assert is_frameshift_gene('23089')  # PEG10, accepts numeric string too
    assert not is_frameshift_gene(_NOT_A_SELENO_GENE_ID)


def test_frameshift_gene_ids_well_formed():
    assert len(FRAMESHIFT_GENE_IDS) >= 2
    for gene_id, spec in FRAMESHIFT_GENE_IDS.items():
        assert isinstance(gene_id, int)
        assert spec['shift_offset'] != 0
        assert spec['shift_residue'] > 0


def test_apply_frameshift_plus_one_skips_a_nucleotide():
    # OAZ1's pattern: naive translation is correct through 2 codons, then an
    # in-frame UGA would normally read as a stop -- skipping 1 nt (the shift)
    # before resuming reveals the real continuation instead.
    cds = 'ATGAAATGACCT'  # ATG AAA | TGA CCT
    shifted = apply_frameshift(cds, shift_residue=2, shift_offset=1)
    assert shifted == cds[:6] + cds[7:]  # nt 6 (the 'T' of TGA) is skipped, never decoded
    assert shifted == 'ATGAAAGACCT'


def test_apply_frameshift_minus_one_overlaps_a_nucleotide():
    # PEG10's pattern: the ribosome re-pairs 1 nt upstream, so that
    # nucleotide is decoded twice -- once in each frame -- not skipped.
    cds = 'ATGAAACCTGGG'  # ATG AAA | CCT GGG
    shifted = apply_frameshift(cds, shift_residue=2, shift_offset=-1)
    assert shifted == cds[:6] + cds[5:]  # nt 5 (the last 'A' of AAA) is decoded twice
    assert shifted == 'ATGAAAACCTGGG'


def test_check_cds_against_protein_frameshift_finds_plus_one_match(monkeypatch):
    monkeypatch.setitem(FRAMESHIFT_GENE_IDS, 999999, {'symbol': 'TEST+1', 'shift_residue': 2, 'shift_offset': 1})
    cds = 'ATGAAATGACCT'
    aa_seq = 'MKD'  # M, K, then Asp (GAC) from the shifted frame -- not a stop
    atg, codvec = check_cds_against_protein_frameshift(cds, [0], aa_seq, gene_id=999999)
    assert atg == 0
    assert codvec == ['ATG', 'AAA', 'GAC']


def test_check_cds_against_protein_frameshift_finds_minus_one_match(monkeypatch):
    monkeypatch.setitem(FRAMESHIFT_GENE_IDS, 999998, {'symbol': 'TEST-1', 'shift_residue': 2, 'shift_offset': -1})
    cds = 'ATGAAACCTGGG'
    aa_seq = 'MKTW'
    atg, codvec = check_cds_against_protein_frameshift(cds, [0], aa_seq, gene_id=999998)
    assert atg == 0
    assert codvec == ['ATG', 'AAA', 'ACC', 'TGG']


def test_check_cds_against_protein_frameshift_rejects_unregistered_gene():
    cds = 'ATGAAATGACCT'
    aa_seq = 'MKD'
    assert check_cds_against_protein_frameshift(cds, [0], aa_seq, gene_id=_NOT_A_SELENO_GENE_ID) == (None, None)
    assert check_cds_against_protein_frameshift(cds, [0], aa_seq, gene_id=None) == (None, None)


def test_check_cds_against_protein_frameshift_still_requires_full_match(monkeypatch):
    monkeypatch.setitem(FRAMESHIFT_GENE_IDS, 999997, {'symbol': 'TEST', 'shift_residue': 2, 'shift_offset': 1})
    cds = 'ATGAAATGACCT'
    aa_seq = 'MKW'  # real protein claims Trp, but the shifted codon (GAC) is Asp
    assert check_cds_against_protein_frameshift(cds, [0], aa_seq, gene_id=999997) == (None, None)


def test_apply_frameshift_to_exons_mid_fragment_plus_one():
    # Same shift as test_apply_frameshift_plus_one_skips_a_nucleotide, but
    # split across two exon fragments instead of one flat string -- the
    # flattened result must be identical either way.
    exsupp = [{'seq': 'ATGAAA'}, {'seq': 'TGACCT'}]
    flat = ''.join(e['seq'] for e in exsupp)
    result = apply_frameshift_to_exons(exsupp, boundary_pos=7, shift_offset=1)
    assert ''.join(e['seq'] for e in result) == flat[:7] + flat[8:]
    assert result[0] == exsupp[0]  # the fragment not straddling the boundary is untouched


def test_apply_frameshift_to_exons_mid_fragment_minus_one():
    exsupp = [{'seq': 'ATGAAA'}, {'seq': 'CCTGGG'}]
    flat = ''.join(e['seq'] for e in exsupp)
    result = apply_frameshift_to_exons(exsupp, boundary_pos=7, shift_offset=-1)
    assert ''.join(e['seq'] for e in result) == flat[:7] + flat[6:7] + flat[7:]
    assert result[0] == exsupp[0]


def test_apply_frameshift_to_exons_matches_flat_splice_at_non_codon_aligned_boundary():
    # Real exon boundaries rarely line up with codon boundaries -- confirm
    # the fragmented splice agrees with the equivalent flat-string splice
    # even when the exon split itself falls mid-codon.
    cds = 'ATGAAACCTGGGTTTAAACCCGGG'
    exsupp = [{'seq': cds[:10]}, {'seq': cds[10:]}]  # exon split mid-codon, deliberately
    boundary_pos = 14
    exon_result = apply_frameshift_to_exons(exsupp, boundary_pos, shift_offset=-1)
    expected = cds[:boundary_pos] + cds[boundary_pos - 1:boundary_pos] + cds[boundary_pos:]
    assert ''.join(e['seq'] for e in exon_result) == expected


def test_apply_frameshift_to_exons_rejects_boundary_at_fragment_start_for_minus_one():
    # boundary_pos sits exactly on a fragment junction -- a -1 overlap would
    # need to reach into the previous fragment, which isn't implemented, so
    # this must raise rather than silently wrap/truncate within one fragment.
    exsupp = [{'seq': 'ATGAAA'}, {'seq': 'TGACCT'}]
    with pytest.raises(ValueError):
        apply_frameshift_to_exons(exsupp, boundary_pos=6, shift_offset=-1)


def test_apply_frameshift_to_exons_rejects_skip_that_overruns_fragment_end():
    # Mirror case: a skip landing near the very end of a fragment, wide
    # enough that it would need to reach into the next fragment (a plain
    # +1 never can -- local's max value is elen-1, so local+1 <= elen
    # always -- this needs a wider hypothetical skip to actually trigger).
    exsupp = [{'seq': 'ATGAAA'}]  # elen=6, boundary at the last nt (local=5)
    with pytest.raises(ValueError):
        apply_frameshift_to_exons(exsupp, boundary_pos=5, shift_offset=2)
