import pytest

from genewriter.codon_tables import (
    AA_CODONS,
    CODON_FREQ_BY_INDEX,
    CODON_FREQS_LIT,
    CODON_LIST,
    CODON_TO_AA,
    CODON_TO_INDEX,
    CODON_TO_VARIABLE_FLAGS,
    GC_FLAGS_BY_INDEX,
    IS_RARE_BY_INDEX,
    RARE_CODON_INDICES,
    RARE_CODONS_LIT,
    STANDARD_AMINO_ACIDS,
    VARIABLE_FLAGS_BY_INDEX,
    decode_codons,
    encode_codons,
    validate_aa_sequence,
)


def test_variable_flags_covers_every_codon():
    assert set(CODON_TO_VARIABLE_FLAGS) == set(CODON_TO_AA)


def test_single_codon_amino_acids_are_never_variable():
    # Met (ATG) and Trp (TGG) each have exactly one codon -- no synonym
    # exists to differ at any base, so every position is fixed.
    assert CODON_TO_VARIABLE_FLAGS['ATG'] == (0, 0, 0)
    assert CODON_TO_VARIABLE_FLAGS['TGG'] == (0, 0, 0)


def test_proline_only_varies_at_third_position():
    # CCT/CCC/CCA/CCG all agree on 'CC' -- only the third base differs.
    for codon in AA_CODONS['P']:
        assert CODON_TO_VARIABLE_FLAGS[codon] == (0, 0, 1)


def test_leucine_varies_at_first_and_third_but_not_second():
    # TTA/TTG/CTT/CTC/CTA/CTG: every codon's second base is T.
    for codon in AA_CODONS['L']:
        assert CODON_TO_VARIABLE_FLAGS[codon] == (1, 0, 1)


def test_serine_varies_at_every_position():
    # TCx and AGy variants together mean all three bases vary.
    for codon in AA_CODONS['S']:
        assert CODON_TO_VARIABLE_FLAGS[codon] == (1, 1, 1)


def test_flags_are_identical_for_every_synonym_of_the_same_amino_acid():
    for aa, codons in AA_CODONS.items():
        flags = {CODON_TO_VARIABLE_FLAGS[c] for c in codons}
        assert len(flags) == 1, f"{aa}'s synonyms disagree on their own variable_flags: {flags}"


def test_codon_list_covers_all_64_codons_with_no_duplicates():
    assert len(CODON_LIST) == 64
    assert len(set(CODON_LIST)) == 64


def test_encode_decode_round_trips():
    sol = ['ATG', 'GCT', 'TGG', 'CCG', 'AAA']
    assert decode_codons(encode_codons(sol)) == sol


def test_encode_gives_indices_matching_codon_to_index():
    sol = ['ATG', 'GCT', 'TGG']
    assert encode_codons(sol) == [CODON_TO_INDEX[c] for c in sol]


def test_index_tables_agree_with_their_dict_counterparts_for_every_codon():
    for codon in CODON_LIST:
        i = CODON_TO_INDEX[codon]
        assert IS_RARE_BY_INDEX[i] == (1 if codon in RARE_CODONS_LIT else 0)
        assert CODON_FREQ_BY_INDEX[i] == CODON_FREQS_LIT[codon]
        assert VARIABLE_FLAGS_BY_INDEX[i] == CODON_TO_VARIABLE_FLAGS[codon]
        assert GC_FLAGS_BY_INDEX[i] == tuple(1 if base in 'GC' else 0 for base in codon)


def test_rare_codon_indices_matches_is_rare_by_index():
    from_indices = {i for i in range(64) if IS_RARE_BY_INDEX[i]}
    assert from_indices == set(RARE_CODON_INDICES)


def test_standard_amino_acids_is_aa_codons_minus_stop():
    assert STANDARD_AMINO_ACIDS == set(AA_CODONS) - {'*'}
    assert len(STANDARD_AMINO_ACIDS) == 20
    assert '*' not in STANDARD_AMINO_ACIDS


def test_validate_aa_sequence_accepts_and_uppercases_valid_sequence():
    assert validate_aa_sequence("mklker") == "MKLKER"
    assert validate_aa_sequence("MKLKER") == "MKLKER"


def test_validate_aa_sequence_rejects_invalid_character_with_position():
    with pytest.raises(ValueError, match=r"'X'.*position 3"):
        validate_aa_sequence("MKLXER")


def test_validate_aa_sequence_rejects_star_anywhere():
    for seq in ("*MKLKER", "MKL*KER", "MKLKER*"):
        with pytest.raises(ValueError, match=r"\*"):
            validate_aa_sequence(seq)


def test_validate_aa_sequence_rejects_empty_string():
    with pytest.raises(ValueError, match="empty"):
        validate_aa_sequence("")
