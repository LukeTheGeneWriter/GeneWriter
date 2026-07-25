import random

import pytest

from genewriter.classes import CodonNgramModel
from genewriter.change_vector import calculate_change_vector
from genewriter.codon_ngram import (
    DEFAULT_NGRAM_CONTEXT_ORDERS,
    _select_context,
    build_codon_ngram_model,
    load_codon_ngram_model,
    sample_codon_ngram_seed,
    save_codon_ngram_model,
)
from genewriter.codon_tables import codon_choices_for_aa

from conftest import make_synthetic_gene, make_synthetic_isoform


def _fixed_codon_choice(aa, i):
    # Deterministic: always the FIRST synonym in AA_CODONS' own ordering,
    # except Leucine, where we force a specific, otherwise-arbitrary choice
    # ('CTG') so tests can assert an exact learned codon rather than "some
    # synonym or other".
    if aa == 'L':
        return 'CTG'
    return codon_choices_for_aa(aa)[0]


def test_build_codon_ngram_model_learns_exact_context_distribution():
    aa_seq = "MKLMKLMKL"  # 3 repeats of "MKL", all interior
    iso = make_synthetic_isoform(aa_seq, _fixed_codon_choice, ['I'] * len(aa_seq))
    genes = [make_synthetic_gene(1, [iso])]

    model = build_codon_ngram_model(genes, context_orders=(3, 2, 1))

    assert model.context_counts['3']['MKL'] == {'CTG': 3}


def test_training_excludes_windows_that_are_not_fully_interior():
    aa_seq = "MKLAAAMKL"
    loc_tags = ['I', 'I', 'I', 'I', 'I', 'I', 'I', 'S', 'I']  # position 7 (K) is splice-spanning
    iso = make_synthetic_isoform(aa_seq, _fixed_codon_choice, loc_tags)
    genes = [make_synthetic_gene(1, [iso])]

    model = build_codon_ngram_model(genes, context_orders=(3, 2, 1))

    # Only the first "MKL" (positions 0-2, all 'I') should count toward k=3;
    # the second (positions 6-8) is broken up by position 7's 'S' tag.
    assert model.context_counts['3']['MKL'] == {'CTG': 1}


def test_build_codon_ngram_model_raises_on_non_descending_context_orders():
    genes = [make_synthetic_gene(1, [make_synthetic_isoform("MKL", _fixed_codon_choice, ['I', 'I', 'I'])])]
    with pytest.raises(ValueError, match="descending"):
        build_codon_ngram_model(genes, context_orders=(1, 2, 3))
    with pytest.raises(ValueError, match="duplicate"):
        build_codon_ngram_model(genes, context_orders=(3, 3, 1))
    with pytest.raises(ValueError, match="empty"):
        build_codon_ngram_model(genes, context_orders=())


def test_build_skips_windows_containing_nonstandard_residue():
    # A stop codon ('*') mid-sequence -- e.g. a readthrough artifact in
    # locate_codons' output -- must not be counted as part of any context,
    # and must not crash. codon_choice_fn below inserts a real stop codon
    # ('TAA') at position 3 regardless of the nominal aa_seq letter there,
    # simulating a non-standard-residue position in otherwise-valid codons.
    aa_seq = "MKLMKLMKL"

    def choice_with_stop(aa, i):
        if i == 3:
            return 'TAA'  # stop codon -- CODON_TO_AA['TAA'] == '*', excluded from STANDARD_AMINO_ACIDS
        return _fixed_codon_choice(aa, i)

    iso = make_synthetic_isoform(aa_seq, choice_with_stop, ['I'] * len(aa_seq))
    genes = [make_synthetic_gene(1, [iso])]

    model = build_codon_ngram_model(genes, context_orders=(3, 2, 1))
    # The second "MKL" (positions 3-5) has its M (position 3) replaced by a
    # stop codon -- that occurrence must not contribute to k=3 "MKL", but
    # the first (0-2) and third (6-8) still should.
    assert model.context_counts['3']['MKL'] == {'CTG': 2}


def test_select_context_prefers_longest_context_meeting_min_observations():
    model = CodonNgramModel(
        organism="test", transcriptome="test", context_orders=(3, 1),
        context_counts={'3': {'MKL': {'CTG': 2}}, '1': {'L': {'CTG': 10, 'TTA': 5}}},
    )
    # k=3 context "MKL" only has 2 observations -- below min_observations=5
    # -- so back off to k=1's well-observed "L" context instead.
    k, counts = _select_context("MKL", 2, model, min_observations=5)
    assert k == 1
    assert counts == {'CTG': 10, 'TTA': 5}


def test_select_context_returns_none_when_nothing_meets_threshold():
    model = CodonNgramModel(
        organism="test", transcriptome="test", context_orders=(3, 1),
        context_counts={'3': {'MKL': {'CTG': 1}}, '1': {'L': {'CTG': 2}}},
    )
    k, counts = _select_context("MKL", 2, model, min_observations=5)
    assert (k, counts) == (None, None)


def test_select_context_none_when_insufficient_preceding_sequence():
    model = CodonNgramModel(
        organism="test", transcriptome="test", context_orders=(3, 1),
        context_counts={'3': {}, '1': {'M': {'ATG': 5}}},
    )
    # Position 0 -- no room for a k=3 context (would need aa_seq[-2:1]) --
    # should skip straight to k=1.
    k, counts = _select_context("MKL", 0, model, min_observations=5)
    assert k == 1
    assert counts == {'ATG': 5}


def test_sample_from_counts_respects_weights():
    model = CodonNgramModel(
        organism="test", transcriptome="test", context_orders=(1,),
        context_counts={'1': {'L': {'CTG': 90, 'TTA': 10}}},
    )
    rng = random.Random(42)
    trials = 2000
    chosen = [sample_codon_ngram_seed("L", model, min_observations=1, rng=rng)[0] for _ in range(trials)]
    ctg_fraction = chosen.count('CTG') / trials
    # Expected ~0.9; loose bound so this isn't flaky.
    assert 0.8 < ctg_fraction < 1.0


def test_sample_codon_ngram_seed_falls_back_to_uniform_random_when_untrained(aa_seq):
    empty_model = CodonNgramModel(
        organism="test", transcriptome="test", context_orders=DEFAULT_NGRAM_CONTEXT_ORDERS,
        context_counts={str(k): {} for k in DEFAULT_NGRAM_CONTEXT_ORDERS},
    )
    rng = random.Random(0)
    seed = sample_codon_ngram_seed(aa_seq, empty_model, rng=rng)
    assert len(seed) == len(aa_seq)
    for codon, aa in zip(seed, aa_seq):
        assert codon in codon_choices_for_aa(aa)


def test_sample_codon_ngram_seed_produces_valid_codons_for_every_residue(aa_seq):
    iso = make_synthetic_isoform(aa_seq, _fixed_codon_choice, ['I'] * len(aa_seq))
    genes = [make_synthetic_gene(1, [iso])]
    model = build_codon_ngram_model(genes)

    rng = random.Random(1)
    seed = sample_codon_ngram_seed(aa_seq, model, rng=rng)
    assert len(seed) == len(aa_seq)
    for codon, aa in zip(seed, aa_seq):
        assert codon in codon_choices_for_aa(aa)


def test_sample_codon_ngram_seed_deterministic_context_always_wins_with_seeded_rng():
    # An overwhelming single-codon count for "MKL" should always win over
    # the fallback/other contexts once min_observations is met.
    model = CodonNgramModel(
        organism="test", transcriptome="test", context_orders=(3, 1),
        context_counts={'3': {'MKL': {'CTG': 1000}}, '1': {}},
    )
    rng = random.Random(7)
    seed = sample_codon_ngram_seed("XMKL".replace('X', 'M'), model, min_observations=5, rng=rng)
    # aa_seq = "MMKL" -> position 3 (L) has context "MKL" (positions 1-3)
    assert seed[3] == 'CTG'


def test_sample_codon_ngram_seed_output_is_valid_input_to_calculate_change_vector(aa_seq, analysis_objects):
    empty_model = CodonNgramModel(
        organism="test", transcriptome="test", context_orders=DEFAULT_NGRAM_CONTEXT_ORDERS,
        context_counts={str(k): {} for k in DEFAULT_NGRAM_CONTEXT_ORDERS},
    )
    seed = sample_codon_ngram_seed(aa_seq, empty_model, rng=random.Random(2))
    result = calculate_change_vector(seed, analysis_objects)
    assert set(result.keys()) == {'RareCodons', 'CodonUsage', 'CodonPairBias', 'GC', 'Kmer'}


def test_save_and_load_codon_ngram_model_round_trips(tmp_path):
    model = CodonNgramModel(
        organism="test", transcriptome="test", context_orders=(3, 2, 1),
        context_counts={'3': {'MKL': {'CTG': 5}}, '2': {'MK': {'AAA': 2}}, '1': {'M': {'ATG': 7}}},
    )
    path = str(tmp_path / "model.json")
    save_codon_ngram_model(model, path)
    loaded = load_codon_ngram_model(path)

    assert loaded.organism == model.organism
    assert loaded.context_orders == model.context_orders
    assert loaded.context_counts == model.context_counts
