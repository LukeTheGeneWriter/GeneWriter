import pytest

from genewriter.weight_calibration import (
    compute_candidate_natural_stats,
    compute_intolerance_weights,
    natural_deviation_zscores,
    term_coefficients_of_variation,
)


def test_term_coefficients_of_variation_covers_every_per_gene_term(analysis_objects):
    cvs = term_coefficients_of_variation(analysis_objects)
    assert set(cvs) == {'RareCodons', 'CodonUsage', 'CodonPairBias', 'GC'}
    assert all(cv > 0 for cv in cvs.values())


def test_tighter_natural_distribution_gets_a_higher_weight(analysis_objects):
    """A term whose per-gene baseline is almost constant (tightly conserved
    in nature -- low CV) should come out with a larger calibrated weight
    than one with a wide natural spread (high CV) -- that's the entire
    point of this calibration."""
    tight = list(analysis_objects.codon_usage.codonUsageScoreByGene)
    mean = sum(tight) / len(tight)
    analysis_objects.codon_usage.codonUsageScoreByGene = [mean + (v - mean) * 0.01 for v in tight]

    loose = list(analysis_objects.gc.gcPerGene)
    gc_mean = sum(loose) / len(loose)
    analysis_objects.gc.gcPerGene = [gc_mean + (v - gc_mean) * 10 for v in loose]

    weights = compute_intolerance_weights(analysis_objects)
    assert weights['CodonUsage'] > weights['GC']


def test_calibrated_weights_average_to_one(analysis_objects):
    weights = compute_intolerance_weights(analysis_objects)
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)


def test_fallback_weights_carried_through_for_uncalibrated_terms(analysis_objects, weights):
    calibrated = compute_intolerance_weights(analysis_objects, fallback_weights=weights)
    assert calibrated['Kmer'] == weights['Kmer']
    assert calibrated['Uracil'] == weights['Uracil']
    # every term compute_intolerance_weights *can* calibrate should have
    # been overwritten, not left at the fallback's flat 1.0/1.0/1.0/1.0
    assert calibrated['GC'] != weights['GC'] or calibrated['CodonUsage'] != weights['CodonUsage']


def test_no_fallback_omits_uncalibrated_terms(analysis_objects):
    calibrated = compute_intolerance_weights(analysis_objects)
    assert 'Kmer' not in calibrated
    assert 'Uracil' not in calibrated


def test_raises_when_no_term_has_a_usable_per_gene_baseline(analysis_objects):
    analysis_objects.rare_codon.usagePerGene = []
    analysis_objects.codon_usage.codonUsageScoreByGene = []
    analysis_objects.codon_pair_bias.cpbPerGene = []
    analysis_objects.gc.gcPerGene = []

    with pytest.raises(ValueError, match="usable per-gene baseline"):
        compute_intolerance_weights(analysis_objects)


def test_compute_candidate_natural_stats_matches_manual_calculation(analysis_objects):
    analysis_objects.codon_usage.codonFreqsLit = {c: 20.0 for c in analysis_objects.codon_usage.codonFreqsLit}
    analysis_objects.codon_pair_bias.cpb_lit = {p: 50000.0 for p in analysis_objects.codon_pair_bias.cpb_lit}

    sol = ['GCG'] * 4  # Ala, rare, all-GC
    stats = compute_candidate_natural_stats(sol, analysis_objects)

    assert stats['RareCodons'] == pytest.approx(1.0)
    assert stats['CodonUsage'] == pytest.approx(20.0)
    assert stats['CodonPairBias'] == pytest.approx(50000.0)
    assert stats['GC'] == pytest.approx(1.0)


def test_compute_candidate_natural_stats_empty_solution_returns_empty_dict(analysis_objects):
    assert compute_candidate_natural_stats([], analysis_objects) == {}


def test_natural_deviation_zscores_zero_at_the_baseline_mean(analysis_objects):
    analysis_objects.codon_usage.codonFreqsLit = {c: 20.0 for c in analysis_objects.codon_usage.codonFreqsLit}
    analysis_objects.codon_usage.codonUsageScoreByGene = [18.0, 22.0] * 100  # mean 20, std 2

    sol = ['CTT'] * 4  # Leu, not rare -- only CodonUsage is being asserted on here
    zscores = natural_deviation_zscores(sol, analysis_objects)

    assert zscores['CodonUsage'] == pytest.approx(0.0)


def test_natural_deviation_zscores_large_for_a_genuine_outlier(analysis_objects):
    analysis_objects.rare_codon.usagePerGene = [0.0, 0.2] * 100  # mean 0.1, std 0.1

    sol = ['GCG'] * 4  # Ala, every codon rare -> fraction 1.0
    zscores = natural_deviation_zscores(sol, analysis_objects)

    assert zscores['RareCodons'] == pytest.approx(9.0)  # |(1.0 - 0.1) / 0.1|


def test_natural_deviation_zscores_skips_categories_with_insufficient_baseline(analysis_objects):
    analysis_objects.gc.gcPerGene = []  # e.g. a real Standards/GCAnalysis.json predating this field

    sol = ['GCG'] * 4
    zscores = natural_deviation_zscores(sol, analysis_objects)

    assert 'GC' not in zscores
    assert 'RareCodons' in zscores
