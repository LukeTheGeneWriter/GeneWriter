"""Derive change-vector term weights from how tightly natural genes conserve
each term's own per-gene score, instead of hand-picking every weight as 1.0.

Every change-vector term (change_vector.py) already z-scores its raw metric
against a natural-gene baseline internally, so within one term a bigger
deviation from the natural mean already scores higher. What that internal
z-scoring does NOT do is tell you how much any one *term* should matter
relative to the others in WEIGHTS -- that's a separate question of how
tightly nature actually conserves this axis at all, gene to gene.

The per-gene arrays each *Analysis baseline already stores (classes.py --
RareCodonAnalysis.usagePerGene, CodonAnalysis.codonUsageScoreByGene,
CodonPairBiasAnalysis.cpbPerGene, GCAnalysis.gcPerGene) answer that directly:
their coefficient of variation (std / |mean|) across the real-gene corpus is
a measure of how much real genes vary in that metric gene-to-gene. A term
real genes barely vary in (low CV) is one nature is intolerant of
perturbing -- deviating from it is rare in nature, so a candidate gene
drifting off it should cost more. A term real genes vary a lot in (high CV)
is one nature tolerates a wide range of -- it should cost less.

This measures a coarser statistic (variation of the term's own per-gene
aggregate) than the position/window-level std each term already divides by
internally (_zscore() in change_vector.py) -- deliberately, so this doesn't
just double-penalize whatever's already reflected in each term's internal
z-scoring.
"""

import numpy as np

from .change_vector import AnalysisObjects, cached_normal_transform
from .codon_tables import RARE_CODONS_LIT

# Per-gene baseline array backing each term's coefficient of variation --
# (AnalysisObjects attribute name, field name on that sub-analysis). Kmer
# has no per-gene array (KmerAnalysis stores corpus-wide k-mer counts, not
# one score per gene) and Uracil isn't a natural-baseline term at all (see
# change_vector._uracil_term's docstring, and its deliberate sign-flip
# design) -- both are left out here and keep whatever fallback weight the
# caller supplies to compute_intolerance_weights().
_PER_GENE_FIELDS = {
    'RareCodons': ('rare_codon', 'usagePerGene'),
    'CodonUsage': ('codon_usage', 'codonUsageScoreByGene'),
    'CodonPairBias': ('codon_pair_bias', 'cpbPerGene'),
    'GC': ('gc', 'gcPerGene'),
}

# Floors a term's coefficient of variation so a near-zero natural spread
# (unlikely on a real ~18,864-gene corpus, but easy to hit on a tiny local
# sample or synthetic test fixture) doesn't blow that term's weight up to
# an arbitrarily large value.
_MIN_CV = 1e-6


def term_coefficients_of_variation(analysis_objects: AnalysisObjects) -> dict:
    """Coefficient of variation (std / |mean|) of each term's per-gene score
    across the natural-gene baseline in analysis_objects -- only for terms
    with a per-gene array available (see _PER_GENE_FIELDS).

    Low value: real genes barely vary in this metric (tightly conserved --
    intolerant of perturbation). High value: real genes vary a lot (loosely
    conserved -- tolerant).

    A term is omitted if its baseline has fewer than 2 genes' worth of data
    or a mean of exactly 0 -- nothing meaningful to divide by either way.
    """
    result = {}
    for term_name, (attr, field_name) in _PER_GENE_FIELDS.items():
        values = getattr(getattr(analysis_objects, attr), field_name)
        arr = np.asarray(values, dtype=float)
        if arr.size < 2:
            continue
        mean = arr.mean()
        if mean == 0:
            continue
        result[term_name] = max(abs(arr.std() / mean), _MIN_CV)
    return result


def compute_intolerance_weights(analysis_objects: AnalysisObjects, fallback_weights: dict = None) -> dict:
    """Change-vector term weights scaled inversely to
    term_coefficients_of_variation() -- a term real genes barely vary in
    gets a proportionally larger weight, one they vary a lot in gets a
    proportionally smaller one.

    Normalized so the calibrated terms' weights average to 1.0 -- matching
    the scale of an all-1.0 WEIGHTS dict, so this is a drop-in replacement
    rather than requiring every other weight to be retuned to a new scale.

    fallback_weights: weight to use for any registered term this can't
    compute a coefficient of variation for (Kmer -- no per-gene baseline;
    Uracil -- not a natural-baseline term at all, see
    change_vector._uracil_term's docstring for why, and pick its sign
    deliberately rather than assuming this function's framing implies one).
    Pass your current WEIGHTS dict here to carry those terms' existing
    values through unchanged; terms it can't compute are simply absent from
    the result if this is omitted.

    Raises ValueError if analysis_objects has no usable per-gene baseline
    for any term at all (e.g. every array is empty or single-valued).
    """
    cvs = term_coefficients_of_variation(analysis_objects)
    if not cvs:
        raise ValueError(
            "No term in analysis_objects has a usable per-gene baseline array "
            "(need at least 2 genes' worth of data and a nonzero mean) -- "
            "can't compute intolerance weights."
        )

    raw = {name: 1.0 / cv for name, cv in cvs.items()}
    scale = len(raw) / sum(raw.values())  # so the calibrated terms average to 1.0

    weights = dict(fallback_weights) if fallback_weights else {}
    for name, value in raw.items():
        weights[name] = value * scale
    return weights


def compute_candidate_natural_stats(sol: list, analysis_objects: AnalysisObjects) -> dict:
    """The same per-gene aggregate metric baseline.py computes for each real
    gene (RareCodons: fraction of codons that are rare; CodonUsage: mean
    codon-frequency score; CodonPairBias: mean adjacent-pair score; GC:
    fraction of G/C bases across all 3 codon positions), computed instead
    for one candidate codon solution -- so it lands on the same footing as
    the corresponding _PER_GENE_FIELDS baseline array and can be compared
    against it directly (see natural_deviation_zscores()).

    sol: list of codon strings (Proposed_Solution.codons), not the
    (codon, location) tuples baseline.py's compute_*_analysis() functions
    read off IsoformGeneBody.codons -- this is scoring a candidate
    genotype, not a loaded real gene.
    """
    if not sol:
        return {}

    stats = {'RareCodons': sum(1 for c in sol if c in RARE_CODONS_LIT) / len(sol)}

    codon_freqs = analysis_objects.codon_usage.codonFreqsLit
    stats['CodonUsage'] = sum(codon_freqs[c] for c in sol) / len(sol)

    if len(sol) >= 2:
        cpb_lit = analysis_objects.codon_pair_bias.cpb_lit
        pairs = [sol[i] + sol[i + 1] for i in range(len(sol) - 1)]
        stats['CodonPairBias'] = sum(cpb_lit[p] for p in pairs) / len(pairs)

    stats['GC'] = sum(1 for c in sol for base in c if base in 'GC') / (3 * len(sol))
    return stats


def natural_deviation_zscores(sol: list, analysis_objects: AnalysisObjects) -> dict:
    """Per-category |z-score| of one candidate solution's own aggregate
    metric (compute_candidate_natural_stats()) against the natural per-gene
    baseline distribution in analysis_objects (same arrays
    term_coefficients_of_variation() reads -- see _PER_GENE_FIELDS).

    Unlike the internal z-scoring every change-vector term already does
    (against position/window-level baselines), this z-scores a single
    whole-candidate aggregate against the distribution of that same
    aggregate across real genes -- "is this candidate's overall rare-codon
    rate/GC content/etc. within the range real genes actually occupy," not
    "is this specific window unusual." Uses the same auto-detected-
    distribution normal-scores transform as the change-vector terms
    (change_vector.cached_normal_transform(), see distribution_fit.py) --
    a per-gene aggregate array (cpbPerGene et al.) is exactly the same kind
    of "might not actually be Gaussian" baseline the terms' own arrays are,
    so it gets the same correction.

    A category is omitted if its baseline array has fewer than 2 genes'
    worth of data or a std of 0 (real Standards/GCAnalysis.json files
    predating GCAnalysis.gcPerGene default to [], so GC is silently
    skipped for those rather than raising -- see classes.py).
    """
    stats = compute_candidate_natural_stats(sol, analysis_objects)
    zscores = {}
    for term_name, (attr, field_name) in _PER_GENE_FIELDS.items():
        if term_name not in stats:
            continue
        baseline_obj = getattr(analysis_objects, attr)
        values = getattr(baseline_obj, field_name)
        if len(values) < 2:
            continue
        transform = cached_normal_transform(baseline_obj, field_name, values)
        if transform.family == 'degenerate':
            continue
        zscores[term_name] = abs(transform.transform(stats[term_name]))
    return zscores
