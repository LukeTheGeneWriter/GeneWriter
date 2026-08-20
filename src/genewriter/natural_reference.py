"""How does a candidate's distance_from_optimal() compare to what nature
itself actually achieves?

Every change-vector term is a "how much does this position deviate from
natural genes" signal, but no real CDS reaches distance_from_optimal() == 0
-- the terms provably conflict (see HANDOFF_2026-08-20.md sec 4). So "is
this candidate good enough" can't mean "close to zero"; it has to mean
"about as good as -- or not unreasonably worse than -- an actual natural
CDS scored the exact same way." This module builds that reference
distribution and the z-score against it.

Luke's framing (2026-08-20 session, after HANDOFF_2026-08-20.md sec 4 was
written up): "if a sequence is z=0 (average) relative to the change
vectors seen in real CDS, that's good enough... They could also go for
z=1 to say... natural CDS have a lot of other pressures on them that
we're not measuring, so even if we're a standard deviation above a
natural change vector's sum, that's okay." This module makes both of
those literal, computable numbers.
"""

from dataclasses import dataclass, field

from .change_vector import AnalysisObjects, calculate_change_vector, distance_from_optimal
from .distribution_fit import FittedDistribution, fit_normal_transform
from .gene_io import protein_coding_isoforms


@dataclass
class NaturalGeneDistance:
    """One natural gene's own distance_from_optimal(), scored on the exact
    same footing (same analysis_objects baseline, same weights) as any GA
    candidate -- so it's directly comparable, not a different metric that
    happens to share a name.

    per_term: {term_name: sum(v_t)} -- the UNWEIGHTED per-position sum for
    each registered term, before `weights` is applied. Kept alongside the
    weighted `distance` so a caller can see which axis actually drove a
    high-distance natural gene, rather than only the blended aggregate --
    see HANDOFF_2026-08-20.md sec 4's warning that the aggregate can hide
    one bad axis behind five good ones.
    """
    gene_id: int
    isoform_number: str
    length: int
    distance: float
    distance_per_codon: float
    per_term: dict = field(default_factory=dict)


def score_natural_genes(genes: list, analysis_objects: AnalysisObjects, weights: dict, locvec: list = None) -> list:
    """distance_from_optimal(), computed exactly (calculate_change_vector,
    not diffed) for every protein-coding isoform in `genes`
    (gene_io.protein_coding_isoforms() -- same corpus-iteration convention
    baseline.py already uses, including its 'X'-isoform and empty-codons
    skip).

    locvec is normally read per-isoform from its own iso.codons location
    tags (real per-gene exon/intron structure) -- the `locvec` parameter
    is an override for the rare case every isoform should be scored
    against one shared tag scheme instead (e.g. comparing against a
    candidate that itself used locvec=None); None (default) uses each
    isoform's own tags, which is what you want for an honest natural-CDS
    reference distribution.

    Returns a NaturalGeneDistance per isoform, in whatever order
    protein_coding_isoforms() yields them. Callers wanting just the
    distance_per_codon array (e.g. to fit a reference distribution) should
    pull it via `[g.distance_per_codon for g in score_natural_genes(...)]`
    rather than this function growing a second "just give me the array"
    variant.
    """
    results = []
    for gene, iso in protein_coding_isoforms(genes):
        sol = [cod for cod, _loc in iso.codons]
        gene_locvec = locvec if locvec is not None else [loc for _cod, loc in iso.codons]
        vecs = calculate_change_vector(sol, analysis_objects, gene_locvec)
        d = distance_from_optimal(vecs, weights)
        results.append(NaturalGeneDistance(
            gene_id=gene.geneID,
            isoform_number=iso.isoformNumber,
            length=len(sol),
            distance=d,
            distance_per_codon=d / len(sol),
            per_term={name: sum(v) for name, v in vecs.items()},
        ))
    return results


@dataclass
class NaturalDistanceReference:
    """A fitted reference distribution of distance_per_codon across a real
    gene corpus, plus the raw per-gene scores it was built from (kept
    around for reporting/plotting, not just the fit -- e.g. percentiles,
    a histogram, or cross-checking against a specific gene's own natural
    CDS, HANDOFF_2026-08-20.md sec 4's single most meaningful reference
    point).

    fit: distribution_fit.FittedDistribution over distance_per_codon --
    the SAME shape-aware normal-scores machinery every other natural-
    baseline comparison in this codebase already uses (see
    distribution_fit.py's module docstring for why raw (x-mean)/std isn't
    safe to assume here either -- a sum of six terms, several themselves
    skewed, has no particular reason to come out Gaussian).
    """
    fit: FittedDistribution
    per_gene: list
    n_genes: int


def build_natural_distance_reference(genes: list, analysis_objects: AnalysisObjects, weights: dict,
                                      locvec: list = None) -> NaturalDistanceReference:
    """score_natural_genes() + fit_normal_transform() over the resulting
    distance_per_codon array -- the one-time, run-once-per-corpus setup
    step. Keep the returned NaturalDistanceReference around and reuse it
    for every natural_zscore()/is_within_natural_range() call rather than
    rebuilding it per candidate -- the fit itself is the expensive part
    (see distribution_fit.py), querying it afterward is cheap.

    Raises ValueError if `genes` has no scorable protein-coding isoforms
    at all -- silently returning a degenerate reference would make every
    future z-score meaningless without any signal that something's wrong.
    """
    per_gene = score_natural_genes(genes, analysis_objects, weights, locvec=locvec)
    if not per_gene:
        raise ValueError(
            "No protein-coding isoforms found in `genes` -- can't build a "
            "natural distance reference from an empty corpus."
        )
    fit = fit_normal_transform([g.distance_per_codon for g in per_gene])
    return NaturalDistanceReference(fit=fit, per_gene=per_gene, n_genes=len(per_gene))


def natural_zscore(sol: list, changevecs: dict, weights: dict, reference: NaturalDistanceReference) -> float:
    """How many natural-CDS-equivalent standard deviations `sol`'s own
    distance_from_optimal() (length-normalized, matching how `reference`
    was built) sits above (positive) or below (negative) the natural
    corpus's typical value. 0.0 means exactly as typical as an average
    real gene, scored the identical way.
    """
    d_per_codon = distance_from_optimal(changevecs, weights) / len(sol)
    return reference.fit.transform(d_per_codon)


def is_within_natural_range(sol: list, changevecs: dict, weights: dict, reference: NaturalDistanceReference,
                             z_threshold: float = 1.0) -> bool:
    """natural_zscore(...) <= z_threshold -- Luke's own "good enough"
    framing (2026-08-20): z_threshold=0.0 means "no worse than the
    average real gene," z_threshold=1.0 (a reasonable default, not a
    universal constant -- pick deliberately per use) means "within one
    natural-equivalent standard deviation of nature's own typical value,"
    which is a genuine pass, not a shortfall -- real CDSs are shaped by
    selective pressures this change vector doesn't model at all (mRNA
    structure, translation kinetics, regulatory motifs, ...), so landing
    a bit above nature's own typical aggregate distance is expected, not
    suspicious.

    Deliberately ONE-SIDED: only an unusually HIGH distance_from_optimal
    is flagged. An unusually LOW one (better than nature on every axis
    this change vector measures) always passes this particular gate --
    that's a separate question (is "better than any real gene on every
    measured axis" itself suspicious?) this function doesn't answer.

    NOT a substitute for ga.kill_off_outside_natural_range()'s existing
    PER-AXIS, two-sided guardrail. A candidate can pass this aggregate
    gate while one single term sits far outside nature's own range,
    hidden by the other five being unusually good -- see
    HANDOFF_2026-08-20.md sec 2.2 ("Cause 2"): CodonPairBias's local layer
    and Kmer specifically have no natural upper bound of their own, so
    they're exactly the terms this aggregate gate is most likely to miss.
    Use both together for a real acceptance decision, not this alone.
    """
    return natural_zscore(sol, changevecs, weights, reference) <= z_threshold
