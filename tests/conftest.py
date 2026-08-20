import random

import pytest

from genewriter.classes import (
    CodonAnalysis,
    CodonPairBiasAnalysis,
    GCAnalysis,
    IsoformGeneBody,
    KmerAnalysis,
    NaturalGene,
    ProteinObj,
    RareCodonAnalysis,
)
from genewriter.change_vector import AnalysisObjects
from genewriter.codon_tables import AA_CODONS, CODON_FREQS_LIT, CODON_TO_AA, generate_codon_vec

AA_SEQ = "MAVLDEFGHIKPQRSTWYCN" * 2  # 40 residues, all 20 standard amino acids twice


def random_solution(aa_seq, seed=0):
    rng = random.Random(seed)
    return [rng.choice(choices) for choices in generate_codon_vec(aa_seq)]


def make_synthetic_isoform(aa_seq: str, codon_choice_fn, loc_tags: list, isoform_number: str = "1") -> IsoformGeneBody:
    """Build an IsoformGeneBody with codons[i] guaranteed to correspond to
    aa_seq[i] by construction -- codon_choice_fn(aa, i) picks the codon for
    each position (e.g. `lambda aa, i: codon_choices_for_aa(aa)[0]` for a
    deterministic "always first synonym" fixture, or something
    randomized+seeded). loc_tags: list[str] of 'F'/'T'/'I'/'S', same length
    as aa_seq.

    This is the ONLY sanctioned way tests get codons<->aaSeq alignment --
    never trust it from a loaded JSON file (GeneDataSourcing.ipynb's
    check_cds_against_protein bug broke that alignment for every gene JSON
    generated before the fix; regenerating that data is a separate,
    out-of-session step, so no local sample data can be trusted for this
    either)."""
    assert len(loc_tags) == len(aa_seq), "loc_tags must be the same length as aa_seq"
    codons = [[codon_choice_fn(aa, i), loc_tags[i]] for i, aa in enumerate(aa_seq)]
    return IsoformGeneBody(
        isoformNumber=isoform_number,
        associatedProtein=ProteinObj(aaSeq=aa_seq, protWeight=0.0, associatedGeneID=0),
        fullSequence="",
        mRNASeq="",
        codons=codons,
        relativeAbundance=1.0,
        geneBody=[],
    )


def make_synthetic_gene(gene_id: int, isoforms: list, gene_name: str = "SYN") -> NaturalGene:
    """Wrap a list of IsoformGeneBody (e.g. from make_synthetic_isoform)
    into a NaturalGene, filling in required-but-unused fields with dummy
    values -- protein_coding_isoforms() and everything downstream of it
    only ever reads .isoforms/.geneID/.geneName."""
    return NaturalGene(
        isoforms=isoforms,
        geneID=gene_id,
        geneName=gene_name,
        organism="test",
        DNASequence="",
        spliceAIDonor=[],
        spliceAIReceptor=[],
        energetics={},
        chromosome=0,
    )


def make_synthetic_genes(n: int, aa_seq: str = AA_SEQ) -> list:
    """n synthetic NaturalGenes, each one isoform whose codons are a seeded,
    per-gene-varied synonymous choice (random_solution) with location tags
    spanning F(5')/I(interior)/T(3') so every bucket is exercised -- for the
    baseline_<test>.py chunk-then-finalize-equals-monolithic tests, which
    need real per-gene/per-window variation to be a meaningful check (unlike
    test_baseline.py's small hand-picked single-value fixtures)."""
    genes = []
    for i in range(n):
        codons = random_solution(aa_seq, seed=i)
        loc_tags = (['F'] * 5) + (['I'] * (len(aa_seq) - 10)) + (['T'] * 5)
        iso = make_synthetic_isoform(aa_seq, lambda aa, j, c=codons: c[j], loc_tags)
        genes.append(make_synthetic_gene(i, [iso]))
    return genes


@pytest.fixture
def aa_seq():
    return AA_SEQ


@pytest.fixture
def rare_codon_analysis():
    rng = random.Random(1)
    return RareCodonAnalysis(
        organism="test",
        transcriptome="test",
        totalCodons=10000,
        rareCodonsByLocation={'ExonL50': [10, 100], 'Exon': [50, 500], 'ExonR50': [10, 100], 'Splice': [5, 50]},
        rare_codon_windows={i: rng.randint(1, 50) for i in range(0, 16)},
        codonFreqsLit=dict(CODON_FREQS_LIT),
        usagePerGene=[rng.uniform(0.0, 0.2) for _ in range(200)],
    )


@pytest.fixture
def codon_usage_analysis():
    rng = random.Random(2)

    # Per-amino-acid, per-codon usage-percentage baseline (see
    # CodonAnalysis.codonUsagePercentByAA's docstring) -- a random "true"
    # preference split per amino acid, then synthetic per-gene samples
    # jittered around it, so the pooled distribution has real spread to
    # z-score against rather than a degenerate single point. Deliberately
    # ~61 separate distributions (one per real (aa, codon) pair, not one
    # shared array the way every other baseline fixture in this file only
    # needs) -- 15 samples each, not 200: distribution_fit.py only tries
    # its 3 iterative-optimizer families (gamma/lognorm/skewnorm) at 20+
    # samples, so staying under that threshold keeps every one of these
    # ~61 fits on the cheap closed-form (normal/expon) path. This function-
    # scoped fixture (like every fixture in this file) refits from scratch
    # for every single test that uses it -- at 200 samples x 61 pairs this
    # measured 220s for one test module; a real GA run pays this cost once
    # per run (cached via change_vector.cached_stat), so it's a test-speed
    # concern specific to fixture rebuilding, not a production one.
    codon_usage_percent_by_aa: dict = {}
    for aa, codons in AA_CODONS.items():
        if aa == '*':
            continue
        if len(codons) == 1:
            # A single-codon amino acid (Met, Trp) has no real choice --
            # every real gene's own percentage for it is exactly 1.0,
            # always, so the true pooled distribution is genuinely
            # degenerate (not jittered) -- matches
            # accumulate_codon_usage_percent_by_aa()'s actual output on
            # real data exactly, which is what change_vector._codon_usage_
            # term's degenerate-transform-relies-on-z=0 behavior assumes.
            codon_usage_percent_by_aa[aa] = {codons[0]: [1.0] * 15}
            continue
        base = [rng.uniform(0.1, 1.0) for _ in codons]
        total = sum(base)
        base = [b / total for b in base]
        for codon, p in zip(codons, base):
            samples = [max(0.0, min(1.0, p + rng.uniform(-0.05, 0.05))) for _ in range(15)]
            codon_usage_percent_by_aa.setdefault(aa, {})[codon] = samples

    return CodonAnalysis(
        organism="test",
        transcriptome="test",
        AAFreqs={aa: rng.uniform(0.01, 0.1) for aa in set(CODON_TO_AA.values())},
        totalCodons=10000,
        codonFreqsByLocation={c: {'ExonL50': 1, 'Exon': 1, 'ExonR50': 1, 'Splice': 1} for c in CODON_FREQS_LIT},
        codonFreqsLit=dict(CODON_FREQS_LIT),
        codonUsageScoreByGene=[rng.uniform(10, 30) for _ in range(200)],
        windowsize=15,
        windowscores=[rng.uniform(10, 30) for _ in range(500)],
        windowdistancesfromoptimal=[rng.uniform(0, 10) for _ in range(500)],
        codonUsagePercentByAA=codon_usage_percent_by_aa,
    )


@pytest.fixture
def codon_pair_bias_analysis():
    rng = random.Random(3)
    codons = sorted(CODON_TO_AA)
    cpb_lit = {c1 + c2: rng.uniform(100, 500000) for c1 in codons for c2 in codons}
    return CodonPairBiasAnalysis(
        organism="test",
        transcriptome="test",
        totalCodonPairs=len(cpb_lit),
        cpbPerGene=[rng.uniform(1000, 100000) for _ in range(200)],
        cpb_lit=cpb_lit,
        cpbPerWindow=[rng.uniform(1000, 100000) for _ in range(500)],
        windowsize=15,
    )


@pytest.fixture
def gc_analysis():
    rng = random.Random(4)
    tagged = {'ExonL50': [rng.uniform(0.3, 0.7) for _ in range(200)],
              'Exon': [rng.uniform(0.3, 0.7) for _ in range(200)],
              'ExonR50': [rng.uniform(0.3, 0.7) for _ in range(200)],
              'Splice': [rng.uniform(0.3, 0.7) for _ in range(200)]}
    windows = {'ExonL50': [rng.uniform(0.3, 0.7) for _ in range(500)],
               'Exon': [rng.uniform(0.3, 0.7) for _ in range(500)],
               'ExonR50': [rng.uniform(0.3, 0.7) for _ in range(500)]}
    return GCAnalysis(
        organism="test",
        transcriptome="test",
        totalCodons=10000,
        windows=windows,
        taggedGC1=tagged,
        taggedGC2={k: list(v) for k, v in tagged.items()},
        taggedGC3={k: list(v) for k, v in tagged.items()},
        windowsize=15,
        gcPerGene=[rng.uniform(0.3, 0.7) for _ in range(200)],
    )


@pytest.fixture
def kmer_analysis():
    rng = random.Random(5)
    bases = "ACGT"
    dimers = {a + b: {'ExonL50': {'p': 0.01, 'fold_enrich': rng.uniform(0.5, 2.0)},
                       'Exon': {'p': 0.01, 'fold_enrich': rng.uniform(0.5, 2.0)},
                       'ExonR50': {'p': 0.01, 'fold_enrich': rng.uniform(0.5, 2.0)}}
              for a in bases for b in bases}
    return KmerAnalysis(
        organism="test",
        transcriptome="test",
        kmer_dict={'2': dimers},
        l_dict={'2': {'ExonL50': 1000, 'Exon': 1000, 'ExonR50': 1000}},
        kmer_score_obs={'2': {'ExonL50': [1.0], 'Exon': [1.0], 'ExonR50': [1.0]}},
    )


@pytest.fixture
def analysis_objects(rare_codon_analysis, codon_usage_analysis, codon_pair_bias_analysis, gc_analysis, kmer_analysis):
    return AnalysisObjects(
        rare_codon=rare_codon_analysis,
        codon_usage=codon_usage_analysis,
        codon_pair_bias=codon_pair_bias_analysis,
        gc=gc_analysis,
        kmer=kmer_analysis,
    )


@pytest.fixture
def weights():
    # Uracil defaults to 0.0 (computed but ignored in scoring) so adding the
    # term doesn't perturb any existing test's selection/scoring outcome --
    # tests targeting the term itself use a nonzero weight deliberately.
    return {'RareCodons': 1.0, 'CodonUsage': 1.0, 'CodonPairBias': 1.0, 'GC': 1.0, 'Kmer': 1.0, 'Uracil': 0.0}
