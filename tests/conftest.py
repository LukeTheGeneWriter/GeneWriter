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
from genewriter.codon_tables import CODON_FREQS_LIT, CODON_TO_AA, generate_codon_vec

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
    return {'RareCodons': 1.0, 'CodonUsage': 1.0, 'CodonPairBias': 1.0, 'GC': 1.0, 'Kmer': 1.0}
