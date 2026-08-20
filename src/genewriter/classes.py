"""Dataclasses ported from GeneClassesCloud.ipynb (the canonical schema).

GeneRider_Cloud.ipynb, GenAlg1.ipynb and CDS_genalg.ipynb each independently
redefine overlapping subsets of these with different field lists. This module
is the one canonical version; code ported from any of the other notebooks has
been rewritten to match these field names.
"""

from dataclasses import dataclass, field


@dataclass
class ProteinObj:
    aaSeq: str
    protWeight: float
    associatedGeneID: int


@dataclass
class NaturalGene:
    isoforms: list
    geneID: int
    geneName: str
    organism: str
    DNASequence: str
    spliceAIDonor: list
    spliceAIReceptor: list
    energetics: dict
    chromosome: int


@dataclass
class IsoformGeneBody:
    isoformNumber: str
    associatedProtein: ProteinObj
    fullSequence: str
    mRNASeq: str
    # list of (codon: str, location_tag: str) pairs. Tags (per
    # GeneDataSourcing.ipynb's locate_codons and GC_Analysis.ipynb's
    # gc_analysis, which is what fixes the F/T convention): 'F' = within
    # 15nt of an exon's 5' end (buckets as 'ExonL50' in the analysis
    # dataclasses), 'T' = within 15nt of an exon's 3' end (buckets as
    # 'ExonR50'), 'I' = interior (buckets as 'Exon'), 'S' = spans a splice
    # junction (buckets as 'Splice').
    codons: list
    relativeAbundance: float
    geneBody: list


@dataclass
class Isoform:
    isoformNumber: str
    associatedProtein: ProteinObj
    fullSequence: str
    codingSeq: str
    relativeAbundance: float


@dataclass
class Proposed_Solution:
    codons: list
    number: int
    change_vecs: dict
    # Shields this individual from ga.kill_off()/kill_off_by_term()/
    # select_survivors() -- set by ga.mark_protected() (see that function's
    # docstring), never inherited by a new genotype (growth's freshly
    # created individuals always start unprotected, same as .number/
    # .change_vecs being freshly computed rather than copied from a
    # parent). Trailing default so every existing positional
    # Proposed_Solution(codons, number, change_vecs) call across the
    # codebase stays valid unchanged.
    protected: bool = False


@dataclass
class SyntheticGene:
    AASeq: str
    desc: str
    DNASeqs: list
    degreeOfDegeneracy: int
    vectors: list


@dataclass
class CodonAnalysis:
    organism: str
    transcriptome: str
    AAFreqs: dict
    totalCodons: int
    codonFreqsByLocation: dict
    codonFreqsLit: dict
    codonUsageScoreByGene: list
    windowsize: int
    windowscores: list
    windowdistancesfromoptimal: list
    # Per-amino-acid, per-synonymous-codon usage-PERCENTAGE distribution,
    # pooled across the whole gene corpus -- {aa: {codon: [percentage
    # samples]}}. Each gene contributes its own "what fraction of this
    # gene's occurrences of this amino acid used this codon" value,
    # repeated in the pooled list proportional to how many times that
    # amino acid actually occurs in that gene -- weights toward genes with
    # more (hence more reliable) observations, rather than counting every
    # gene's percentage equally regardless of sample size. See
    # baseline.compute_codon_usage_analysis()'s matching comment for the
    # full reasoning.
    #
    # This is what change_vector._codon_usage_term's 2026-08-19 redesign
    # z-scores a candidate's own usage-% against -- NOT
    # codonUsageScoreByGene/windowscores/windowdistancesfromoptimal above,
    # which are now vestigial for scoring purposes (still computed,
    # unchanged, because weight_calibration.py's intolerance-weight
    # calibration still reads codonUsageScoreByGene as CodonUsage's
    # per-gene aggregate -- a deliberate scope limit, not an oversight;
    # see memory/codon_usage_term_percentage_redesign.md).
    #
    # Trailing default (not a positional field) so existing
    # CodonAnalysis(...) construction call sites that predate this field
    # keep working unchanged.
    codonUsagePercentByAA: dict = field(default_factory=dict)


@dataclass
class RareCodonAnalysis:
    organism: str
    transcriptome: str
    totalCodons: int
    rareCodonsByLocation: dict
    rare_codon_windows: dict
    codonFreqsLit: dict
    usagePerGene: list


@dataclass
class CodonPairBiasAnalysis:
    organism: str
    transcriptome: str
    totalCodonPairs: int
    cpbPerGene: list
    cpb_lit: dict
    cpbPerWindow: list
    windowsize: int


@dataclass
class GCAnalysis:
    organism: str
    transcriptome: str
    totalCodons: int
    windows: dict
    taggedGC1: dict
    taggedGC2: dict
    taggedGC3: dict
    windowsize: int
    # Overall per-gene GC fraction (all 3 codon positions pooled), one value
    # per gene -- the GC counterpart of CodonAnalysis.codonUsageScoreByGene /
    # CodonPairBiasAnalysis.cpbPerGene (see weight_calibration.py, which
    # needs a per-gene array for every term to gauge how tightly natural
    # genes conserve it). Defaulted to [] rather than required: existing
    # real Standards/GCAnalysis.json files (generated by the external
    # GC_Analysis.ipynb notebook, hours to regenerate) predate this field
    # and don't have it -- standards_io.load_standards() would otherwise
    # break on every one of them until they're regenerated.
    gcPerGene: list = field(default_factory=list)


@dataclass
class KmerAnalysis:
    organism: str
    transcriptome: str
    kmer_dict: dict
    l_dict: dict
    kmer_score_obs: dict


@dataclass
class AAMotifIndex:
    """Amino-acid subsequence ("motif") occurrence counts across a
    proteome, for a fixed range of lengths -- see aa_motif_index.py.
    Unlike KmerAnalysis, this is unrestricted by codon location tag (the
    report tool built on this answers "how common is this motif overall",
    not a codon-choice model)."""
    organism: str
    transcriptome: str
    k_values: tuple
    motif_counts: dict   # motif_counts[str(k)][motif_str] -> occurrence count
    total_windows: dict  # total_windows[str(k)] -> windows scanned, for a future frequency-rate use


@dataclass
class CodonNgramModel:
    """Codon-choice distribution conditioned on local amino-acid context,
    learned from natural genes' interior ('I'-tagged) codon positions by
    default -- see codon_ngram.py. context_orders is descending
    (longest-first); context_counts[str(k)][aa_context] -> {codon: count},
    where aa_context is the k-length AA substring ending at (and
    including) the position being predicted."""
    organism: str
    transcriptome: str
    context_orders: tuple
    context_counts: dict


@dataclass
class EnergeticAnalysis:
    organism: str
    transcriptome: str
    el50model_path: str
    er50model_path: str


@dataclass
class RNAFoldingAnalysis:
    organism: str
    transcriptome: str
    folding_dict: dict
    loc_dict: dict
    observations: dict
