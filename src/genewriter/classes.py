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


@dataclass
class KmerAnalysis:
    organism: str
    transcriptome: str
    kmer_dict: dict
    l_dict: dict
    kmer_score_obs: dict


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
