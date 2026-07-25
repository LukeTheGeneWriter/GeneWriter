"""Shared codon tables.

Consolidated from the ~15 copies of these same literals scattered across the
notebooks (GeneRider_Cloud, GeneRiderStandards, GenAlg1, CDS_genalg, and each
of the Standards analysis notebooks all defined their own).
"""

AA_CODONS = {
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'C': ['TGT', 'TGC'],
    'W': ['TGG'],
    'E': ['GAA', 'GAG'],
    'D': ['GAT', 'GAC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'N': ['AAT', 'AAC'],
    'M': ['ATG'],
    'K': ['AAA', 'AAG'],
    'Y': ['TAT', 'TAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'Q': ['CAA', 'CAG'],
    'F': ['TTT', 'TTC'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    '*': ['TAA', 'TAG', 'TGA'],
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
}

CODON_TO_AA = {codon: aa for aa, codons in AA_CODONS.items() for codon in codons}

# Human codon usage frequency per 1000 codons (literature baseline used
# throughout the notebooks as a fallback when no organism-specific reference
# set has been computed yet).
CODON_FREQS_LIT = {
    'TTT': 17.14, 'TCT': 16.93, 'TAT': 12.11, 'TGT': 10.40, 'TTC': 17.48, 'TCC': 17.32, 'TAC': 13.49,
    'TGC': 10.81, 'TTA': 8.71, 'TCA': 14.14, 'TAA': 0.44, 'TGA': 0.79, 'TTG': 13.44, 'TCG': 4.03,
    'TAG': 0.35, 'TGG': 11.60, 'CTT': 14.08, 'CCT': 19.31, 'CAT': 11.83, 'CGT': 4.55, 'CTC': 17.81,
    'CCC': 19.11, 'CAC': 14.65, 'CGC': 8.71, 'CTA': 7.44, 'CCA': 18.92, 'CAA': 14.06, 'CGA': 6.42,
    'CTG': 36.10, 'CCG': 6.22, 'CAG': 35.53, 'CGG': 10.79, 'ATT': 16.48, 'ACT': 14.26, 'AAT': 18.43,
    'AGT': 14.05, 'ATC': 18.67, 'ACC': 17.85, 'AAC': 18.30, 'AGC': 19.69, 'ATA': 8.08, 'ACA': 16.52,
    'AAA': 27.48, 'AGA': 13.28, 'ATG': 21.53, 'ACG': 5.59, 'AAG': 31.77, 'AGG': 12.13, 'GTT': 11.74,
    'GCT': 18.99, 'GAT': 24.03, 'GGT': 10.83, 'GTC': 13.44, 'GCC': 25.84, 'GAC': 24.27, 'GGC': 19.79,
    'GTA': 7.66, 'GCA': 17.04, 'GAA': 33.65, 'GGA': 17.12, 'GTG': 25.87, 'GCG': 5.91, 'GAG': 39.67,
    'GGG': 15.35,
}

RARE_CODONS_LIT = ['GCG', 'CCG', 'CGT', 'CGC', 'TCG', 'ACG']

# Per-codon location tags (assigned by GeneDataSourcing.ipynb's
# locate_codons) -> the bucket keys used across the analysis dataclasses.
# 'F' is the exon's 5' end -> 'ExonL50', 'T' is the exon's 3' end ->
# 'ExonR50', 'I' is interior -> 'Exon', 'S' spans a splice junction ->
# 'Splice'. Several of the *Analysis dataclasses (GCAnalysis.windows,
# KmerAnalysis.kmer_dict) never track a 'Splice' bucket at all, so
# TAG_TO_WINDOW_BUCKET folds splice-spanning windows into 'Exon' instead.
TAG_TO_BUCKET = {'F': 'ExonL50', 'T': 'ExonR50', 'I': 'Exon', 'S': 'Splice'}
TAG_TO_WINDOW_BUCKET = {'F': 'ExonL50', 'T': 'ExonR50', 'I': 'Exon', 'S': 'Exon'}


def get_aa(codon: str) -> str:
    return CODON_TO_AA[codon.upper()]


def codon_choices_for_aa(aa: str) -> list:
    aa = aa.upper()
    if aa not in AA_CODONS:
        raise ValueError(f"Unrecognized amino acid: {aa!r}")
    return AA_CODONS[aa]


def generate_codon_vec(aa_seq: str) -> list:
    """For each residue in aa_seq, the list of synonymous codons that encode it."""
    return [codon_choices_for_aa(aa) for aa in aa_seq.upper()]


def _variable_flags(codon: str) -> tuple:
    """Which of a codon's 3 bases can even change under a synonymous swap.

    True whenever some other codon for the same amino acid differs at that
    base -- e.g. Leucine's synonyms (TTA/TTG/CTT/CTC/CTA/CTG) disagree at
    every position, but Serine's AGT/AGC agree at positions 1-2 and only
    ever differ at position 3.
    """
    choices = codon_choices_for_aa(get_aa(codon))
    v1 = 0 if all(c[0] == choices[0][0] for c in choices) else 1
    v2 = 0 if all(c[1] == choices[0][1] for c in choices) else 1
    v3 = 0 if all(c[2] == choices[0][2] for c in choices) else 1
    return (v1, v2, v3)


# This only depends on which amino acid a codon encodes, which never
# changes for the lifetime of a GA run (or a process, for that matter) --
# precomputed once for all 64 codons here rather than recomputed from
# scratch (codon_choices_for_aa + get_aa + three all()-comparisons) every
# time change_vector.py's GC term is called for every candidate sequence.
CODON_TO_VARIABLE_FLAGS = {codon: _variable_flags(codon) for codon in CODON_TO_AA}

# A fixed, arbitrary-but-stable ordering over all 64 codons, and the
# integer<->codon mappings derived from it. Needed for any array-based
# (numpy/CuPy) batch representation of a population of sequences: a GPU
# kernel can't operate on Python strings, so a population becomes a 2D
# array of codon *indices* (population_size x sequence_length) instead of
# a list of lists of 3-letter strings. Sorted so the mapping is
# deterministic across processes/runs without needing to persist it.
CODON_LIST = sorted(CODON_TO_AA)
CODON_TO_INDEX = {codon: i for i, codon in enumerate(CODON_LIST)}
INDEX_TO_CODON = {i: codon for codon, i in CODON_TO_INDEX.items()}

# Per-codon-index versions of the lookup tables above, as flat arrays
# indexed directly by codon index (position i = the codon CODON_LIST[i]) --
# the array-based counterparts of RARE_CODONS_LIT / CODON_FREQS_LIT /
# CODON_TO_VARIABLE_FLAGS / CODON_TO_AA, for O(1)-per-element vectorized
# lookup (`table[index_array]`) instead of a per-codon dict lookup in a
# Python loop. Also includes the "precompute per-codon GC" item flagged for
# later: GC_FLAGS_BY_INDEX[i] is (is_G_or_C(base1), is_G_or_C(base2),
# is_G_or_C(base3)) for CODON_LIST[i], so a batch's per-position GC1/2/3
# indicators are a single fancy-index lookup rather than re-inspecting each
# codon's characters every call.
RARE_CODON_INDICES = frozenset(CODON_TO_INDEX[c] for c in RARE_CODONS_LIT)
IS_RARE_BY_INDEX = [1 if codon in RARE_CODONS_LIT else 0 for codon in CODON_LIST]
CODON_FREQ_BY_INDEX = [CODON_FREQS_LIT[codon] for codon in CODON_LIST]
VARIABLE_FLAGS_BY_INDEX = [CODON_TO_VARIABLE_FLAGS[codon] for codon in CODON_LIST]
GC_FLAGS_BY_INDEX = [tuple(1 if base in 'GC' else 0 for base in codon) for codon in CODON_LIST]

# Base-4 digit per nucleotide (A/C/G/T -> 0..3) -- the encoding
# gpu_change_vector.batch_kmer_term() uses to turn each k-mer window into a
# single integer in [0, 4**k) for array-indexed lookup, instead of hashing
# the k-mer substring itself (the previously-unbatchable part of the Kmer
# term -- see that function's docstring). Per-codon-index so a batch's
# per-nucleotide base-4 digits are one fancy-index lookup, same pattern as
# GC_FLAGS_BY_INDEX above.
NT_TO_BASE4 = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
NT_BASE4_BY_CODON_INDEX = [[NT_TO_BASE4[base] for base in codon] for codon in CODON_LIST]


def encode_codons(codons) -> list:
    """Map an iterable of codon strings to their integer indices (see
    CODON_TO_INDEX). The inverse of decode_codons()."""
    return [CODON_TO_INDEX[c] for c in codons]


def decode_codons(indices) -> list:
    """Map an iterable of codon indices back to codon strings. The inverse
    of encode_codons()."""
    return [INDEX_TO_CODON[int(i)] for i in indices]
