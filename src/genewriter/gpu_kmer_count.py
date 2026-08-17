"""Corpus-side k-mer counting: the counting mirror-image of
gpu_change_vector.batch_kmer_term()'s scoring code.

batch_kmer_term() looks candidate windows up against an already-built
fold_enrich table (table[bucket, code]) -- this module does the other half of
that pipeline, the one the legacy Kmer_Analysis.ipynb notebook used to do with
a per-window Python dict insert (and, worse, a per-gene np.meshgrid rebuild of
the entire 4**k k-mer universe just to prime that dict): given a gene's
isoforms, count how many times each k-mer actually occurs, per location
bucket, so baseline_kmer.py can later turn those raw counts into fold_enrich
values.

Reuses the exact same encoding building blocks gpu_change_vector.py already
has for the scoring direction -- codon_tables.encode_codons()/
NT_BASE4_BY_CODON_INDEX for turning codons into base-4 nucleotide digits,
gpu_change_vector._sliding_window_view() for overlapping windows, and
gpu_change_vector.majority_window_bucket() for "which of ExonL50/Exon/ExonR50
does this window mostly fall in" -- so a window becomes an integer code in
[0, 4**k) and the count becomes one xp.bincount() call instead of a dict
insert. No 4**k universe is ever materialized: xp.bincount(minlength=4**k)
covers the full domain implicitly (an unobserved code just gets a 0), the same
"observed-only tracked, dense output" property baseline.compute_kmer_analysis's
dict-based version already has.

Dispatch is per-isoform (one encode+count call per protein_coding_isoforms()
yield), not per-chunk-concatenated, since different genes have different
(ragged) sequence lengths -- a single flat cross-gene buffer for one kernel
launch per k is a plausible follow-up if per-isoform launch overhead proves
significant in practice, not built here.
"""

import numpy as np

from .codon_tables import NT_BASE4_BY_CODON_INDEX, encode_codons
from .gene_io import protein_coding_isoforms
from .gpu_change_vector import _sliding_window_view, majority_window_bucket

_WINDOW_BUCKET_NAMES = ('ExonL50', 'Exon', 'ExonR50')

_BASE4_TABLE = np.asarray(NT_BASE4_BY_CODON_INDEX, dtype=np.int64)  # (64, 3)


def encode_isoform_nt_base4(xp, iso) -> tuple:
    """iso.codons -> (nt_base4: (3N,) int array on the given backend,
    location_string: str of length 3N). Same `continuous`/`loc_string`
    construction baseline.compute_kmer_analysis uses (''.join over
    iso.codons), but encoded via NT_BASE4_BY_CODON_INDEX (one array lookup)
    instead of NT_TO_BASE4 (a per-character dict) -- the single-isoform
    counterpart of gpu_change_vector.encode_population()/batch_kmer_term()'s
    codon_idx encoding."""
    codon_idx = np.asarray(encode_codons([cod for cod, _loc in iso.codons]), dtype=np.int64)
    nt_base4 = _BASE4_TABLE[codon_idx].reshape(-1)  # (3N,)
    loc_string = ''.join(loc * 3 for _cod, loc in iso.codons)
    return (nt_base4 if xp is np else xp.asarray(nt_base4)), loc_string


def count_kmers_for_isoform(xp, nt_base4, loc_string: str, k: int) -> tuple:
    """One isoform, one k. Returns (counts, totals):
    counts[bucket_name] -> (4**k,) int64 array of raw occurrence counts,
    totals[bucket_name] -> int, the number of windows whose majority bucket
    was that name (matches baseline.compute_kmer_analysis's loc_totals,
    needed later to compute each bucket's `expected = total / 4**k`).

    Window count follows the same `num_wins = max(M - k, 0)` convention as
    baseline.compute_kmer_analysis and gpu_change_vector.batch_kmer_term
    (both deliberately stop one window short of the naive M-k+1 -- a
    long-standing convention inherited from the original notebook, kept here
    so this module's counts agree with baseline.py's byte-for-byte, which
    tests/test_gpu_kmer_count.py cross-checks directly)."""
    M = nt_base4.shape[0]
    num_wins = max(M - k, 0)
    counts = {name: xp.zeros(4 ** k, dtype=xp.int64) for name in _WINDOW_BUCKET_NAMES}
    totals = {name: 0 for name in _WINDOW_BUCKET_NAMES}
    if num_wins == 0:
        return counts, totals

    windows = _sliding_window_view(xp, nt_base4, k, axis=0)[:num_wins]  # (num_wins, k)
    powers = xp.asarray(np.asarray([4 ** (k - 1 - j) for j in range(k)], dtype=np.int64))
    codes = (windows * powers).sum(axis=1)  # (num_wins,), each in [0, 4**k)

    majority = majority_window_bucket(loc_string, k, num_wins)  # (num_wins,) always numpy
    majority = majority if xp is np else xp.asarray(majority)

    for b, name in enumerate(_WINDOW_BUCKET_NAMES):
        mask = majority == b
        n = int(mask.sum())
        if n == 0:
            continue
        counts[name] = counts[name] + xp.bincount(codes[mask], minlength=4 ** k)
        totals[name] = n

    return counts, totals


def count_kmers_for_chunk(xp, genes: list, k_values=range(2, 11)) -> tuple:
    """One in-memory chunk of NaturalGene objects, every k in k_values.
    Loops isoforms via gene_io.protein_coding_isoforms() (the same
    "skip computationally-predicted-only transcripts, skip empty codon
    streams" rule every baseline_*.py test already uses) and accumulates raw
    counts across every isoform in the chunk.

    Returns (raw_counts, loc_totals):
      raw_counts[k][bucket_name] -> (4**k,) int64 array
      loc_totals[k][bucket_name] -> int
    Callers (baseline_kmer.compute_and_write_shard) persist these as one
    chunk's shard; corpus-wide fold_enrich is only computed once, in
    baseline_kmer.finalize(), from the sum of every chunk's raw counts."""
    raw = {k: {name: xp.zeros(4 ** k, dtype=xp.int64) for name in _WINDOW_BUCKET_NAMES} for k in k_values}
    loc_totals = {k: {name: 0 for name in _WINDOW_BUCKET_NAMES} for k in k_values}

    for _gene, iso in protein_coding_isoforms(genes):
        nt_base4, loc_string = encode_isoform_nt_base4(xp, iso)
        for k in k_values:
            counts, totals = count_kmers_for_isoform(xp, nt_base4, loc_string, k)
            for name in _WINDOW_BUCKET_NAMES:
                raw[k][name] = raw[k][name] + counts[name]
                loc_totals[k][name] += totals[name]

    return raw, loc_totals
