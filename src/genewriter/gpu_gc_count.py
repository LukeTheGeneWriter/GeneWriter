"""GCAnalysis baseline counting -- the mirror-image of
gpu_change_vector.batch_gc_term()'s scoring code. See
gpu_rare_codon_count.py's module docstring for the shared shape/rationale.

Unlike the other 3 new counting modules, this one needs BOTH codon- and
nucleotide-granularity views of each isoform at once: gcPerGene/windows
operate at nucleotide granularity (matching kmer's convention), while
taggedGC1/2/3 are per-codon-position values bucketed by a per-CODON (not
per-nt) location tag. Rather than encoding/concatenating twice, the
nucleotide-level GC-flags array and location string are both *derived* from
the codon-level batch (GC_FLAGS_BY_INDEX[codon_idx].reshape(-1) for the
flags; ''.join(c*3 for c in codon_loc_string) for the locations) -- so only
one concat_isoform_batch() call is needed per batch.

windows' majority-vote uses gpu_corpus_batch.majority_raw_tag_bucket(), NOT
gpu_change_vector.majority_window_bucket() -- see that function's own
docstring and memory/majority_vote_bucket_discrepancy.md for why the two are
not interchangeable (gpu_kmer_count.py's counting side already made this
mistake; this module deliberately does not repeat it).
"""

import numpy as np

from .classes import GCAnalysis
from .codon_tables import GC_FLAGS_BY_INDEX, TAG_TO_BUCKET, encode_codons
from .gene_io import protein_coding_isoforms
from .gpu_change_vector import _sliding_window_view
from .gpu_corpus_batch import (
    _WINDOW_BUCKET_NAMES,
    concat_isoform_batch,
    majority_raw_tag_bucket,
    segment_mean,
    valid_window_mask,
    vram_aware_batch_size,
)

_GC_FLAGS_TABLE = np.asarray(GC_FLAGS_BY_INDEX, dtype=float)  # (64, 3)
_BUCKET_NAMES_4WAY = ('ExonL50', 'ExonR50', 'Exon', 'Splice')
_BUCKET_INDEX_4WAY = {name: i for i, name in enumerate(_BUCKET_NAMES_4WAY)}
_TAG_TO_BUCKET_INDEX = {tag: _BUCKET_INDEX_4WAY[bucket] for tag, bucket in TAG_TO_BUCKET.items()}

_VRAM_SAFETY_FACTOR = 2
_DEFAULT_CPU_BATCH_CODONS = 5_000_000
_MIN_BATCH_CODONS = 100_000


def encode_isoform_codons(xp, iso) -> tuple:
    """Codon granularity (untripled loc_string) -- the nt-level view this
    module also needs is derived from this in count_gc_for_batch(), not
    encoded separately."""
    codon_idx = np.asarray(encode_codons([cod for cod, _loc in iso.codons]), dtype=np.int64)
    loc_string = ''.join(loc for _cod, loc in iso.codons)
    return (codon_idx if xp is np else xp.asarray(codon_idx)), loc_string


def _estimate_bytes_per_codon(winsize: int) -> int:
    # nt-level sliding window is the dominant term (winsize is in
    # nucleotides here, 3x the codon count) -- see gpu_kmer_count's
    # matching estimate for the same reasoning, scaled to codon units.
    return _VRAM_SAFETY_FACTOR * (3 * (8 * winsize + 8) + 8)


def count_gc_for_batch(xp, codon_idx: 'np.ndarray', codon_loc_string: str,
                        starts: list, lengths: list, winsize: int) -> dict:
    """One VRAM sub-batch. Returns a dict:
      gc_per_gene: list[float], one per isoform.
      tagged: dict[str, list[list[int]]] -- 4-way bucket -> [gc1s, gc2s, gc3s],
        raw per-codon-position flags, order-preserving (isoform order, then
        codon order within isoform -- matches baseline.py's append order).
      windows: dict[str, list[float]] -- 3-way bucket -> raw per-window GC
        fractions, boundary-masked, order-preserving.
      total_codons: int.
    """
    M = int(codon_idx.shape[0])
    gc_table = xp.asarray(_GC_FLAGS_TABLE)
    per_pos_gc = gc_table[codon_idx]  # (M, 3)
    flat_nt_gc = per_pos_gc.reshape(-1)  # (3M,)

    nt_loc_string = ''.join(c * 3 for c in codon_loc_string)
    nt_starts = [3 * s for s in starts]
    nt_lengths = [3 * length for length in lengths]

    n_isoforms = len(lengths)
    if n_isoforms:
        nt_segment_ids = np.repeat(np.arange(n_isoforms), nt_lengths)
        means = segment_mean(xp, flat_nt_gc, nt_segment_ids, n_isoforms)
        means = means if xp is np else xp.asnumpy(means)
        gc_per_gene = means.tolist()
    else:
        gc_per_gene = []

    bucket_of_codon = np.fromiter(
        (_TAG_TO_BUCKET_INDEX[c] for c in codon_loc_string), dtype=np.int64, count=len(codon_loc_string),
    ) if codon_loc_string else np.zeros(0, dtype=np.int64)
    per_pos_gc_np = per_pos_gc if xp is np else xp.asnumpy(per_pos_gc)
    tagged = {name: [[], [], []] for name in _BUCKET_NAMES_4WAY}
    for b, name in enumerate(_BUCKET_NAMES_4WAY):
        mask = bucket_of_codon == b
        for pos in range(3):
            tagged[name][pos] = per_pos_gc_np[mask, pos].astype(int).tolist()

    windows = {name: [] for name in _WINDOW_BUCKET_NAMES}
    max_start = (3 * M) - winsize + 1
    if max_start > 0 and n_isoforms:
        valid = valid_window_mask(3 * M, nt_starts, nt_lengths, winsize)
        if valid.any():
            win_gc_all = _sliding_window_view(xp, flat_nt_gc, winsize, axis=0)[:max_start].mean(axis=1)
            majority_all = majority_raw_tag_bucket(nt_loc_string, winsize, max_start)  # always numpy
            win_gc_all_np = win_gc_all if xp is np else xp.asnumpy(win_gc_all)
            for b, name in enumerate(_WINDOW_BUCKET_NAMES):
                mask = valid & (majority_all == b)
                windows[name] = win_gc_all_np[mask].tolist()

    return {'gc_per_gene': gc_per_gene, 'tagged': tagged, 'windows': windows, 'total_codons': M}


def count_gc_for_chunk(xp, genes: list, organism: str = "human",
                        winsize: int = 21, vram_fraction: float = 0.5) -> GCAnalysis:
    """One in-memory chunk of NaturalGene objects. windowsize default is 21
    here, not 15 -- matches baseline.compute_gc_analysis's own default,
    which differs from the other 3 baseline.py window functions."""
    gc_per_gene: list = []
    tagged_gc1 = {name: [] for name in _BUCKET_NAMES_4WAY}
    tagged_gc2 = {name: [] for name in _BUCKET_NAMES_4WAY}
    tagged_gc3 = {name: [] for name in _BUCKET_NAMES_4WAY}
    windows = {name: [] for name in _WINDOW_BUCKET_NAMES}
    total_codons = 0

    max_batch_codons = vram_aware_batch_size(
        xp, _estimate_bytes_per_codon(winsize), vram_fraction=vram_fraction,
        default_cpu_batch=_DEFAULT_CPU_BATCH_CODONS, min_batch=_MIN_BATCH_CODONS,
    )

    def _flush(batch):
        if not batch:
            return
        codon_idx, loc_string, starts, lengths = concat_isoform_batch(xp, batch)
        result = count_gc_for_batch(xp, codon_idx, loc_string, starts, lengths, winsize)
        gc_per_gene.extend(result['gc_per_gene'])
        for name in _BUCKET_NAMES_4WAY:
            tagged_gc1[name].extend(result['tagged'][name][0])
            tagged_gc2[name].extend(result['tagged'][name][1])
            tagged_gc3[name].extend(result['tagged'][name][2])
        for name in _WINDOW_BUCKET_NAMES:
            windows[name].extend(result['windows'][name])
        nonlocal total_codons
        total_codons += result['total_codons']

    batch = []
    batch_codons = 0
    for _gene, iso in protein_coding_isoforms(genes):
        encoded = encode_isoform_codons(xp, iso)
        n = int(encoded[0].shape[0])
        if batch and batch_codons + n > max_batch_codons:
            _flush(batch)
            batch = []
            batch_codons = 0
        batch.append(encoded)
        batch_codons += n
    _flush(batch)

    return GCAnalysis(
        organism=organism,
        transcriptome="local-sample",  # matches baseline.compute_gc_analysis exactly -- see gpu_rare_codon_count.py's matching comment
        totalCodons=total_codons,
        windows=windows,
        taggedGC1=tagged_gc1,
        taggedGC2=tagged_gc2,
        taggedGC3=tagged_gc3,
        windowsize=winsize,
        gcPerGene=gc_per_gene,
    )
