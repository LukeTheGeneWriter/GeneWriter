"""RareCodonAnalysis baseline counting -- the mirror-image of
gpu_change_vector.batch_rare_codon_term()'s scoring code, the same
relationship gpu_kmer_count.py already has to batch_kmer_term(). Given a
chunk of NaturalGene objects, counts the raw statistics
baseline.compute_rare_codon_analysis() computes with pure Python loops,
batched across many isoforms per kernel launch via gpu_corpus_batch.py's
shared VRAM-aware concatenation machinery.

Codon granularity throughout (encode_codons(), untripled loc strings) --
unlike gpu_kmer_count.py/a future gpu_gc_count.py, which work at nucleotide
granularity. Window count uses the full, never-drop-a-window convention
(gpu_corpus_batch.valid_window_mask(), `M - winsize + 1`) -- matches
baseline.compute_rare_codon_analysis's own loop, which already used this
convention (the OTHER 3 baseline.py window loops were fixed to match it
2026-08-19, see that module's own comments).
"""

import numpy as np

from .classes import RareCodonAnalysis
from .codon_tables import CODON_FREQS_LIT, IS_RARE_BY_INDEX, TAG_TO_BUCKET, encode_codons
from .gene_io import protein_coding_isoforms
from .gpu_change_vector import _sliding_window_view
from .gpu_corpus_batch import concat_isoform_batch, segment_mean, valid_window_mask, vram_aware_batch_size

_IS_RARE_TABLE = np.asarray(IS_RARE_BY_INDEX, dtype=np.int64)
_BUCKET_NAMES_4WAY = ('ExonL50', 'ExonR50', 'Exon', 'Splice')
_BUCKET_INDEX_4WAY = {name: i for i, name in enumerate(_BUCKET_NAMES_4WAY)}
_TAG_TO_BUCKET_INDEX = {tag: _BUCKET_INDEX_4WAY[bucket] for tag, bucket in TAG_TO_BUCKET.items()}

_VRAM_SAFETY_FACTOR = 2
_DEFAULT_CPU_BATCH_CODONS = 5_000_000
_MIN_BATCH_CODONS = 100_000


def encode_isoform_codons(xp, iso) -> tuple:
    """iso.codons -> (codon_idx: (N,) int array on the given backend,
    loc_string: str of length N, NOT tripled -- codon granularity, unlike
    gpu_kmer_count.encode_isoform_nt_base4's nucleotide granularity)."""
    codon_idx = np.asarray(encode_codons([cod for cod, _loc in iso.codons]), dtype=np.int64)
    loc_string = ''.join(loc for _cod, loc in iso.codons)
    return (codon_idx if xp is np else xp.asarray(codon_idx)), loc_string


def _estimate_bytes_per_codon(winsize: int) -> int:
    """Conservative worst-case GPU bytes needed per codon of batch content,
    for one count_rare_codons_for_batch() call. Dominant term: the sliding-
    window view over rare_flags, elementwise-summed, materializes a
    (num_positions, winsize) int64 array -- 8*winsize bytes/position. Plus
    rare_flags/codon_idx themselves (8 bytes/position each). Doubled for
    allocator overhead/temporaries, same margin gpu_kmer_count.py's
    _estimate_bytes_per_nt uses."""
    return _VRAM_SAFETY_FACTOR * (8 * winsize + 8 + 8)


def count_rare_codons_for_batch(xp, codon_idx: 'np.ndarray', loc_string: str,
                                 starts: list, lengths: list, winsize: int) -> dict:
    """One VRAM sub-batch (many isoforms concatenated). Returns a dict:
      usage_per_gene: list[float], one entry per isoform in this batch --
        fraction of codons that are rare.
      rare_codon_windows: dict[int, int] -- histogram of window rare-counts
        (0..winsize), boundary-masked so no window crosses an isoform join.
      rare_by_location: dict[str, list[int, int]] -- [n_rare, n_total] per
        4-way TAG_TO_BUCKET bucket, per-codon (not windowed).
      total_codons: int.
    """
    M = int(codon_idx.shape[0])
    is_rare_table = xp.asarray(_IS_RARE_TABLE)
    rare_flags = is_rare_table[codon_idx]  # (M,)

    n_isoforms = len(lengths)
    if n_isoforms:
        segment_ids = np.repeat(np.arange(n_isoforms), lengths)
        means = segment_mean(xp, rare_flags.astype(float), segment_ids, n_isoforms)
        means = means if xp is np else xp.asnumpy(means)
        usage_per_gene = means.tolist()
    else:
        usage_per_gene = []

    rare_codon_windows = {i: 0 for i in range(winsize + 1)}
    max_start = M - winsize + 1
    if max_start > 0:
        valid = valid_window_mask(M, starts, lengths, winsize)
        if valid.any():
            windows = _sliding_window_view(xp, rare_flags, winsize, axis=0)[:max_start]  # (max_start, winsize)
            window_sums = windows.sum(axis=1)  # (max_start,)
            valid_arr = valid if xp is np else xp.asarray(valid)
            window_sums_valid = window_sums[valid_arr]
            hist = xp.bincount(window_sums_valid, minlength=winsize + 1)
            hist = hist if xp is np else xp.asnumpy(hist)
            for count, n in enumerate(hist[:winsize + 1].tolist()):
                rare_codon_windows[count] = n

    rare_flags_np = rare_flags if xp is np else xp.asnumpy(rare_flags)
    bucket_of_codon = np.fromiter(
        (_TAG_TO_BUCKET_INDEX[c] for c in loc_string), dtype=np.int64, count=len(loc_string),
    ) if loc_string else np.zeros(0, dtype=np.int64)
    n_total_by_bucket = np.bincount(bucket_of_codon, minlength=4)
    n_rare_by_bucket = np.bincount(bucket_of_codon, weights=rare_flags_np, minlength=4)
    rare_by_location = {
        name: [int(n_rare_by_bucket[i]), int(n_total_by_bucket[i])]
        for i, name in enumerate(_BUCKET_NAMES_4WAY)
    }

    return {
        'usage_per_gene': usage_per_gene,
        'rare_codon_windows': rare_codon_windows,
        'rare_by_location': rare_by_location,
        'total_codons': M,
    }


def count_rare_codons_for_chunk(xp, genes: list, organism: str = "human",
                                 winsize: int = 15, vram_fraction: float = 0.5) -> RareCodonAnalysis:
    """One in-memory chunk of NaturalGene objects. Batches isoforms into
    VRAM-aware sub-batches (same shape as gpu_kmer_count.count_kmers_for_chunk),
    accumulating usagePerGene (concatenated), rare_codon_windows (summed),
    rareCodonsByLocation (elementwise-summed), totalCodons (summed) across
    every sub-batch. Returns a RareCodonAnalysis for this chunk (a shard,
    not the final corpus-wide result -- baseline_rare_codon.finalize() still
    merges across chunks, unchanged)."""
    usage_per_gene = []
    rare_codon_windows = {i: 0 for i in range(winsize + 1)}
    rare_by_location = {name: [0, 0] for name in _BUCKET_NAMES_4WAY}
    total_codons = 0

    max_batch_codons = vram_aware_batch_size(
        xp, _estimate_bytes_per_codon(winsize), vram_fraction=vram_fraction,
        default_cpu_batch=_DEFAULT_CPU_BATCH_CODONS, min_batch=_MIN_BATCH_CODONS,
    )

    def _flush(batch):
        if not batch:
            return
        codon_idx, loc_string, starts, lengths = concat_isoform_batch(xp, batch)
        result = count_rare_codons_for_batch(xp, codon_idx, loc_string, starts, lengths, winsize)
        usage_per_gene.extend(result['usage_per_gene'])
        for count, n in result['rare_codon_windows'].items():
            rare_codon_windows[count] += n
        for name, (n_rare, n_total) in result['rare_by_location'].items():
            rare_by_location[name][0] += n_rare
            rare_by_location[name][1] += n_total
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

    return RareCodonAnalysis(
        organism=organism,
        # "local-sample", not "corpus-wide" -- matches
        # baseline.compute_rare_codon_analysis's own constant exactly (this
        # module computes the same thing for whatever `genes` it's given,
        # one chunk's worth), needed for test_chunked_finalize_matches_
        # monolithic's exact-equality check on this field to hold, since
        # baseline_rare_codon.finalize() carries shard[0]'s transcriptome
        # forward rather than overriding it (unlike baseline_kmer.finalize()).
        transcriptome="local-sample",
        totalCodons=total_codons,
        rareCodonsByLocation=rare_by_location,
        rare_codon_windows=rare_codon_windows,
        codonFreqsLit=dict(CODON_FREQS_LIT),
        usagePerGene=usage_per_gene,
    )
