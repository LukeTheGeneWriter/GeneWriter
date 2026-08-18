"""Shared, granularity-agnostic batching/aggregation primitives for the 5
corpus-side baseline-counting modules (gpu_kmer_count.py + the 4
gpu_<name>_count.py siblings it's the template for).

Extracted from gpu_kmer_count.py's original, already-verified batching
machinery: concatenating a chunk's ragged isoforms into one flat buffer per
kernel launch (VRAM-aware batch sizing, real GPU memory queried once per
chunk via cupy.cuda.Device().mem_info) instead of dispatching one tiny
launch per isoform. The concatenation/masking logic here only ever touches
array shapes and integer offsets -- it doesn't care whether a "unit" is a
nucleotide (kmer, gc) or a codon (rare_codon, codon_usage, codon_pair_bias),
which is what makes it shareable across all 5 modules unchanged.

NOT shared: gpu_kmer_count.py's own "stop one window short" (`M - k`) window
count convention. The other 4 modules use the full, never-drop-a-window
convention (`M - span + 1`) per an explicit correction -- kmer's own
convention is deliberately left alone (a separate, not-yet-decided follow-up,
see majority_raw_tag_bucket()'s docstring below and
memory/majority_vote_bucket_discrepancy.md for the related majority-vote
finding). valid_window_mask() below implements only the new/correct
never-drop convention; gpu_kmer_count.py keeps its own local, unchanged
"M - k" mask construction rather than calling this function.
"""

import numpy as np

from .codon_tables import TAG_TO_WINDOW_BUCKET

_WINDOW_BUCKET_NAMES = ('ExonL50', 'Exon', 'ExonR50')
_RAW_LOCATION_TAGS = ('F', 'T', 'I', 'S')


def select_backend(use_gpu: bool, label: str):
    """cupy-or-numpy resolution + device select + warn-and-fallback print,
    factored out of baseline_kmer._select_backend so all 5 baseline_<name>.py
    modules share one implementation instead of 5 near-identical copies.

    Import cupy only inside this function body -- load-bearing fork-safety
    invariant (see baseline_pipeline.py's own docstring): the parent process
    must never import cupy, since forking after CUDA init is a crash hazard.
    This function only ever executes already inside a forked child.

    label: feeds the fallback print message (e.g. "baseline_gc" ->
    "[baseline_gc] GPU unavailable (...) -- falling back to numpy...").
    """
    if not use_gpu:
        return np
    try:
        import cupy as xp
        xp.cuda.Device(0).use()
        return xp
    except Exception as e:
        print(f"[{label}] GPU unavailable ({e!r}) -- falling back to numpy for this chunk.")
        return np


def vram_aware_batch_size(xp, bytes_per_unit: int, vram_fraction: float = 0.5,
                           default_cpu_batch: int = 5_000_000, min_batch: int = 100_000) -> int:
    """Max total units (nucleotides or codons, whatever the caller's
    bytes_per_unit is priced in) to concatenate into one batched kernel-
    launch call. Generalized gpu_kmer_count._vram_aware_batch_size: the
    caller supplies its own bytes_per_unit estimate (module-specific -- e.g.
    kmer's own _estimate_bytes_per_nt(max(k_values))) instead of this
    function having any k-mer-specific knowledge.

    On the numpy backend there's no VRAM ceiling to respect, so this just
    returns default_cpu_batch. On cupy, queries *actually free* GPU memory
    (cupy.cuda.Device().mem_info -- a property, not a method) once and
    budgets a fraction of it (vram_fraction -- default 0.5, leaving headroom
    rather than planning to consume every free byte), so batch sizing adapts
    automatically to whatever GPU is attached (a T4's free memory looks
    nothing like an RTX 3050's or an A100's) instead of a fixed count that
    would either underutilize a big GPU or risk OOM on a small one.
    """
    if xp is np:
        return default_cpu_batch
    free_bytes, _total_bytes = xp.cuda.Device().mem_info  # property, not a method -- no ()
    budget = int(free_bytes * vram_fraction)
    return max(budget // bytes_per_unit, min_batch)


def concat_isoform_batch(xp, encoded_isoforms: list) -> tuple:
    """encoded_isoforms: list of (values, loc_string) pairs, each already
    produced by a module's own encode-one-isoform step (codon- or
    nucleotide-granularity, this function doesn't care which). Returns
    (flat_values, flat_loc, starts, lengths):
      flat_values: every isoform's array concatenated on `xp`, one array.
      flat_loc: every isoform's loc_string concatenated (plain Python str
        concat -- cheap; majority-vote helpers always run on plain
        numpy/Python regardless of xp).
      starts/lengths: plain Python lists (one entry per isoform in this
        batch -- small, never worth moving to the GPU), each isoform's own
        span within flat_values/flat_loc. Consumed by valid_window_mask()
        and by per-gene segment_mean() (via a parallel segment-id array the
        caller builds from these same starts/lengths).
    """
    lengths = [int(values.shape[0]) for values, _loc in encoded_isoforms]
    if lengths:
        starts = list(np.cumsum([0] + lengths[:-1]))
        flat_values = xp.concatenate([values for values, _loc in encoded_isoforms])
    else:
        starts = []
        flat_values = xp.zeros(0, dtype=xp.int64)
    flat_loc = ''.join(loc for _values, loc in encoded_isoforms)
    return flat_values, flat_loc, starts, lengths


def valid_window_mask(total_len: int, starts: list, lengths: list, span: int) -> np.ndarray:
    """Boolean mask over every flat window-start position in [0, total_len -
    span + 1) -- True only where a span-length window starting there falls
    entirely within ONE isoform's own [start, start+length) span, never
    crossing into a neighboring (unrelated) gene's concatenated data.

    Full/never-drop convention (`length - span + 1` windows per isoform, not
    `length - span`) -- see module docstring: this is the corrected
    convention every counting module except kmer uses. A pair (e.g.
    CodonPairBias's adjacent-codon-pair validity) is just span=2 in this
    framing.
    """
    max_start = total_len - span + 1
    if max_start <= 0:
        return np.zeros(0, dtype=bool)
    valid = np.zeros(max_start, dtype=bool)
    for start, length in zip(starts, lengths):
        n = length - span + 1
        if n > 0:
            valid[start:start + n] = True
    return valid


def segment_mean(xp, values, segment_ids, n_segments: int):
    """Per-gene reduce: mean of `values` grouped by `segment_ids` (one
    integer segment id per element of `values`, e.g. built from
    concat_isoform_batch()'s starts/lengths via
    np.repeat(np.arange(len(lengths)), lengths)). Implemented via
    bincount(weights=values)/bincount(counts) -- the idiom that's portable
    to cupy (ufunc.reduceat isn't reliably supported there), same pattern
    already used elsewhere in this codebase
    (batch_rare_codon_term's odds_by_count[window_sums] lookup).

    A batch never splits an isoform mid-span (concat_isoform_batch always
    appends one whole isoform at a time), so every segment's mean is always
    computable within a single call -- no cross-batch complexity; the
    resulting per-gene list across sub-batches/chunks is just list
    concatenation (baseline_shard_util.concat_lists).
    """
    segment_ids = segment_ids if xp is np else xp.asarray(segment_ids)
    sums = xp.bincount(segment_ids, weights=values, minlength=n_segments)
    counts = xp.bincount(segment_ids, minlength=n_segments)
    safe_counts = xp.where(counts > 0, counts, 1)
    return xp.where(counts > 0, sums / safe_counts, 0.0)


def _np_sliding_sum(bool_arr_1d, span):
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(bool_arr_1d.astype(np.int64), span).sum(axis=1)


def majority_raw_tag_bucket(location_string: str, span: int, num_windows: int) -> np.ndarray:
    """Majority location-bucket per window, matching baseline.py's ORIGINAL
    algorithm exactly (compute_gc_analysis/compute_kmer_analysis's
    `max(set(loc_window), key=loc_window.count)`) -- NOT a reuse of
    gpu_change_vector.majority_window_bucket(), and not equivalent to it.

    majority_window_bucket() folds every nucleotide's raw tag (F/T/I/S)
    through TAG_TO_WINDOW_BUCKET FIRST (merging S and I into the same
    'Exon' vote-bin), then votes among the 3 already-folded buckets. This
    function votes among the 4 RAW tags first, then folds only the WINNER
    through TAG_TO_WINDOW_BUCKET. These disagree whenever a window mixes S
    and I against an F/T plurality -- e.g. 5xS + 4xI + 6xF: this function
    picks F (6 is the largest raw count, no tie) -> ExonL50;
    majority_window_bucket() sums S+I=9 > F=6 -> Exon. A real, confirmed,
    non-tie divergence -- see memory/majority_vote_bucket_discrepancy.md for
    the full investigation (it also covers change_vector.py's per-individual
    _gc_term/_kmer_term, and gpu_kmer_count.py's counting side, which
    already uses majority_window_bucket() and so already disagrees with
    baseline.compute_kmer_analysis for this same reason -- not fixed there,
    out of scope; this function exists specifically so gpu_gc_count.py does
    NOT repeat that mistake).

    Always plain numpy (same reasoning as majority_window_bucket(): depends
    only on locvec, no batch/population dimension to move to the GPU).
    Returns an int array of shape (num_windows,), values indexing
    gpu_corpus_batch._WINDOW_BUCKET_NAMES. Assumes num_windows > 0 (callers
    already guard the num_windows == 0 case, matching majority_window_bucket's
    own convention).

    TIE-BREAKING CAVEAT, confirmed while building gpu_gc_count.py's tests:
    when two raw tags are EXACTLY tied for the window's max count (not just
    "close" -- a genuine equal count), this function's tie-break (numpy
    argmax's "first occurrence wins" over _RAW_LOCATION_TAGS's fixed F/T/I/S
    order) is NOT guaranteed to agree with baseline.py's own tie-break
    (`max(set(loc_window), key=loc_window.count)`, whose result depends on
    Python's set iteration order -- itself governed by string hashing, not
    any fixed rule; change_vector.py:470-474 has a matching comment making
    the same observation for a different, but analogous, tie-break). This is
    an inherent, irreducible ambiguity in baseline.py's own algorithm, not a
    bug introduced here -- no fixed-order re-implementation can guarantee
    bit-for-bit agreement in a genuine tie, only in a clear (non-tie)
    plurality, which is what this function's whole design otherwise
    guarantees exactly. In practice this only bites a small fraction of
    windows in real gene data (an exact count tie between two specific raw
    tags is a coincidence, not the common case) -- test_gpu_gc_count.py's
    integration test picks its synthetic tag run-lengths specifically to
    avoid manufacturing one, rather than working around it with a looser
    comparison.
    """
    tag_index = {tag: i for i, tag in enumerate(_RAW_LOCATION_TAGS)}
    tag_of_nt = np.fromiter(
        (tag_index[loc] for loc in location_string),
        dtype=np.int8, count=len(location_string),
    )
    counts = np.stack([
        _np_sliding_sum(tag_of_nt == t, span)[:num_windows]
        for t in range(len(_RAW_LOCATION_TAGS))
    ], axis=1)
    winning_raw_tag = counts.argmax(axis=1)

    bucket_index = {name: i for i, name in enumerate(_WINDOW_BUCKET_NAMES)}
    fold_table = np.asarray([bucket_index[TAG_TO_WINDOW_BUCKET[tag]] for tag in _RAW_LOCATION_TAGS])
    return fold_table[winning_raw_tag]
