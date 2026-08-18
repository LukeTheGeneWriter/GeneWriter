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

count_kmers_for_chunk() batches many isoforms into one flat cross-gene buffer
per kernel launch (see _concat_isoform_batch()/count_kmers_for_batch()) rather
than dispatching one tiny launch per isoform -- real GPU utilization needs
enough work per launch to be worth the fixed overhead, and a chunk's isoforms
are wildly ragged in length (a few hundred nt to a few hundred thousand),
which is exactly why concatenation (not padding to a common length, the usual
batching trick) is the right move here: no wasted compute on padding, and a
window is only ever counted if it falls entirely within one isoform's own
span (see count_kmers_for_batch's `valid` mask) so concatenation never
manufactures a bogus cross-gene k-mer at a join point.

Batch size (in total nucleotides, not isoform count -- isoform length varies
too much for a fixed isoform-count cap to bound memory usefully) is derived
from *actually free* GPU memory when running on cupy
(gpu_corpus_batch.vram_aware_batch_size() queries cupy.cuda.Device().mem_info()
directly, once per chunk) -- enough to batch aggressively for real
utilization without risking an OOM on whatever GPU tier happens to be
attached (T4 vs. an RTX 3050 vs. an A100 all report very different
free_bytes). Falls back to a large fixed default on the numpy backend, where
there's no VRAM ceiling to respect but batching is still worth it for fewer,
bigger Python-level dispatches.

concat_isoform_batch()/vram_aware_batch_size() are gpu_corpus_batch.py's
shared, granularity-agnostic versions (this module was their original
source, factored out 2026-08-19 when 4 sibling gpu_<name>_count.py modules
needed the same scaffolding at codon granularity). This module's OWN "stop
one window short" (`M - k`) window-count convention and its `valid`-mask
construction are deliberately NOT delegated to gpu_corpus_batch.valid_window_mask()
-- that function implements the different, "never drop a window" convention
the other 4 modules use; kmer's own convention is left alone on purpose (a
separate, not-yet-decided follow-up, see
memory/majority_vote_bucket_discrepancy.md).
"""

import numpy as np

from .codon_tables import NT_BASE4_BY_CODON_INDEX, encode_codons
from .gene_io import protein_coding_isoforms
from .gpu_change_vector import _sliding_window_view, majority_window_bucket
from .gpu_corpus_batch import concat_isoform_batch as _shared_concat_isoform_batch
from .gpu_corpus_batch import vram_aware_batch_size as _shared_vram_aware_batch_size

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
    long-standing convention inherited from the original notebook).

    Majority-bucket assignment (majority_window_bucket(), fold-then-vote)
    does NOT always agree with baseline.compute_kmer_analysis's own
    raw-tag-vote-then-fold algorithm -- they can disagree in real, non-tie
    cases whenever a window mixes 'S' and 'I' tags against an 'F'/'T'
    plurality (see memory/majority_vote_bucket_discrepancy.md). This
    module's own tests (tests/test_gpu_kmer_count.py) check against
    hand-counted expectations, not a direct comparison against
    baseline.compute_kmer_analysis -- there is currently no test asserting
    byte-for-byte agreement between the two (test_baseline_kmer.py uses
    pytest.approx specifically because of this)."""
    M = nt_base4.shape[0]
    num_wins = max(M - k, 0)
    counts = {name: xp.zeros(4 ** k, dtype=xp.int64) for name in _WINDOW_BUCKET_NAMES}
    totals = {name: 0 for name in _WINDOW_BUCKET_NAMES}
    if num_wins == 0:
        return counts, totals

    windows = _sliding_window_view(xp, nt_base4, k, axis=0)[:num_wins]  # (num_wins, k)
    powers = xp.asarray(np.asarray([4 ** (k - 1 - j) for j in range(k)], dtype=np.int64))
    codes = (windows * powers).sum(axis=1)  # (num_wins,), each in [0, 4**k)

    majority = majority_window_bucket(loc_string, k, num_wins)  # (num_wins,) always numpy -- fold-then-vote, see this function's own docstring
    majority = majority if xp is np else xp.asarray(majority)

    for b, name in enumerate(_WINDOW_BUCKET_NAMES):
        mask = majority == b
        n = int(mask.sum())
        if n == 0:
            continue
        counts[name] = counts[name] + xp.bincount(codes[mask], minlength=4 ** k)
        totals[name] = n

    return counts, totals


_DEFAULT_CPU_BATCH_NT = 5_000_000  # no VRAM ceiling on numpy; still batch for fewer, bigger dispatches
_MIN_BATCH_NT = 100_000  # floor so a very tight VRAM estimate never shrinks batches into per-isoform-again territory
_VRAM_SAFETY_FACTOR = 2  # doubles the raw per-nt estimate below to cover cupy allocator overhead/temporaries this doesn't explicitly account for



def _estimate_bytes_per_nt(max_k: int) -> int:
    """Conservative worst-case GPU bytes needed per nucleotide of batch
    content, for one count_kmers_for_batch() call at the largest k in play.

    Dominant term: `windows * powers` in count_kmers_for_batch materializes
    a (num_positions, k) int64 array (elementwise ops on a strided
    sliding-window VIEW still allocate a real, non-strided output) -- the
    biggest per-k intermediate, 8*max_k bytes/position. `codes` and the
    batch's own `flat_nt` buffer each add another 8 bytes/position;
    majority/valid arrays are smaller-dtype and not double-counted here.
    Doubled by _VRAM_SAFETY_FACTOR as a margin for whatever this rough
    accounting misses (allocator fragmentation, other temporaries) --
    better to under-batch a little than risk a real OOM on whatever GPU
    tier happens to be attached.
    """
    return _VRAM_SAFETY_FACTOR * (8 * max_k + 8 + 8)


def _vram_aware_batch_size(xp, k_values, vram_fraction: float = 0.5) -> int:
    """Max total nucleotides to concatenate into one count_kmers_for_batch()
    call. Thin wrapper over the shared gpu_corpus_batch.vram_aware_batch_size()
    -- this function's own job is just translating k_values into the
    k-mer-specific bytes_per_nt estimate (_estimate_bytes_per_nt()); the
    actual VRAM query / fixed-default / floor logic all lives in the shared
    function now (see module docstring)."""
    return _shared_vram_aware_batch_size(
        xp, _estimate_bytes_per_nt(max(k_values)), vram_fraction=vram_fraction,
        default_cpu_batch=_DEFAULT_CPU_BATCH_NT, min_batch=_MIN_BATCH_NT,
    )


def _concat_isoform_batch(xp, encoded_isoforms: list) -> tuple:
    """encoded_isoforms: list of (nt_base4, loc_string) pairs, each already
    produced by encode_isoform_nt_base4(). Thin wrapper over the shared
    gpu_corpus_batch.concat_isoform_batch() -- see module docstring. Kept as
    a local name (rather than importing concat_isoform_batch directly at
    every call site) so tests/test_gpu_kmer_count.py's existing references
    to this name don't need to change."""
    return _shared_concat_isoform_batch(xp, encoded_isoforms)


def count_kmers_for_batch(xp, flat_nt, flat_loc: str, starts: list, lengths: list, k: int) -> tuple:
    """Batched counterpart of count_kmers_for_isoform(): counts every valid
    k-mer window across MANY isoforms concatenated into flat_nt/flat_loc, in
    one kernel launch instead of one per isoform. starts/lengths (plain
    Python, one pair per isoform -- see _concat_isoform_batch()) mark each
    isoform's own span; a window is only counted if it falls entirely
    within a single isoform's own valid range (the same `max(length - k, 0)`
    "stop one window short" convention count_kmers_for_isoform uses,
    applied per-isoform here via an explicit boolean mask -- concatenation
    means a window position near a join point would otherwise look like a
    perfectly ordinary in-range window straddling two unrelated genes'
    sequences, which the mask exists specifically to exclude).

    Returns (counts, totals), same shape as count_kmers_for_isoform().
    """
    M = int(flat_nt.shape[0])
    counts = {name: xp.zeros(4 ** k, dtype=xp.int64) for name in _WINDOW_BUCKET_NAMES}
    totals = {name: 0 for name in _WINDOW_BUCKET_NAMES}
    max_start = M - k  # upper bound over every flat window start position (includes invalid, cross-isoform ones)
    if max_start <= 0:
        return counts, totals

    valid = np.zeros(max_start, dtype=bool)
    for start, length in zip(starts, lengths):
        n = length - k
        if n > 0:
            valid[start:start + n] = True
    if not valid.any():
        return counts, totals

    windows = _sliding_window_view(xp, flat_nt, k, axis=0)[:max_start]  # (max_start, k)
    powers = xp.asarray(np.asarray([4 ** (k - 1 - j) for j in range(k)], dtype=np.int64))
    codes = (windows * powers).sum(axis=1)  # (max_start,)

    majority = majority_window_bucket(flat_loc, k, max_start)  # (max_start,) always numpy -- fold-then-vote, see count_kmers_for_isoform's docstring
    valid_arr = valid if xp is np else xp.asarray(valid)
    majority = majority if xp is np else xp.asarray(majority)

    for b, name in enumerate(_WINDOW_BUCKET_NAMES):
        mask = valid_arr & (majority == b)
        n = int(mask.sum())
        if n == 0:
            continue
        counts[name] = counts[name] + xp.bincount(codes[mask], minlength=4 ** k)
        totals[name] = n

    return counts, totals


def count_kmers_for_chunk(xp, genes: list, k_values=range(2, 11), vram_fraction: float = 0.5) -> tuple:
    """One in-memory chunk of NaturalGene objects, every k in k_values.
    Loops isoforms via gene_io.protein_coding_isoforms() (the same
    "skip computationally-predicted-only transcripts, skip empty codon
    streams" rule every baseline_*.py test already uses), grouping them into
    VRAM-aware sub-batches (_vram_aware_batch_size()) -- each sub-batch gets
    ONE count_kmers_for_batch() call per k instead of one per isoform, for
    real GPU utilization instead of many tiny launches.

    A single isoform larger than the whole batch budget (e.g. titin's CDS,
    the one real outlier gene long enough to matter here) still just gets
    its own one-isoform batch -- no worse than the old strictly-per-isoform
    behavior for that rare case, and every other isoform still batches
    normally around it.

    Returns (raw_counts, loc_totals):
      raw_counts[k][bucket_name] -> (4**k,) int64 array
      loc_totals[k][bucket_name] -> int
    Callers (baseline_kmer.compute_and_write_shard) persist these as one
    chunk's shard; corpus-wide fold_enrich is only computed once, in
    baseline_kmer.finalize(), from the sum of every chunk's raw counts."""
    raw = {k: {name: xp.zeros(4 ** k, dtype=xp.int64) for name in _WINDOW_BUCKET_NAMES} for k in k_values}
    loc_totals = {k: {name: 0 for name in _WINDOW_BUCKET_NAMES} for k in k_values}
    max_batch_nt = _vram_aware_batch_size(xp, k_values, vram_fraction=vram_fraction)

    def _flush(batch):
        if not batch:
            return
        flat_nt, flat_loc, starts, lengths = _concat_isoform_batch(xp, batch)
        for k in k_values:
            counts, totals = count_kmers_for_batch(xp, flat_nt, flat_loc, starts, lengths, k)
            for name in _WINDOW_BUCKET_NAMES:
                raw[k][name] = raw[k][name] + counts[name]
                loc_totals[k][name] += totals[name]

    batch = []
    batch_nt = 0
    for _gene, iso in protein_coding_isoforms(genes):
        encoded = encode_isoform_nt_base4(xp, iso)
        n = int(encoded[0].shape[0])
        if batch and batch_nt + n > max_batch_nt:
            _flush(batch)
            batch = []
            batch_nt = 0
        batch.append(encoded)
        batch_nt += n
    _flush(batch)

    return raw, loc_totals
