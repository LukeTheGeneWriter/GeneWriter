"""Population-batched change-vector computation: the whole GA population's
change vectors computed in one shot, instead of Python looping
calculate_change_vector() once per individual.

This is the "axis 1" parallelism from the CUDA discussion: every
individual's change vector is independent of every other's (same
read-only baseline data, no cross-individual dependency), so it's an
embarrassingly parallel batch of identical computations -- exactly the
shape GPU array libraries are built for. Every function here is written
against an injected array module `xp` (numpy or cupy) rather than hardcoding
either, so the *same code* runs as a batched-but-CPU numpy implementation
(fast to test, no GPU required) or genuinely on the GPU by passing
`xp=cupy` -- see batch_calculate_change_vectors().

Correctness oracle: change_vector.py's per-individual calculate_change_vector,
itself checked against the pre-vectorization original in
tests/test_change_vector_reference.py. tests/test_gpu_change_vector.py runs
this module's numpy-backend output through the same population and checks
it matches calculate_change_vector() call-by-call, then (when a GPU is
available) checks the cupy backend matches the numpy backend.

Population representation: sequences are encoded as a 2D integer array of
codon indices (population_size x sequence_length), matching the SoA layout
Handoff.md's original proposed encoding.py module called for -- see
codon_tables.encode_codons()/decode_codons(). A GPU kernel can't operate on
Python strings, so this conversion happens once at the batch boundary, not
per-term.

Kmer is now batched too (batch_kmer_term()): its substring->score lookup
was a hash-map operation over arbitrary-length k-mer strings, not a natural
fit for array vectorization -- fixed by re-encoding every k-mer window as a
base-4 integer (A/C/G/T -> 0..3, codon_tables.NT_TO_BASE4) so the baseline's
fold_enrich scores become one flat (num_buckets, 4**k) array, and the
per-window lookup becomes array indexing (`table[bucket, code]`) instead of
a dict lookup -- see batch_kmer_term()'s docstring for the full design and
its memory-scaling caveat at large k.
"""

import numpy as np

from .change_vector import cached_normal_transform, cached_stat, calculate_change_vector, registered_terms
from .codon_tables import (
    AA_CODONS,
    CODON_FREQ_BY_INDEX,
    CODON_LIST,
    CODON_TO_AA,
    CODON_TO_INDEX,
    GC_FLAGS_BY_INDEX,
    NT_BASE4_BY_CODON_INDEX,
    NT_TO_BASE4,
    RARE_CODON_INDICES,
    TAG_TO_WINDOW_BUCKET,
    VARIABLE_FLAGS_BY_INDEX,
    encode_codons,
)

_WINDOW_BUCKET_NAMES = ('ExonL50', 'Exon', 'ExonR50')

# The only terms this module has a hand-written batched (xp-array)
# implementation for. Unlike change_vector.py's @register_term registry,
# there is no generic way to "batch" an arbitrary custom term -- each one
# needs its own array-based reimplementation, written here by hand. Any
# registered term NOT in this set (e.g. change_vector.Uracil, added without
# ever touching this file) is still computed correctly by
# batch_calculate_change_vectors() -- see its per-individual fallback loop
# -- just not accelerated.
_NATIVELY_BATCHED_TERMS = frozenset({'RareCodons', 'CodonUsage', 'CodonPairBias', 'GC', 'Kmer'})

# Per-codon-index "best synonym frequency for this codon's amino acid" --
# the batched counterpart of change_vector.dist_from_optimal(), which only
# depends on the amino acid, not the specific codon (see its per-individual
# use in change_vector._codon_usage_term for the same observation).
_BEST_FREQ_BY_AA = {aa: max(CODON_FREQ_BY_INDEX[CODON_TO_INDEX[c]] for c in codons) for aa, codons in AA_CODONS.items()}
BEST_FREQ_BY_INDEX = [_BEST_FREQ_BY_AA[CODON_TO_AA[codon]] for codon in CODON_LIST]


def _sliding_window_view(xp, arr, window, axis):
    if xp is np:
        from numpy.lib.stride_tricks import sliding_window_view
    else:
        from cupy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(arr, window, axis=axis)


def windowed_average_batched(xp, values, target_len: int, winsize: int):
    """Batched counterpart of change_vector._windowed_average: values has
    shape (P, M) (M scores per individual), returns shape (P, target_len).

    Regime classification (which positions get the fixed-radius fast path
    vs. the boundary/clipped fallback) depends only on position index, not
    on the batch -- computed once and applied across every individual, same
    principle as the per-individual version's hybrid vectorized/fallback
    split (see its docstring for why sliding_window_view rather than
    cumsum: avoids a subtraction-of-large-sums rounding risk that would
    undermine diff_change_vector's exact-locality guarantee for some
    terms)."""
    P, M = values.shape
    half = max(winsize // 2, 1)
    if M == 0 or target_len == 0:
        return xp.zeros((P, target_len), dtype=float)

    idx = np.arange(target_len)
    regime_a = idx < winsize
    regime_b = (~regime_a) & (idx > target_len - winsize)
    regime_c = ~(regime_a | regime_b)
    safe = regime_c & (idx - half >= 0) & (idx + half <= M) & (M >= 2 * half)

    out = xp.zeros((P, target_len), dtype=float)

    if safe.any():
        windows = _sliding_window_view(xp, values, 2 * half, axis=1)  # (P, M-2half+1, 2half)
        safe_idx = idx[safe]
        row_starts = safe_idx - half
        # safe_idx is always a single contiguous ascending run: regime_c is
        # itself a contiguous middle band, and the extra (idx-half>=0) /
        # (idx+half<=M) conditions are just tighter monotonic bounds on the
        # same idx array -- intersecting monotonic bounds can only narrow
        # an interval, never split it. That makes row_starts contiguous
        # too, so this can be a plain slice into `windows` (a view, no
        # allocation) instead of fancy-indexing it -- fancy-indexing here
        # forces materializing a full (P, len(row_starts), 2*half) copy,
        # confirmed as the exact cause of a real CUDA OOM on real Colab
        # data (population=50,000, a 758-residue gene): the failed
        # allocation was 50000*2238*14*8 == 12,532,800,000 bytes, an exact
        # match for that shape.
        lo, hi = int(row_starts[0]), int(row_starts[-1]) + 1
        means = windows[:, lo:hi, :].mean(axis=2)
        out[:, safe_idx[0]:safe_idx[-1] + 1] = means

    for i in idx[~safe].tolist():
        if i < winsize:
            start, end = 0, i
        elif i > target_len - winsize:
            start, end = i, M
        else:
            start, end = max(i - half, 0), min(i + half, M)
        if end > start:
            out[:, i] = values[:, start:end].mean(axis=1)

    return out


def encode_population(xp, pop_codons: list):
    """pop_codons: list of P individuals, each a list of N codon strings
    (all the same length N -- same amino acid sequence). Returns a (P, N)
    int array of codon indices on the given backend."""
    encoded = np.asarray([encode_codons(sol) for sol in pop_codons], dtype=np.int16)
    return encoded if xp is np else xp.asarray(encoded)


def batch_rare_codon_term(xp, codon_idx, rc, winsize: int = 15):
    """codon_idx: (P, N) int array. rc: RareCodonAnalysis. Returns (P, N)."""
    P, N = codon_idx.shape
    rare_mask = np.zeros(len(CODON_LIST), dtype=np.int64)
    rare_mask[list(RARE_CODON_INDICES)] = 1
    rare_mask = xp.asarray(rare_mask)
    rvec = rare_mask[codon_idx]  # (P, N)

    num_windows = max(N - winsize, 0)
    if num_windows > 0:
        window_sums = _sliding_window_view(xp, rvec, winsize, axis=1)[:, :num_windows].sum(axis=2)
    else:
        window_sums = xp.zeros((P, 0), dtype=xp.int64)

    # Cached per (rc, winsize) -- same key change_vector._rare_codon_term
    # uses, shared cache -- see cached_stat()'s docstring for why this
    # matters at real-scale baseline sizes.
    def _build_odds_by_count():
        total_windows = sum(rc.rare_codon_windows.values())
        table = np.full(winsize + 1, np.inf)
        for count, n in rc.rare_codon_windows.items():
            if 0 <= count <= winsize:
                table[count] = np.inf if n == 0 else total_windows / n
        return table
    odds_by_count = xp.asarray(cached_stat(rc, ('odds_by_count', winsize), _build_odds_by_count))
    scorewins = odds_by_count[window_sums] if window_sums.shape[1] else xp.zeros((P, 0), dtype=float)

    scorevec = windowed_average_batched(xp, scorewins, N, winsize)  # (P, N)

    overall_rc = rvec.sum(axis=1) / N  # (P,)
    usage_transform = cached_normal_transform(rc, 'usagePerGene', rc.usagePerGene)
    zscore = usage_transform.transform_array(overall_rc, xp=xp) ** 2  # (P,)

    # rvec gates the signal to rare-codon positions only -- select 0 rather
    # than multiply by the gate (inf * 0 == NaN, see the per-individual
    # term's identical guard in change_vector.py).
    return xp.where(rvec.astype(bool), scorevec * zscore.reshape(P, 1), 0.0)


def batch_codon_usage_term(xp, codon_idx, ca, winsize: int = 15):
    """codon_idx: (P, N) int array. ca: CodonAnalysis. Returns (P, N)."""
    P, N = codon_idx.shape
    freq_table = xp.asarray(np.asarray(CODON_FREQ_BY_INDEX, dtype=float))
    best_freq_table = xp.asarray(np.asarray(BEST_FREQ_BY_INDEX, dtype=float))
    cuvec = freq_table[codon_idx]  # (P, N)
    best_freq = best_freq_table[codon_idx]  # (P, N)

    span = max(ca.windowsize - 1, 1)
    num_windows = max(N - ca.windowsize, 0)
    if num_windows > 0:
        cuwinscores = _sliding_window_view(xp, cuvec, span, axis=1)[:, :num_windows].sum(axis=2) / ca.windowsize
        cudists = _sliding_window_view(xp, best_freq, span, axis=1)[:, :num_windows].mean(axis=2)
    else:
        cuwinscores = xp.zeros((P, 0), dtype=float)
        cudists = xp.zeros((P, 0), dtype=float)

    dist_transform = cached_normal_transform(ca, 'windowdistancesfromoptimal', ca.windowdistancesfromoptimal)
    score_transform = cached_normal_transform(ca, 'windowscores', ca.windowscores)
    dist_vals = windowed_average_batched(xp, cudists, N, winsize)
    score_vals = windowed_average_batched(xp, cuwinscores, N, winsize)
    # transform_array() always returns a real xp array shaped like its
    # input (the degenerate-baseline case still returns an all-zero array
    # of that shape, not a bare Python 0.0), so anything feeding back into
    # windowed_average_batched() below always gets a real array with a
    # .shape -- see distribution_fit.FittedDistribution's docstring.
    dist_zs = dist_transform.transform_array(dist_vals, xp=xp) ** 2
    score_zs = score_transform.transform_array(score_vals, xp=xp) ** 2

    gene_transform = cached_normal_transform(ca, 'codonUsageScoreByGene', ca.codonUsageScoreByGene)
    straight_z = gene_transform.transform_array(cuvec, xp=xp) / 2

    return straight_z * score_zs + straight_z * dist_zs


def batch_codon_pair_bias_term(xp, codon_idx, cpb):
    """codon_idx: (P, N) int array. cpb: CodonPairBiasAnalysis. Returns (P, N).

    Batched counterpart of change_vector._codon_pair_bias_term() -- see its
    docstring for the local-distance/global-multiplier design and why it
    replaced the old windowed-z-times-raw-score formula. Same two layers
    here, vectorized across the population (P) dimension."""
    P, N = codon_idx.shape
    if N < 2:
        return xp.zeros((P, N), dtype=float)

    def _build_cpb_table():
        num_codons = len(CODON_LIST)
        table = np.zeros((num_codons, num_codons), dtype=float)
        for pair, score in cpb.cpb_lit.items():
            i, j = CODON_TO_INDEX[pair[:3]], CODON_TO_INDEX[pair[3:]]
            table[i, j] = score
        return table
    cpb_table_np = cached_stat(cpb, 'cpb_table', _build_cpb_table)
    cpb_table = xp.asarray(cpb_table_np)

    pair_scores = cpb_table[codon_idx[:, :-1], codon_idx[:, 1:]]  # (P, N-1), raw favorability, high=good

    # Local layer: distance from optimal (see change_vector._codon_pair_
    # bias_term's docstring) -- cpb_lit is a fixed literal table, so its max
    # is cached once rather than rescanned every call.
    max_score = cached_stat(cpb, 'cpb_max_score', lambda: max(cpb.cpb_lit.values()))
    pair_distance = max_score - pair_scores  # (P, N-1)

    pair_change = xp.zeros((P, N), dtype=float)
    pair_change[:, 0] = pair_distance[:, 0]
    pair_change[:, -1] = pair_distance[:, -1]
    if N > 2:
        pair_change[:, 1:-1] = (pair_distance[:, :-1] + pair_distance[:, 1:]) / 2

    # Global multiplier: how far BELOW cpbPerGene's mean (natural genes'
    # own per-gene average pair scores) this candidate's own average sits
    # -- asymmetric on purpose (see change_vector._codon_pair_bias_term's
    # matching comment): under-mean CPB amplifies local mutation pressure,
    # above-mean doesn't, since over-optimizing CPB specifically is already
    # checked by the change vector's overall term interplay, not by this
    # term punishing "too good." Clamping z to <=0 before squaring means
    # at/above the mean the multiplier is exactly 1.0 (neutral, not zero).
    # overall_cpb matches cpbPerGene's own construction exactly (mean of a
    # gene's own pair scores).
    overall_cpb = pair_scores.mean(axis=1)  # (P,)
    seq_transform = cached_normal_transform(cpb, 'cpbPerGene', cpb.cpbPerGene)
    below_mean_z = xp.minimum(seq_transform.transform_array(overall_cpb, xp=xp), 0.0)  # (P,)
    global_multiplier = 1.0 + below_mean_z ** 2  # (P,)

    return pair_change * global_multiplier.reshape(P, 1)


def batch_gc_term(xp, codon_idx, gc, locvec: list, winsize: int = 15):
    """codon_idx: (P, N) int array. gc: GCAnalysis. locvec: list[str] of
    length N, shared across the whole batch (same candidate sequence
    length/gene body for every individual in one calculate_change_vector-
    style call). Returns (P, N)."""
    P, N = codon_idx.shape
    gc_flags = xp.asarray(np.asarray(GC_FLAGS_BY_INDEX, dtype=float))  # (64, 3), 1=G/C, normal polarity
    per_pos_gc = gc_flags[codon_idx]  # (P, N, 3)

    # Per-SEQUENCE GC deviation -- one scalar per individual, added (not
    # multiplied) to every position's local windowed score below. Replaces
    # the old gc1/gc2/gc3-by-bucket signal, which was a whole-candidate
    # aggregate applied identically to every position sharing a bucket
    # (not a real per-position signal) and barely moved when scoring
    # synonymous alternatives at a single position -- see
    # change_vector._gc_term's matching comment for the full reasoning.
    # overall_gc matches GCAnalysis.gcPerGene's own construction exactly
    # (baseline.py: gc_bases / (3 * len(codons)), same normal polarity).
    overall_gc = per_pos_gc.reshape(P, N * 3).mean(axis=1)  # (P,)
    seq_transform = cached_normal_transform(gc, 'gcPerGene', gc.gcPerGene)
    seq_deviation = seq_transform.transform_array(overall_gc, xp=xp) ** 2  # (P,)

    # A position is only worth mutating for GC purposes if some synonymous
    # codon actually differs at some base there -- see change_vector._gc_
    # term's matching comment.
    variable_flags = xp.asarray(np.asarray(VARIABLE_FLAGS_BY_INDEX, dtype=float))  # (64, 3)
    per_pos_var = variable_flags[codon_idx]  # (P, N, 3)
    mutable = (per_pos_var.sum(axis=2) > 0).astype(float)  # (P, N)

    # Flatten to per-nucleotide (P, 3N) to match the window span (in
    # nucleotides) the baseline was computed over.
    nt_gc_flat = per_pos_gc.reshape(P, N * 3)

    location_string = ''.join(loc * 3 for loc in locvec)  # length 3N, shared
    span = max(gc.windowsize, 1)
    num_windows = max(len(location_string) - span, 0)

    if num_windows > 0:
        # Fold-then-vote convention -- NOT baseline.compute_gc_analysis's
        # raw-tag-vote-then-fold. See majority_window_bucket()'s own
        # docstring and memory/majority_vote_bucket_discrepancy.md.
        majority_bucket = majority_window_bucket(location_string, span, num_windows)

        window_gc = _sliding_window_view(xp, nt_gc_flat, span, axis=1)[:, :num_windows].mean(axis=2)  # (P, num_windows)

        win_z = xp.zeros((P, num_windows), dtype=float)
        for b, name in enumerate(_WINDOW_BUCKET_NAMES):
            mask = majority_bucket == b
            if not mask.any():
                continue
            transform = cached_normal_transform(gc, ('windows', name), gc.windows[name])
            cols = np.asarray(np.nonzero(mask)[0])
            val = window_gc[:, cols]
            win_z[:, cols] = transform.transform_array(val, xp=xp) ** 2
    else:
        win_z = xp.zeros((P, 0), dtype=float)

    win_z_by_pos = windowed_average_batched(xp, win_z, N * 3, winsize)  # (P, 3N)
    per_codon = win_z_by_pos.reshape(P, N, 3).mean(axis=2)  # (P, N)

    return mutable * (per_codon + seq_deviation.reshape(P, 1))


def _np_sliding_sum(bool_arr_1d, span):
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(bool_arr_1d.astype(np.int64), span).sum(axis=1)


def majority_window_bucket(location_string: str, span: int, num_windows: int):
    """Majority location-bucket per window (3-way argmax over per-bucket
    windowed counts) -- shared by batch_gc_term() and batch_kmer_term()
    (and, outside this module, gpu_kmer_count.py's counting side, which
    needs the identical "which of ExonL50/Exon/ExonR50 does this window
    mostly fall in" logic for the same location_string/span shape -- not
    underscore-prefixed for that reason). Always plain numpy regardless of
    the caller's xp: this depends only on locvec (one string, shared by the
    whole batch), not on population size, so there's no batch dimension to
    move to the GPU here -- computed once per call and reused across every
    individual via the callers' own indexing.

    FOLD-THEN-VOTE, not baseline.py's raw-tag-vote-then-fold: folds every
    nucleotide's raw tag (F/T/I/S) through TAG_TO_WINDOW_BUCKET FIRST
    (merging S and I into the same 'Exon' vote-bin), then votes among the 3
    already-folded buckets. baseline.compute_gc_analysis()/
    compute_kmer_analysis() vote among the 4 RAW tags first, then fold only
    the winner -- a genuinely different algorithm (not just a different
    tie-break), confirmed to disagree in real, non-tie cases whenever a
    window mixes S and I against an F/T plurality. gpu_kmer_count.py's
    counting side already calls this function and so already disagrees with
    its own stated baseline.compute_kmer_analysis oracle for this reason --
    not fixed, out of scope; gpu_gc_count.py deliberately does NOT call this
    function for the same reason (uses
    gpu_corpus_batch.majority_raw_tag_bucket() instead). See
    memory/majority_vote_bucket_discrepancy.md for the full investigation.

    Returns an int array of shape (num_windows,), values indexing
    _WINDOW_BUCKET_NAMES. Assumes num_windows > 0 (callers already guard
    the num_windows == 0 case before calling)."""
    bucket_index = {name: i for i, name in enumerate(_WINDOW_BUCKET_NAMES)}
    bucket_of_nt = np.fromiter(
        (bucket_index[TAG_TO_WINDOW_BUCKET[loc]] for loc in location_string),
        dtype=np.int8, count=len(location_string),
    )
    counts = np.stack([
        _np_sliding_sum(bucket_of_nt == b, span)[:num_windows]
        for b in range(len(_WINDOW_BUCKET_NAMES))
    ], axis=1)
    return counts.argmax(axis=1)


def _encode_kmer_string(seq: str, k: int) -> int:
    """A k-length A/C/G/T string -> its base-4 integer in [0, 4**k), using
    codon_tables.NT_TO_BASE4 -- must stay the same mapping
    batch_kmer_term() uses to encode candidate windows, since this is only
    ever used to build that function's lookup table from the same kind of
    string the candidates are encoded from."""
    code = 0
    for base in seq:
        code = code * 4 + NT_TO_BASE4[base]
    return code


def _build_kmer_score_table(kmer_by_seq: dict, k: int) -> np.ndarray:
    """(num_buckets, 4**k) float array of fold_enrich scores for one k,
    built once per batch_kmer_term() call (not per individual -- same cost
    class as batch_codon_pair_bias_term() building its cpb_table from a
    dict). Default 1.0 for a k-mer/bucket combination never observed in the
    baseline, matching _kmer_term's identical `entry is None` /
    `.get(bucket, {}).get('fold_enrich', 1.0)` defaults exactly."""
    table = np.ones((len(_WINDOW_BUCKET_NAMES), 4 ** k), dtype=float)
    for seq, buckets in kmer_by_seq.items():
        idx = _encode_kmer_string(seq, k)
        for b, name in enumerate(_WINDOW_BUCKET_NAMES):
            entry = buckets.get(name)
            if entry is not None:
                table[b, idx] = entry.get('fold_enrich', 1.0)
    return table


def batch_kmer_term(xp, codon_idx, kmer, locvec: list, winsize: int = 15):
    """codon_idx: (P, N) int array. kmer: KmerAnalysis. locvec: list[str] of
    length N, shared across the whole batch. Returns (P, N).

    The one piece of calculate_change_vector() that used to resist batching
    entirely (see this module's own history / docstring): _kmer_term's
    substring->score lookup is keyed by an arbitrary-length k-mer string,
    which doesn't fit a numeric array lookup the way every other term's
    baseline data does. Fixed here by re-encoding every k-mer window as a
    base-4 integer (A/C/G/T -> 0..3, codon_tables.NT_TO_BASE4/
    NT_BASE4_BY_CODON_INDEX) so the baseline's fold_enrich scores become one
    flat (num_buckets, 4**k) table (_build_kmer_score_table()) and the
    per-window score becomes one `table[bucket, code]` fancy-index gather
    across the *whole batch at once*, instead of a per-window,
    per-individual Python dict lookup.

    Memory scales as 4**k per k value in kmer.kmer_dict -- fine for the k
    range real Standards baselines actually use (up to 9-10, see
    Handoff.md's environment notes on how long a genome-wide k=9 baseline
    takes to compute in the first place): 4**10 * 3 buckets * 8 bytes is
    ~25MB, built once per call. This would stop being fine well before
    k=15 (4**15 * 3 * 8 bytes ~= 400GB) -- not a real concern for any
    baseline this codebase can currently produce, but worth knowing if a
    much larger k is ever considered.
    """
    P, N = codon_idx.shape
    base4_table = xp.asarray(np.asarray(NT_BASE4_BY_CODON_INDEX, dtype=np.int64))  # (64, 3)
    nt_base4 = base4_table[codon_idx].reshape(P, N * 3)  # (P, 3N)

    location_string = ''.join(loc * 3 for loc in locvec)  # length 3N, shared
    M = N * 3

    per_nt_total = xp.zeros((P, M), dtype=float)
    for k_str, kmer_by_seq in kmer.kmer_dict.items():
        k = int(k_str)
        num_wins = max(M - k, 0)
        if num_wins == 0:
            continue

        windows = _sliding_window_view(xp, nt_base4, k, axis=1)[:, :num_wins]  # (P, num_wins, k)
        powers = xp.asarray(np.asarray([4 ** (k - 1 - j) for j in range(k)], dtype=np.int64))
        codes = (windows * powers).sum(axis=2)  # (P, num_wins), each in [0, 4**k)

        # Fold-then-vote convention -- NOT baseline.compute_kmer_analysis's
        # raw-tag-vote-then-fold. See majority_window_bucket()'s own
        # docstring and memory/majority_vote_bucket_discrepancy.md.
        majority_bucket = majority_window_bucket(location_string, k, num_wins)  # (num_wins,) shared
        bucket_rows = xp.asarray(majority_bucket)

        # Cached per (kmer, k) -- see this function's docstring for the
        # per-k table-build cost, and cached_stat()'s docstring for why
        # rebuilding it from kmer_by_seq on every call was worth avoiding.
        table_np = cached_stat(kmer, ('kmer_score_table', k_str), lambda kbs=kmer_by_seq, kk=k: _build_kmer_score_table(kbs, kk))
        table = xp.asarray(table_np)  # (num_buckets, 4**k)
        win_scores = table[xp.broadcast_to(bucket_rows, codes.shape), codes]  # (P, num_wins)

        overall = win_scores.mean(axis=1)  # (P,)
        by_pos = windowed_average_batched(xp, win_scores, M, winsize)  # (P, M)
        # Summing each k's per-nucleotide contribution before the final
        # reshape-to-per-codon-and-mean (done once, after this loop) is
        # equivalent to _kmer_term's per-k reshape-then-sum-at-the-end:
        # reshape+mean is linear, so it commutes with the sum over k.
        per_nt_total += by_pos * overall.reshape(P, 1)

    return per_nt_total.reshape(P, N, 3).mean(axis=2)


def _estimate_scoring_bytes_per_individual(seq_len_nt: int, max_k: int, safety_factor: int = 2) -> int:
    """Conservative worst-case GPU bytes needed per POPULATION ROW (one
    individual) for one batch_calculate_change_vectors() call -- dominated
    by batch_kmer_term()'s own per-k `windows * powers` intermediate (see
    that function's docstring): a (num_wins, max_k) int64 array per row,
    num_wins ~= seq_len_nt - max_k. Same accounting shape as
    gpu_kmer_count._estimate_bytes_per_nt() (that module's counting-side
    counterpart), just priced per-individual here instead of per-corpus-
    nucleotide -- P (population size) is this module's batch dimension,
    not a flattened multi-gene nucleotide count. Doubled by safety_factor
    for the same reason that function doubles: codes/majority/table-lookup
    intermediates this rough accounting doesn't explicitly price, allocator
    fragmentation, etc. -- better to under-batch a little than risk a real
    OOM on whatever GPU tier happens to be attached."""
    num_wins = max(seq_len_nt - max_k, 1)
    return safety_factor * 8 * num_wins * max_k


def suggest_chunk_size(xp, aa_seq_len: int, kmer_k_values=range(2, 11), vram_fraction: float = 0.5,
                        default_cpu_chunk: int = 50_000, min_chunk: int = 100) -> int:
    """VRAM-aware default for batch_calculate_change_vectors()'s chunk_size
    -- gpu_corpus_batch.vram_aware_batch_size()'s exact free-memory-query
    pattern (already proven on the baseline-counting side, see
    gpu_kmer_count.py), applied here to the GA-SCORING side instead:
    queries *actually free* GPU memory once (xp.cuda.Device().mem_info,
    a property not a method) and budgets a fraction of it, so a bigger
    GPU automatically gets a bigger chunk_size (fewer, bigger batched
    calls -- higher throughput, no manual tuning) instead of a fixed
    constant that either underutilizes a big GPU or risks the exact real
    OOM this codebase already hit once on a small one (see
    memory/colab_stress_test_stale_paste_oom.md) -- chunk_size=None used
    to mean "one unchunked batch across the whole population," which is
    the shape that OOM'd; run_schedule() now calls this instead of leaving
    chunk_size at that unbounded default whenever xp is set.

    aa_seq_len: codon-sequence length (residues), NOT nucleotides -- this
    function itself converts (aa_seq_len * 3) before pricing.
    On the numpy backend there's no VRAM ceiling to respect, so this just
    returns default_cpu_chunk (still worth batching for fewer, bigger
    Python-level dispatches, same reasoning as gpu_corpus_batch's own
    default_cpu_batch)."""
    if xp is np:
        return default_cpu_chunk
    free_bytes, _total_bytes = xp.cuda.Device().mem_info  # property, not a method -- no ()
    budget = int(free_bytes * vram_fraction)
    bytes_per_individual = _estimate_scoring_bytes_per_individual(aa_seq_len * 3, max(kmer_k_values))
    return max(budget // bytes_per_individual, min_chunk)


def batch_calculate_change_vectors(pop_codons: list, analysis_objects, locvec: list = None, xp=np, progress_every: int = None, chunk_size: int = None) -> list:
    """Compute change vectors for a whole population in one batched pass.

    pop_codons: list of P individuals, each a list of N codon strings (same
        N for every individual -- one aa_seq per call, like
        calculate_change_vector()).
    xp: array module to compute with -- numpy (default, still much faster
        than the per-individual Python loop) or cupy (genuinely on the
        GPU). Pass the cupy module object, not a string.
    progress_every: if set, print elapsed-time progress marking when each
        term's batched pass finishes -- diagnostic only, no effect on the
        returned values. None (default): silent, matching the original
        behavior exactly.
    chunk_size: if set, process pop_codons in sequential chunks of at most
        this many individuals each, through this same function, instead of
        building one xp array sized to the whole population in a single
        pass. Every intermediate array below scales with the population
        dimension P -- some (batch_kmer_term's sliding-window codes, in
        particular, see its docstring) multiplicatively with sequence
        length N and k-mer length k too -- so a long protein can blow past
        available GPU memory at a population size that was previously fine.
        Bounding P via chunk_size bounds peak xp memory independent of both
        P and N, trading a few more (individually cheap -- see Handoff.md
        sec 6) batched calls for a hard cap on peak allocation. Output is
        identical to an unchunked call, just assembled chunk by chunk. None
        (default): one pass over the whole population, unchanged behavior.

    Returns a list of P dicts, one per individual, in the same shape
    calculate_change_vector() returns for a single individual -- so this is
    a genuine drop-in replacement for `[calculate_change_vector(sol, ...)
    for sol in pop_codons]`, just computed as one batch instead of P
    separate calls -- including any custom term registered via
    change_vector.register_term() after this module was written (see the
    fallback loop below), not just the 5 built-ins.

    All 5 built-in terms are natively batched, including Kmer (see
    batch_kmer_term() and this module's docstring for how -- it used to
    fall back to a per-individual Python loop, which is why progress_every
    used to report throughput "through the Kmer loop" specifically; that
    per-individual fallback is gone, so a slow call now means something
    else -- rerun with progress_every set to see which term's batched pass
    is slow).

    Any OTHER registered term (see _NATIVELY_BATCHED_TERMS) has no
    hand-written batched implementation here -- there's no generic way to
    "batch" an arbitrary (sol, analysis_objects, locvec) -> list[float]
    function the way change_vector.calculate_change_vector() genuinely is
    generic over the registry. Caught by test_gpu_change_vector.py's
    per-individual-vs-batched comparison the moment a 6th term (Uracil) was
    registered: the batched path was silently dropping it entirely whenever
    xp was set, even though weights['Uracil'] might be nonzero -- a real
    correctness gap, not just a test mismatch, since every batch-computed
    change vector (seeding, refresh, growth's new-genotype storage) would
    have silently never scored that term. Fixed with a per-individual
    fallback loop, below, for any such term -- still correct (unlike
    dropping it), just not accelerated. Extend _NATIVELY_BATCHED_TERMS with
    a real xp-array implementation if a custom term also needs to be fast
    at scale.
    """
    if not pop_codons:
        return []

    if chunk_size is not None and chunk_size < len(pop_codons):
        results = []
        for start in range(0, len(pop_codons), chunk_size):
            results.extend(batch_calculate_change_vectors(
                pop_codons[start:start + chunk_size], analysis_objects, locvec, xp=xp,
                progress_every=progress_every,
            ))
        return results

    N = len(pop_codons[0])
    if locvec is None:
        locvec = ['I'] * N

    if progress_every:
        import time as _time
        _t0 = _time.perf_counter()

        def _mark(label):
            nonlocal _t0
            now = _time.perf_counter()
            print(f"  [batch_calculate_change_vectors] {label} for {len(pop_codons)} "
                  f"individuals done in {now - _t0:.2f}s", flush=True)
            _t0 = now

    codon_idx = encode_population(xp, pop_codons)

    rare = batch_rare_codon_term(xp, codon_idx, analysis_objects.rare_codon)
    if progress_every:
        _mark("RareCodons")
    usage = batch_codon_usage_term(xp, codon_idx, analysis_objects.codon_usage)
    if progress_every:
        _mark("CodonUsage")
    cpb = batch_codon_pair_bias_term(xp, codon_idx, analysis_objects.codon_pair_bias)
    if progress_every:
        _mark("CodonPairBias")
    gc = batch_gc_term(xp, codon_idx, analysis_objects.gc, locvec)
    if progress_every:
        _mark("GC")
    kmer = batch_kmer_term(xp, codon_idx, analysis_objects.kmer, locvec)
    if progress_every:
        _mark("Kmer")

    rare_np = rare if xp is np else xp.asnumpy(rare)
    usage_np = usage if xp is np else xp.asnumpy(usage)
    cpb_np = cpb if xp is np else xp.asnumpy(cpb)
    gc_np = gc if xp is np else xp.asnumpy(gc)
    kmer_np = kmer if xp is np else xp.asnumpy(kmer)

    results = [
        {
            'RareCodons': rare_np[p].tolist(),
            'CodonUsage': usage_np[p].tolist(),
            'CodonPairBias': cpb_np[p].tolist(),
            'GC': gc_np[p].tolist(),
            'Kmer': kmer_np[p].tolist(),
        }
        for p in range(len(pop_codons))
    ]

    # Any registered term without a hand-written batched implementation
    # (see _NATIVELY_BATCHED_TERMS and this function's docstring) -- a
    # per-individual Python loop, but still correct, which a silent
    # omission would not be.
    extra_terms = {name: fn for name, fn in registered_terms().items() if name not in _NATIVELY_BATCHED_TERMS}
    if extra_terms:
        for sol, entry in zip(pop_codons, results):
            for name, fn in extra_terms.items():
                entry[name] = fn(sol, analysis_objects, locvec)
        if progress_every:
            _mark(f"extra terms ({', '.join(sorted(extra_terms))})")

    return results
