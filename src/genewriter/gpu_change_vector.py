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

Not yet batched here: Kmer (its substring->score lookup is a hash-map
operation over arbitrary-length k-mer strings, not a natural fit for either
numpy-style array vectorization or a first GPU pass -- would need k-mers
re-encoded as base-4 integers for an array-indexed lookup table, left as
follow-up). batch_calculate_change_vectors() falls back to the existing
per-individual Kmer term for that piece.
"""

import numpy as np

from .change_vector import _zscore, calculate_change_vector
from .codon_tables import (
    AA_CODONS,
    CODON_FREQ_BY_INDEX,
    CODON_LIST,
    CODON_TO_AA,
    CODON_TO_INDEX,
    GC_FLAGS_BY_INDEX,
    RARE_CODON_INDICES,
    TAG_TO_BUCKET,
    TAG_TO_WINDOW_BUCKET,
    VARIABLE_FLAGS_BY_INDEX,
    encode_codons,
)

_WINDOW_BUCKET_NAMES = ('ExonL50', 'Exon', 'ExonR50')
_LOCATION_BUCKET_NAMES = ('ExonL50', 'Exon', 'ExonR50', 'Splice')

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
        means = windows[:, row_starts, :].mean(axis=2)
        out[:, safe_idx] = means

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

    total_windows = sum(rc.rare_codon_windows.values())
    odds_by_count = np.full(winsize + 1, np.inf)
    for count, n in rc.rare_codon_windows.items():
        if 0 <= count <= winsize:
            odds_by_count[count] = np.inf if n == 0 else total_windows / n
    odds_by_count = xp.asarray(odds_by_count)
    scorewins = odds_by_count[window_sums] if window_sums.shape[1] else xp.zeros((P, 0), dtype=float)

    scorevec = windowed_average_batched(xp, scorewins, N, winsize)  # (P, N)

    overall_rc = rvec.sum(axis=1) / N  # (P,)
    distrib = np.asarray(rc.usagePerGene, dtype=float)
    dmean, dstd = float(distrib.mean()), float(distrib.std())
    zscore = (xp.zeros_like(overall_rc) if dstd == 0 else (overall_rc - dmean) / dstd) ** 2  # (P,)

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

    dist_mean, dist_std = float(np.mean(ca.windowdistancesfromoptimal)), float(np.std(ca.windowdistancesfromoptimal))
    score_mean, score_std = float(np.mean(ca.windowscores)), float(np.std(ca.windowscores))
    dist_vals = windowed_average_batched(xp, cudists, N, winsize)
    score_vals = windowed_average_batched(xp, cuwinscores, N, winsize)
    # std==0 branches return a same-shaped zero array, not a bare Python
    # 0.0 -- see batch_gc_term's z_batch() docstring for why that matters
    # (anything feeding back into windowed_average_batched needs a real
    # array with a .shape).
    dist_zs = (xp.zeros_like(dist_vals) if dist_std == 0 else (dist_vals - dist_mean) / dist_std) ** 2
    score_zs = (xp.zeros_like(score_vals) if score_std == 0 else (score_vals - score_mean) / score_std) ** 2

    gene_mean, gene_std = float(np.mean(ca.codonUsageScoreByGene)), float(np.std(ca.codonUsageScoreByGene))
    straight_z = (xp.zeros_like(cuvec) if gene_std == 0 else (cuvec - gene_mean) / gene_std) / 2

    return straight_z * score_zs + straight_z * dist_zs


def batch_codon_pair_bias_term(xp, codon_idx, cpb, winsize: int = 15):
    """codon_idx: (P, N) int array. cpb: CodonPairBiasAnalysis. Returns (P, N)."""
    P, N = codon_idx.shape
    if N < 2:
        return xp.zeros((P, N), dtype=float)

    num_codons = len(CODON_LIST)
    cpb_table = np.zeros((num_codons, num_codons), dtype=float)
    for pair, score in cpb.cpb_lit.items():
        i, j = CODON_TO_INDEX[pair[:3]], CODON_TO_INDEX[pair[3:]]
        cpb_table[i, j] = score
    cpb_table = xp.asarray(cpb_table)

    pair_scores = cpb_table[codon_idx[:, :-1], codon_idx[:, 1:]]  # (P, N-1)

    span = max(cpb.windowsize - 1, 2)
    pair_span = span - 1
    num_windows = max(N - cpb.windowsize, 0)
    if num_windows > 0 and pair_span >= 1 and pair_scores.shape[1] >= pair_span:
        win_scores = _sliding_window_view(xp, pair_scores, pair_span, axis=1)[:, :num_windows].mean(axis=2)
    else:
        win_scores = xp.zeros((P, num_windows), dtype=float)

    win_mean, win_std = float(np.mean(cpb.cpbPerWindow)), float(np.std(cpb.cpbPerWindow))
    win_z = (xp.zeros_like(win_scores) if win_std == 0 else (win_scores - win_mean) / win_std) ** 2
    win_change = windowed_average_batched(xp, win_z, N, winsize)  # (P, N)

    pair_change = xp.zeros((P, N), dtype=float)
    pair_change[:, 0] = pair_scores[:, 0]
    pair_change[:, -1] = pair_scores[:, -1]
    if N > 2:
        pair_change[:, 1:-1] = (pair_scores[:, :-1] + pair_scores[:, 1:]) / 2

    return win_change * pair_change


def batch_gc_term(xp, codon_idx, gc, locvec: list, winsize: int = 15):
    """codon_idx: (P, N) int array. gc: GCAnalysis. locvec: list[str] of
    length N, shared across the whole batch (same candidate sequence
    length/gene body for every individual in one calculate_change_vector-
    style call). Returns (P, N)."""
    P, N = codon_idx.shape
    gc_flags = xp.asarray(np.asarray(GC_FLAGS_BY_INDEX, dtype=float))  # (64, 3), 1=G/C, 0=A/T
    per_pos_gc = gc_flags[codon_idx]  # (P, N, 3)
    # The per-individual term's gc1_mean/gc2_mean/gc3_mean use an *inverted*
    # polarity from everywhere else in this function (and from the baseline
    # data itself, gc.taggedGC1/2/3, and window_gc below): `0 if c[0] in
    # 'GC' else 1`, i.e. 0 for G/C and 1 for A/T. That's how the original
    # notebook computed it and change_vector.py's _gc_term faithfully
    # preserves it rather than "fixing" an inconsistency that isn't this
    # port's call to make (see change_vector.py's module docstring) --
    # matched here by inverting the (normal-polarity) per_pos_gc average.
    gc1_mean = 1.0 - per_pos_gc[:, :, 0].mean(axis=1)  # (P,)
    gc2_mean = 1.0 - per_pos_gc[:, :, 1].mean(axis=1)
    gc3_mean = 1.0 - per_pos_gc[:, :, 2].mean(axis=1)

    def z_batch(value_mean, loc, tagged):
        """(value_mean - baseline_mean) / baseline_std, elementwise over
        the batch. Always returns an xp array of shape (P,), even when
        std==0 (a same-shaped zero array, not a bare Python 0.0) so every
        caller can treat the result uniformly."""
        arr = np.asarray(tagged[loc], dtype=float)
        m, s = float(arr.mean()), float(arr.std())
        if s == 0:
            return xp.zeros_like(value_mean, dtype=float)
        return (value_mean - m) / s

    z_by_bucket = {}  # bucket -> (z1, z2, z3), each shape (P,)
    for bucket in _LOCATION_BUCKET_NAMES:
        z_by_bucket[bucket] = (
            z_batch(gc1_mean, bucket, gc.taggedGC1),
            z_batch(gc2_mean, bucket, gc.taggedGC2),
            z_batch(gc3_mean, bucket, gc.taggedGC3),
        )

    variable_flags = xp.asarray(np.asarray(VARIABLE_FLAGS_BY_INDEX, dtype=float))  # (64, 3)
    per_pos_var = variable_flags[codon_idx]  # (P, N, 3)

    bucket_of_codon = [TAG_TO_BUCKET[loc] for loc in locvec]  # length N, shared
    gc_change = xp.zeros((P, N), dtype=float)
    for bucket in set(bucket_of_codon):
        cols = np.asarray([i for i, b in enumerate(bucket_of_codon) if b == bucket])
        z1, z2, z3 = z_by_bucket[bucket]
        contribution = (
            per_pos_var[:, cols, 0] * (z1.reshape(P, 1)) ** 2
            + per_pos_var[:, cols, 1] * (z2.reshape(P, 1)) ** 2
            + per_pos_var[:, cols, 2] * (z3.reshape(P, 1)) ** 2
        )
        gc_change[:, cols] = contribution

    # Flatten to per-nucleotide (P, 3N) to match the window span (in
    # nucleotides) the baseline was computed over.
    nt_gc_flat = per_pos_gc.reshape(P, N * 3)

    location_string = ''.join(loc * 3 for loc in locvec)  # length 3N, shared
    span = max(gc.windowsize, 1)
    num_windows = max(len(location_string) - span, 0)

    if num_windows > 0:
        bucket_index = {name: i for i, name in enumerate(_WINDOW_BUCKET_NAMES)}
        bucket_of_nt = np.fromiter(
            (bucket_index[TAG_TO_WINDOW_BUCKET[loc]] for loc in location_string),
            dtype=np.int8, count=len(location_string),
        )
        counts = np.stack([
            _np_sliding_sum(bucket_of_nt == b, span)[:num_windows]
            for b in range(len(_WINDOW_BUCKET_NAMES))
        ], axis=1)
        majority_bucket = counts.argmax(axis=1)  # (num_windows,) shared across batch

        window_gc = _sliding_window_view(xp, nt_gc_flat, span, axis=1)[:, :num_windows].mean(axis=2)  # (P, num_windows)

        win_z = xp.zeros((P, num_windows), dtype=float)
        for b, name in enumerate(_WINDOW_BUCKET_NAMES):
            mask = majority_bucket == b
            if not mask.any():
                continue
            baseline = np.asarray(gc.windows[name], dtype=float)
            m, s = float(baseline.mean()), float(baseline.std())
            cols = np.asarray(np.nonzero(mask)[0])
            val = window_gc[:, cols]
            win_z[:, cols] = (0.0 if s == 0 else (val - m) / s) ** 2
    else:
        win_z = xp.zeros((P, 0), dtype=float)

    win_z_by_pos = windowed_average_batched(xp, win_z, N * 3, winsize)  # (P, 3N)
    per_codon = win_z_by_pos.reshape(P, N, 3).mean(axis=2)  # (P, N)

    return per_codon * gc_change


def _np_sliding_sum(bool_arr_1d, span):
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(bool_arr_1d.astype(np.int64), span).sum(axis=1)


def batch_calculate_change_vectors(pop_codons: list, analysis_objects, locvec: list = None, xp=np, progress_every: int = None) -> list:
    """Compute change vectors for a whole population in one batched pass.

    pop_codons: list of P individuals, each a list of N codon strings (same
        N for every individual -- one aa_seq per call, like
        calculate_change_vector()).
    xp: array module to compute with -- numpy (default, still much faster
        than the per-individual Python loop) or cupy (genuinely on the
        GPU). Pass the cupy module object, not a string.
    progress_every: if set, print elapsed-time progress every this-many
        individuals through the Kmer loop below (plus one line marking when
        the batched rare/usage/cpb/gc section finishes) -- diagnostic only,
        no effect on the returned values. Kmer is the one term *not*
        batched (see module docstring), so at large population sizes it is
        the most likely place this function appears to hang; this is here
        to make that visible instead of guessed at. None (default): silent,
        matching the original behavior exactly.

    Returns a list of P dicts, one per individual, in the same shape
    calculate_change_vector() returns for a single individual -- so this is
    a drop-in replacement for `[calculate_change_vector(sol, ...) for sol
    in pop_codons]`, just computed as one batch instead of P separate calls.

    Kmer is not yet batched (see module docstring) -- computed per
    individual via the existing implementation and merged in.
    """
    if not pop_codons:
        return []
    N = len(pop_codons[0])
    if locvec is None:
        locvec = ['I'] * N

    if progress_every:
        import time as _time
        _t0 = _time.perf_counter()

    codon_idx = encode_population(xp, pop_codons)

    rare = batch_rare_codon_term(xp, codon_idx, analysis_objects.rare_codon)
    usage = batch_codon_usage_term(xp, codon_idx, analysis_objects.codon_usage)
    cpb = batch_codon_pair_bias_term(xp, codon_idx, analysis_objects.codon_pair_bias)
    gc = batch_gc_term(xp, codon_idx, analysis_objects.gc, locvec)

    rare_np = rare if xp is np else xp.asnumpy(rare)
    usage_np = usage if xp is np else xp.asnumpy(usage)
    cpb_np = cpb if xp is np else xp.asnumpy(cpb)
    gc_np = gc if xp is np else xp.asnumpy(gc)

    if progress_every:
        print(f"  [batch_calculate_change_vectors] batched terms (RareCodons/CodonUsage/"
              f"CodonPairBias/GC) for {len(pop_codons)} individuals done in "
              f"{_time.perf_counter() - _t0:.2f}s -- starting per-individual Kmer loop "
              f"(the known unbatched hotspot)", flush=True)
        _t_kmer0 = _time.perf_counter()

    from .change_vector import _kmer_term

    results = []
    for p, sol in enumerate(pop_codons):
        kmer_vals = _kmer_term(sol, analysis_objects, locvec)
        results.append({
            'RareCodons': rare_np[p].tolist(),
            'CodonUsage': usage_np[p].tolist(),
            'CodonPairBias': cpb_np[p].tolist(),
            'GC': gc_np[p].tolist(),
            'Kmer': kmer_vals,
        })
        if progress_every and (p + 1) % progress_every == 0:
            elapsed = _time.perf_counter() - _t_kmer0
            rate = (p + 1) / elapsed if elapsed > 0 else float('inf')
            remaining = (len(pop_codons) - (p + 1)) / rate if rate > 0 else float('inf')
            print(f"  [batch_calculate_change_vectors] Kmer {p + 1}/{len(pop_codons)} "
                  f"({rate:.1f} indiv/s, ~{remaining:.0f}s remaining)", flush=True)

    if progress_every:
        print(f"  [batch_calculate_change_vectors] Kmer loop done in "
              f"{_time.perf_counter() - _t_kmer0:.2f}s", flush=True)

    return results
