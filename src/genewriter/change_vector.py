"""Position-wise mutation-propensity ("change vector") scoring.

Ported from calculate_change_vector() in GeneRider_Cloud.ipynb and fixed.
That version never ran to completion (the one recorded execution was
manually interrupted before reaching this cell) and has several bugs that
would crash on first call:

  - Builds `ProposedSolution(...)` but only `Proposed_Solution` (underscore)
    is ever defined, anywhere in any notebook.
  - `cpbchangevec` is referenced in the codon-pair-bias term's final list
    comprehension but is never appended to -- always an empty list, so
    indexing it raises IndexError immediately.
  - Looks up `acobj.windowlength` on a CodonPairBiasAnalysis, whose actual
    field (per GeneClassesCloud.ipynb) is `windowsize`.
  - Calls dist_from_optimal() on windows of plain codon strings, but that
    function unpacks each element as a `(codon, location)` tuple.
  - The k-mer term's `range(0, len(sol) - 1)` silently drops the sequence's
    last codon.
  - `.mean()` / `.std()` are called directly on dataclass fields that are
    plain Python lists (RareCodonAnalysis.usagePerGene,
    CodonAnalysis.codonUsageScoreByGene/windowscores/windowdistancesfromoptimal,
    GCAnalysis.taggedGC1/2/3, GCAnalysis.windows) -- those methods only exist
    on numpy arrays.

All fixed below by operating on numpy arrays and by keeping the change-vector
terms working on plain codon strings throughout (no location tuples).

One thing this port does NOT fix silently: the original GC and k-mer terms
hardcoded the exon/intron layout of one specific gene ("For Immismo only" /
CD74, per the source comment) via a `locvec` of 'T'/'I' tags, and even then
only ever branched on those two (never 'F' or 'S', despite both existing in
the tagging scheme and in the analysis dataclasses). Here locvec is a
parameter accepting all four real tags ('F'/'T'/'I'/'S', see classes.py);
callers with no real exon-boundary information for their candidate sequence
should pass locvec=None, which scores every position as generic interior
coding sequence ('I').

Terms are pluggable: calculate_change_vector() runs whatever is in the
module-level registry rather than five hardcoded calls, so a new term (e.g.
a future RNA-folding or SpliceAI signal) is just a function decorated with
@register_term("Name") with signature (sol, analysis_objects, locvec) ->
list[float] -- see register_term()'s docstring.

calculate_change_vector() costs ~389ms/call on a 398-residue sequence
before vectorization (pure Python loops building and reducing windows) --
expensive when called for every candidate in a GA's reproduction step.
Every term below is now vectorized with numpy (mainly sliding_window_view
for the windowing that dominated the cost), bringing that same call down
to ~6ms -- a ~68x speedup, verified against the original pure-Python
implementation with a 550-case randomized regression test
(tests/test_windowed_average.py) plus the existing correctness suite.
Deliberately NOT done via cumsum/prefix-sum, despite that being the more
obvious vectorization: a cumsum-then-subtract window sum computes each
result as the *difference* of two large accumulated sums, which can round
differently than summing that window's own elements directly (the Handoff's
own warning that floating-point addition isn't associative) -- risking
exactly the kind of subtle drift that would undermine
diff_change_vector()'s "some terms are exactly locally reproducible"
guarantee. sliding_window_view avoids that: each window's reduction is
still independent and bit-for-bit what a manual slice-and-reduce would give,
just batched in numpy's C loop instead of Python's.

diff_change_vector() approximates a child's change vector from its
parent's by recomputing only a local excerpt around the mutated
position(s) -- see its docstring for exactly which terms this is exact for
and which it approximates. It benefits from the same vectorization (it
just calls calculate_change_vector() on a shorter excerpt).
"""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .classes import CodonAnalysis, CodonPairBiasAnalysis, GCAnalysis, KmerAnalysis, RareCodonAnalysis
from .codon_tables import (
    CODON_FREQS_LIT,
    CODON_TO_INDEX,
    CODON_TO_VARIABLE_FLAGS,
    GC_FLAGS_BY_INDEX,
    RARE_CODONS_LIT,
    TAG_TO_WINDOW_BUCKET,
    codon_choices_for_aa,
    get_aa,
)

# The 3 bucket names GCAnalysis.windows/KmerAnalysis.kmer_dict actually
# track (see TAG_TO_WINDOW_BUCKET) -- fixed order, used to vectorize
# "majority location tag in this window" as a 3-way argmax over per-bucket
# windowed counts instead of Python's max(set(...), key=...count) per window.
_WINDOW_BUCKET_NAMES = ('ExonL50', 'Exon', 'ExonR50')


def _windowed_average(values: list, target_len: int, winsize: int) -> list:
    """Smear a list of per-window scores back onto per-position scores.

    Position i gets the average of the windows overlapping it. Same pattern
    used (inconsistently) throughout the original for every change-vector
    term; consolidated here into one helper.

    Vectorized: the bulk of positions (interior ones whose window
    [i-half, i+half) is fully in-bounds) are computed via
    sliding_window_view -- each window's mean is still an independent
    reduction over exactly that window's elements, batched in numpy's C
    loop instead of Python's, so this is not just faster but bit-for-bit
    identical to summing that same slice by hand (no prefix-sum/cumsum
    trick is used here specifically to avoid that: a cumsum-then-subtract
    approach computes each window sum as the *difference* of two large
    accumulated sums, which can round differently than summing the window's
    own elements directly -- see the Handoff's own warning that floating-
    point addition isn't associative). Positions near either boundary, or
    where the window would need clipping, fall back to the original
    per-position slice+mean -- a small, bounded number of positions
    regardless of target_len, so this costs nothing next to vectorizing
    the bulk.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    half = max(winsize // 2, 1)
    if n == 0 or target_len == 0:
        return [0.0] * target_len

    out = [0.0] * target_len
    idx = np.arange(target_len)
    regime_a = idx < winsize
    regime_b = (~regime_a) & (idx > target_len - winsize)
    regime_c = ~(regime_a | regime_b)
    # Within regime C, only positions whose fixed-radius window is fully
    # in-bounds are safe for the batched path (see docstring).
    safe = regime_c & (idx - half >= 0) & (idx + half <= n) & (n >= 2 * half)

    if safe.any():
        windows = sliding_window_view(values, 2 * half)  # shape (n - 2*half + 1, 2*half)
        safe_idx = idx[safe]
        means = windows[safe_idx - half].mean(axis=1)
        for pos, m in zip(safe_idx.tolist(), means.tolist()):
            out[pos] = m

    fallback_idx = idx[~safe]
    for i in fallback_idx.tolist():
        if i < winsize:
            relevant = values[0:i]
        elif i > target_len - winsize:
            relevant = values[i:n]
        else:
            relevant = values[max(i - half, 0):min(i + half, n)]
        out[i] = float(relevant.mean()) if relevant.size else 0.0

    return out


def _zscore(value: float, mean: float, std: float) -> float:
    return 0.0 if std == 0 else (value - mean) / std


def cached_stat(obj, key, compute_fn):
    """Lazily compute and cache compute_fn() on obj, keyed by key -- reused
    across every subsequent call for the life of obj instead of
    recomputing from scratch every time.

    Real-world impact this fixes: RareCodons/CodonUsage/CodonPairBias/GC
    all recomputed mean/std (or, for CodonPairBias/Kmer, a whole lookup
    table) from a baseline array on *every single call* -- cheap against
    the tiny synthetic baselines every local test uses, but on a real
    Colab run against real genome-wide Standards baselines (windowdistances-
    fromoptimal/windowscores/taggedGC1-3/cpbPerWindow are window-level
    arrays across the whole corpus, potentially huge), this dominated
    real-scale cost so completely that CodonUsage/GC/CodonPairBias each
    took a near-*constant* ~5-13s per batched call regardless of whether
    the population was 200 or 50,000 -- a dead giveaway it was baseline-
    sized work, not population-sized work. `rare_codon.usagePerGene` (one
    value per gene, not per window) is the one baseline small enough that
    this went unnoticed there. `analysis_objects` (and each of its 5
    sub-analyses) is loaded once per GA run and never mutated afterward by
    any term in this module or gpu_change_vector.py, so caching per-object
    is safe -- a caller that genuinely needs different baseline data
    mid-run must load/construct a new object rather than editing fields on
    this one in place, or the cache goes stale (no test in this repo does
    that; each mutates a baseline field before the object's first use, on
    a fresh per-test fixture instance).

    Cached directly on the instance (a private dict attribute), not a
    module-level dict keyed by id(obj) -- so it is garbage-collected
    together with the object (no unbounded growth across many different
    baseline objects over a long-running process) and immune to id()
    reuse after garbage collection (a real risk a global id()-keyed cache
    would have). Shared between change_vector.py's per-individual terms
    and gpu_change_vector.py's batched counterparts (same obj, same key)
    -- whichever path runs first populates the cache for both.
    """
    cache = getattr(obj, '_stat_cache', None)
    if cache is None:
        cache = {}
        obj._stat_cache = cache
    if key not in cache:
        cache[key] = compute_fn()
    return cache[key]


def cached_mean_std(obj, key, values) -> tuple:
    """cached_stat() specialized for the extremely common "(mean, std) of
    this baseline array/list" case -- see cached_stat()'s docstring."""
    def _compute():
        arr = np.asarray(values, dtype=float)
        return (float(arr.mean()), float(arr.std()))
    return cached_stat(obj, key, _compute)


def cached_normal_transform(obj, key, values):
    """cached_stat() specialized for building a distribution_fit.
    FittedDistribution from a baseline array -- the normal-scores-transform
    counterpart of cached_mean_std(), and every term below now uses this
    instead. See distribution_fit.py's module docstring for why: a plain
    (x - mean) / std z-score, squared, silently misrepresents rarity when
    the baseline array isn't symmetric (confirmed against real
    CodonPairBias data, which is Boltzmann/exponential-tailed, not
    Gaussian). fit_normal_transform() auto-detects the actual shape and
    returns a lookup table that reduces to the classic z-score exactly when
    the shape really is normal, so this is a strict correctness upgrade,
    not a behavior change for baselines that were already Gaussian.

    Same object-keyed caching as cached_mean_std() (and cached_stat() in
    general) -- the fit is genuinely heavyweight (an MLE + AIC comparison
    across several candidate families), which is exactly why it's cached
    for the life of `obj` rather than recomputed per call. Shared between
    change_vector.py's per-individual terms, gpu_change_vector.py's batched
    counterparts, and weight_calibration.py wherever they pass the same
    (obj, key) -- whichever runs first populates the cache for all three.
    """
    from .distribution_fit import fit_normal_transform

    def _compute():
        return fit_normal_transform(values)
    return cached_stat(obj, key, _compute)


_TERM_REGISTRY = {}


def register_term(name: str):
    """Decorator: register a new change-vector term under `name`.

    The decorated function must have the signature
    `(sol: list[str], analysis_objects: AnalysisObjects, locvec: list[str]) -> list[float]`,
    returning exactly one score per codon in `sol`. Pull whatever baseline
    data the term needs off `analysis_objects` (add a field to that
    dataclass first if the term needs a baseline type that doesn't exist
    yet); ignore `locvec` if the term has no location-dependence.

    Registration happens at import time, so the module defining a custom
    term must be imported before calculate_change_vector() runs (e.g.
    import it in your driver script) for the term to take effect.
    """
    def decorator(fn):
        if name in _TERM_REGISTRY:
            raise ValueError(f"Change-vector term {name!r} is already registered")
        _TERM_REGISTRY[name] = fn
        return fn
    return decorator


def registered_terms() -> dict:
    return dict(_TERM_REGISTRY)


def require_weights(term_names, weights: dict) -> None:
    missing = [name for name in term_names if name not in weights]
    if missing:
        raise ValueError(
            f"No weight configured for change-vector term(s): {missing}. "
            f"Weights dict must have an entry for every registered term "
            f"(currently: {sorted(_TERM_REGISTRY)})."
        )


def dist_from_optimal(codons: list) -> float:
    """Average, over a window of codons, of the best synonymous-codon score
    available for each position's amino acid (i.e. what CAI-optimal usage
    would have scored there)."""
    total = 0.0
    for codon in codons:
        aa = get_aa(codon)
        total += max(CODON_FREQS_LIT[c] for c in codon_choices_for_aa(aa))
    return total / len(codons)


@register_term('RareCodons')
def _rare_codon_term(sol: list, analysis_objects: 'AnalysisObjects', locvec: list, winsize: int = 15) -> list:
    rc = analysis_objects.rare_codon
    rvec = np.fromiter((1 if c in RARE_CODONS_LIT else 0 for c in sol), dtype=np.int64, count=len(sol))

    # Sum of a binary window is an int in [0, winsize] -- exact regardless
    # of summation order, so vectorizing this via sliding_window_view has
    # no floating-point-reduction-order concerns at all (unlike the
    # float-valued windows elsewhere). Window *count* matches the original
    # exactly: range(0, len(rvec)-winsize) gives len(rvec)-winsize windows,
    # but sliding_window_view gives len(rvec)-winsize+1 (one more, at the
    # tail) -- trimmed off here. Caught late, by directly diffing against
    # the original reference implementation on random inputs (this term had
    # no test asserting exact equality with the pre-vectorization version,
    # only "no NaN" and "approximately close" -- the gap that let a
    # genuine off-by-one ship two turns ago).
    num_windows = max(len(rvec) - winsize, 0)
    if num_windows > 0:
        window_sums = sliding_window_view(rvec, winsize).sum(axis=1)[:num_windows]
    else:
        window_sums = np.empty(0, dtype=np.int64)

    # odds.get(sum(win), inf) as an array lookup: window sums can only be
    # in [0, winsize], so a fixed-size table covers every possible key;
    # counts the baseline dict doesn't have an entry for (e.g. windows of
    # that composition were never observed) default to inf, same as .get().
    # Cached per (rc, winsize) -- built from rc.rare_codon_windows, which
    # never changes for the life of an analysis object -- see
    # cached_stat()'s docstring for why recomputing this every call was a
    # real cost at real-scale baseline sizes.
    def _build_odds_by_count():
        total_windows = sum(rc.rare_codon_windows.values())
        table = np.full(winsize + 1, np.inf)
        for count, n in rc.rare_codon_windows.items():
            if 0 <= count <= winsize:
                table[count] = np.inf if n == 0 else total_windows / n
        return table
    odds_by_count = cached_stat(rc, ('odds_by_count', winsize), _build_odds_by_count)
    scorewins = odds_by_count[window_sums].tolist() if window_sums.size else []
    scorevec = _windowed_average(scorewins, len(sol), winsize)

    overall_rc = rvec.sum() / len(rvec)
    usage_transform = cached_normal_transform(rc, 'usagePerGene', rc.usagePerGene)
    zscore = usage_transform.transform(overall_rc) ** 2

    # rvec[i] gates the signal to rare-codon positions only. Guard the
    # multiplication explicitly rather than `scorevec[i] * rvec[i] * zscore`:
    # scorevec[i] can be float('inf') (a window composition never observed
    # in the baseline), and inf * 0 is NaN, not 0, in IEEE 754.
    return [(scorevec[i] * zscore) if rvec[i] else 0.0 for i in range(len(sol))]


@register_term('CodonUsage')
def _codon_usage_term(sol: list, analysis_objects: 'AnalysisObjects', locvec: list, winsize: int = 15) -> list:
    ca = analysis_objects.codon_usage
    cuvec = np.fromiter((ca.codonFreqsLit[c] for c in sol), dtype=float, count=len(sol))
    # dist_from_optimal's per-codon term (best synonymous-codon score for
    # that position's amino acid) depends only on the AA, not which codon
    # was actually chosen there -- precompute once per position instead of
    # letting dist_from_optimal() redundantly recompute it inside every
    # overlapping window that touches that position.
    best_freq = np.fromiter(
        (max(CODON_FREQS_LIT[c] for c in codon_choices_for_aa(get_aa(codon))) for codon in sol),
        dtype=float, count=len(sol),
    )

    span = max(ca.windowsize - 1, 1)
    # Matches the original's window count exactly: range(0, len(sol) -
    # ca.windowsize) -- note that's windowsize, not span, as the bound,
    # even though each window itself has length span=windowsize-1, so
    # sliding_window_view's own window count (len(sol)-span+1) is trimmed
    # down to match rather than "corrected" to the seemingly-intended count.
    num_windows = max(len(sol) - ca.windowsize, 0)
    if num_windows > 0:
        cuwinscores = (sliding_window_view(cuvec, span)[:num_windows].sum(axis=1) / ca.windowsize).tolist()
        cudists = sliding_window_view(best_freq, span)[:num_windows].mean(axis=1).tolist()
    else:
        cuwinscores = []
        cudists = []

    dist_transform = cached_normal_transform(ca, 'windowdistancesfromoptimal', ca.windowdistancesfromoptimal)
    score_transform = cached_normal_transform(ca, 'windowscores', ca.windowscores)
    dist_vals = _windowed_average(cudists, len(sol), winsize)
    score_vals = _windowed_average(cuwinscores, len(sol), winsize)
    dist_zs = [dist_transform.transform(v) ** 2 for v in dist_vals]
    score_zs = [score_transform.transform(v) ** 2 for v in score_vals]

    gene_transform = cached_normal_transform(ca, 'codonUsageScoreByGene', ca.codonUsageScoreByGene)
    straight_z = [gene_transform.transform(o) / 2 for o in cuvec]

    return [straight_z[i] * score_zs[i] + straight_z[i] * dist_zs[i] for i in range(len(sol))]


@register_term('CodonPairBias')
def _codon_pair_bias_term(sol: list, analysis_objects: 'AnalysisObjects', locvec: list, winsize: int = 15) -> list:
    cpb = analysis_objects.codon_pair_bias
    if len(sol) < 2:
        return [0.0] * len(sol)

    pairs = [sol[i] + sol[i + 1] for i in range(len(sol) - 1)]
    pair_scores = [cpb.cpb_lit[p] for p in pairs]

    span = max(cpb.windowsize - 1, 2)
    # A window of `span` codons produces span-1 adjacent pairs, which are
    # exactly pair_scores[i:i+span-1] (each window's own pair sequence is
    # just a slice of the pairs already computed above) -- so this reduces
    # to a plain windowed mean of pair_scores, instead of rebuilding and
    # re-scoring each window's pairs from scratch.
    num_windows = max(len(sol) - cpb.windowsize, 0)
    pair_span = span - 1
    pair_scores_arr = np.asarray(pair_scores, dtype=float)
    if num_windows > 0 and pair_span >= 1 and len(pair_scores_arr) >= pair_span:
        win_scores = sliding_window_view(pair_scores_arr, pair_span)[:num_windows].mean(axis=1).tolist()
    else:
        win_scores = [0.0] * num_windows

    win_transform = cached_normal_transform(cpb, 'cpbPerWindow', cpb.cpbPerWindow)
    win_z = [win_transform.transform(v) ** 2 for v in win_scores]
    win_change = _windowed_average(win_z, len(sol), winsize)

    # Score at position i is the average of the two codon-pairs touching it.
    pair_change = []
    for i in range(len(sol)):
        if i == 0:
            pair_change.append(pair_scores[0])
        elif i == len(sol) - 1:
            pair_change.append(pair_scores[-1])
        else:
            pair_change.append((pair_scores[i - 1] + pair_scores[i]) / 2)

    return [win_change[i] * pair_change[i] for i in range(len(sol))]


@register_term('GC')
def _gc_term(sol: list, analysis_objects: 'AnalysisObjects', locvec: list, winsize: int = 15) -> list:
    gc = analysis_objects.gc
    gc_flags = np.asarray(GC_FLAGS_BY_INDEX, dtype=float)[[CODON_TO_INDEX[c] for c in sol]]  # (N, 3), 1=G/C, normal polarity

    # Per-SEQUENCE GC deviation: one scalar for the whole candidate, added
    # (not multiplied) to every position's local windowed score below.
    # Replaces the old gc1/gc2/gc3-by-bucket signal, which turned out to be
    # a whole-candidate aggregate masquerading as a per-position one (every
    # position sharing a bucket got the identical value, regardless of
    # which codon was actually there) and -- worse -- barely moved at all
    # when the GA scored synonymous alternatives at a position, since
    # swapping one codon shifts a whole-sequence mean by ~1/(3*len(sol))
    # (see diff_change_vector()'s and directed_evolution_batch()'s own
    # docstrings, which already documented this dilution for exactly this
    # kind of whole-sequence aggregate). overall_gc here matches
    # GCAnalysis.gcPerGene's own construction exactly (baseline.py:
    # gc_bases / (3 * len(codons)), same normal polarity) so the two are
    # directly comparable through the fitted transform.
    #
    # This makes GC's overall priority scale with how far off nature's
    # typical GC% the WHOLE candidate is: badly off overall -> every
    # position's GC score gets a large constant boost (GC matters more,
    # relative to the other terms, everywhere); roughly on-spec overall ->
    # that boost is ~0 and GC only matters where the *local* window
    # (below) is itself off -- still-independent local smoothing pressure,
    # not gated by the sequence-wide state.
    overall_gc = gc_flags.mean()
    seq_transform = cached_normal_transform(gc, 'gcPerGene', gc.gcPerGene)
    seq_deviation = seq_transform.transform(overall_gc) ** 2

    # A position is only worth mutating for GC purposes if some synonymous
    # codon actually differs at some base there -- if every synonym shares
    # the same base 1/2/3, no change at this position can move GC at all,
    # local or global. CODON_TO_VARIABLE_FLAGS is precomputed once at
    # import time (depends only on the amino acid, not which codon is
    # chosen -- see codon_tables.py).
    mutable = [1.0 if any(CODON_TO_VARIABLE_FLAGS[c]) else 0.0 for c in sol]

    continuous = ''.join(sol)
    location_string = ''.join(loc * 3 for loc in locvec)
    span = max(gc.windowsize, 1)
    num_windows = max(len(continuous) - span, 0)

    if num_windows > 0:
        is_gc = np.fromiter((1 if ch in 'GC' else 0 for ch in continuous), dtype=np.int64, count=len(continuous))
        window_gc = sliding_window_view(is_gc, span)[:num_windows].mean(axis=1)

        # Majority location-bucket per window, vectorized as a 3-way argmax
        # over per-bucket windowed counts, instead of Python's
        # max(set(loc_window), key=loc_window.count) once per window.
        #
        # CORRECTION, 2026-08-19 (this comment previously claimed the
        # rewrite below was tie-break-only -- it is NOT, confirmed with a
        # concrete non-tie counter-example, see
        # memory/majority_vote_bucket_discrepancy.md): folding every
        # nucleotide's raw tag (F/T/I/S) through TAG_TO_WINDOW_BUCKET FIRST
        # (as bucket_of_nt does, next line) merges S and I into the same
        # 'Exon' vote-bin BEFORE counting, then votes among the 3
        # already-folded buckets. baseline.py's original
        # `max(set(loc_window), key=loc_window.count)` votes among the 4
        # RAW tags first, then folds only the winner -- a genuinely
        # different algorithm, not just a different tie-break rule. They
        # disagree whenever a window mixes S and I against an F/T
        # plurality, in real (non-tie) cases, not only in ties. This
        # function (scoring a GA candidate) intentionally still uses the
        # fold-first convention -- unifying it with baseline.py's
        # counting-side convention is a deliberate, not-yet-made decision,
        # see that memory file for why it isn't simply "fixed" here.
        bucket_index = {name: i for i, name in enumerate(_WINDOW_BUCKET_NAMES)}
        bucket_of_nt = np.fromiter(
            (bucket_index[TAG_TO_WINDOW_BUCKET[loc]] for loc in location_string),
            dtype=np.int8, count=len(location_string),
        )
        counts = np.stack([
            sliding_window_view((bucket_of_nt == b).astype(np.int64), span)[:num_windows].sum(axis=1)
            for b in range(len(_WINDOW_BUCKET_NAMES))
        ], axis=1)
        majority_bucket = counts.argmax(axis=1)

        win_z = np.empty(num_windows, dtype=float)
        for b, name in enumerate(_WINDOW_BUCKET_NAMES):
            mask = majority_bucket == b
            if not mask.any():
                continue
            base_transform = cached_normal_transform(gc, ('windows', name), gc.windows[name])
            win_z[mask] = base_transform.transform_array(window_gc[mask]) ** 2
        win_z = win_z.tolist()
    else:
        win_z = []

    win_z_by_pos = _windowed_average(win_z, len(continuous), winsize)
    # Collapse per-nucleotide window scores back down to per-codon. Every
    # codon contributes exactly 3 nucleotides, so len(win_z_by_pos) is
    # always a multiple of 3 in practice; the truncate/pad below is a
    # defensive no-op that only matters for a malformed (non-triplet) sol.
    win_z_by_pos_arr = np.asarray(win_z_by_pos, dtype=float)
    num_full_codons = len(win_z_by_pos_arr) // 3
    per_codon = win_z_by_pos_arr[:num_full_codons * 3].reshape(-1, 3).mean(axis=1).tolist() if num_full_codons else []
    per_codon = per_codon[:len(sol)] + [0.0] * max(len(sol) - len(per_codon), 0)

    return [mutable[i] * (per_codon[i] + seq_deviation) for i in range(len(sol))]


@register_term('Kmer')
def _kmer_term(sol: list, analysis_objects: 'AnalysisObjects', locvec: list, winsize: int = 15) -> list:
    kmer = analysis_objects.kmer
    continuous = ''.join(sol)
    location_string = ''.join(loc * 3 for loc in locvec)
    assert len(continuous) == len(location_string)

    per_k_vecs = []
    for k_str, kmer_by_seq in kmer.kmer_dict.items():
        k = int(k_str)
        num_wins = max(len(continuous) - k, 0)
        seq_wins = [continuous[i:i + k] for i in range(num_wins)]

        # Majority location-bucket per window, vectorized the same way as
        # the GC term (3-way argmax over per-bucket windowed counts) --
        # the substring->score lookup just below stays a per-window dict
        # lookup, since it's keyed by an arbitrary-length k-mer string:
        # hashing that string is the real cost, not something a numpy
        # windowed reduction helps with (unlike the numeric GC/location
        # reductions this shares its majority-vote logic with).
        #
        # Fold-then-vote convention, NOT baseline.compute_kmer_analysis's
        # raw-tag-vote-then-fold -- same divergence as _gc_term's identical
        # block above, see that function's own (corrected 2026-08-19)
        # comment and memory/majority_vote_bucket_discrepancy.md.
        if num_wins > 0:
            bucket_index = {name: i for i, name in enumerate(_WINDOW_BUCKET_NAMES)}
            bucket_of_nt = np.fromiter(
                (bucket_index[TAG_TO_WINDOW_BUCKET[loc]] for loc in location_string),
                dtype=np.int8, count=len(location_string),
            )
            counts = np.stack([
                sliding_window_view((bucket_of_nt == b).astype(np.int64), k)[:num_wins].sum(axis=1)
                for b in range(len(_WINDOW_BUCKET_NAMES))
            ], axis=1)
            majority_buckets = [_WINDOW_BUCKET_NAMES[b] for b in counts.argmax(axis=1)]
        else:
            majority_buckets = []

        win_scores = []
        for seq_win, bucket in zip(seq_wins, majority_buckets):
            entry = kmer_by_seq.get(seq_win)
            win_scores.append(1.0 if entry is None else entry.get(bucket, {}).get('fold_enrich', 1.0))

        overall = sum(win_scores) / len(win_scores) if win_scores else 1.0
        by_pos = _windowed_average(win_scores, len(continuous), winsize)
        by_pos_arr = np.asarray(by_pos, dtype=float) * overall
        num_full_codons = len(by_pos_arr) // 3
        per_codon = by_pos_arr[:num_full_codons * 3].reshape(-1, 3).mean(axis=1).tolist() if num_full_codons else []
        per_codon = per_codon[:len(sol)] + [0.0] * max(len(sol) - len(per_codon), 0)
        per_k_vecs.append(per_codon)

    if not per_k_vecs:
        return [0.0] * len(sol)
    return [sum(vec[i] for vec in per_k_vecs) for i in range(len(sol))]


@register_term('Uracil')
def _uracil_term(sol: list, analysis_objects: 'AnalysisObjects', locvec: list) -> list:
    """Per-position count of removable uracil (T, since sol is DNA-alphabet
    codons -- what would be U in the transcribed mRNA): how many of a
    codon's bases are both 'T' and actually variable via some synonymous
    swap (codon_tables.CODON_TO_VARIABLE_FLAGS -- the same "does a synonym
    actually differ here" gate _gc_term uses), in [0, 3] per position.

    Unlike every other term here, this is deliberately NOT a natural-gene-
    baseline z-score (no UracilAnalysis dataclass, no change to
    AnalysisObjects) -- future_work_items describes uracil minimization as
    a flat "reduce U content" objective, not a "match real genes" one, so
    there's no natural distribution to compare against. `analysis_objects`
    is accepted (matching every registered term's required signature) but
    unused.

    Design call, flagged rather than silently resolved (per this module's
    stated policy): the score returned is a *positive* "opportunity to
    remove a U here" count, consistent with every other term scoring "this
    position is worth mutating" rather than "this position is good" --
    same "positive weight = counts normally" convention documented on
    WEIGHTS itself. kill_off()/select_survivors() use score_changevec()
    directly as a death weight via _finite_nonneg() (ga.py), which clamps
    any negative score to ~0 -- so a *negative* weights['Uracil'] doesn't
    invert anything, it just floors a high-U individual's death pressure
    to the same near-zero value as a low-U one, neutering selection
    pressure on this term (or worse: once other positive-weighted terms
    contribute their own positive baseline, a negative Uracil weight can
    make a high-U individual's *total* score lower than a low-U one's,
    making it die LESS). Symmetrically, directed_evolution()'s lookahead
    picks whichever synonymous alternative *minimizes* total score
    (ga.py's `candidate_score < best_score`) -- with a negative weight that
    means preferring the alternative with MORE removable U, actively
    growing uracil content. A *positive* weights['Uracil'] (the normal
    convention, matching every other registered term's default) is what
    actually pushes uracil content down: confirmed empirically -- an
    isolated die-weight comparison and a synonymous-codon-choice trace
    (Serine, alternatives spanning 0-2 removable U's) both show weight=+1
    favoring low-U individuals/alternatives and weight=-1 favoring
    high-U ones.
    """
    scores = []
    for codon in sol:
        v1, v2, v3 = CODON_TO_VARIABLE_FLAGS[codon]
        count = (v1 if codon[0] == 'T' else 0) + (v2 if codon[1] == 'T' else 0) + (v3 if codon[2] == 'T' else 0)
        scores.append(float(count))
    return scores


@dataclass
class AnalysisObjects:
    rare_codon: RareCodonAnalysis
    codon_usage: CodonAnalysis
    codon_pair_bias: CodonPairBiasAnalysis
    gc: GCAnalysis
    kmer: KmerAnalysis


def calculate_change_vector(sol: list, analysis_objects: AnalysisObjects, locvec: list = None) -> dict:
    """Per-position mutation-propensity scores for one candidate codon solution.

    sol: list of codon strings, one per residue.
    locvec: per-position exon/intron location tags ('T' near an exon 5' end,
        'I' interior). Defaults to all 'I' if the caller has no real
        exon-boundary information yet -- see module docstring.

    Runs every term in the registry (see register_term()) -- built-in terms
    plus any custom ones registered elsewhere and imported before this runs.
    """
    if locvec is None:
        locvec = ['I'] * len(sol)
    if len(locvec) != len(sol):
        raise ValueError("locvec must be the same length as sol")

    return {name: term_fn(sol, analysis_objects, locvec) for name, term_fn in _TERM_REGISTRY.items()}


def diff_change_vector(
    parent_codons: list,
    parent_vecs: dict,
    child_codons: list,
    analysis_objects: AnalysisObjects,
    locvec: list = None,
    margin: int = 40,
) -> dict:
    """Approximate a child's change vector from its parent's, recomputing
    only a local excerpt around the positions that actually changed instead
    of the whole sequence.

    Why this is safe for some terms and approximate for others: CodonUsage
    and CodonPairBias depend only on fixed baseline constants and windows
    within one window-width of each position -- recomputed on an excerpt,
    positions away from the excerpt's own artificial edges come out exactly
    equal to a full recompute. RareCodons, GC, and Kmer each also fold in a
    population-wide aggregate over the *whole* sequence (overall rare-codon
    rate, GC1/2/3 composition, overall k-mer fold-enrichment) into a z-score
    or multiplier applied to every position -- recomputed on an excerpt,
    that aggregate is estimated from the excerpt's local composition instead
    of the true whole-sequence composition, which drifts a small amount
    (bounded by how unrepresentative the excerpt is of the full sequence)
    from what a full recompute would give. That drift does not compound
    across positions within one call, but it does carry forward if you keep
    diffing children of children of children -- periodically call
    calculate_change_vector() for an exact reset (this is why
    schedule.py's kill_off/select/flatten steps always do a full recompute
    first: population is smaller there, and those are exactly the steps
    that make survival decisions based on the scores).

    Not safe for a term with genuine long-range structure (e.g. a future
    RNA-folding term, where a mutation can change base-pairing partners
    arbitrarily far away): such a term would need to opt out of excerpt-based
    diffing, e.g. by returning None/raising to force a caller fallback, or
    this function would need a per-term "diffable: bool" flag. Not needed
    for any term registered today.

    parent_vecs is trusted as-is (not re-validated against parent_codons).
    """
    if locvec is None:
        locvec = ['I'] * len(child_codons)
    if len(child_codons) != len(parent_codons):
        raise ValueError("diff_change_vector requires parent and child of the same length (synonymous substitutions only)")
    if len(locvec) != len(child_codons):
        raise ValueError("locvec must be the same length as child_codons")

    changed = [i for i in range(len(child_codons)) if child_codons[i] != parent_codons[i]]
    if not changed:
        return {name: list(values) for name, values in parent_vecs.items()}

    lo = max(0, min(changed) - margin)
    hi = min(len(child_codons), max(changed) + margin + 1)
    # Too spread out for a local excerpt to be worth it -- just recompute
    # exactly rather than pay excerpt overhead for near-total coverage.
    if hi - lo > len(child_codons) * 0.5:
        return calculate_change_vector(child_codons, analysis_objects, locvec)

    trusted_lo = max(0, min(changed) - margin // 2)
    trusted_hi = min(len(child_codons), max(changed) + margin // 2 + 1)

    excerpt_vecs = calculate_change_vector(child_codons[lo:hi], analysis_objects, locvec[lo:hi])

    result = {name: list(values) for name, values in parent_vecs.items()}
    for name, excerpt_values in excerpt_vecs.items():
        target = result.setdefault(name, [0.0] * len(child_codons))
        for j in range(trusted_lo, trusted_hi):
            target[j] = excerpt_values[j - lo]
    return result


def score_changevec(changevecs: dict, weights: dict) -> float:
    require_weights(changevecs.keys(), weights)
    return sum(weights[key] * sum(changevecs[key]) for key in changevecs)
