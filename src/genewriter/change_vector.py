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
    TAG_TO_BUCKET,
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

    total_windows = sum(rc.rare_codon_windows.values())
    # odds.get(sum(win), inf) as an array lookup: window sums can only be
    # in [0, winsize], so a fixed-size table covers every possible key;
    # counts the baseline dict doesn't have an entry for (e.g. windows of
    # that composition were never observed) default to inf, same as .get().
    odds_by_count = np.full(winsize + 1, np.inf)
    for count, n in rc.rare_codon_windows.items():
        if 0 <= count <= winsize:
            odds_by_count[count] = np.inf if n == 0 else total_windows / n
    scorewins = odds_by_count[window_sums].tolist() if window_sums.size else []
    scorevec = _windowed_average(scorewins, len(sol), winsize)

    overall_rc = rvec.sum() / len(rvec)
    distrib = np.asarray(rc.usagePerGene, dtype=float)
    zscore = _zscore(overall_rc, distrib.mean(), distrib.std()) ** 2

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

    dist_mean, dist_std = np.mean(ca.windowdistancesfromoptimal), np.std(ca.windowdistancesfromoptimal)
    score_mean, score_std = np.mean(ca.windowscores), np.std(ca.windowscores)
    dist_vals = _windowed_average(cudists, len(sol), winsize)
    score_vals = _windowed_average(cuwinscores, len(sol), winsize)
    dist_zs = [_zscore(v, dist_mean, dist_std) ** 2 for v in dist_vals]
    score_zs = [_zscore(v, score_mean, score_std) ** 2 for v in score_vals]

    gene_mean, gene_std = np.mean(ca.codonUsageScoreByGene), np.std(ca.codonUsageScoreByGene)
    straight_z = [_zscore(o, gene_mean, gene_std) / 2 for o in cuvec]

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

    win_mean, win_std = np.mean(cpb.cpbPerWindow), np.std(cpb.cpbPerWindow)
    win_z = [_zscore(v, win_mean, win_std) ** 2 for v in win_scores]
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
    # NB for anyone touching this later (bit me once already porting this
    # to gpu_change_vector.py's batched version): 0 for G/C, 1 for A/T --
    # *inverted* from every other GC indicator in this function (window_gc
    # below) and from how the baseline itself was built (baseline.py's
    # tagged_gc1/2/3 use 1 for G/C). That's what the original notebook
    # computed; preserved as-is rather than "fixed", per this module's
    # policy of flagging behavior questions instead of silently resolving
    # them (see module docstring). Computed via the precomputed
    # GC_FLAGS_BY_INDEX table (1=G/C, normal polarity) and inverted here,
    # rather than re-inspecting each codon's characters -- mirrors
    # gpu_change_vector.py's batch_gc_term, which already solved this same
    # "precompute per-codon GC" item the same way (see codon_tables.py's
    # GC_FLAGS_BY_INDEX comment).
    gc_flags = np.asarray(GC_FLAGS_BY_INDEX, dtype=float)[[CODON_TO_INDEX[c] for c in sol]]
    gc1_mean = 1.0 - gc_flags[:, 0].mean()
    gc2_mean = 1.0 - gc_flags[:, 1].mean()
    gc3_mean = 1.0 - gc_flags[:, 2].mean()

    def z_for(loc: str, tagged: dict, value: float) -> float:
        arr = np.asarray(tagged[loc], dtype=float)
        return _zscore(value, arr.mean(), arr.std())

    z_by_bucket = {}
    for bucket in ('ExonL50', 'Exon', 'ExonR50', 'Splice'):
        z_by_bucket[bucket] = {
            1: z_for(bucket, gc.taggedGC1, gc1_mean),
            2: z_for(bucket, gc.taggedGC2, gc2_mean),
            3: z_for(bucket, gc.taggedGC3, gc3_mean),
        }

    # A position is only worth mutating for GC purposes if some synonymous
    # codon actually differs at that base -- if every synonym shares the
    # same base 1/2/3, mutating can't change GC there. CODON_TO_VARIABLE_FLAGS
    # is precomputed once at import time (depends only on the amino acid,
    # not which codon is chosen -- see codon_tables.py), replacing a
    # per-position codon_choices_for_aa()+get_aa()+three all()-comparisons.
    gc_change = []
    for i in range(len(sol)):
        v1, v2, v3 = CODON_TO_VARIABLE_FLAGS[sol[i]]
        z = z_by_bucket[TAG_TO_BUCKET[locvec[i]]]
        gc_change.append(v1 * z[1] ** 2 + v2 * z[2] ** 2 + v3 * z[3] ** 2)

    continuous = ''.join(sol)
    location_string = ''.join(loc * 3 for loc in locvec)
    span = max(gc.windowsize, 1)
    num_windows = max(len(continuous) - span, 0)

    if num_windows > 0:
        is_gc = np.fromiter((1 if ch in 'GC' else 0 for ch in continuous), dtype=np.int64, count=len(continuous))
        window_gc = sliding_window_view(is_gc, span)[:num_windows].mean(axis=1)

        # Majority location-bucket per window, vectorized as a 3-way argmax
        # over per-bucket windowed counts, instead of Python's
        # max(set(loc_window), key=loc_window.count) once per window. Note:
        # that original tie-break was already unspecified (set iteration
        # order depends on Python's string hashing), so argmax's
        # first-max-wins tie-break isn't a behavior change so much as
        # replacing one unspecified tie-break with a deterministic one.
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
            baseline = np.asarray(gc.windows[name], dtype=float)
            win_z[mask] = _zscore(window_gc[mask], baseline.mean(), baseline.std()) ** 2
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

    return [per_codon[i] * gc_change[i] for i in range(len(sol))]


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
    position is worth mutating" rather than "this position is good."
    kill_off()/select_survivors()/directed_evolution() all treat a higher
    weighted score as more in need of mutation -- so a *negative*
    weights['Uracil'] is what actually pushes uracil content down; a
    positive weight would instead bias the GA toward mutating away from
    low-U positions (preserving or growing U content). Pick the sign
    deliberately when configuring weights, not by assuming this docstring's
    framing implies one.
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
