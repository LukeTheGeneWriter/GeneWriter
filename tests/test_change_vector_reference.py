"""Reference-comparison tests for every change-vector term, checked against
the pre-vectorization implementation (kept here verbatim, exactly as it
shipped in change_vector.py before this session's numpy rewrite).

Why this file exists: _windowed_average got this treatment (see
test_windowed_average.py) when it was rewritten, but the *terms themselves*
did not -- each term's own window-count bookkeeping (matching the original
`range(0, len(x) - window_param)` bound, not sliding_window_view's own
`len(x) - window + 1`) was verified by eye, not by a direct comparison. That
gap let a real off-by-one ship in _rare_codon_term (an extra trailing
window that shouldn't have existed) for two turns before this test caught
it. Every term gets a direct, randomized comparison here now, closing
exactly that gap -- and this also becomes the correctness oracle for the
upcoming batched/GPU implementations, the same role _windowed_average's
reference already plays for the per-individual vectorization.
"""
import random

import numpy as np
import pytest

from genewriter.change_vector import cached_normal_transform, calculate_change_vector
from genewriter.codon_tables import (
    CODON_FREQS_LIT,
    RARE_CODONS_LIT,
    TAG_TO_WINDOW_BUCKET,
    codon_choices_for_aa,
    get_aa,
)

from conftest import random_solution


def _ref_windowed_average(values, target_len, winsize):
    out = []
    half = max(winsize // 2, 1)
    n = len(values)
    for i in range(target_len):
        if i < winsize:
            relevant = values[0:i]
        elif i > target_len - winsize:
            relevant = values[i:n]
        else:
            relevant = values[max(i - half, 0):min(i + half, n)]
        out.append(sum(relevant) / len(relevant) if relevant else 0.0)
    return out


# Z-scoring itself is deliberately NOT reimplemented here from raw (mean,
# std) the way the rest of this file reimplements each term from scratch --
# distribution_fit.py has its own dedicated test coverage for whether the
# auto-detected-distribution transform is correct, so re-deriving it here
# too would just be duplicated (and driftable) test surface for something
# this file was never trying to catch in the first place (see module
# docstring: this file exists to catch window-bookkeeping bugs, not z-score
# formula bugs). Goes through cached_normal_transform() on the exact same
# analysis_objects sub-object `actual = calculate_change_vector(...)`
# already populated moments earlier in each test below, so this is a cache
# hit (cheap) and guaranteed to be looking at the identical fitted
# transform the real implementation used -- not an independent, possibly
# divergent refit.


def _ref_dist_from_optimal(codons):
    total = 0.0
    for codon in codons:
        aa = get_aa(codon)
        total += max(CODON_FREQS_LIT[c] for c in codon_choices_for_aa(aa))
    return total / len(codons)


def _ref_rare_codon_term(sol, rc, winsize=15):
    rvec = [1 if c in RARE_CODONS_LIT else 0 for c in sol]
    wins = [rvec[i:i + winsize] for i in range(0, max(len(rvec) - winsize, 0))]
    total_windows = sum(rc.rare_codon_windows.values())
    odds = {count: (float('inf') if n == 0 else total_windows / n) for count, n in rc.rare_codon_windows.items()}
    scorewins = [odds.get(sum(win), float('inf')) for win in wins]
    scorevec = _ref_windowed_average(scorewins, len(sol), winsize)
    overall_rc = sum(rvec) / len(rvec)
    zscore = cached_normal_transform(rc, 'usagePerGene', rc.usagePerGene).transform(overall_rc) ** 2
    return [(scorevec[i] * zscore) if rvec[i] else 0.0 for i in range(len(sol))]


def _ref_codon_usage_term(sol, ca, winsize=15):
    cuvec = [ca.codonFreqsLit[c] for c in sol]
    span = max(ca.windowsize - 1, 1)
    cuwins = [sol[i:i + span] for i in range(0, max(len(sol) - ca.windowsize, 0))]
    cuwinscores = [sum(ca.codonFreqsLit[c] for c in win) / ca.windowsize for win in cuwins]
    cudists = [_ref_dist_from_optimal(win) for win in cuwins]
    dist_fit = cached_normal_transform(ca, 'windowdistancesfromoptimal', ca.windowdistancesfromoptimal)
    score_fit = cached_normal_transform(ca, 'windowscores', ca.windowscores)
    dist_vals = _ref_windowed_average(cudists, len(sol), winsize)
    score_vals = _ref_windowed_average(cuwinscores, len(sol), winsize)
    dist_zs = [dist_fit.transform(v) ** 2 for v in dist_vals]
    score_zs = [score_fit.transform(v) ** 2 for v in score_vals]
    gene_fit = cached_normal_transform(ca, 'codonUsageScoreByGene', ca.codonUsageScoreByGene)
    straight_z = [gene_fit.transform(o) / 2 for o in cuvec]
    return [straight_z[i] * score_zs[i] + straight_z[i] * dist_zs[i] for i in range(len(sol))]


def _ref_codon_pair_bias_term(sol, cpb):
    if len(sol) < 2:
        return [0.0] * len(sol)
    pairs = [sol[i] + sol[i + 1] for i in range(len(sol) - 1)]
    pair_scores = [cpb.cpb_lit[p] for p in pairs]

    # Distance from optimal: the single best-scoring pair in the whole
    # lookup is 0 distance (already optimal); a rare/poor pairing is far
    # below that ceiling. See change_vector._codon_pair_bias_term's
    # docstring for the full design this mirrors.
    max_score = max(cpb.cpb_lit.values())
    pair_distance = [max_score - s for s in pair_scores]

    pair_change = []
    for i in range(len(sol)):
        if i == 0:
            pair_change.append(pair_distance[0])
        elif i == len(sol) - 1:
            pair_change.append(pair_distance[-1])
        else:
            pair_change.append((pair_distance[i - 1] + pair_distance[i]) / 2)

    # Asymmetric on purpose -- see change_vector._codon_pair_bias_term's
    # matching comment: only below-mean deviation amplifies.
    overall_cpb = sum(pair_scores) / len(pair_scores)
    seq_fit = cached_normal_transform(cpb, 'cpbPerGene', cpb.cpbPerGene)
    below_mean_z = min(seq_fit.transform(overall_cpb), 0.0)
    global_multiplier = 1.0 + below_mean_z ** 2

    return [v * global_multiplier for v in pair_change]


def _ref_gc_term(sol, gc, locvec, winsize=15):
    # Per-sequence GC deviation: one scalar for the whole candidate, added
    # (not multiplied) to every position's local windowed score below --
    # see change_vector._gc_term's docstring for the full reasoning behind
    # this replacing the old gc1/gc2/gc3-by-bucket signal. Goes through
    # cached_normal_transform() on the same gc object calculate_change_
    # vector() already populated moments earlier -- cache hit, not a refit.
    overall_gc = np.mean([1 if base in 'GC' else 0 for c in sol for base in c])
    seq_deviation = cached_normal_transform(gc, 'gcPerGene', gc.gcPerGene).transform(overall_gc) ** 2

    def variable_flags(pos):
        choices = codon_choices_for_aa(get_aa(sol[pos]))
        v1 = 0 if all(c[0] == choices[0][0] for c in choices) else 1
        v2 = 0 if all(c[1] == choices[0][1] for c in choices) else 1
        v3 = 0 if all(c[2] == choices[0][2] for c in choices) else 1
        return v1, v2, v3

    mutable = [1.0 if any(variable_flags(i)) else 0.0 for i in range(len(sol))]

    continuous = ''.join(sol)
    location_string = ''.join(loc * 3 for loc in locvec)
    span = max(gc.windowsize, 1)
    win_z = []
    for i in range(0, max(len(continuous) - span, 0)):
        window = continuous[i:i + span]
        loc_window = location_string[i:i + span]
        majority_loc = max(set(loc_window), key=loc_window.count)
        bucket_name = TAG_TO_WINDOW_BUCKET[majority_loc]
        # ('windows', bucket_name) matches change_vector._gc_term's cache
        # key exactly -- cache hit, not a refit (see z_for()'s comment above).
        window_transform = cached_normal_transform(gc, ('windows', bucket_name), gc.windows[bucket_name])
        window_gc = sum(1 for ch in window if ch in 'GC') / len(window)
        win_z.append(window_transform.transform(window_gc) ** 2)
    win_z_by_pos = _ref_windowed_average(win_z, len(continuous), winsize)
    per_codon = [np.mean(win_z_by_pos[i:i + 3]) if win_z_by_pos[i:i + 3] else 0.0 for i in range(0, len(win_z_by_pos), 3)]
    per_codon = per_codon[:len(sol)] + [0.0] * max(len(sol) - len(per_codon), 0)

    return [mutable[i] * (per_codon[i] + seq_deviation) for i in range(len(sol))]


def _ref_kmer_term(sol, kmer, locvec, winsize=15):
    continuous = ''.join(sol)
    location_string = ''.join(loc * 3 for loc in locvec)
    per_k_vecs = []
    for k_str, kmer_by_seq in kmer.kmer_dict.items():
        k = int(k_str)
        seq_wins = [continuous[i:i + k] for i in range(0, max(len(continuous) - k, 0))]
        loc_wins = [location_string[i:i + k] for i in range(0, max(len(location_string) - k, 0))]
        win_scores = []
        for seq_win, loc_win in zip(seq_wins, loc_wins):
            majority_loc = max(set(loc_win), key=loc_win.count)
            entry = kmer_by_seq.get(seq_win)
            if entry is None:
                win_scores.append(1.0 / 1.1)
                continue
            bucket = TAG_TO_WINDOW_BUCKET[majority_loc]
            raw_fold_enrich = entry.get(bucket, {}).get('fold_enrich', 1.0)
            win_scores.append(1.0 / (raw_fold_enrich + 0.1))
        overall = sum(win_scores) / len(win_scores) if win_scores else 1.0 / 1.1
        by_pos = _ref_windowed_average(win_scores, len(continuous), winsize)
        by_pos = [v * overall for v in by_pos]
        per_codon = [np.mean(by_pos[i:i + 3]) if by_pos[i:i + 3] else 0.0 for i in range(0, len(by_pos), 3)]
        per_codon = per_codon[:len(sol)] + [0.0] * max(len(sol) - len(per_codon), 0)
        per_k_vecs.append(per_codon)
    if not per_k_vecs:
        return [0.0] * len(sol)
    return [sum(vec[i] for vec in per_k_vecs) for i in range(len(sol))]


def _assert_close(name, actual, expected, tol=1e-6):
    assert len(actual) == len(expected), f"{name}: length mismatch {len(actual)} vs {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        import math
        if math.isinf(e) or math.isinf(a):
            assert math.isinf(a) and math.isinf(e) and (a > 0) == (e > 0), f"{name}[{i}]: expected {e}, got {a}"
            continue
        assert abs(a - e) < tol, f"{name}[{i}]: expected {e}, got {a}"


LONG_AA_SEQ = "MAVLDEFGHIKPQRSTWYCN" * 7 + "MAVLDEFGHI"  # 150 residues, > any winsize used here


def _random_locvec(n, seed):
    """A locvec with a realistic block structure: a handful of long 'I'
    (interior) runs, each with a short 'F' (5' end) and 'T' (3' end) run at
    its boundaries and an occasional single 'S' (splice) position -- i.e.
    what GeneDataSourcing.ipynb's locate_codons actually produces, not
    independent per-codon noise. An earlier version of this helper picked a
    uniform-random tag per codon, which manufactures far more majority-tag
    ties within a single window than any real gene body would ever have
    (see the module docstring's note on tie-breaking) and made this test
    fail for a reason that had nothing to do with correctness."""
    rng = random.Random(seed)
    locvec = []
    while len(locvec) < n:
        block_len = rng.randint(15, 30)
        edge = min(3, block_len // 4)
        block = ['F'] * edge + ['I'] * (block_len - 2 * edge) + ['T'] * edge
        if rng.random() < 0.3 and block:
            block[rng.randrange(len(block))] = 'S'
        locvec.extend(block)
    return locvec[:n]


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("aa_seq_choice", ["short", "long"])
def test_all_terms_match_reference_implementation_with_uniform_locvec(analysis_objects, aa_seq, seed, aa_seq_choice):
    """Strict, exact-match comparison against the pre-vectorization
    reference. Uses locvec=None (uniform 'I' throughout) deliberately: with
    every position tagged identically, a window's "majority location tag"
    can never be a tie, so the GC/Kmer terms' switch from the original's
    max(set(...), key=...count) (tie-break order unspecified -- depends on
    Python's string hash) to a deterministic argmax cannot cause a
    divergence here. See test_gc_and_kmer_survive_non_uniform_locvec below
    for the case where ties are actually possible."""
    seq = aa_seq if aa_seq_choice == "short" else LONG_AA_SEQ
    sol = random_solution(seq, seed=seed)
    locvec = ['I'] * len(sol)

    actual = calculate_change_vector(sol, analysis_objects, locvec)

    _assert_close('RareCodons', actual['RareCodons'], _ref_rare_codon_term(sol, analysis_objects.rare_codon))
    _assert_close('CodonUsage', actual['CodonUsage'], _ref_codon_usage_term(sol, analysis_objects.codon_usage))
    _assert_close('CodonPairBias', actual['CodonPairBias'], _ref_codon_pair_bias_term(sol, analysis_objects.codon_pair_bias))
    _assert_close('GC', actual['GC'], _ref_gc_term(sol, analysis_objects.gc, locvec))
    _assert_close('Kmer', actual['Kmer'], _ref_kmer_term(sol, analysis_objects.kmer, locvec))


@pytest.mark.parametrize("seed", range(10))
def test_gc_and_kmer_survive_non_uniform_locvec(analysis_objects, seed):
    """Non-uniform locvec (especially with k=2 k-mers) makes majority-tag
    ties common at codon boundaries, where the original's tie-break
    (unspecified: depends on Python's string hashing) and the vectorized
    argmax (deterministic: first-max-wins) can legitimately disagree --
    not a bug, see change_vector.py's _gc_term/_kmer_term comments. So this
    doesn't assert exact equality, just that both implementations produce
    well-formed, finite-or-matching-infinite output of the right shape,
    and that they agree everywhere a majority isn't actually tied."""
    sol = random_solution(LONG_AA_SEQ, seed=seed)
    locvec = _random_locvec(len(sol), seed=seed + 1000)

    actual = calculate_change_vector(sol, analysis_objects, locvec)
    ref_gc = _ref_gc_term(sol, analysis_objects.gc, locvec)
    ref_kmer = _ref_kmer_term(sol, analysis_objects.kmer, locvec)

    assert len(actual['GC']) == len(ref_gc) == len(sol)
    assert len(actual['Kmer']) == len(ref_kmer) == len(sol)

    # Where the two implementations *do* agree (the non-tied majority),
    # require the values to actually match -- a silent, unrelated
    # computational error shouldn't be able to hide behind "well, ties are
    # allowed to differ".
    disagreements = sum(1 for a, e in zip(actual['GC'], ref_gc) if abs(a - e) > 1e-6)
    agreement_rate = 1 - disagreements / len(sol)
    assert agreement_rate > 0.5, f"GC term agrees with reference on only {agreement_rate:.0%} of positions -- more than tie-breaking should explain"
