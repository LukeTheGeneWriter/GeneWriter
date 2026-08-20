"""t-SNE visualization of a GA population's codon-choice ("genotype")
neighborhoods, colored by distance from optimal or any individual
change-vector term --
literally see hotspots: clusters of genotypes the GA converged on, colored
by why they're doing well (or badly).

Two things are deliberately decoupled, so picking a color mode never
re-runs the (expensive) embedding: the 2D t-SNE layout depends only on
codon-choice similarity (Hamming distance over synonymous codon choices --
the same "one mutation apart" relationship ga.nearest_neighbors() already
walks), while color is a separate scalar per point, read straight off each
individual's already-computed change_vecs. Request as many color modes as
you want (plot_population_tsne's color_by) and every subplot reuses the
one embedding, so different modes are directly comparable point-for-point.

Real-scale populations (tens of thousands of individuals) are subsampled
before embedding: exact t-SNE on a precomputed distance matrix is O(P^2)
memory (a 50,000-individual population would need a ~20GB float64
distance matrix), and a scatter plot of that many overlapping points
isn't visually useful anyway -- see plot_population_tsne's max_points.

Also here: plot_population_similarity_to_reference(), a different question
from the rest of the module -- not "why is the GA favoring this genotype"
but "how much of the population sits within X% nucleotide identity of one
fixed reference sequence" (e.g. a patent attorney's actual filing
candidate). Reuses the same t-SNE embedding machinery for spatial layout,
but colors by discrete similarity band to that one reference instead of a
continuous distance-from-optimal/term scale.
"""

import glob
import json
import os
import random
import re

import numpy as np

from .change_vector import calculate_change_vector, registered_terms, distance_from_optimal
from .classes import Proposed_Solution
from .codon_tables import encode_codons


def load_population_json(path: str) -> list:
    """Reload a population saved by ga.save_gen() -- lets this be pointed
    at a previously-saved generation, not just an in-memory `pop` right
    after a live run_ga()/run_schedule() call. entry.get('protected',
    False) rather than entry['protected'] -- a checkpoint saved before
    Proposed_Solution.protected existed has no such key at all."""
    with open(path) as f:
        data = json.load(f)
    return [Proposed_Solution(entry['codons'], entry['number'], entry['change_vecs'], entry.get('protected', False)) for entry in data]


_GEN_FILENAME_RE = re.compile(r"gen(\d+)\.json$")


def load_population_trajectory(save_dir: str, run_name: str) -> list:
    """Every generation checkpoint ga.save_gen() wrote for one run
    (save_dir/run_name/gen{gen}.json -- see that function's docstring),
    loaded and sorted into generation order.

    Returns list of (gen: int, pop: list[Proposed_Solution]) tuples.
    Raises FileNotFoundError if no checkpoint matches -- e.g. save_dir was
    None for the run that produced final_pop, so ga.save_gen()/schedule.py's
    "save" step never actually wrote anything to reload here.
    """
    pattern = os.path.join(save_dir, run_name, "gen*.json")
    entries = []
    for path in glob.glob(pattern):
        match = _GEN_FILENAME_RE.search(path)
        if not match:
            continue
        entries.append((int(match.group(1)), load_population_json(path)))
    if not entries:
        raise FileNotFoundError(
            f"No generation checkpoints found matching {pattern!r} -- was save_dir set "
            f"for the run that produced them (see ga.run_ga/schedule.run_schedule's "
            f"save_dir/run_name, or schedule.py's \"save\" step)?"
        )
    entries.sort(key=lambda pair: pair[0])
    return entries


def subsample_population(pop: list, max_points: int, rng: random.Random = None) -> list:
    """Uniform random subsample of distinct individuals down to at most
    max_points -- see module docstring for why this matters at real
    scale. Each sampled individual keeps its own `number` (replicate
    count) as-is (not resampled proportionally), so point-size encoding
    (see plot_population_tsne's size_by_count) still reflects the real
    population, just over fewer distinct genotypes."""
    if len(pop) <= max_points:
        return list(pop)
    rng = rng if rng is not None else random
    return rng.sample(list(pop), max_points)


def codon_distance_matrix(pop: list) -> np.ndarray:
    """(P, P) pairwise Hamming distance (fraction of differing positions)
    over each individual's codon choices -- the genotype-similarity metric
    the t-SNE embedding is built on. Every individual must share the same
    sequence length (true within one GA run/schedule -- synonymous
    substitutions only)."""
    from scipy.spatial.distance import pdist, squareform

    if len(pop) < 2:
        return np.zeros((len(pop), len(pop)), dtype=float)
    codon_idx = np.asarray([encode_codons(p.codons) for p in pop])
    return squareform(pdist(codon_idx, metric='hamming'))


def nucleotide_identity_to_reference(pop: list, reference_codons: list) -> np.ndarray:
    """Per-individual nucleotide-level percent identity (0-100) to one fixed
    reference codon sequence -- e.g. a specific sequence a patent attorney
    is evaluating for filing, or the natural CDS itself.

    Deliberately literal-nucleotide, not codon_distance_matrix()'s codon-
    CHOICE Hamming distance: two different codons that still share 1-2 of
    their 3 nucleotides (e.g. CTG vs CTC) count as partially identical here,
    matching how percent identity is actually reported for nucleic acid
    sequence claims (BLAST-style alignment identity), not the GA's own
    coarser internal genotype-similarity proxy used for t-SNE clustering.

    Every individual (and the reference) must have the same codon-list
    length -- true within one GA run/schedule, since every individual there
    encodes the same fixed amino acid sequence via synonymous substitutions
    only. Raises ValueError on a length mismatch rather than silently
    truncating/broadcasting, since a coverage number computed against the
    wrong-length pair would be meaningless (a patent-claim-scope figure
    someone might actually act on).
    """
    ref_len = len(reference_codons)
    if ref_len == 0:
        raise ValueError("reference_codons is empty")
    ref_nt = ''.join(reference_codons)
    values = np.empty(len(pop), dtype=float)
    for i, p in enumerate(pop):
        if len(p.codons) != ref_len:
            raise ValueError(
                f"individual {i} has {len(p.codons)} codons, reference has {ref_len} -- "
                f"nucleotide identity requires matching lengths (same amino acid sequence)"
            )
        nt = ''.join(p.codons)
        matches = sum(a == b for a, b in zip(nt, ref_nt))
        values[i] = 100.0 * matches / len(ref_nt)
    return values


def similarity_band_counts(identities: np.ndarray, thresholds=(99.9, 99, 95, 90, 85)) -> dict:
    """For each threshold T in `thresholds`, how many individuals have
    nucleotide identity >= T -- the "how many viable outputs would a claim
    of at least T% identity cover" number a patent attorney actually wants.

    Cumulative by construction (an individual 99.9% identical to the
    reference is also counted in the >=99%, >=95%, ... bins), matching how
    a percent-identity claim's coverage nests: a tighter claim's coverage is
    always a subset of a looser one's. Returns a dict keyed by threshold
    (sorted descending), not a list, so callers don't have to remember
    positional order.
    """
    return {t: int(np.sum(identities >= t)) for t in sorted(set(thresholds), reverse=True)}


def _assign_similarity_band(identities: np.ndarray, thresholds_desc: list) -> np.ndarray:
    """Per-individual index into thresholds_desc (already sorted descending)
    -- 0 = meets the tightest threshold, ..., len(thresholds_desc) = meets
    none of them. Unlike similarity_band_counts()'s cumulative counts, this
    assigns each point to exactly one (its best/tightest-met) band, for
    discrete per-point scatter coloring.
    """
    bands = np.full(identities.shape, len(thresholds_desc), dtype=int)
    unassigned = np.ones(identities.shape, dtype=bool)
    for band_idx, t in enumerate(thresholds_desc):
        meets = unassigned & (identities >= t)
        bands[meets] = band_idx
        unassigned &= ~meets
    return bands


def plot_population_similarity_to_reference(
    pop: list,
    reference_codons: list,
    thresholds: tuple = (99.9, 99, 95, 90, 85),
    max_points: int = 4000,
    perplexity: float = 30.0,
    random_state: int = 0,
    size_by_count: bool = True,
    size_range: tuple = (4.0, 24.0),
    figsize: tuple = (9.0, 8.0),
):
    """t-SNE scatter (same codon-choice-Hamming embedding as
    plot_population_tsne()) with points discretely colored by which
    similarity-to-reference band (see similarity_band_counts()) each falls
    into. Built for the "patent attorney wants to file a percent-identity
    claim on `reference_codons`" case: shows both how many viable outputs a
    given threshold would cover (the legend counts) and whether the ones
    falling outside a tighter band cluster together spatially -- a distinct
    genotype neighborhood that might be worth its own separate claim -- or
    are just scattered noise near the boundary.

    reference_codons: the specific sequence being evaluated for coverage --
        e.g. one individual from `pop` picked as the actual filing
        candidate, or an externally supplied sequence (must have the same
        codon count as every individual in pop -- same amino acid
        sequence). Not required to itself be a member of pop.
    thresholds: percent-identity cutoffs, e.g. the default
        (99.9, 99, 95, 90, 85) -- order doesn't matter, sorted descending
        internally.

    Band counts and the full per-individual identity array are computed
    against the FULL population (O(P * sequence length), cheap -- not the
    O(P^2) t-SNE embedding), not just the plotted subsample, so the
    coverage numbers are exact even when max_points forces the scatter
    itself to subsample for rendering.

    Returns (fig, band_counts, identities): band_counts is
    similarity_band_counts()'s dict over the FULL population; identities is
    the full per-individual percent-identity array (same order as pop) --
    both returned alongside the figure so the real numbers are available
    without having to read them off the plot.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if not pop:
        raise ValueError("pop is empty -- nothing to plot")

    thresholds_desc = sorted(set(thresholds), reverse=True)
    identities = nucleotide_identity_to_reference(pop, reference_codons)
    band_counts = similarity_band_counts(identities, thresholds_desc)

    sampled = subsample_population(pop, max_points, rng=random.Random(random_state))
    sampled_identities = nucleotide_identity_to_reference(sampled, reference_codons)
    n = len(sampled)
    effective_perplexity = min(perplexity, max(n - 1, 1))

    dist = codon_distance_matrix(sampled)
    coords = TSNE(
        n_components=2, metric='precomputed', init='random',
        perplexity=effective_perplexity, random_state=random_state,
    ).fit_transform(dist)

    bands = _assign_similarity_band(sampled_identities, thresholds_desc)
    sizes = _size_by_count(sampled, size_range) if size_by_count else None

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap('viridis', len(thresholds_desc) + 1)
    band_labels = [f">= {t:g}%" for t in thresholds_desc] + [f"< {thresholds_desc[-1]:g}%"]
    band_full_counts = [band_counts[t] for t in thresholds_desc] + [len(pop) - band_counts[thresholds_desc[-1]]]

    for band_idx, label in enumerate(band_labels):
        mask = bands == band_idx
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=sizes[mask] if sizes is not None else None,
            color=cmap(band_idx),
            label=f"{label} ({band_full_counts[band_idx]} of {len(pop)} total)",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='best', fontsize=8, title='Identity to reference')
    ax.set_title(f"Population similarity to reference sequence ({n} of {len(pop)} shown)")
    fig.tight_layout()
    return fig, band_counts, identities


def metric_color_values(pop: list, weights: dict, color_by: str) -> np.ndarray:
    """Per-individual color value for one color_by mode:

    - 'distance_from_optimal': the same weighted aggregate (change_vector.
      distance_from_optimal()) that ga.kill_off()/ga.select_survivors()
      actually use to decide who dies -- the real selection pressure
      driving the run, not a separate ad-hoc metric. Called 'fitness'
      until 2026-08-20.
    - a registered change-vector term name (e.g. 'GC', 'RareCodons',
      'Uracil'): that term's own raw (unweighted) per-individual sum --
      deliberately NOT multiplied by weights[term], so a term configured
      at weight 0 (Uracil defaults to 0.0 -- see Handoff.md) still shows
      its real signal instead of an all-zero color axis.

    Non-finite values are clamped to the finite extremes before returning
    -- see _sanitize_for_color()'s docstring for why this matters (caught
    on a real render against real sample gene data: RareCodons -- and, by
    summation, 'distance_from_optimal' -- came back all-inf and every point
    silently vanished from that panel).
    """
    if color_by == 'distance_from_optimal':
        values = np.asarray([distance_from_optimal(p.change_vecs, weights) for p in pop], dtype=float)
    else:
        terms = registered_terms()
        if color_by not in terms:
            hint = (" -- 'fitness' was renamed to 'distance_from_optimal' on 2026-08-20"
                    if color_by == 'fitness' else "")
            raise ValueError(
                f"Unknown color_by {color_by!r}{hint} -- must be "
                f"'distance_from_optimal' or one of {sorted(terms)}"
            )
        values = np.asarray([sum(p.change_vecs[color_by]) for p in pop], dtype=float)
    return _sanitize_for_color(values)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Raw color values (metric_color_values()'s output) rescaled to
    percentile rank in [0, 100] -- fixes the "everything looks the same
    shade" problem a raw min/max-normalized colormap has whenever a few
    outliers (e.g. an inf-clamped RareCodons score -- see
    _sanitize_for_color()) dominate the scale and compress everyone else
    into one end of it. Percentile rank only cares about each point's
    relative position in the sorted order, so contrast stays spread out
    across the full population regardless of the term's raw distribution.

    Ties get the average rank (matches scipy.stats.rankdata's default),
    so identical raw scores still render as the identical color.
    """
    if values.size <= 1:
        return np.zeros_like(values)
    from scipy.stats import rankdata
    return (rankdata(values, method='average') - 1) / (values.size - 1) * 100.0


def _size_by_count(pop: list, size_range: tuple = (4.0, 24.0)) -> np.ndarray:
    """Scatter point size from each individual's `number` (replicate
    count), log-scaled so a handful of massively-converged genotypes don't
    make every singly-visited one invisible by comparison."""
    counts = np.asarray([p.number for p in pop], dtype=float)
    peak = counts.max() if counts.max() > 0 else 1.0
    base, scale = size_range
    return base + scale * (np.log1p(counts) / np.log1p(peak))


def _sanitize_for_color(values: np.ndarray) -> np.ndarray:
    """Replace +-inf with the min/max *finite* value actually present (NaN
    with the mean of the finite values), so a point whose score happens to
    be infinite still renders as a visible extreme instead of silently
    vanishing.

    Why this happens at all: change_vector._rare_codon_term's odds_by_count
    defaults to float('inf') for a window composition never observed in
    the baseline (a small/local baseline -- exactly what real sample gene
    data or a quick synthetic test corpus gives you -- hits this far more
    than a real genome-wide Standards baseline would). 'distance_from_optimal' inherits
    it via distance_from_optimal()'s summation across terms. Passing an array
    containing inf as a matplotlib scatter `c=` value produces NaN after
    color normalization (min/max-based), which matplotlib then silently
    does not draw -- no error, no warning, just missing points. Caught by
    actually rendering a figure against real sample gene data, not by the
    smoke tests alone (their small synthetic baselines never happened to
    produce an inf term score).

    If every value is non-finite, falls back to all zeros rather than
    crashing or returning an unusable array.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    out = values.copy()
    out[np.isposinf(out)] = finite.max()
    out[np.isneginf(out)] = finite.min()
    out[np.isnan(out)] = finite.mean()
    return out


def plot_population_tsne(
    pop: list,
    weights: dict,
    color_by=('distance_from_optimal',),
    max_points: int = 4000,
    perplexity: float = 30.0,
    random_state: int = 0,
    size_by_count: bool = True,
    size_range: tuple = (4.0, 24.0),
    figsize_per_plot: tuple = (8.0, 8.0),
):
    """One t-SNE embedding (codon-choice Hamming distance -- see
    codon_distance_matrix()), plotted once per requested color_by mode,
    all sharing the same 2D layout.

    pop: list of Proposed_Solution -- a live run's population (e.g.
        ga.run_ga()'s return value) or load_population_json()'s output.
    weights: only consulted for color_by='distance_from_optimal' -- same weights dict
        passed to run_ga()/run_schedule().
    color_by: iterable of 'distance_from_optimal' and/or any registered change-vector
        term name (see change_vector.registered_terms()) -- one subplot
        per entry. Each mode's raw score (metric_color_values()) is
        rescaled to percentile rank (see _percentile_rank()) before
        coloring, so contrast stays spread across the full colormap
        instead of collapsing toward one shade whenever the raw
        distribution is skewed or outlier-dominated.
    max_points: pop is subsampled to at most this many distinct
        individuals before embedding -- see module docstring. Population
        size before subsampling is reported in the figure title.
    perplexity: t-SNE's own perplexity parameter, capped automatically if
        the (possibly subsampled) population is smaller than it.
    size_by_count: if True, scatter point size scales with each plotted
        individual's `number` (replicate count) -- a cluster of large
        points is a genotype the GA converged on hard, not just visited
        once.
    size_range: (min, max) point size passed to _size_by_count() when
        size_by_count is True; ignored (every point drawn at size_range[0])
        when False.

    Returns the matplotlib Figure -- caller decides whether to show/save
    it (e.g. `plt.show()` in a Colab cell, or `fig.savefig(...)`).
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if not pop:
        raise ValueError("pop is empty -- nothing to plot")

    color_by = list(color_by)
    if not color_by:
        raise ValueError("color_by must list at least one mode")

    sampled = subsample_population(pop, max_points, rng=random.Random(random_state))
    n = len(sampled)
    effective_perplexity = min(perplexity, max(n - 1, 1))

    dist = codon_distance_matrix(sampled)
    coords = TSNE(
        n_components=2, metric='precomputed', init='random',
        perplexity=effective_perplexity, random_state=random_state,
    ).fit_transform(dist)

    sizes = _size_by_count(sampled, size_range) if size_by_count else None

    fig, axes = plt.subplots(
        1, len(color_by), figsize=(figsize_per_plot[0] * len(color_by), figsize_per_plot[1]), squeeze=False,
    )
    axes = axes[0]

    for ax, mode in zip(axes, color_by):
        ranks = _percentile_rank(metric_color_values(sampled, weights, mode))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=ranks, s=sizes, cmap='viridis', vmin=0, vmax=100)
        ax.set_title(mode)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(scatter, ax=ax, shrink=0.8, label='percentile rank')

    fig.suptitle(f"t-SNE over codon-choice Hamming distance ({n} of {len(pop)} individuals shown)")
    fig.tight_layout()
    return fig


def _pool_and_embed_trajectory(
    save_dir: str,
    run_name: str,
    analysis_objects=None,
    natural_codons: list = None,
    locvec: list = None,
    max_points_per_gen: int = 500,
    perplexity: float = 30.0,
    random_state: int = 0,
):
    """Shared pooling + embedding step behind plot_population_trajectory()
    and animate_population_trajectory(): load every generation checkpoint,
    subsample each independently (see plot_population_trajectory's
    docstring for why per-generation, not global), optionally append the
    natural-CDS reference point, and embed everything in ONE shared t-SNE
    space so positions are comparable across generations.

    Returns (entries, pooled, gen_of, natural_idx, coords):
      entries: load_population_trajectory()'s (gen, pop) list, in order.
      pooled: list[Proposed_Solution], every subsampled individual across
        every generation, plus the natural reference point if given (last).
      gen_of: list[int | None], same length/order as pooled -- gen_of[i]
        is pooled[i]'s generation, or None for the natural reference point.
      natural_idx: index into pooled of the natural reference point, or
        None if natural_codons wasn't given.
      coords: (len(pooled), 2) shared t-SNE embedding.
    """
    if natural_codons is not None and analysis_objects is None:
        raise ValueError("natural_codons requires analysis_objects (to score it) to also be given")

    from sklearn.manifold import TSNE

    entries = load_population_trajectory(save_dir, run_name)
    rng = random.Random(random_state)

    pooled = []
    gen_of = []
    for gen, pop in entries:
        sampled = subsample_population(pop, max_points_per_gen, rng=rng)
        pooled.extend(sampled)
        gen_of.extend([gen] * len(sampled))

    natural_idx = None
    if natural_codons is not None:
        natural_vecs = calculate_change_vector(list(natural_codons), analysis_objects, locvec)
        natural_idx = len(pooled)
        pooled.append(Proposed_Solution(list(natural_codons), 1, natural_vecs))
        gen_of.append(None)

    n = len(pooled)
    if n < 2:
        raise ValueError("fewer than 2 points across every generation (plus the optional natural reference) -- nothing to embed")
    effective_perplexity = min(perplexity, max(n - 1, 1))

    dist = codon_distance_matrix(pooled)
    coords = TSNE(
        n_components=2, metric='precomputed', init='random',
        perplexity=effective_perplexity, random_state=random_state,
    ).fit_transform(dist)

    return entries, pooled, gen_of, natural_idx, coords


def plot_population_trajectory(
    save_dir: str,
    run_name: str,
    weights: dict,
    analysis_objects=None,
    natural_codons: list = None,
    locvec: list = None,
    color_by=('distance_from_optimal',),
    max_points_per_gen: int = 500,
    perplexity: float = 30.0,
    random_state: int = 0,
    size_by_count: bool = True,
    size_range: tuple = (4.0, 24.0),
    figsize_per_plot: tuple = (8.0, 8.0),
):
    """One shared t-SNE embedding across EVERY saved generation checkpoint
    of a run (ga.save_gen()'s save_dir/run_name -- see
    load_population_trajectory()), so you can see the population actually
    migrate across generations instead of one frozen snapshot. Sharing one
    embedding across generations matters even more here than across
    plot_population_tsne's color_by modes: point positions are only
    comparable generation-to-generation if the SAME embedding produced
    them, which is why every generation is pooled and embedded together
    rather than one t-SNE run per checkpoint.

    Each generation is independently subsampled to at most
    max_points_per_gen distinct individuals BEFORE pooling (see module
    docstring on why real-scale populations need subsampling at all) --
    kept separate per generation rather than one global subsample, so an
    early, larger generation can't drown out a later, smaller one.

    The first panel is always the trajectory itself: every point colored by
    its generation number (sequential colormap), with each generation's
    centroid connected by a line -- the actual "did the population move,
    and which way" signal this function exists for. One further panel is
    drawn per color_by entry (same distance-from-optimal/term semantics as
    plot_population_tsne, including the percentile-rank color rescaling --
    see _percentile_rank()), on the identical embedding, so you can check
    whether the migration in panel 1 lines up with a real
    distance-from-optimal/term
    improvement rather than random drift.

    natural_codons: the real target gene's own codon list (list[str]), if
    you have one -- e.g. the actual CDS being rewritten, so you can see
    where nature's own sequence already sits relative to where the GA's
    population has moved (useful for judging whether a rewrite target even
    needed to move very far). Embedded as one extra point in the SAME
    space, flagged with a distinct star marker in every panel, colored like
    every other point in the color_by panels but excluded from the
    generation panel's color scale (it has no generation). Requires
    analysis_objects (to score it) if given; None (default) omits it
    entirely -- e.g. for a TARGET_MODE="custom" run with no real gene to
    compare against.
    locvec: real 'F'/'T'/'I'/'S' location tags for natural_codons, if
    available (see gene_io/pick_target) -- None scores it as generic
    interior sequence, same default calculate_change_vector itself uses.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    entries, pooled, gen_of, natural_idx, coords = _pool_and_embed_trajectory(
        save_dir, run_name, analysis_objects=analysis_objects, natural_codons=natural_codons, locvec=locvec,
        max_points_per_gen=max_points_per_gen, perplexity=perplexity, random_state=random_state,
    )
    n = len(pooled)

    has_gen = np.asarray([g is not None for g in gen_of])
    gen_values = np.asarray([g if g is not None else 0 for g in gen_of], dtype=float)

    sizes = _size_by_count(pooled, size_range) if size_by_count else None

    color_by = list(color_by)
    panels = ['generation'] + color_by
    fig, axes = plt.subplots(
        1, len(panels), figsize=(figsize_per_plot[0] * len(panels), figsize_per_plot[1]), squeeze=False,
    )
    axes = axes[0]

    for ax, mode in zip(axes, panels):
        if mode == 'generation':
            scatter = ax.scatter(
                coords[has_gen, 0], coords[has_gen, 1], c=gen_values[has_gen],
                s=sizes[has_gen] if sizes is not None else None, cmap='plasma',
            )
            fig.colorbar(scatter, ax=ax, shrink=0.8, label='generation')

            unique_gens = sorted(set(g for g in gen_of if g is not None))
            centroids = np.asarray([coords[[i for i, g in enumerate(gen_of) if g == gg]].mean(axis=0) for gg in unique_gens])
            ax.plot(centroids[:, 0], centroids[:, 1], '-o', color='black', linewidth=1.5, markersize=4, alpha=0.7,
                    label='generation centroid')
            ax.set_title('generation (trajectory)')
        else:
            ranks = _percentile_rank(metric_color_values(pooled, weights, mode))
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=ranks, s=sizes, cmap='viridis', vmin=0, vmax=100)
            fig.colorbar(scatter, ax=ax, shrink=0.8, label='percentile rank')
            ax.set_title(mode)

        if natural_idx is not None:
            ax.scatter(
                coords[natural_idx, 0], coords[natural_idx, 1], marker='*', s=350,
                facecolors='none', edgecolors='red', linewidths=2, zorder=5, label='Natural CDS',
            )

        ax.set_xticks([])
        ax.set_yticks([])
        if natural_idx is not None or mode == 'generation':
            ax.legend(loc='best', fontsize=8)

    gens_str = ", ".join(str(g) for g, _pop in entries)
    fig.suptitle(f"Population trajectory over generations [{gens_str}] ({n} points embedded, run={run_name!r})")
    fig.tight_layout()
    return fig


def animate_population_trajectory(
    save_dir: str,
    run_name: str,
    weights: dict,
    analysis_objects=None,
    natural_codons: list = None,
    locvec: list = None,
    color_by=('distance_from_optimal',),
    max_points_per_gen: int = 500,
    perplexity: float = 30.0,
    random_state: int = 0,
    size_by_count: bool = True,
    size_range: tuple = (4.0, 24.0),
    figsize_per_plot: tuple = (6.0, 6.0),
    trail: str = 'both',
    trail_alpha: float = 0.15,
    fps: float = 2.0,
    out_path: str = None,
):
    """plot_population_trajectory(), animated: same shared t-SNE embedding
    across every saved generation checkpoint (see _pool_and_embed_trajectory()
    -- positions are only comparable generation-to-generation because every
    generation is pooled and embedded together), but instead of overlaying
    every generation on one static scatter, renders one frame per
    generation -- the population's fixed 2D space stays put (axis limits
    are set once, from every point across every generation) while which
    dots are lit up on it changes frame to frame, so you actually see it
    move rather than staring at one frozen superposition of every
    generation at once.

    color_by: iterable of 'distance_from_optimal' and/or any registered change-vector
        term name (see change_vector.registered_terms()) -- one column of
        panels per entry. Each mode's raw score is rescaled to percentile
        rank (see _percentile_rank()) ONCE, up front, so the color scale
        stays fixed across every frame -- a point's shade means the same
        thing in frame 1 as it does in frame 40.
    trail: what a frame shows besides the current generation:
      'current': only that generation's points are drawn -- positions
        reset each frame, the cleanest read of "where is the population
        RIGHT NOW".
      'fade': the current generation drawn at full opacity, every EARLIER
        generation also drawn (at trail_alpha) so a persistent low-opacity
        trail builds up across frames -- motion context, where the
        population CAME from, not just where it is.
      'both' (default): renders both side by side (one row per variant,
        one column per color_by mode) in the same gif, so you get either
        read without re-rendering.
    trail_alpha: opacity of earlier-generation points when trail is 'fade'
        or 'both'; ignored for the 'current'-only row.
    fps: playback speed, both for the live FuncAnimation interval and for
        the saved gif (when out_path is given).
    out_path: if given, saves the animation there (Pillow gif writer --
        picked automatically, no imageio/ffmpeg dependency needed).

    Returns the matplotlib.animation.FuncAnimation. Caller decides whether
    to display it inline (e.g. `HTML(anim.to_jshtml())` in a notebook) or
    save it (`anim.save(path, writer='pillow', fps=fps)`) if out_path
    wasn't given.
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    trail_variants = []  # False = current-generation-only row, True = +fading-trail row
    if trail in ('current', 'both'):
        trail_variants.append(False)
    if trail in ('fade', 'both'):
        trail_variants.append(True)
    if not trail_variants:
        raise ValueError(f"trail must be one of 'current', 'fade', 'both' -- got {trail!r}")

    entries, pooled, gen_of, natural_idx, coords = _pool_and_embed_trajectory(
        save_dir, run_name, analysis_objects=analysis_objects, natural_codons=natural_codons, locvec=locvec,
        max_points_per_gen=max_points_per_gen, perplexity=perplexity, random_state=random_state,
    )
    gen_of_arr = np.asarray([g if g is not None else -1 for g in gen_of])
    has_gen = gen_of_arr >= 0
    unique_gens = sorted(set(g for g in gen_of if g is not None))

    sizes = _size_by_count(pooled, size_range) if size_by_count else np.full(len(pooled), sum(size_range) / 2.0)

    color_by = list(color_by)
    rank_colors = {mode: _percentile_rank(metric_color_values(pooled, weights, mode)) for mode in color_by}

    nrows, ncols = len(trail_variants), len(color_by)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows), squeeze=False,
    )

    pad = 0.05 * max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1e-9)
    xlim = (coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ylim = (coords[:, 1].min() - pad, coords[:, 1].max() + pad)

    trail_scatters = {}
    current_scatters = {}
    for row, show_trail in enumerate(trail_variants):
        for col, mode in enumerate(color_by):
            ax = axes[row][col]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{mode}" + (" (+ trail)" if show_trail else ""))

            if show_trail:
                trail_scatters[(mode, show_trail)] = ax.scatter(
                    [], [], s=[], c=[], cmap='viridis', vmin=0, vmax=100, alpha=trail_alpha,
                )
            current_scatters[(mode, show_trail)] = ax.scatter([], [], s=[], c=[], cmap='viridis', vmin=0, vmax=100)
            fig.colorbar(current_scatters[(mode, show_trail)], ax=ax, shrink=0.8, label='percentile rank')

            if natural_idx is not None:
                ax.scatter(
                    coords[natural_idx, 0], coords[natural_idx, 1], marker='*', s=350,
                    facecolors='none', edgecolors='red', linewidths=2, zorder=5, label='Natural CDS',
                )
                ax.legend(loc='best', fontsize=8)

    def update(frame_idx):
        gen = unique_gens[frame_idx]
        cur_mask = has_gen & (gen_of_arr == gen)
        artists = []
        for (mode, show_trail), scatter in current_scatters.items():
            if show_trail:
                trail_mask = has_gen & (gen_of_arr < gen)
                trail_scatter = trail_scatters[(mode, show_trail)]
                trail_scatter.set_offsets(coords[trail_mask])
                trail_scatter.set_array(rank_colors[mode][trail_mask])
                trail_scatter.set_sizes(sizes[trail_mask])
                artists.append(trail_scatter)
            scatter.set_offsets(coords[cur_mask])
            scatter.set_array(rank_colors[mode][cur_mask])
            scatter.set_sizes(sizes[cur_mask])
            artists.append(scatter)
        fig.suptitle(f"Population trajectory -- generation {gen} ({int(cur_mask.sum())} points shown, run={run_name!r})")
        return artists

    anim = animation.FuncAnimation(fig, update, frames=len(unique_gens), interval=1000.0 / fps, blit=False)

    if out_path:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))

    return anim
