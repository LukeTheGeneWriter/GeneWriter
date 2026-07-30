"""t-SNE visualization of a GA population's codon-choice ("genotype")
neighborhoods, colored by fitness or any individual change-vector term --
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
"""

import glob
import json
import os
import random
import re

import numpy as np

from .change_vector import calculate_change_vector, registered_terms, score_changevec
from .classes import Proposed_Solution
from .codon_tables import encode_codons


def load_population_json(path: str) -> list:
    """Reload a population saved by ga.save_gen() -- lets this be pointed
    at a previously-saved generation, not just an in-memory `pop` right
    after a live run_ga()/run_schedule() call."""
    with open(path) as f:
        data = json.load(f)
    return [Proposed_Solution(entry['codons'], entry['number'], entry['change_vecs']) for entry in data]


_GEN_FILENAME_RE = re.compile(r"_gen(\d+)\.json$")


def load_population_trajectory(save_dir: str, run_name: str) -> list:
    """Every generation checkpoint ga.save_gen() wrote for one run
    (save_dir/run_name -- see that function's `{run_name}_gen{gen}.json`
    naming), loaded and sorted into generation order.

    Returns list of (gen: int, pop: list[Proposed_Solution]) tuples.
    Raises FileNotFoundError if no checkpoint matches -- e.g. save_dir was
    None for the run that produced final_pop, so ga.save_gen()/schedule.py's
    "save" step never actually wrote anything to reload here.
    """
    pattern = os.path.join(save_dir, f"{run_name}_gen*.json")
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


def fitness_color_values(pop: list, weights: dict, color_by: str) -> np.ndarray:
    """Per-individual color value for one color_by mode:

    - 'fitness': the same weighted score_changevec() total ga.kill_off()/
      ga.select_survivors() actually use to decide who dies -- the real
      selection pressure driving the run, not a separate ad-hoc metric.
    - a registered change-vector term name (e.g. 'GC', 'RareCodons',
      'Uracil'): that term's own raw (unweighted) per-individual sum --
      deliberately NOT multiplied by weights[term], so a term configured
      at weight 0 (Uracil defaults to 0.0 -- see Handoff.md) still shows
      its real signal instead of an all-zero color axis.

    Non-finite values are clamped to the finite extremes before returning
    -- see _sanitize_for_color()'s docstring for why this matters (caught
    on a real render against real sample gene data: RareCodons -- and, by
    summation, 'fitness' -- came back all-inf and every point silently
    vanished from that panel).
    """
    if color_by == 'fitness':
        values = np.asarray([score_changevec(p.change_vecs, weights) for p in pop], dtype=float)
    else:
        terms = registered_terms()
        if color_by not in terms:
            raise ValueError(f"Unknown color_by {color_by!r} -- must be 'fitness' or one of {sorted(terms)}")
        values = np.asarray([sum(p.change_vecs[color_by]) for p in pop], dtype=float)
    return _sanitize_for_color(values)


def _sanitize_for_color(values: np.ndarray) -> np.ndarray:
    """Replace +-inf with the min/max *finite* value actually present (NaN
    with the mean of the finite values), so a point whose score happens to
    be infinite still renders as a visible extreme instead of silently
    vanishing.

    Why this happens at all: change_vector._rare_codon_term's odds_by_count
    defaults to float('inf') for a window composition never observed in
    the baseline (a small/local baseline -- exactly what real sample gene
    data or a quick synthetic test corpus gives you -- hits this far more
    than a real genome-wide Standards baseline would). 'fitness' inherits
    it via score_changevec()'s summation across terms. Passing an array
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
    color_by=('fitness',),
    max_points: int = 4000,
    perplexity: float = 30.0,
    random_state: int = 0,
    size_by_count: bool = True,
    figsize_per_plot: tuple = (8.0, 8.0),
):
    """One t-SNE embedding (codon-choice Hamming distance -- see
    codon_distance_matrix()), plotted once per requested color_by mode,
    all sharing the same 2D layout.

    pop: list of Proposed_Solution -- a live run's population (e.g.
        ga.run_ga()'s return value) or load_population_json()'s output.
    weights: only consulted for color_by='fitness' -- same weights dict
        passed to run_ga()/run_schedule().
    color_by: iterable of 'fitness' and/or any registered change-vector
        term name (see change_vector.registered_terms()) -- one subplot
        per entry.
    max_points: pop is subsampled to at most this many distinct
        individuals before embedding -- see module docstring. Population
        size before subsampling is reported in the figure title.
    perplexity: t-SNE's own perplexity parameter, capped automatically if
        the (possibly subsampled) population is smaller than it.
    size_by_count: if True, scatter point size scales with each plotted
        individual's `number` (replicate count) -- a cluster of large
        points is a genotype the GA converged on hard, not just visited
        once.

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

    sizes = None
    if size_by_count:
        counts = np.asarray([p.number for p in sampled], dtype=float)
        peak = counts.max() if counts.max() > 0 else 1.0
        sizes = 10.0 + 60.0 * (np.log1p(counts) / np.log1p(peak))

    fig, axes = plt.subplots(
        1, len(color_by), figsize=(figsize_per_plot[0] * len(color_by), figsize_per_plot[1]), squeeze=False,
    )
    axes = axes[0]

    for ax, mode in zip(axes, color_by):
        colors = fitness_color_values(sampled, weights, mode)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, cmap='viridis')
        ax.set_title(mode)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(scatter, ax=ax, shrink=0.8)

    fig.suptitle(f"t-SNE over codon-choice Hamming distance ({n} of {len(pop)} individuals shown)")
    fig.tight_layout()
    return fig


def plot_population_trajectory(
    save_dir: str,
    run_name: str,
    weights: dict,
    analysis_objects=None,
    natural_codons: list = None,
    locvec: list = None,
    color_by=('fitness',),
    max_points_per_gen: int = 500,
    perplexity: float = 30.0,
    random_state: int = 0,
    size_by_count: bool = True,
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
    drawn per color_by entry (same fitness/term semantics as
    plot_population_tsne), on the identical embedding, so you can check
    whether the migration in panel 1 lines up with a real fitness/term
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
    from sklearn.manifold import TSNE

    if natural_codons is not None and analysis_objects is None:
        raise ValueError("natural_codons requires analysis_objects (to score it) to also be given")

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

    has_gen = np.asarray([g is not None for g in gen_of])
    gen_values = np.asarray([g if g is not None else 0 for g in gen_of], dtype=float)

    sizes = None
    if size_by_count:
        counts = np.asarray([p.number for p in pooled], dtype=float)
        peak = counts.max() if counts.max() > 0 else 1.0
        sizes = 10.0 + 60.0 * (np.log1p(counts) / np.log1p(peak))

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
            colors = fitness_color_values(pooled, weights, mode)
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, cmap='viridis')
            fig.colorbar(scatter, ax=ax, shrink=0.8)
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
