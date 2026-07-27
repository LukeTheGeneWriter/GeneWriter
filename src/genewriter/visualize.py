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

import json
import random

import numpy as np

from .change_vector import registered_terms, score_changevec
from .classes import Proposed_Solution
from .codon_tables import encode_codons


def load_population_json(path: str) -> list:
    """Reload a population saved by ga.save_gen() -- lets this be pointed
    at a previously-saved generation, not just an in-memory `pop` right
    after a live run_ga()/run_schedule() call."""
    with open(path) as f:
        data = json.load(f)
    return [Proposed_Solution(entry['codons'], entry['number'], entry['change_vecs']) for entry in data]


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
    max_points: int = 2000,
    perplexity: float = 30.0,
    random_state: int = 0,
    size_by_count: bool = True,
    figsize_per_plot: tuple = (5.0, 5.0),
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
        sizes = 20.0 + 60.0 * (np.log1p(counts) / np.log1p(peak))

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
