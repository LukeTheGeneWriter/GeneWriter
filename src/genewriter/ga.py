"""Genetic algorithm loop over synonymous codon choices.

Ported from GA()/kill_off()/replicate_and_mutate_random()/directed_evolution()
in GeneRider_Cloud.ipynb, fixed to:

  - Use Proposed_Solution (the dataclass that actually exists in
    GeneClassesCloud.ipynb) instead of the undefined `ProposedSolution`.
  - Thread `analysis_objects` and `weights` through explicitly instead of
    reading them off an implicit global (`aobjs`) set up earlier in the
    notebook -- makes this testable and usable outside a single Colab
    session.
  - Take a `save_dir` parameter instead of a hardcoded Google Drive path.
  - Clamp change-vector scores to a finite range before using them as
    random.choices() weights. calculate_change_vector's RareCodons term can
    legitimately return float('inf') (a window's exact rare-codon count was
    never observed in the baseline), and random.choices() raises ValueError
    on a non-finite (inf or nan) weight -- unclamped, that crashes the GA
    the first time it encounters a rare-codon-window composition the
    baseline never saw. Python's builtin min()/max() do NOT reliably filter
    NaN (comparisons against NaN are always False), so this checks
    math.isfinite() explicitly rather than just clamping with min/max.
  - The original kill_off() floored every individual's replicate count at 1
    (`if pop[i].number > 1: ...`), so nothing was ever fully culled and the
    distinct-individual count could only grow generation over generation --
    confirmed against real sample data: 3 unbounded generations went
    29 -> 180 -> 491 distinct individuals, 2.8s -> 6.8s -> 41s. select_survivors()
    now caps the population to a target size each generation (a *soft*,
    weighted top-N cut, not a hard sort-and-slice -- see its docstring), and
    kill_off() itself can now fully remove an individual instead of
    flooring at 1.

select_survivors() and flatten_generation() are new, not ports of anything
in the source notebooks -- see their docstrings.

Two further hardening passes, motivated by discussing real-scale runs (tens
of thousands of individuals) rather than any specific bug report:

  - Existence checks against the population (`next((q for q in pop if
    q.codons == rep), None)`) were an O(n) linear scan per lookup -- O(n^2)
    per generation once you add up every rep. Replaced with a dict keyed by
    tuple(codons) (see merge_replicate()), which also fixes a latent
    correctness bug: the old scan only checked the *pre-generation*
    population, so two different individuals' offspring landing on the same
    new genotype in the same generation became two separate duplicate
    entries instead of merging.
  - select_survivors() used to call random.choices() in a loop, deleting
    one element and rebuilding the whole cumulative-weight table each time
    -- also O(n^2). Replaced with the Efraimidis-Spirakis weighted
    sampling-without-replacement algorithm (see its docstring), O(n log k).
"""

import heapq
import json
import math
import os
import random

from .change_vector import AnalysisObjects, calculate_change_vector, diff_change_vector, require_weights, score_changevec
from .classes import Proposed_Solution
from .codon_tables import generate_codon_vec

_MAX_FINITE_WEIGHT = 1e12


def _finite_nonneg(value: float) -> float:
    if math.isnan(value):
        return 1e-9
    if math.isinf(value):
        return _MAX_FINITE_WEIGHT if value > 0 else 1e-9
    return min(max(value, 0.0), _MAX_FINITE_WEIGHT) + 1e-9


def merge_replicate(
    pop_index: dict,
    sol: list,
    analysis_objects: AnalysisObjects,
    locvec: list = None,
    parent: Proposed_Solution = None,
) -> None:
    """Add one replicate of sol into pop_index (keyed by tuple(codons)),
    incrementing an existing entry's count instead of duplicating it.

    Existence checks against the population used to be a linear scan
    (`next((q for q in pop if q.codons == rep), None)`), i.e. O(n) per
    lookup -- fine for a handful of individuals, but quadratic overall at
    the population sizes an actual run needs (thousands to tens of
    thousands). A dict keyed by the codon tuple makes this O(1).

    parent: if given, a genuinely new genotype's change vector is
    approximated from parent.change_vecs via diff_change_vector() instead
    of a full (expensive) calculate_change_vector() -- see that function's
    docstring for what this trades away. Omit when there's no real parent
    to diff against (e.g. a freshly-generated random seed).
    """
    key = tuple(sol)
    existing = pop_index.get(key)
    if existing is not None:
        existing.number += 1
    elif parent is not None:
        vecs = diff_change_vector(parent.codons, parent.change_vecs, sol, analysis_objects, locvec)
        pop_index[key] = Proposed_Solution(sol, 1, vecs)
    else:
        vecs = calculate_change_vector(sol, analysis_objects, locvec)
        pop_index[key] = Proposed_Solution(sol, 1, vecs)


def refresh_change_vectors(pop: list, analysis_objects: AnalysisObjects, locvec: list = None) -> list:
    """Recompute every individual's change vector exactly (in place) rather
    than trusting whatever approximate diff it may have accumulated during
    growth. Meant to be called right before a step that makes a survival
    decision based on those scores (kill_off, select_survivors) or that
    otherwise warrants a clean baseline (flatten_generation) -- see
    schedule.py's kill_off/select/flatten steps, which always do this
    first."""
    for p in pop:
        p.change_vecs = calculate_change_vector(p.codons, analysis_objects, locvec)
    return pop


def degree_of_degeneracy(aa_seq: str) -> int:
    deg_dict = {
        'S': 6, 'L': 6, 'C': 2, 'W': 1, 'E': 2, 'D': 2, 'P': 4, 'V': 4, 'N': 2, 'M': 1,
        'K': 2, 'Y': 2, 'I': 3, 'Q': 2, 'F': 2, 'R': 6, 'T': 4, '*': 3, 'A': 4, 'G': 4, 'H': 2,
    }
    deg = 1
    for aa in aa_seq:
        deg *= deg_dict[aa]
    return deg


def generate_seed(aa_seq: str) -> list:
    codon_vec = generate_codon_vec(aa_seq)
    return [random.choice(choices) for choices in codon_vec]


def codvec_to_str(codvecs: list) -> list:
    return [''.join(cv) for cv in codvecs]


def str_to_codvec(seqs: list) -> list:
    return [[seq[i:i + 3] for i in range(0, len(seq), 3)] for seq in seqs]


def replicate_and_mutate_random(sol: list, aa_seq: str, nreplicates: int = 10, mutation_rate: float = 0.05) -> list:
    codon_vec = generate_codon_vec(aa_seq)
    replicates = []
    for _ in range(nreplicates):
        new_sol = sol.copy()
        for j in range(len(sol)):
            if random.random() < mutation_rate:
                new_sol[j] = random.choice(codon_vec[j])
        replicates.append(new_sol)
    return replicates


def directed_evolution(
    sol: list,
    changevecs: dict,
    weights: dict,
    aa_seq: str,
    analysis_objects: AnalysisObjects,
    locvec: list = None,
    nreplicates: int = 10,
    lookahead: bool = True,
) -> list:
    """Bias mutation toward high-change-vector positions.

    lookahead=True (default): at the chosen position, greedily pick the
    synonymous substitution with the best one-step lookahead score --
    scored via diff_change_vector() (cheap: each alternative is a
    single-codon change from sol, exactly what that function is for), not a
    full calculate_change_vector() per alternative.
    lookahead=False: skip scoring alternatives entirely and pick one
    uniformly at random. Still "directed" in the sense that *which
    position* gets mutated is still weighted by the change vector -- this
    only removes the expensive part (comparing every synonymous option at
    that position).
    """
    require_weights(changevecs.keys(), weights)
    codon_vec = generate_codon_vec(aa_seq)
    weighted_positions = [
        sum(weights[key] * changevecs[key][i] for key in changevecs) for i in range(len(sol))
    ]
    ranked = sorted(range(len(sol)), key=lambda i: weighted_positions[i], reverse=True)
    position_weights = [_finite_nonneg(weighted_positions[i]) for i in ranked]

    replicates = []
    for _ in range(nreplicates):
        choice_idx = random.choices(range(len(ranked)), weights=position_weights)[0]
        position = ranked[choice_idx]

        current_codon = sol[position]
        alternatives = [c for c in codon_vec[position] if c != current_codon]
        if not alternatives:
            continue

        if not lookahead:
            new_sol = sol.copy()
            new_sol[position] = random.choice(alternatives)
            replicates.append(new_sol)
            continue

        best_codon, best_score = None, None
        for alt in alternatives:
            candidate = sol.copy()
            candidate[position] = alt
            candidate_vecs = diff_change_vector(sol, changevecs, candidate, analysis_objects, locvec)
            candidate_score = score_changevec(candidate_vecs, weights)
            if best_score is None or candidate_score < best_score:
                best_codon, best_score = alt, candidate_score

        new_sol = sol.copy()
        new_sol[position] = best_codon
        replicates.append(new_sol)
    return replicates


def nearest_neighbors(sol: list, aa_seq: str) -> list:
    """All single-codon-substitution neighbors of sol: same amino acid
    sequence, exactly one position swapped to a different synonymous codon."""
    codon_vec = generate_codon_vec(aa_seq)
    neighbors = []
    for i in range(len(sol)):
        for alt in codon_vec[i]:
            if alt != sol[i]:
                neighbor = sol.copy()
                neighbor[i] = alt
                neighbors.append(neighbor)
    return neighbors


def select_survivors(pop: list, weights: dict, target_size: int) -> list:
    """Stochastically cut the population down to at most target_size
    individuals, weighted toward removing individuals whose change vector
    says they most need mutation.

    This is a *soft* top-N: each removal is (in effect) a weighted random
    draw, not a hard sort-and-slice, so a currently-weak individual can
    survive by chance and a currently-strong one can occasionally be cut --
    preserving some diversity rather than collapsing to a single local
    optimum.

    Implemented as weighted random sampling *without* replacement
    (Efraimidis & Spirakis, 2006) rather than repeatedly calling
    random.choices() and deleting one element at a time: that loop rebuilds
    a cumulative-weight table from scratch on every single removal, which is
    O(n) per removal and O(n^2) overall -- at a population of even a few
    thousand this is already the dominant cost of a generation, and at the
    tens-of-thousands scale a real run needs it's unusable. The
    Efraimidis-Spirakis trick assigns each individual a one-time priority
    key = ln(u)/die_weight (u ~ Uniform(0,1)) and removes the target_size
    - len(pop) individuals with the largest keys; this is a proven exact
    algorithm for weighted sampling without replacement (ranking by
    ln(u)/w is order-equivalent to ranking by u**(1/w), just numerically
    stable), and runs in O(n log k) via a heap.
    """
    pop = list(pop)
    num_to_remove = len(pop) - target_size
    if num_to_remove <= 0:
        return pop
    die_weights = [_finite_nonneg(score_changevec(p.change_vecs, weights)) for p in pop]
    keys = [math.log(max(random.random(), 1e-300)) / w for w in die_weights]
    victim_indices = set(heapq.nlargest(num_to_remove, range(len(pop)), key=lambda i: keys[i]))
    return [p for i, p in enumerate(pop) if i not in victim_indices]


def _flatten_round(pop: list, aa_seq: str, analysis_objects: AnalysisObjects, locvec: list = None) -> list:
    """One round of flatten_generation's cash-in: every individual's
    round-start replicate count is redistributed, one unit at a time, onto
    a randomly-drawn single-mutation neighbor (incrementing it if already
    present in the population, creating it at count 1 otherwise).

    This step alone is conservative: the total replicate count across the
    population is unchanged (every unit removed from an individual becomes
    exactly one unit added somewhere else). flatten_generation's *final*
    collapse-to-1 step is not part of this -- that intentionally changes
    the total to match the distinct-individual count, by design.
    """
    pop = list(pop)
    by_codons = {tuple(p.codons): p for p in pop}
    # Snapshot counts before any cashing-in happens this round: if we read
    # p.number at iteration time instead, an individual processed later in
    # the round could cash in copies it only just received from an earlier
    # individual's draws this same round, contradicting the "only cashes in
    # its round-start count" contract.
    starting_counts = {tuple(p.codons): p.number for p in pop}
    for p in list(pop):
        cash_in = starting_counts[tuple(p.codons)]
        if cash_in <= 0:
            continue
        neighbors = nearest_neighbors(p.codons, aa_seq)
        if not neighbors:
            continue
        p.number -= cash_in
        for _ in range(cash_in):
            neighbor = random.choice(neighbors)
            key = tuple(neighbor)
            existing = by_codons.get(key)
            if existing is not None:
                existing.number += 1
            else:
                vecs = calculate_change_vector(neighbor, analysis_objects, locvec)
                new_ind = Proposed_Solution(neighbor, 1, vecs)
                by_codons[key] = new_ind
                pop.append(new_ind)
    return [p for p in pop if p.number > 0]


def flatten_generation(
    pop: list,
    aa_seq: str,
    analysis_objects: AnalysisObjects,
    locvec: list = None,
    recursion_limit: int = 3,
) -> list:
    """Trade replicate-count concentration for search breadth.

    Every individual's replicate count is "cashed in" for that many draws
    from its single-mutation neighborhood: an individual with 45 copies
    stops being 45 copies of the same genotype and becomes (up to) 45
    distinct neighbor genotypes instead -- new ones start at count 1, and a
    draw that lands on a genotype already in the population just increments
    its count (see _flatten_round(), which this repeats recursion_limit
    times), then every remaining individual's count is collapsed to 1.

    An individual with no viable neighbors (e.g. every position is a
    single-codon amino acid) is left untouched rather than cashed in.
    """
    pop = list(pop)
    for _ in range(recursion_limit):
        pop = _flatten_round(pop, aa_seq, analysis_objects, locvec)

    for p in pop:
        p.number = 1
    return pop


def kill_off(pop: list, weights: dict, percent_cut: int = 30) -> list:
    """Reduce the population's total replicate count by percent_cut%,
    weighted toward individuals whose change vector says they most need it,
    and drop any individual whose count reaches zero.

    The original floored `number` at 1 (`if pop[i].number > 1: ...`), so no
    individual was ever fully culled -- the distinct-individual count could
    only grow generation over generation, and with it the cost of
    directed_evolution's per-individual lookahead. It also had no fallback
    once every individual was stuck at number==1, which is an infinite loop
    (`while num_to_kill > 0` with no way to make progress). Both fixed here:
    number can reach 0, dead individuals are dropped, and the loop stops
    once nothing is left to kill.
    """
    total = sum(p.number for p in pop)
    num_to_kill = total * percent_cut // 100
    scores = [_finite_nonneg(score_changevec(p.change_vecs, weights)) for p in pop]

    while num_to_kill > 0 and any(p.number > 0 for p in pop):
        idx = random.choices(range(len(pop)), weights=scores)[0]
        if pop[idx].number > 0:
            pop[idx].number -= 1
            num_to_kill -= 1
            if pop[idx].number == 0:
                scores[idx] = 0.0
    return [p for p in pop if p.number > 0]


def save_gen(pop: list, gen: int, save_dir: str, run_name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{run_name}_gen{gen}.json")
    with open(path, 'w') as f:
        json.dump([{'codons': p.codons, 'number': p.number, 'change_vecs': p.change_vecs} for p in pop], f)
    return path


def run_ga(
    aa_seq: str,
    seeds: list,
    weights: dict,
    analysis_objects: AnalysisObjects,
    num_gens: int = 30,
    locvec: list = None,
    save_dir: str = None,
    run_name: str = "run",
    target_size: int = None,
    flatten_every: int = None,
    flatten_recursion_limit: int = 3,
    lookahead: bool = True,
    refresh_every: int = 5,
) -> list:
    """Run the genetic algorithm and return the final population.

    seeds: list of codon-list solutions (e.g. from generate_seed()).
    weights: dict of term-name -> weight, matching calculate_change_vector's
        output keys (RareCodons, CodonUsage, CodonPairBias, GC, Kmer).
    target_size: population is stochastically cut down to this many distinct
        individuals (see select_survivors()) after each generation's
        reproduction step. Reproduction can add far more than target_size
        candidates per generation (each individual can spawn up to 10), so
        without this the population grows unboundedly. Defaults to the
        number of distinct seed genotypes.
    flatten_every: if set, run flatten_generation() every this-many
        generations (before selection) instead of straightforward
        reproduction that generation -- trades replicate-count
        concentration for neighborhood breadth. Off (None) by default; this
        is a search-strategy choice, not something to default silently.
    lookahead: passed through to directed_evolution() -- see its docstring.
    refresh_every: change vectors are exactly refreshed (see
        refresh_change_vectors()) every this-many generations, right before
        that generation's selection. Growth always produces approximate,
        diffed change vectors for new genotypes (see merge_replicate() /
        diff_change_vector()) -- cheap, but the drift this trades away
        would otherwise compound silently across all num_gens generations.
        Note this is a real tradeoff, not a free win: run_ga pairs one
        growth step with one selection step per generation (unlike a
        schedule.py schedule, which can run several cheap growth steps
        between precision checkpoints), so refreshing *every* generation
        here would cost the same as never diffing at all -- refresh_every
        controls how many generations run on approximate scores between
        each exact reset. Set to 1 for always-exact selection (same
        behavior/cost as before diffing existed), or 0/None to never force
        a reset (fastest, but drift is unbounded over a long run).
    """
    pop_index = {}
    for seed in seeds:
        merge_replicate(pop_index, seed, analysis_objects, locvec)
    pop = list(pop_index.values())

    if target_size is None:
        target_size = len(pop)

    for gen in range(num_gens):
        if flatten_every and gen > 0 and gen % flatten_every == 0:
            pop = refresh_change_vectors(pop, analysis_objects, locvec)
            pop = flatten_generation(pop, aa_seq, analysis_objects, locvec, flatten_recursion_limit)

        # Dedup against the *whole generation's* output, not just the
        # pre-generation population: two different individuals' offspring
        # landing on the same new genotype in the same generation must
        # merge into one entry, not become two duplicate Proposed_Solution
        # objects with identical codons (a latent bug in the original
        # per-rep `next((q for q in pop if q.codons == rep), None)` scan,
        # which only ever checked the pre-generation pop -- fixed as a
        # side effect of switching to a dict keyed by codons).
        pop_index = {tuple(p.codons): p for p in pop}
        for p in pop:
            if random.random() < 0.5:
                reps = replicate_and_mutate_random(p.codons, aa_seq)
            else:
                reps = directed_evolution(
                    p.codons, p.change_vecs, weights, aa_seq, analysis_objects, locvec, lookahead=lookahead,
                )
            for rep in reps:
                merge_replicate(pop_index, rep, analysis_objects, locvec, parent=p)

        pop = list(pop_index.values())
        if refresh_every and gen % refresh_every == 0:
            pop = refresh_change_vectors(pop, analysis_objects, locvec)
        pop = select_survivors(pop, weights, target_size)
        if save_dir:
            save_gen(pop, gen, save_dir, run_name)

    return pop
