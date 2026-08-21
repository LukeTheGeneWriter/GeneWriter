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

import dataclasses
import heapq
import json
import math
import os
import random
import sys

import numpy as np

from .change_vector import AnalysisObjects, calculate_change_vector, diff_change_vector, require_weights, distance_from_optimal
from .classes import Proposed_Solution
from .codon_tables import generate_codon_vec, sequence_space_size
from .gpu_change_vector import batch_calculate_change_vectors

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


def merge_replicate_exact(pop_index: dict, sol: list, vecs: dict) -> None:
    """Like merge_replicate(), but for a genotype whose exact change vector
    is already known -- no diffing or computing needed, just the same
    dedup-by-genotype-key bookkeeping. Intended consumer:
    directed_evolution_batch(vecs_out=...), which already computes each
    replicate's exact change vector while choosing the best synonymous
    alternative -- recomputing it again via merge_replicate()'s
    diff_change_vector() would both waste that work and be less accurate
    than the exact result already in hand.
    """
    key = tuple(sol)
    existing = pop_index.get(key)
    if existing is not None:
        existing.number += 1
    else:
        pop_index[key] = Proposed_Solution(sol, 1, vecs)


def merge_replicates_batch(
    pop_index: dict,
    new_solutions: list,
    analysis_objects: AnalysisObjects,
    locvec: list,
    xp,
    progress_every: int = None,
    chunk_size: int = None,
) -> None:
    """Batched counterpart of merge_replicate() for genotypes with no
    precomputed vecs and no parent to diff against usefully in bulk (e.g. a
    generation's random-mutation replicates from replicate_and_mutate_random()) --
    mirrors seed_population()'s dedup-then-batch shape rather than one
    diff_change_vector() Python call per replicate.

    Duplicates within new_solutions itself, and genotypes already present in
    pop_index, are merged (counted) exactly like repeated merge_replicate()
    calls would be -- only genuinely new, unique genotypes get a
    batch_calculate_change_vectors() call, and that call happens once for
    all of them together rather than once each.

    xp: array module (numpy or cupy) for the batched pass -- required, not
    optional, like directed_evolution_batch(): this function only exists
    for its batching, callers decide whether to use it at all (vs.
    merge_replicate() in a loop) by choosing to call it.
    chunk_size: passed straight through to
    gpu_change_vector.batch_calculate_change_vectors() -- see its docstring.
    None (default): unchanged, one batched call across every unique
    genotype.
    """
    counts = {}
    order = []
    for sol in new_solutions:
        key = tuple(sol)
        existing = pop_index.get(key)
        if existing is not None:
            existing.number += 1
            continue
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1

    if not order:
        return

    unique_codons = [list(key) for key in order]
    vecs_list = batch_calculate_change_vectors(unique_codons, analysis_objects, locvec, xp=xp, progress_every=progress_every, chunk_size=chunk_size)
    for key, codons, vecs in zip(order, unique_codons, vecs_list):
        pop_index[key] = Proposed_Solution(codons, counts[key], vecs)


def _progress_print(label: str, i: int, total: int, t0: float) -> None:
    import time as _time
    elapsed = _time.perf_counter() - t0
    rate = i / elapsed if elapsed > 0 else float('inf')
    remaining = (total - i) / rate if rate > 0 else float('inf')
    print(f"  [{label}] {i}/{total} ({rate:.1f} indiv/s, ~{remaining:.0f}s remaining)", flush=True)


def seed_population(seeds: list, analysis_objects: AnalysisObjects, locvec: list = None, xp=None, progress_every: int = None, chunk_size: int = None) -> dict:
    """Build a pop_index (dict keyed by tuple(codons) -> Proposed_Solution)
    from a batch of fresh genotypes that have no parent to diff against --
    e.g. run_ga's initial seeds, or schedule.py's "input" step. Duplicate
    seeds are merged (counted) exactly like repeated merge_replicate() calls
    would, but as one pass over the unique genotypes rather than one
    calculate_change_vector() call per seed (including duplicates).

    xp: array module (numpy or cupy) to batch the exact computation through
    gpu_change_vector.batch_calculate_change_vectors() instead of the
    per-individual Python loop -- None (default) keeps the original
    per-individual path. This is where batching actually pays off for
    seeding: a fresh population of tens of thousands of individuals used to
    mean tens of thousands of separate calculate_change_vector() calls with
    no diffing available to cheapen any of them.
    progress_every: if set, print elapsed-time progress every this-many
        individuals -- diagnostic only (see gpu_change_vector.py's
        batch_calculate_change_vectors, which this passes it through to
        when xp is set). None (default): silent.
    chunk_size: passed straight through to
    gpu_change_vector.batch_calculate_change_vectors() when xp is set -- see
    its docstring. None (default): unchanged, one batched call across every
    unique seed.
    """
    counts = {}
    order = []
    for sol in seeds:
        key = tuple(sol)
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1

    unique_codons = [list(key) for key in order]
    if xp is not None:
        vecs_list = batch_calculate_change_vectors(unique_codons, analysis_objects, locvec, xp=xp, progress_every=progress_every, chunk_size=chunk_size)
    else:
        if progress_every:
            import time as _time
            t0 = _time.perf_counter()
        vecs_list = []
        for i, codons in enumerate(unique_codons):
            vecs_list.append(calculate_change_vector(codons, analysis_objects, locvec))
            if progress_every and (i + 1) % progress_every == 0:
                _progress_print("seed_population", i + 1, len(unique_codons), t0)

    return {
        key: Proposed_Solution(codons, counts[key], vecs)
        for key, codons, vecs in zip(order, unique_codons, vecs_list)
    }


def refresh_change_vectors(pop: list, analysis_objects: AnalysisObjects, locvec: list = None, xp=None, progress_every: int = None, chunk_size: int = None) -> list:
    """Recompute every individual's change vector exactly (in place) rather
    than trusting whatever approximate diff it may have accumulated during
    growth. Meant to be called right before a step that makes a survival
    decision based on those scores (kill_off, select_survivors) or that
    otherwise warrants a clean baseline (flatten_generation) -- see
    schedule.py's kill_off/select/flatten steps, which always do this
    first.

    xp: array module (numpy or cupy) to batch this refresh through
    gpu_change_vector.batch_calculate_change_vectors() instead of the
    per-individual Python loop -- None (default) keeps the original
    per-individual path (no dependency on gpu_change_vector.py's numpy
    import if a caller never opts in). Both paths compute exactly the same
    thing -- see tests/test_gpu_change_vector.py's call-by-call equivalence
    check -- so this is a pure performance choice, not a behavior change.
    All of `pop` must share one sequence length (true within a single GA
    run: synonymous substitutions only), same requirement as
    batch_calculate_change_vectors().
    progress_every: if set, print elapsed-time progress every this-many
        individuals -- diagnostic only. None (default): silent.
    chunk_size: passed straight through to
    gpu_change_vector.batch_calculate_change_vectors() when xp is set -- see
    its docstring. None (default): unchanged, one batched call across the
    whole population.
    """
    if xp is not None and pop:
        pop_codons = [p.codons for p in pop]
        vecs_list = batch_calculate_change_vectors(pop_codons, analysis_objects, locvec, xp=xp, progress_every=progress_every, chunk_size=chunk_size)
        for p, vecs in zip(pop, vecs_list):
            p.change_vecs = vecs
        return pop

    if progress_every:
        import time as _time
        t0 = _time.perf_counter()
    for i, p in enumerate(pop):
        p.change_vecs = calculate_change_vector(p.codons, analysis_objects, locvec)
        if progress_every and (i + 1) % progress_every == 0:
            _progress_print("refresh_change_vectors", i + 1, len(pop), t0)
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


def _deep_sizeof(obj, _seen: set = None) -> int:
    """Recursive memory footprint via sys.getsizeof(), walking dict/list/
    tuple/set containers AND dataclass instances (none of which count their
    own contents/fields in their own getsizeof() -- a bare
    sys.getsizeof(Proposed_Solution(...)) is just its small object header,
    a real bug caught live: an earlier version of this function returned
    the same tiny constant regardless of aa_seq length, since it never
    descended into a dataclass instance's fields at all). Used to measure
    a real Proposed_Solution's actual RAM cost rather than guessing at an
    analytical formula, so the estimate automatically stays correct if
    change_vector.py's registered terms ever change (more/fewer terms,
    e.g. Uracil, or different per-position value types) without needing a
    matching update here. _seen guards against double-counting an object
    reachable two ways -- not expected for a single Proposed_Solution's
    own tree, but cheap insurance."""
    seen = _seen if _seen is not None else set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(_deep_sizeof(k, seen) + _deep_sizeof(v, seen) for k, v in obj.items())
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        size += sum(_deep_sizeof(getattr(obj, f.name), seen) for f in dataclasses.fields(obj))
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_deep_sizeof(item, seen) for item in obj)
    return size


def estimate_bytes_per_individual(aa_seq: str, analysis_objects: AnalysisObjects, locvec: list = None) -> int:
    """Real measured in-memory footprint of one Proposed_Solution for this
    aa_seq -- builds one real seed and scores it exactly once
    (calculate_change_vector(), the cheap per-individual path -- this is a
    single call, not a batch), then walks the result with _deep_sizeof()
    rather than an analytical guess. Used by suggest_population_size() to
    size "input"/"select" defaults against available system RAM."""
    sol = generate_seed(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects, locvec)
    return _deep_sizeof(Proposed_Solution(sol, 1, vecs))


def _cgroup_ram_headroom_bytes():
    """Bytes still available inside THIS PROCESS'S cgroup memory limit
    (limit - current usage), or None if there is no enforced limit or the
    cgroup files aren't readable.

    Why this exists: os.sysconf('SC_AVPHYS_PAGES') reports free pages as
    the *kernel* sees them, which in a container is the host's view, not
    the limit the container will actually be OOM-killed at. A Colab
    runtime is exactly that shape -- a cgroup-limited sandbox -- so
    sysconf alone can happily report far more free RAM than the process is
    permitted to touch, and sizing a population against it produces a run
    that dies partway through instead of one that never started. Same
    reasoning as gpu_change_vector.suggest_chunk_size() querying *actually
    free* VRAM rather than total VRAM, applied to the host side.

    Checks cgroup v2 first (the unified /sys/fs/cgroup/memory.max +
    memory.current), then v1 (/sys/fs/cgroup/memory/memory.limit_in_bytes
    + memory.usage_in_bytes). Both report "no limit" differently: v2 uses
    the literal string "max", v1 uses a sentinel near 2**63 -- both are
    treated as None (unlimited) rather than as an absurdly large budget.
    """
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            with open(limit_path) as f:
                raw_limit = f.read().strip()
            if raw_limit == "max":
                return None
            limit = int(raw_limit)
            if limit >= 2 ** 62:
                return None
            with open(usage_path) as f:
                usage = int(f.read().strip())
        except (OSError, ValueError):
            continue
        return max(limit - usage, 0)
    return None


def available_ram_bytes(default: int = 2 * 1024 ** 3) -> int:
    """RAM in bytes this process can actually still allocate -- the smaller
    of what the kernel reports free (os.sysconf, SC_AVPHYS_PAGES *
    SC_PAGE_SIZE) and what's left inside its cgroup memory limit
    (_cgroup_ram_headroom_bytes()).

    sysconf works with no extra dependency on any POSIX system, which
    covers every environment this codebase actually runs GA work on (WSL,
    Colab -- see memory/dev_environment.md: no native Windows Python at
    all). But inside a container it describes the *host*, not the limit
    this process gets OOM-killed at, which is why the cgroup headroom is
    taken as a second, usually tighter ceiling -- see
    _cgroup_ram_headroom_bytes() for why that distinction is the one that
    actually kills long Colab runs.

    Falls back to `default` (2GB, a conservative guess) if sysconf isn't
    available or doesn't expose these keys, rather than raising and
    blocking sizing entirely on an unexpected platform. A readable cgroup
    limit still applies on top of that fallback.
    """
    try:
        free = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (ValueError, OSError, AttributeError):
        free = default
    cgroup_headroom = _cgroup_ram_headroom_bytes()
    if cgroup_headroom is not None:
        return min(free, cgroup_headroom)
    return free


def suggest_population_size(aa_seq: str, analysis_objects: AnalysisObjects, already_covered: int = 0,
                             locvec: list = None, ram_fraction: float = 0.5,
                             peak_multiplier: float = 1.0) -> int:
    """How many distinct individuals it's reasonable to seed/hold for
    aa_seq, maximizing use of available hardware instead of a hand-picked
    fixed constant -- the smaller of two real ceilings:

      - codon_tables.sequence_space_size(aa_seq) minus already_covered --
        can't produce more distinct sequences than mathematically exist
        (same ceiling schedule._remaining_space() already enforces on top
        of whatever this suggests, e.g. against an explicit count).
      - available system RAM (available_ram_bytes() * ram_fraction)
        divided by one individual's real measured footprint
        (estimate_bytes_per_individual()) -- the population lives entirely
        in host memory as a list of Proposed_Solution for the life of a
        run, so this is the real constraint on how large it can grow
        without risking the process (or the whole Colab VM) running out of
        RAM.

    ram_fraction (default 0.5) leaves headroom rather than planning to
    consume every free byte, the same reasoning gpu_corpus_batch.
    vram_aware_batch_size()'s vram_fraction already uses on the VRAM side
    -- other things (the interpreter itself, gene corpus data still
    resident, Colab's own overhead) already share that RAM.

    peak_multiplier (default 1.0, i.e. off) divides the RAM budget by the
    factor a schedule's population is expected to TRANSIENTLY exceed its
    steady-state size by. This function otherwise answers "how many
    individuals fit in RAM," which is only the right question when the
    steady-state size IS the peak -- true for run_ga() (one growth step
    per selection step) and for a schedule whose every growth step is
    immediately followed by a select. It is NOT true for an explore/exploit
    alternation: schedule.explore() deliberately inflates the population
    (flatten multiplies distinct individuals, kick adds colonists) and does
    not select, so the peak lands in the middle of the cycle at some
    multiple of the size select() later returns it to. Sizing the
    steady state to fill RAM and then tripling it mid-cycle is precisely
    how a long run dies partway through instead of never starting.
    schedule.default_schedule() passes its own DEFAULT_PEAK_MULTIPLIER
    through to the "input"/"select" steps for exactly this reason; measure
    the real factor for your schedule rather than trusting the default.

    Raises ValueError for peak_multiplier < 1.0 -- a population that peaks
    *below* its steady-state size is not a thing, and silently accepting
    the value would quietly hand back a LARGER budget than the un-adjusted
    call, which is the opposite of what anyone reaching for this parameter
    wants.
    """
    if peak_multiplier < 1.0:
        raise ValueError(
            f"peak_multiplier must be >= 1.0 (got {peak_multiplier!r}) -- it divides the RAM "
            "budget by how far the population transiently EXCEEDS its steady-state size; a "
            "value below 1.0 would enlarge the budget instead of shrinking it."
        )
    space_ceiling = max(sequence_space_size(aa_seq) - already_covered, 0)
    bytes_per_individual = max(estimate_bytes_per_individual(aa_seq, analysis_objects, locvec), 1)
    ram_budget = int(available_ram_bytes() * ram_fraction / peak_multiplier)
    ram_ceiling = ram_budget // bytes_per_individual
    return min(space_ceiling, ram_ceiling)


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


def _ranked_positions(sol: list, changevecs: dict, weights: dict) -> tuple:
    """Every position in sol, ranked by weighted change-vector score
    (highest first) with a matching list of finite, non-negative
    random.choices() weights -- the position-selection logic shared by
    directed_evolution() and directed_evolution_batch()."""
    weighted_positions = [
        sum(weights[key] * changevecs[key][i] for key in changevecs) for i in range(len(sol))
    ]
    ranked = sorted(range(len(sol)), key=lambda i: weighted_positions[i], reverse=True)
    position_weights = [_finite_nonneg(weighted_positions[i]) for i in ranked]
    return ranked, position_weights


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

    This is the per-individual path -- see directed_evolution_batch() for
    the batched counterpart used when growth is run with xp set, which
    scores every candidate exactly (via batch_calculate_change_vectors())
    instead of diff_change_vector()'s cheaper-per-call-but-many-calls
    excerpt approximation; measured as the dominant real-scale cost here
    (many small numpy calls, dispatch-overhead-bound) -- see Handoff.md
    sec 6.
    """
    require_weights(changevecs.keys(), weights)
    codon_vec = generate_codon_vec(aa_seq)
    ranked, position_weights = _ranked_positions(sol, changevecs, weights)

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
            candidate_score = distance_from_optimal(candidate_vecs, weights)
            if best_score is None or candidate_score < best_score:
                best_codon, best_score = alt, candidate_score

        new_sol = sol.copy()
        new_sol[position] = best_codon
        replicates.append(new_sol)
    return replicates


def directed_evolution_batch(
    individuals: list,
    weights: dict,
    aa_seq: str,
    analysis_objects: AnalysisObjects,
    locvec: list = None,
    nreplicates: int = 10,
    xp=None,
    progress_every: int = None,
    vecs_out: dict = None,
    chunk_size: int = None,
) -> dict:
    """Batched counterpart of directed_evolution(lookahead=True) across
    MULTIPLE individuals at once.

    Why this exists: directed_evolution()'s lookahead scores every
    synonymous alternative at the chosen position via one
    diff_change_vector() Python call each -- cheap in isolation, but
    measured as the actual real-scale bottleneck (~20-27 individuals/sec,
    ~60-85x slower per individual than the batched exact-refresh path --
    see Handoff.md sec 6), because it's thousands of small numpy calls
    (dispatch-overhead-bound, not compute-bound), not because the per-call
    math is expensive. This collects every individual's candidate
    alternatives into one batch and scores them all with a single
    gpu_change_vector.batch_calculate_change_vectors() call instead.

    individuals: list of Proposed_Solution, each with a valid change_vecs
        (used for position-selection weighting, same as directed_evolution).
    xp: array module (numpy or cupy) for the batched scoring pass -- unlike
        seed_population()/refresh_change_vectors(), there's no
        per-individual fallback mode here; this function only exists for
        its batching. None defaults to numpy. Callers decide *whether* to
        batch growth at all by choosing to call this instead of
        directed_evolution() in a loop -- see run_ga()/schedule.py's
        "growth" step, both gated on `xp is not None`.

    Position selection (which position each of nreplicates replicate slots
    mutates) is unchanged from directed_evolution() -- same weighted-random
    logic via _ranked_positions(), same number and order of
    random.choices() calls per individual. What's batched is choosing the
    *best* synonymous alternative at that position: a given (individual,
    position) pair's best alternative is deterministic from that
    individual's own change_vecs (it doesn't depend on which replicate slot
    drew the position), so repeat draws of the same position -- likely,
    since draws are weighted toward a handful of high-scoring positions --
    are memoized instead of re-scored.

    Accuracy trade-off worth knowing: scoring here recomputes each
    candidate's change vector exactly (full sequence, and -- now that
    batch_calculate_change_vectors() batches all 5 terms including Kmer --
    cheaply) rather than diff_change_vector's local-excerpt approximation.
    Strictly more accurate (true whole-sequence aggregates instead of
    excerpt-estimated ones), numerically different results.

    Performance reality check, found by profiling rather than assumed:
    this function's own batched scoring is fast (measured: 1.45s for
    ~10,000 candidates across ~1,000 individuals on real gene data, this
    RTX 3050 -- see Handoff.md sec 6), but a naive caller storing each
    accepted replicate one at a time via merge_replicate() -- one
    diff_change_vector() call per replicate -- previously dominated
    growth's total time instead (measured: ~12s for that same ~10,000
    replicates in the same run, capping net growth speedup at a modest
    ~3x despite the 25-48x batch_calculate_change_vectors() itself
    achieves). That storage step is no longer naive: pass `vecs_out` (see
    below) and use merge_replicate_exact() instead of merge_replicate() to
    store each replicate's already-known exact vecs directly, skipping the
    redundant diff_change_vector() recompute entirely -- see
    ga.run_ga()/schedule.py's "growth" step for the wiring.

    vecs_out: if given, populated (mutated in place) with
        {tuple(new_sol): vecs} for every returned replicate -- the exact
        change vector this function already computed while scoring that
        candidate, which the caller would otherwise have to recompute (via
        merge_replicate()'s diff_change_vector() approximation) despite it
        being sitting right here. See merge_replicate_exact(), the intended
        consumer. None (default): behaves exactly as before this parameter
        existed -- purely additive, no existing caller needs to change.

    chunk_size: passed straight through to
    gpu_change_vector.batch_calculate_change_vectors() -- see its
    docstring. None (default): unchanged, one batched call across every
    candidate.

    Returns {id(individual): [new_sol, ...]}, matching what looping
    directed_evolution() per individual would return (list length up to
    nreplicates, fewer entries for slots whose drawn position had no
    synonymous alternative).
    """
    xp = xp if xp is not None else np
    codon_vec = generate_codon_vec(aa_seq)

    for ind in individuals:
        require_weights(ind.change_vecs.keys(), weights)

    unique_pairs = {}  # (id(ind), position) -> True, insertion order
    per_individual_positions = {}  # id(ind) -> [position, ...], one per replicate slot

    for ind in individuals:
        sol = ind.codons
        key = id(ind)
        ranked, position_weights = _ranked_positions(sol, ind.change_vecs, weights)

        positions = []
        for _ in range(nreplicates):
            choice_idx = random.choices(range(len(ranked)), weights=position_weights)[0]
            position = ranked[choice_idx]
            positions.append(position)
            current_codon = sol[position]
            if any(c != current_codon for c in codon_vec[position]):
                unique_pairs[(key, position)] = True
        per_individual_positions[key] = positions

    if not unique_pairs:
        return {id(ind): [] for ind in individuals}

    by_key = {id(ind): ind for ind in individuals}
    candidates = []
    candidate_owner = []  # parallel: (key, position, alt) per candidate row
    for key, position in unique_pairs:
        sol = by_key[key].codons
        current_codon = sol[position]
        for alt in codon_vec[position]:
            if alt == current_codon:
                continue
            candidate = list(sol)
            candidate[position] = alt
            candidates.append(candidate)
            candidate_owner.append((key, position, alt))

    vecs_list = batch_calculate_change_vectors(candidates, analysis_objects, locvec, xp=xp, progress_every=progress_every, chunk_size=chunk_size)

    best_alt = {}  # (key, position) -> (best_alt, best_score, vecs)
    for (key, position, alt), vecs in zip(candidate_owner, vecs_list):
        score = distance_from_optimal(vecs, weights)
        current = best_alt.get((key, position))
        if current is None or score < current[1]:
            best_alt[(key, position)] = (alt, score, vecs)

    results = {}
    for ind in individuals:
        key = id(ind)
        sol = ind.codons
        reps = []
        for position in per_individual_positions[key]:
            entry = best_alt.get((key, position))
            if entry is None:
                continue
            new_sol = sol.copy()
            new_sol[position] = entry[0]
            reps.append(new_sol)
            if vecs_out is not None:
                vecs_out[tuple(new_sol)] = entry[2]
        results[key] = reps
    return results


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

    Individuals with .protected == True (see mark_protected()) are never
    candidates for removal -- if there are more protected individuals than
    target_size allows, the returned population exceeds target_size rather
    than removing a protected one; protection is a hard guarantee,
    target_size a soft cap it can only be exceeded by, never violated.
    """
    pop = list(pop)
    num_to_remove = len(pop) - target_size
    if num_to_remove <= 0:
        return pop
    removable = [i for i, p in enumerate(pop) if not p.protected]
    num_to_remove = min(num_to_remove, len(removable))
    if num_to_remove <= 0:
        return pop
    die_weights = [_finite_nonneg(distance_from_optimal(pop[i].change_vecs, weights)) for i in removable]
    keys = [math.log(max(random.random(), 1e-300)) / w for w in die_weights]
    victims = {removable[j] for j in heapq.nlargest(num_to_remove, range(len(removable)), key=lambda j: keys[j])}
    return [p for i, p in enumerate(pop) if i not in victims]


def _flatten_round(pop: list, aa_seq: str, analysis_objects: AnalysisObjects, locvec: list = None, xp=None, chunk_size: int = None) -> list:
    """One round of flatten_generation's cash-in: every individual's
    round-start replicate count is redistributed, one unit at a time, onto
    a randomly-drawn single-mutation neighbor (incrementing it if already
    present in the population, creating it at count 1 otherwise).

    This step alone is conservative: the total replicate count across the
    population is unchanged (every unit removed from an individual becomes
    exactly one unit added somewhere else). flatten_generation's *final*
    collapse-to-1 step is not part of this -- that intentionally changes
    the total to match the distinct-individual count, by design.

    xp: array module (numpy or cupy) to batch every brand-new neighbor's
    change vector into one batch_calculate_change_vectors() call at the end
    of the round, instead of one calculate_change_vector() call per new
    neighbor as they're drawn -- the identical one-exact-call-per-new-
    genotype shape merge_replicates_batch() batches for growth (see its
    docstring), except every neighbor here is already exact (no diffing),
    so batching it is a pure speed win with no accuracy trade-off. None
    (default) keeps the original per-neighbor exact loop unchanged.
    chunk_size: passed straight through to
    gpu_change_vector.batch_calculate_change_vectors() when xp is set -- see
    its docstring. None (default): unchanged, one batched call across every
    brand-new neighbor found this round.
    """
    pop = list(pop)
    by_codons = {tuple(p.codons): p for p in pop}
    # Snapshot counts before any cashing-in happens this round: if we read
    # p.number at iteration time instead, an individual processed later in
    # the round could cash in copies it only just received from an earlier
    # individual's draws this same round, contradicting the "only cashes in
    # its round-start count" contract.
    starting_counts = {tuple(p.codons): p.number for p in pop}
    pending_counts = {}  # key -> cash-in count, for brand-new genotypes not yet given a change vector
    pending_order = []
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
            elif xp is not None:
                if key not in pending_counts:
                    pending_counts[key] = 0
                    pending_order.append(neighbor)
                pending_counts[key] += 1
            else:
                vecs = calculate_change_vector(neighbor, analysis_objects, locvec)
                new_ind = Proposed_Solution(neighbor, 1, vecs)
                by_codons[key] = new_ind
                pop.append(new_ind)

    if xp is not None and pending_order:
        vecs_list = batch_calculate_change_vectors(pending_order, analysis_objects, locvec, xp=xp, chunk_size=chunk_size)
        for neighbor, vecs in zip(pending_order, vecs_list):
            new_ind = Proposed_Solution(neighbor, pending_counts[tuple(neighbor)], vecs)
            by_codons[tuple(neighbor)] = new_ind
            pop.append(new_ind)

    return [p for p in pop if p.number > 0]


def flatten_generation(
    pop: list,
    aa_seq: str,
    analysis_objects: AnalysisObjects,
    locvec: list = None,
    recursion_limit: int = 3,
    xp=None,
    chunk_size: int = None,
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

    xp: array module (numpy or cupy), passed through to _flatten_round() to
    batch new-neighbor change-vector computation -- see its docstring. None
    (default) keeps the original per-neighbor path.
    chunk_size: passed straight through to _flatten_round() (and from there
    to gpu_change_vector.batch_calculate_change_vectors()) when xp is set --
    see its docstring. None (default): unchanged, one batched call per
    round.
    """
    pop = list(pop)
    for _ in range(recursion_limit):
        pop = _flatten_round(pop, aa_seq, analysis_objects, locvec, xp=xp, chunk_size=chunk_size)

    for p in pop:
        p.number = 1
    return pop


def _score_by_metric(pop: list, weights: dict, metric: str) -> list:
    """One score per individual for `metric` -- "distance_from_optimal"
    (change_vector.distance_from_optimal(), the full weighted aggregate) or
    a single registered change-vector term name (summed the same way
    distance_from_optimal() reduces each term before weighting: sum(), not
    mean, for a directly comparable scale). Shared by mark_protected()/
    _top_fraction_indices() and kill_off_by_term(). Raises ValueError naming
    the valid options for an unrecognized term, rather than a bare KeyError.

    "distance_from_optimal" was called "fitness" until 2026-08-20 -- that
    exact string is special-cased in the error below, since saved schedules
    and pasted Colab configs predating the rename will still carry it."""
    if metric == "distance_from_optimal":
        return [_finite_nonneg(distance_from_optimal(p.change_vecs, weights)) for p in pop]
    if pop and metric not in pop[0].change_vecs:
        hint = (" -- 'fitness' was renamed to 'distance_from_optimal' on 2026-08-20"
                if metric == "fitness" else "")
        raise ValueError(
            f"Unknown metric {metric!r}{hint} -- must be 'distance_from_optimal' "
            f"or one of {sorted(pop[0].change_vecs)}."
        )
    return [_finite_nonneg(sum(p.change_vecs[metric])) for p in pop]


def _top_fraction_indices(pop: list, weights: dict, criteria: list) -> set:
    """Indices into pop that rank in the top `top_fraction` (best -- i.e.
    lowest score, "needs the least mutation") for ANY of the given
    (metric, top_fraction) pairs in `criteria` -- a UNION across criteria,
    not an intersection. Read-only ranking step shared by two different
    USES: mark_protected() turns the result into a permanent
    Proposed_Solution.protected flag; kill_off()/kill_off_by_term()'s
    optional `protect_criteria` uses it to exempt those indices from just
    one cull, without ever touching .protected. See mark_protected()'s
    docstring for what "lowest score = best" means and why."""
    qualifying = set()
    for metric, top_fraction in criteria:
        scores = _score_by_metric(pop, weights, metric)
        cutoff_count = math.ceil(len(pop) * top_fraction)
        if cutoff_count <= 0:
            continue
        qualifying.update(heapq.nsmallest(cutoff_count, range(len(pop)), key=lambda i: scores[i]))
    return qualifying


def _kill_off_by_scores(pop: list, scores: list, percent_cut: int, immune: set = None) -> list:
    """Shared weighted-random mass-removal loop behind kill_off() and
    kill_off_by_term() -- percent_cut% of TOTAL replicate mass removed,
    weighted toward high-score individuals (score = "how much this
    individual needs mutation", whatever the caller's `scores` measures),
    one unit at a time; an individual whose count reaches 0 is dropped.
    The two callers differ only in what "most in need" means -- kill_off()'s
    weighted aggregate distance_from_optimal() vs. kill_off_by_term()'s single
    term.

    Individuals with .protected == True, OR whose index is in `immune`
    (default none -- see kill_off()/kill_off_by_term()'s `protect_criteria`,
    computed via _top_fraction_indices()), are never candidates: their
    score is forced to 0 up front (same mechanism the loop already uses to
    retire an exhausted individual), so random.choices() can never select
    them regardless of how badly they'd otherwise score. The loop's own
    stop condition checks the (possibly zeroed) scores directly rather
    than raw .number, so it still terminates correctly even when every
    remaining individual with mass left is protected/immune.
    """
    immune = immune or ()
    total = sum(p.number for p in pop)
    num_to_kill = total * percent_cut // 100
    scores = [0.0 if (p.protected or i in immune or p.number <= 0) else s for i, (p, s) in enumerate(zip(pop, scores))]

    while num_to_kill > 0 and any(s > 0 for s in scores):
        idx = random.choices(range(len(pop)), weights=scores)[0]
        pop[idx].number -= 1
        num_to_kill -= 1
        if pop[idx].number <= 0:
            scores[idx] = 0.0
    return [p for p in pop if p.number > 0]


def kill_off(pop: list, weights: dict, percent_cut: int = 30, protect_criteria: list = None) -> list:
    """Reduce the population's total replicate count by percent_cut%,
    weighted toward individuals whose change vector says they most need it,
    and drop any individual whose count reaches zero. See
    _kill_off_by_scores() for the shared mechanics, including how
    .protected individuals (mark_protected()) are shielded.

    protect_criteria: optional list of (metric, top_fraction) pairs, same
    shape/semantics as mark_protected()'s `criteria` -- individuals that
    qualify are exempt from THIS cull only (see _top_fraction_indices()),
    without ever setting .protected. Use this for "don't let this
    particular stochastic cull touch the top 10% by distance from optimal"
    as a one-off,
    vs. mark_protected()/the "protect" schedule step's permanent-until-
    replaced version of the same ranking. None (default): unchanged, only
    .protected individuals are exempt.

    The original floored `number` at 1 (`if pop[i].number > 1: ...`), so no
    individual was ever fully culled -- the distinct-individual count could
    only grow generation over generation, and with it the cost of
    directed_evolution's per-individual lookahead. It also had no fallback
    once every individual was stuck at number==1, which is an infinite loop
    (`while num_to_kill > 0` with no way to make progress). Both fixed here:
    number can reach 0, dead individuals are dropped, and the loop stops
    once nothing is left to kill.
    """
    scores = _score_by_metric(pop, weights, "distance_from_optimal")
    immune = _top_fraction_indices(pop, weights, protect_criteria) if protect_criteria else None
    return _kill_off_by_scores(pop, scores, percent_cut, immune=immune)


def kill_off_by_term(pop: list, term: str, percent_cut: int = 30, protect_criteria: list = None, weights: dict = None) -> list:
    """Like kill_off(), but weighted toward a single change-vector term's
    own score instead of the full weighted aggregate -- e.g. "remove 20%
    of total mass, weighted toward the worst CodonPairBias offenders"
    without that pressure being diluted/rebalanced by every other term the
    way weights-based kill_off() is. See _kill_off_by_scores() for the
    shared removal mechanics, including how .protected individuals are
    shielded.

    `term` must be a real registered change-vector term name (e.g.
    'CodonPairBias', 'Uracil') -- raises ValueError naming the valid
    options otherwise, rather than a bare KeyError.

    protect_criteria/weights: same one-off exemption as kill_off()'s
    `protect_criteria` -- see its docstring. `weights` is only actually
    read if protect_criteria includes a "distance_from_optimal" entry (term-only
    criteria never need it); required in that case, ignored otherwise."""
    scores = _score_by_metric(pop, weights, term)
    immune = _top_fraction_indices(pop, weights, protect_criteria) if protect_criteria else None
    return _kill_off_by_scores(pop, scores, percent_cut, immune=immune)


def mark_protected(pop: list, weights: dict, criteria: list) -> list:
    """Sets .protected = True on every individual that ranks in the top
    `top_fraction` (best -- i.e. lowest score, "needs the least mutation")
    for ANY of the given (metric, top_fraction) pairs in `criteria` -- a
    UNION across criteria, not an intersection (see _top_fraction_indices(),
    the shared ranking step this uses). e.g. criteria=[("distance_from_optimal", 0.10),
    ("Uracil", 0.40)] protects whichever individuals are in the best 10% by
    overall weighted distance from optimal, PLUS whichever are in the best
    40% by Uracil
    depletion alone, even if those two groups barely overlap.

    metric is "distance_from_optimal" (distance_from_optimal(), the same weighted aggregate
    kill_off()/select_survivors() rank by) or any single registered
    change-vector term name, summed the same way kill_off_by_term() does.
    "Lowest score = best" throughout this codebase's convention (every
    term is a "how much does this position/candidate need fixing" signal)
    -- top_fraction=0.10 keeps the 10% with the SMALLEST score, not the
    largest.

    Monotonic: only ever sets .protected True, never clears it -- calling
    this multiple times across a schedule (or passing multiple criteria at
    once) only ever grows the protected set -- a PERMANENT shield, unlike
    kill_off()/kill_off_by_term()'s `protect_criteria`, which uses the
    identical ranking for a one-cull-only exemption instead. Protected
    individuals are shielded from kill_off()/kill_off_by_term()/
    select_survivors() until a brand-new genotype replaces them -- growth's
    new genotypes always start unprotected, same as
    Proposed_Solution.protected's own field default.

    Mutates pop's individuals in place and returns pop, matching
    kill_off()/select_survivors()'s existing mutate-then-filter
    convention."""
    for i in _top_fraction_indices(pop, weights, criteria):
        pop[i].protected = True
    return pop


def release_protection(pop: list) -> list:
    """Clear .protected on every individual, returning pop.

    The counterpart mark_protected() deliberately does not have:
    mark_protected() is monotonic (only ever sets True), which is correct
    for a one-shot "shield the best of this population" call but makes any
    *cyclic* schedule structurally non-viable. A schedule that protects
    once per cycle -- which is exactly what an explore/exploit alternation
    needs, both to preserve the incumbent across a destructive explore
    phase and to give kick colonists a grace period -- would otherwise grow
    the immortal set every cycle until every individual is protected,
    select_survivors()/kill_off() become no-ops (both skip protected
    individuals unconditionally, and select_survivors() exceeds
    target_size rather than remove one), and the population grows without
    bound. See schedule.explore()/schedule.exploit(), which pair every
    protect with exactly one release.

    Blunt on purpose: it clears everything, with no notion of *why* an
    individual was protected or how long its grace was meant to last. The
    schedule controls the lifetime by *where* it places the release --
    which is why schedule.exploit() puts it after its descent steps and
    before its selection step, making the grace period exactly "one
    descent," the semantically meaningful unit for a kick colonist (it is
    worse than its parent by construction and needs to descend into its
    new basin before anything judges it).

    A richer per-individual TTL (Proposed_Solution.protected_until compared
    against a step counter) would allow overlapping grace periods of
    different lengths and is the natural upgrade if grace periods longer
    than one descent turn out to matter; it touches both cull paths and
    every existing reader of .protected, which this does not.
    """
    for p in pop:
        p.protected = False
    return pop


def kill_off_outside_natural_range(pop: list, analysis_objects: AnalysisObjects, threshold: float) -> list:
    """Hard cutoff, unlike kill_off()'s soft proportional cull: drops any
    individual whose own aggregate metric (see
    weight_calibration.natural_deviation_zscores()) is more than
    `threshold` standard deviations from the natural per-gene baseline mean
    in ANY category with a usable baseline -- a candidate can score well on
    every other axis and still get cut for being a `threshold`+ sigma
    outlier on just one.

    Complements kill_off(): that one culls proportionally, weighted toward
    the worst overall weighted score, but never guarantees any individual
    genotype survives or dies. This one is an absolute guardrail -- "no
    candidate this far outside nature's own observed range for any single
    metric survives," independent of how good its overall weighted distance
    from optimal looks. An individual's entire replicate count is dropped (not
    proportionally reduced like kill_off()) -- the cut is a property of the
    genotype itself, not something a fraction of its copies can be exempt
    from.

    threshold: no default -- pick deliberately for your baseline's scale
    (see natural_deviation_zscores()'s docstring for what's being
    thresholded) rather than inheriting an arbitrary cutoff.
    """
    from .weight_calibration import natural_deviation_zscores

    survivors = []
    for p in pop:
        zscores = natural_deviation_zscores(p.codons, analysis_objects)
        if zscores and max(zscores.values()) > threshold:
            continue
        survivors.append(p)
    return survivors


def save_gen(pop: list, gen: int, save_dir: str, run_name: str) -> str:
    """Writes one generation checkpoint to save_dir/run_name/gen{gen}.json --
    one subdirectory per run_name (created if needed), rather than every
    run's checkpoints mixed flat into save_dir distinguished only by a
    filename prefix. See visualize.load_population_trajectory(), the
    matching reader for this layout."""
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"gen{gen}.json")
    with open(path, 'w') as f:
        json.dump([{'codons': p.codons, 'number': p.number, 'change_vecs': p.change_vecs, 'protected': p.protected} for p in pop], f)
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
    xp=None,
    progress: bool = False,
    progress_every: int = None,
    chunk_size: int = None,
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
    xp: array module (numpy or cupy) to batch the initial seeding, every
        exact refresh, and (when lookahead=True) growth's alternative
        scoring, through gpu_change_vector.batch_calculate_change_vectors()
        instead of per-individual Python loops -- see seed_population() /
        refresh_change_vectors() / directed_evolution_batch(). None
        (default) keeps the original per-individual path throughout,
        including growth. When xp is set and lookahead=True, growth's
        best-alternative selection is scored by an exact full recompute
        batched across the whole generation instead of
        diff_change_vector()'s cheaper-per-call-but-many-calls excerpt
        approximation -- measurably more accurate, numerically different
        results. Storage of both directed and random-mutation replicates is
        now batched too when xp is set (merge_replicate_exact()/
        merge_replicates_batch(), instead of one merge_replicate() ->
        diff_change_vector() call per replicate) -- this used to be
        growth's actual dominant cost once scoring itself got fast (see
        Handoff.md sec 5/6), not touched by xp until now. lookahead=False
        growth is unaffected by xp either way -- it was already cheap (no
        per-alternative scoring or exact storage to batch).
    progress: if True, print per-generation timing (seeding, growth,
        refresh, select, each separately) plus population size before/after
        -- diagnostic only, meant for tracking down which phase a slow or
        seemingly-hung real-scale run is actually spending its time in.
        False (default): silent, identical to the original behavior.
    progress_every: passed through to seed_population()/refresh_change_vectors()
        for finer-grained (every-N-individuals) progress within those calls
        -- see their docstrings. Independent of `progress` above (you can
        have generation-level timing without individual-level, or vice
        versa); None (default): no individual-level progress either way.
    chunk_size: passed straight through to every xp-batched call this makes
        (seed_population(), refresh_change_vectors(), directed_evolution_batch(),
        merge_replicates_batch(), flatten_generation()) -- see
        gpu_change_vector.batch_calculate_change_vectors()'s docstring for
        what it bounds and why. None (default): unchanged, each call still
        batches its whole input in one xp pass. Only relevant when xp is
        set; ignored by the per-individual path.
    """
    import time as _time

    if progress:
        print(f"[run_ga] seeding {len(seeds)} initial solutions (xp={_xp_label(xp)})...", flush=True)
        _t0 = _time.perf_counter()
    pop_index = seed_population(seeds, analysis_objects, locvec, xp=xp, progress_every=progress_every, chunk_size=chunk_size)
    pop = list(pop_index.values())
    if progress:
        print(f"[run_ga] seeded: {len(seeds)} seeds -> {len(pop)} distinct in {_time.perf_counter() - _t0:.2f}s", flush=True)

    if target_size is None:
        target_size = len(pop)

    for gen in range(num_gens):
        if progress:
            print(f"[run_ga] generation {gen + 1}/{num_gens} starting, pop={len(pop)}", flush=True)

        if flatten_every and gen > 0 and gen % flatten_every == 0:
            if progress:
                _t0 = _time.perf_counter()
            pop = refresh_change_vectors(pop, analysis_objects, locvec, xp=xp, progress_every=progress_every, chunk_size=chunk_size)
            pop = flatten_generation(pop, aa_seq, analysis_objects, locvec, flatten_recursion_limit, xp=xp, chunk_size=chunk_size)
            if progress:
                print(f"[run_ga]   flatten: pop={len(pop)} in {_time.perf_counter() - _t0:.2f}s", flush=True)

        # Dedup against the *whole generation's* output, not just the
        # pre-generation population: two different individuals' offspring
        # landing on the same new genotype in the same generation must
        # merge into one entry, not become two duplicate Proposed_Solution
        # objects with identical codons (a latent bug in the original
        # per-rep `next((q for q in pop if q.codons == rep), None)` scan,
        # which only ever checked the pre-generation pop -- fixed as a
        # side effect of switching to a dict keyed by codons).
        if progress:
            _t0 = _time.perf_counter()
        pop_index = {tuple(p.codons): p for p in pop}

        if xp is not None and lookahead:
            # Batched growth path -- see directed_evolution_batch()'s
            # docstring and Handoff.md sec 6 for why this exists: lookahead
            # scoring, not anything else in the loop, was the measured
            # real-scale bottleneck. Classification (random vs. directed)
            # keeps the same one-random.random()-per-individual-in-pop-order
            # sequence as the per-individual path below; what changes is
            # that directed individuals' replicates are all scored in one
            # batched pass instead of individually, AND (below) storage of
            # both kinds of replicate is now batched too instead of one
            # merge_replicate() call each -- see merge_replicate_exact()/
            # merge_replicates_batch()'s docstrings: after Kmer batching,
            # this per-replicate storage diffing became growth's actual
            # dominant cost (12.25s vs. 1.45s scoring for ~10,000
            # candidates -- Handoff.md sec 5/6), not anything scoring
            # touches.
            is_random = [random.random() < 0.5 for _ in pop]
            random_individuals = [p for p, r in zip(pop, is_random) if r]
            directed_individuals = [p for p, r in zip(pop, is_random) if not r]

            random_reps = []
            for p in random_individuals:
                random_reps.extend(replicate_and_mutate_random(p.codons, aa_seq))
            merge_replicates_batch(pop_index, random_reps, analysis_objects, locvec, xp, progress_every=progress_every, chunk_size=chunk_size)

            vecs_out = {}
            batch_reps = directed_evolution_batch(
                directed_individuals, weights, aa_seq, analysis_objects, locvec, xp=xp, progress_every=progress_every,
                vecs_out=vecs_out, chunk_size=chunk_size,
            )
            for p in directed_individuals:
                for rep in batch_reps[id(p)]:
                    merge_replicate_exact(pop_index, rep, vecs_out[tuple(rep)])
        else:
            for i, p in enumerate(pop):
                if random.random() < 0.5:
                    reps = replicate_and_mutate_random(p.codons, aa_seq)
                else:
                    reps = directed_evolution(
                        p.codons, p.change_vecs, weights, aa_seq, analysis_objects, locvec, lookahead=lookahead,
                    )
                for rep in reps:
                    merge_replicate(pop_index, rep, analysis_objects, locvec, parent=p)
                if progress_every and (i + 1) % progress_every == 0:
                    _progress_print("run_ga growth", i + 1, len(pop), _t0)

        pop = list(pop_index.values())
        if progress:
            print(f"[run_ga]   growth: pop={len(pop)} in {_time.perf_counter() - _t0:.2f}s", flush=True)

        if refresh_every and gen % refresh_every == 0:
            if progress:
                _t0 = _time.perf_counter()
            pop = refresh_change_vectors(pop, analysis_objects, locvec, xp=xp, progress_every=progress_every, chunk_size=chunk_size)
            if progress:
                print(f"[run_ga]   refresh: pop={len(pop)} in {_time.perf_counter() - _t0:.2f}s", flush=True)

        if progress:
            _t0 = _time.perf_counter()
        pop = select_survivors(pop, weights, target_size)
        if progress:
            print(f"[run_ga]   select: pop={len(pop)} in {_time.perf_counter() - _t0:.2f}s", flush=True)

        if save_dir:
            save_gen(pop, gen, save_dir, run_name)

    return pop


def _xp_label(xp) -> str:
    return "none (per-individual)" if xp is None else getattr(xp, "__name__", str(xp))
