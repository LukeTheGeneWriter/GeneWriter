"""Declarative GA schedules: a JSON-able list of step dicts driving the
population through generations, the way an ML training config drives a
model through epochs.

    schedule = [
        {"kind": "input", "count": 50000},
        {"kind": "growth", "rate": 4, "mutation_chance": 0.1},
        {"kind": "kill_off"},
        {"kind": "growth", "rate": 10, "mutation_chance": 0.1},
        {"kind": "flatten"},
        {"kind": "repeat", "times": 10, "steps": [
            {"kind": "growth", "rate": 4, "mutation_chance": 0.1},
            {"kind": "select", "target_size": 5000},
        ]},
    ]
    final_pop = run_schedule(aa_seq, weights, analysis_objects, schedule)

Every step is a plain dict (`"kind"` plus whatever params that kind takes) so
a schedule can be written in a config file, saved alongside a run's results
for reproducibility, diffed between runs, or built programmatically -- not
just hardcoded as a sequence of Python calls.

New step kinds are pluggable the same way change-vector terms are: decorate
a function with @register_step("my_kind"); its signature is
`(pop: list, ctx: ScheduleContext, params: dict) -> list`, params being the
step's dict with "kind" already removed. Registration happens at import
time, same caveat as register_term() -- the defining module must be
imported before run_schedule() runs.

Performance: "growth" always produces new genotypes with an *approximate*
change vector, diffed from their parent (see ga.merge_replicate() /
change_vector.diff_change_vector()) rather than a full recompute -- cheap,
which is the point of running several growth steps in a row. "kill_off",
"select", and "flatten" always do a full exact refresh (see
ga.refresh_change_vectors()) of the population *first*, before their own
logic -- those are the steps that make a survival decision based on the
scores, and this is also where the schedule format pays for itself: stack
several cheap "growth" steps before one precision checkpoint (as in the
`repeat` block in the example above) and the exact refresh only runs on
however many individuals are left at that checkpoint, not every generation.

Every exact refresh -- "input"'s seeding and "kill_off"/"select"/"flatten"'s
pre-step refresh -- can be routed through gpu_change_vector's population-
batched computation instead of the per-individual loop: pass
`xp=numpy`/`xp=cupy` to run_schedule() (see ScheduleContext.xp). Measured
25-48x speedup on real hardware for that path (see Handoff.md sec 6 -- all
5 terms are batched now, including Kmer).

"growth" used to benefit less than that number might suggest. When xp is
set and a "growth" step's `lookahead` is True (the default), *choosing*
each replicate's best synonymous alternative is also batched (see
ga.directed_evolution_batch()) -- this was originally the actual real-scale
bottleneck (~20-27 individuals/sec, tens of times slower per individual
than the batched refresh path). "growth"'s new-genotype *storage* is now
batched too when xp is set (ga.merge_replicate_exact()/
ga.merge_replicates_batch(), instead of diffing each accepted replicate
from its parent one at a time via merge_replicate()) -- once the
alternative-scoring itself got fast, that per-replicate storage diffing had
become the new dominant cost (see directed_evolution_batch()'s docstring
and Handoff.md sec 5/6 for the profiling that found it).

"directed_growth" is "growth" with the random-mutation branch removed
entirely -- every replicate is directed/lookahead, none is a plain random
mutation. Meant for strategic placement rather than as a blanket swap for
"growth", e.g. right after "flatten": flatten trades concentration for
neighborhood breadth, and a pure directed-growth step right after pushes
each of those newly-diversified individuals toward its locally-best
synonymous alternative instead of spending half its replicates on
undirected mutation again.
"""

import random
from dataclasses import dataclass, field

from .change_vector import AnalysisObjects
from .codon_tables import sequence_space_size
from .gpu_change_vector import suggest_chunk_size
from .ga import (
    directed_evolution,
    directed_evolution_batch,
    flatten_generation,
    generate_seed,
    kill_off,
    kill_off_outside_natural_range,
    merge_replicate,
    merge_replicate_exact,
    merge_replicates_batch,
    refresh_change_vectors,
    replicate_and_mutate_random,
    save_gen,
    seed_population,
    select_survivors,
    suggest_population_size,
)

_STEP_REGISTRY = {}


def _remaining_space(aa_seq: str, already_covered: int) -> int:
    """How many more distinct codon sequences for aa_seq are even possible
    to generate, given `already_covered` distinct individuals already in
    play. codon_tables.sequence_space_size(aa_seq) is a hard, finite
    ceiling -- the exact count of synonymous sequences that exist at all --
    so requesting more than this remaining headroom guarantees wasted
    duplicate-collision work (a seed/replicate that can only ever land on a
    genotype already covered). Real-scale proteins' space is astronomically
    larger than any population size used here (100 residues alone is
    already ~3**100), so this is a no-op in practice for anything but a
    short peptide, where over-requesting a round-number count like 50,000
    is otherwise easy to do by accident."""
    return max(sequence_space_size(aa_seq) - already_covered, 0)


def register_step(kind: str):
    """Decorator: register a new schedule step kind. See module docstring
    for the required signature."""
    def decorator(fn):
        if kind in _STEP_REGISTRY:
            raise ValueError(f"Schedule step kind {kind!r} is already registered")
        _STEP_REGISTRY[kind] = fn
        return fn
    return decorator


def registered_steps() -> dict:
    return dict(_STEP_REGISTRY)


@dataclass
class ScheduleContext:
    aa_seq: str
    weights: dict
    analysis_objects: AnalysisObjects
    locvec: list = None
    save_dir: str = None
    run_name: str = "run"
    step_count: int = field(default=0)
    # Array module (numpy or cupy) to batch "input"'s seeding and every
    # exact refresh (kill_off/select/flatten) through
    # gpu_change_vector.batch_calculate_change_vectors() instead of a
    # per-individual Python loop. None (default) keeps the original
    # per-individual path. "growth" is unaffected either way -- its new
    # genotypes are cheaply diffed from their parent, not batched.
    xp: object = None
    # Passed straight through to every xp-batched
    # gpu_change_vector.batch_calculate_change_vectors() call this makes
    # (via seed_population()/refresh_change_vectors()/
    # directed_evolution_batch()/merge_replicates_batch()/
    # flatten_generation()) -- see that function's docstring for what it
    # bounds and why: population size and protein length both multiply
    # into some of its intermediate arrays, so a long enough protein can
    # OOM at a population size that was previously fine. None here (a
    # ScheduleContext built directly rather than via run_schedule()) keeps
    # the original unbounded-batch behavior -- run_schedule() itself
    # replaces None with gpu_change_vector.suggest_chunk_size()'s VRAM-aware
    # default before constructing this dataclass, so callers going through
    # the normal entry point get the safer/faster default automatically;
    # see run_schedule()'s own docstring. Only relevant when xp is set.
    chunk_size: int = None
    # Diagnostic-only, no effect on results: `progress` prints each step's
    # kind, timing, and resulting population size as it runs (see
    # run_steps()); `progress_every` is passed through to
    # seed_population()/refresh_change_vectors() for finer-grained
    # every-N-individuals progress inside "input"/"kill_off"/"select"/
    # "flatten". Both default off/None -- silent, matching original
    # behavior.
    progress: bool = False
    progress_every: int = None
    # Callable aa_seq -> list[codon_str], used by the "input" step to
    # generate each new seed genotype. None (default) uses ga.generate_seed
    # -- unchanged behavior. Pass e.g.
    # functools.partial(codon_ngram.sample_codon_ngram_seed, model=model)
    # to seed from a trained CodonNgramModel instead -- same opt-in pattern
    # as xp above.
    seed_fn: object = None


@register_step("input")
def _step_input(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Add `count` freshly-generated random seed genotypes to the
    population (merging into any genotype already present rather than
    replacing the population), the way a training run's initial batch of
    samples is drawn.

    Uses ga.seed_population() -- one batched (or per-individual, per
    ctx.xp) pass over the new seeds -- rather than one merge_replicate()
    call per seed; this is the step where a large ctx.xp batch pays off
    most, since none of these genotypes has a parent to cheaply diff
    against. Seeds come from ctx.seed_fn (ga.generate_seed by default --
    see ScheduleContext.seed_fn) rather than a hardcoded call, so a trained
    codon_ngram.CodonNgramModel can be substituted.

    `count` is optional -- if omitted, ga.suggest_population_size() picks a
    hardware-aware default (as many distinct seeds as available system RAM
    can hold, up to the protein's finite synonymous-codon space) instead of
    a hand-picked constant, so short runs don't accidentally under-explore
    a large protein's neighborhood just because nobody tuned `count` for
    that specific target. An explicit `count` is still clamped to
    _remaining_space(ctx.aa_seq, len(pop)) as before -- can't usefully
    generate more distinct seeds than the protein's space has left
    uncovered, whether that number came from the caller or from the
    auto-sized default."""
    requested = params.get("count")
    if requested is None:
        requested = suggest_population_size(ctx.aa_seq, ctx.analysis_objects, already_covered=len(pop), locvec=ctx.locvec)
    count = min(requested, _remaining_space(ctx.aa_seq, len(pop)))
    seed_fn = ctx.seed_fn or generate_seed
    new_seeds = [seed_fn(ctx.aa_seq) for _ in range(count)]
    new_index = seed_population(new_seeds, ctx.analysis_objects, ctx.locvec, xp=ctx.xp, progress_every=ctx.progress_every, chunk_size=ctx.chunk_size)

    pop_index = {tuple(p.codons): p for p in pop}
    for key, new_sol in new_index.items():
        existing = pop_index.get(key)
        if existing is not None:
            existing.number += new_sol.number
        else:
            pop_index[key] = new_sol
    return list(pop_index.values())


def _do_growth(
    pop: list, ctx: ScheduleContext, rate: int, mutation_chance: float, directed_fraction: float, lookahead: bool,
) -> list:
    """Shared implementation behind the "growth" and "directed_growth" step
    kinds (see both registered functions' docstrings) -- every difference
    between them is just what directed_fraction is: "growth" takes it from
    `params` (default 0.5, a real mix), "directed_growth" hardcodes 1.0
    (every replicate directed, no random-mutation branch at all).

    directed_fraction >= 1.0 skips the random-mutation classification/
    branch entirely rather than just routing 0 individuals into it -- both
    to avoid a wasted random.random() draw per individual (keeps
    "directed_growth"'s RNG consumption minimal and predictable, not "the
    same classification draw as growth, just always landing on directed")
    and because merge_replicates_batch()/replicate_and_mutate_random()
    have no useful work to do on an empty individual list anyway.

    `rate` is clamped to _remaining_space(ctx.aa_seq, len(pop)) -- no
    individual can be usefully asked for more replicates than there are
    distinct codon sequences left uncovered in the whole protein's space,
    since every replicate beyond that is guaranteed to land on a genotype
    already in pop (deduped away for free by pop_index, but still a wasted
    draw/score). Only binds for very short peptides in practice -- see
    _remaining_space()'s docstring.
    """
    rate = min(rate, _remaining_space(ctx.aa_seq, len(pop)))
    pop_index = {tuple(p.codons): p for p in pop}

    if ctx.xp is not None and lookahead:
        # Batched growth path -- see ga.directed_evolution_batch() and this
        # module's docstring. Classification keeps the same
        # one-random.random()-per-individual-in-pop-order sequence as the
        # per-individual path below (skipped entirely, not just
        # short-circuited, when directed_fraction >= 1.0).
        if directed_fraction >= 1.0:
            directed_individuals = list(pop)
            random_individuals = []
        else:
            is_directed = [random.random() < directed_fraction for _ in pop]
            directed_individuals = [p for p, d in zip(pop, is_directed) if d]
            random_individuals = [p for p, d in zip(pop, is_directed) if not d]

        random_reps = []
        for p in random_individuals:
            random_reps.extend(replicate_and_mutate_random(p.codons, ctx.aa_seq, nreplicates=rate, mutation_rate=mutation_chance))
        merge_replicates_batch(pop_index, random_reps, ctx.analysis_objects, ctx.locvec, ctx.xp, progress_every=ctx.progress_every, chunk_size=ctx.chunk_size)

        vecs_out = {}
        batch_reps = directed_evolution_batch(
            directed_individuals, ctx.weights, ctx.aa_seq, ctx.analysis_objects, ctx.locvec,
            nreplicates=rate, xp=ctx.xp, progress_every=ctx.progress_every, vecs_out=vecs_out,
            chunk_size=ctx.chunk_size,
        )
        for p in directed_individuals:
            for rep in batch_reps[id(p)]:
                merge_replicate_exact(pop_index, rep, vecs_out[tuple(rep)])
    else:
        if ctx.progress_every:
            import time as _time
            _t0 = _time.perf_counter()
        for i, p in enumerate(pop):
            # Short-circuits (no random.random() call at all) when
            # directed_fraction >= 1.0, same reasoning as the batched
            # branch above.
            if directed_fraction >= 1.0 or random.random() < directed_fraction:
                # This, not anything xp/GPU-batching touches (unless xp is
                # set -- see above), is the known per-individual hotspot at
                # real scale: `lookahead=True` scores every synonymous
                # alternative at the chosen position via diff_change_vector()
                # -- many small numpy calls per individual, dispatch-
                # overhead-bound rather than compute-bound. See
                # ga.run_ga's identical instrumentation and Handoff.md sec 6
                # for measured throughput (~20-27 individuals/s here vs.
                # 25-48x faster for the batched refresh path on real
                # hardware).
                reps = directed_evolution(
                    p.codons, p.change_vecs, ctx.weights, ctx.aa_seq, ctx.analysis_objects, ctx.locvec,
                    nreplicates=rate, lookahead=lookahead,
                )
            else:
                reps = replicate_and_mutate_random(p.codons, ctx.aa_seq, nreplicates=rate, mutation_rate=mutation_chance)
            if ctx.progress_every and (i + 1) % ctx.progress_every == 0:
                from .ga import _progress_print
                _progress_print("schedule growth", i + 1, len(pop), _t0)
            for rep in reps:
                merge_replicate(pop_index, rep, ctx.analysis_objects, ctx.locvec, parent=p)

    result = list(pop_index.values())
    ctx.step_count += 1
    if ctx.save_dir:
        save_gen(result, ctx.step_count, ctx.save_dir, ctx.run_name)
    return result


@register_step("growth")
def _step_growth(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Reproduce every individual. `rate` offspring per individual (default
    10), each independently either a random-mutation replicate (with
    probability `mutation_chance` per codon, default 0.05) or a
    directed-evolution replicate (greedy one-step lookahead biased toward
    high-change-vector positions, unless `lookahead` is False -- see
    ga.directed_evolution()), chosen per-offspring with probability
    `directed_fraction` (default 0.5) of being directed.

    Cheap: every new genotype's change vector is approximated by diffing
    against its parent rather than fully recomputed -- see module
    docstring. When ctx.xp is set and lookahead=True, *which* synonymous
    alternative gets chosen is instead decided by a batched exact
    recompute across every directed individual in this step at once (see
    ga.directed_evolution_batch()), and storage of both directed and
    random-mutation replicates is batched too (see
    ga.merge_replicate_exact()/ga.merge_replicates_batch()) -- see module
    docstring.

    See "directed_growth" for a step that skips the random-mutation
    branch entirely (100% directed) instead of mixing the two."""
    rate = params.get("rate", 10)
    mutation_chance = params.get("mutation_chance", 0.05)
    directed_fraction = params.get("directed_fraction", 0.5)
    lookahead = params.get("lookahead", True)
    return _do_growth(pop, ctx, rate, mutation_chance, directed_fraction, lookahead)


@register_step("directed_growth")
def _step_directed_growth(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Pure directed-evolution growth: every replicate comes from the
    lookahead/directed-evolution path (ga.directed_evolution() /
    ga.directed_evolution_batch()) -- no random-mutation replicates at
    all, unlike "growth" (which mixes both via `directed_fraction`).

    Meant to be placed strategically in a schedule rather than used as a
    blanket replacement for "growth" -- e.g. right after a "flatten" step:
    flatten trades replicate-count concentration for neighborhood breadth
    (cashing each individual's copies in for single-mutation neighbors),
    and a pure directed-growth step right after aggressively pushes each
    of those newly-diversified individuals toward its locally-best
    synonymous alternative instead of spending half the step's replicates
    on undirected random mutation again.

    `rate` (replicates per individual, default 10) and `lookahead`
    (default True -- greedy one-step-lookahead alternative selection vs.
    a uniform-random one at the chosen position, see
    ga.directed_evolution()'s docstring) are the only params -- no
    `mutation_chance`/`directed_fraction`, since there's no random-mutation
    branch here to configure. Same xp-batching behavior as "growth" when
    ctx.xp is set and lookahead=True (see ga.directed_evolution_batch()
    and this module's docstring), just with every individual routed
    through the directed/batched path instead of a random subset."""
    rate = params.get("rate", 10)
    lookahead = params.get("lookahead", True)
    return _do_growth(pop, ctx, rate, mutation_chance=0.0, directed_fraction=1.0, lookahead=lookahead)


@register_step("kill_off")
def _step_kill_off(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Proportional cull: remove `percent_cut`% (default 30) of total
    replicate count, weighted toward individuals most in need of mutation.
    See ga.kill_off(). Refreshes change vectors exactly first -- see module
    docstring."""
    percent_cut = params.get("percent_cut", 30)
    pop = refresh_change_vectors(pop, ctx.analysis_objects, ctx.locvec, xp=ctx.xp, progress_every=ctx.progress_every, chunk_size=ctx.chunk_size)
    return kill_off(pop, ctx.weights, percent_cut=percent_cut)


@register_step("natural_range_cutoff")
def _step_natural_range_cutoff(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Hard cutoff, not a survival-pressure step like kill_off/select/
    flatten: drops any individual whose own aggregate per-category metric
    is more than `threshold` standard deviations from the natural per-gene
    baseline mean in any category (see
    ga.kill_off_outside_natural_range()). No pre-step exact refresh --
    unlike kill_off/select/flatten, this doesn't consult change_vecs at
    all, only each individual's raw codons against analysis_objects.

    `threshold` is required, no default -- pick deliberately (see
    weight_calibration.natural_deviation_zscores()'s docstring for what's
    being thresholded) rather than inheriting an arbitrary cutoff."""
    threshold = params["threshold"]
    return kill_off_outside_natural_range(pop, ctx.analysis_objects, threshold)


@register_step("select")
def _step_select(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Cap the population to `target_size` distinct individuals. See
    ga.select_survivors(). Refreshes change vectors exactly first -- see
    module docstring.

    `target_size` is optional -- if omitted, ga.suggest_population_size()
    picks a hardware-aware default (as large as available system RAM
    reasonably allows) instead of a hand-picked constant, same reasoning as
    "input"'s `count`. An explicit `target_size` is still clamped to
    codon_tables.sequence_space_size(ctx.aa_seq) as before -- a population
    can never usefully hold more distinct individuals than the protein's
    finite synonymous-codon space contains in total."""
    requested = params.get("target_size")
    if requested is None:
        requested = suggest_population_size(ctx.aa_seq, ctx.analysis_objects, locvec=ctx.locvec)
    target_size = min(requested, sequence_space_size(ctx.aa_seq))
    pop = refresh_change_vectors(pop, ctx.analysis_objects, ctx.locvec, xp=ctx.xp, progress_every=ctx.progress_every, chunk_size=ctx.chunk_size)
    return select_survivors(pop, ctx.weights, target_size)


@register_step("flatten")
def _step_flatten(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Trade replicate-count concentration for neighborhood breadth.
    `recursion_limit` (default 3). See ga.flatten_generation(), which this
    threads ctx.xp into -- when set, every new neighbor genotype created
    this step gets its change vector via one batched
    batch_calculate_change_vectors() call instead of one
    calculate_change_vector() call each (see ga._flatten_round()'s
    docstring). Refreshes change vectors exactly first -- see module
    docstring."""
    recursion_limit = params.get("recursion_limit", 3)
    pop = refresh_change_vectors(pop, ctx.analysis_objects, ctx.locvec, xp=ctx.xp, progress_every=ctx.progress_every, chunk_size=ctx.chunk_size)
    return flatten_generation(pop, ctx.aa_seq, ctx.analysis_objects, ctx.locvec, recursion_limit, xp=ctx.xp, chunk_size=ctx.chunk_size)


@register_step("save")
def _step_save(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Checkpoint the current population to ctx.save_dir regardless of
    whether a growth step already would have (e.g. right before a risky
    step). No-op if ctx.save_dir wasn't set."""
    if ctx.save_dir:
        save_gen(pop, ctx.step_count, ctx.save_dir, ctx.run_name)
    return pop


@register_step("repeat")
def _step_repeat(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Run a nested sub-schedule `times` times in a row."""
    times = params["times"]
    steps = params["steps"]
    for _ in range(times):
        pop = run_steps(pop, ctx, steps)
    return pop


def run_steps(pop: list, ctx: ScheduleContext, schedule: list) -> list:
    for step in schedule:
        step = dict(step)
        try:
            kind = step.pop("kind")
        except KeyError:
            raise ValueError(f"Schedule step missing required 'kind' key: {step!r}")
        executor = _STEP_REGISTRY.get(kind)
        if executor is None:
            raise ValueError(f"Unknown schedule step kind {kind!r}. Registered kinds: {sorted(_STEP_REGISTRY)}")

        if ctx.progress:
            import time as _time
            print(f"[schedule] {kind} {step} starting, pop={len(pop)}", flush=True)
            t0 = _time.perf_counter()
            pop = executor(pop, ctx, step)
            print(f"[schedule] {kind} done: pop={len(pop)} in {_time.perf_counter() - t0:.2f}s", flush=True)
        else:
            pop = executor(pop, ctx, step)
    return pop


def run_schedule(
    aa_seq: str,
    weights: dict,
    analysis_objects: AnalysisObjects,
    schedule: list,
    locvec: list = None,
    save_dir: str = None,
    run_name: str = "run",
    xp=None,
    progress: bool = False,
    progress_every: int = None,
    seed_fn=None,
    chunk_size: int = None,
) -> list:
    """Run a declarative schedule end to end and return the final population.

    schedule: list of step dicts, e.g. [{"kind": "input", "count": 50000},
        {"kind": "growth", "rate": 4}, ...] -- see module docstring, and
        registered_steps() for the currently-available kinds.
    xp: array module (numpy or cupy) to batch "input"'s seeding and every
        exact refresh ("kill_off"/"select"/"flatten") through
        gpu_change_vector.batch_calculate_change_vectors() -- see
        ScheduleContext.xp. None (default) keeps the original
        per-individual path.
    progress / progress_every: diagnostic-only step-level and
        individual-level progress logging -- see ScheduleContext. Both
        default off/None, matching the original silent behavior.
    seed_fn: callable aa_seq -> list[codon_str] used by "input" to generate
        new seeds -- see ScheduleContext.seed_fn. None (default) uses
        ga.generate_seed.
    chunk_size: caps how many individuals any single xp-batched call in this
        schedule processes at once -- see ScheduleContext.chunk_size. None
        (default, and only relevant when xp is set) no longer means "one
        unchunked pass across the whole population" -- that shape is what
        actually OOM'd on a real Colab run once population size grew large
        enough (see memory/colab_stress_test_stale_paste_oom.md). Instead,
        gpu_change_vector.suggest_chunk_size() picks a VRAM-aware default:
        queries actually-free GPU memory and sizes chunk_size to use it,
        so a bigger GPU automatically gets bigger (faster) batches with no
        manual tuning, while still bounding peak memory. Pass an explicit
        chunk_size to override this and get the old fixed-cap behavior.
    """
    if chunk_size is None and xp is not None:
        chunk_size = suggest_chunk_size(xp, len(aa_seq))
    ctx = ScheduleContext(aa_seq=aa_seq, weights=weights, analysis_objects=analysis_objects,
                           locvec=locvec, save_dir=save_dir, run_name=run_name, xp=xp,
                           progress=progress, progress_every=progress_every, seed_fn=seed_fn,
                           chunk_size=chunk_size)
    return run_steps([], ctx, schedule)
