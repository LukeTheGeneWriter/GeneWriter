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
"""

import random
from dataclasses import dataclass, field

from .change_vector import AnalysisObjects
from .ga import (
    directed_evolution,
    flatten_generation,
    generate_seed,
    kill_off,
    merge_replicate,
    refresh_change_vectors,
    replicate_and_mutate_random,
    save_gen,
    select_survivors,
)

_STEP_REGISTRY = {}


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


@register_step("input")
def _step_input(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Add `count` freshly-generated random seed genotypes to the
    population (merging into any genotype already present rather than
    replacing the population), the way a training run's initial batch of
    samples is drawn."""
    count = params["count"]
    pop_index = {tuple(p.codons): p for p in pop}
    for _ in range(count):
        merge_replicate(pop_index, generate_seed(ctx.aa_seq), ctx.analysis_objects, ctx.locvec)
    return list(pop_index.values())


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
    docstring."""
    rate = params.get("rate", 10)
    mutation_chance = params.get("mutation_chance", 0.05)
    directed_fraction = params.get("directed_fraction", 0.5)
    lookahead = params.get("lookahead", True)

    pop_index = {tuple(p.codons): p for p in pop}
    for p in pop:
        if random.random() < directed_fraction:
            reps = directed_evolution(
                p.codons, p.change_vecs, ctx.weights, ctx.aa_seq, ctx.analysis_objects, ctx.locvec,
                nreplicates=rate, lookahead=lookahead,
            )
        else:
            reps = replicate_and_mutate_random(p.codons, ctx.aa_seq, nreplicates=rate, mutation_rate=mutation_chance)
        for rep in reps:
            merge_replicate(pop_index, rep, ctx.analysis_objects, ctx.locvec, parent=p)

    result = list(pop_index.values())
    ctx.step_count += 1
    if ctx.save_dir:
        save_gen(result, ctx.step_count, ctx.save_dir, ctx.run_name)
    return result


@register_step("kill_off")
def _step_kill_off(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Proportional cull: remove `percent_cut`% (default 30) of total
    replicate count, weighted toward individuals most in need of mutation.
    See ga.kill_off(). Refreshes change vectors exactly first -- see module
    docstring."""
    percent_cut = params.get("percent_cut", 30)
    pop = refresh_change_vectors(pop, ctx.analysis_objects, ctx.locvec)
    return kill_off(pop, ctx.weights, percent_cut=percent_cut)


@register_step("select")
def _step_select(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Cap the population to `target_size` distinct individuals. See
    ga.select_survivors(). Refreshes change vectors exactly first -- see
    module docstring."""
    target_size = params["target_size"]
    pop = refresh_change_vectors(pop, ctx.analysis_objects, ctx.locvec)
    return select_survivors(pop, ctx.weights, target_size)


@register_step("flatten")
def _step_flatten(pop: list, ctx: ScheduleContext, params: dict) -> list:
    """Trade replicate-count concentration for neighborhood breadth.
    `recursion_limit` (default 3). See ga.flatten_generation(). Refreshes
    change vectors exactly first -- see module docstring."""
    recursion_limit = params.get("recursion_limit", 3)
    pop = refresh_change_vectors(pop, ctx.analysis_objects, ctx.locvec)
    return flatten_generation(pop, ctx.aa_seq, ctx.analysis_objects, ctx.locvec, recursion_limit)


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
) -> list:
    """Run a declarative schedule end to end and return the final population.

    schedule: list of step dicts, e.g. [{"kind": "input", "count": 50000},
        {"kind": "growth", "rate": 4}, ...] -- see module docstring, and
        registered_steps() for the currently-available kinds.
    """
    ctx = ScheduleContext(aa_seq=aa_seq, weights=weights, analysis_objects=analysis_objects,
                           locvec=locvec, save_dir=save_dir, run_name=run_name)
    return run_steps([], ctx, schedule)
