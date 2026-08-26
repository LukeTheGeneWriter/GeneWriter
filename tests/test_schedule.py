import json
import os

import pytest

from genewriter import ga
from genewriter import schedule as sched
from genewriter.classes import Proposed_Solution
from genewriter.change_vector import calculate_change_vector
from genewriter.codon_tables import generate_codon_vec


@pytest.fixture
def ctx(aa_seq, analysis_objects, weights):
    return sched.ScheduleContext(aa_seq=aa_seq, weights=weights, analysis_objects=analysis_objects)


def test_registered_steps_includes_the_builtin_kinds():
    assert set(sched.registered_steps()) == {
        'input', 'growth', 'directed_growth', 'kill_off', 'kill_off_by_term', 'protect',
        'release_protection', 'natural_range_cutoff', 'select', 'flatten', 'save', 'repeat',
    }


def test_step_input_adds_the_requested_number_of_seeds(ctx):
    result = sched.run_steps([], ctx, [{"kind": "input", "count": 25}])
    assert sum(p.number for p in result) == 25
    assert all(isinstance(p, Proposed_Solution) for p in result)


def test_step_input_uses_default_seed_fn_when_unset(ctx, aa_seq):
    # Regression guard: ctx.seed_fn defaults to None -> generate_seed,
    # unchanged behavior -- every produced genotype must be a well-formed
    # random seed for aa_seq (same shape ga.generate_seed produces).
    result = sched.run_steps([], ctx, [{"kind": "input", "count": 10}])
    for p in result:
        assert len(p.codons) == len(aa_seq)


def test_step_input_uses_custom_seed_fn_when_provided(aa_seq, analysis_objects, weights):
    fixed_sol = [c[0] for c in generate_codon_vec(aa_seq)]
    custom_ctx = sched.ScheduleContext(
        aa_seq=aa_seq, weights=weights, analysis_objects=analysis_objects,
        seed_fn=lambda aa_seq: list(fixed_sol),
    )
    result = sched.run_steps([], custom_ctx, [{"kind": "input", "count": 5}])
    # All 5 draws are the identical fixed genotype -- should merge into one
    # Proposed_Solution with number=5, not 5 distinct (random) entries.
    assert len(result) == 1
    assert result[0].codons == fixed_sol
    assert result[0].number == 5


def test_step_input_merges_into_existing_population_rather_than_replacing_it(ctx, aa_seq, analysis_objects):
    existing_sol = [c[0] for c in generate_codon_vec(aa_seq)]
    existing = Proposed_Solution(existing_sol, 7, calculate_change_vector(existing_sol, analysis_objects))
    result = sched.run_steps([existing], ctx, [{"kind": "input", "count": 10}])
    assert sum(p.number for p in result) == 17


def test_step_input_clamps_count_to_remaining_sequence_space(monkeypatch, analysis_objects, weights):
    # "MC": M has 1 synonymous codon, C has 2 -- only 2 distinct sequences
    # exist for this aa_seq at all, so a requested count of 1000 must be
    # clamped down to 2, not passed through unchanged.
    tiny_ctx = sched.ScheduleContext(aa_seq="MC", weights=weights, analysis_objects=analysis_objects)
    real_seed_population = sched.seed_population

    def _spy_seed_population(new_seeds, *args, **kwargs):
        _spy_seed_population.seen = len(new_seeds)
        return real_seed_population(new_seeds, *args, **kwargs)

    monkeypatch.setattr(sched, "seed_population", _spy_seed_population)
    sched.run_steps([], tiny_ctx, [{"kind": "input", "count": 1000}])
    assert _spy_seed_population.seen == 2


def test_step_input_auto_sizes_count_when_omitted(analysis_objects, weights):
    """No `count` in params -- ga.suggest_population_size() picks a
    hardware-aware default instead of raising KeyError the way the
    required-count behavior used to. "MC" (space of exactly 2) keeps this
    fast and deterministic regardless of how much RAM the test machine
    actually has -- the sequence-space ceiling is what should bind here,
    not the RAM half of the suggestion."""
    tiny_ctx = sched.ScheduleContext(aa_seq="MC", weights=weights, analysis_objects=analysis_objects)
    result = sched.run_steps([], tiny_ctx, [{"kind": "input"}])
    assert 0 < sum(p.number for p in result) <= 2


def test_step_input_keep_caps_the_surviving_population(ctx):
    """Seed wide, keep narrow: `keep` cuts the freshly-seeded population
    down the same way a following "select" step would, so the next "growth"
    step multiplies a working-sized population rather than every seed RAM
    could hold (see _step_input's docstring)."""
    result = sched.run_steps([], ctx, [{"kind": "input", "count": 60, "keep": 10}])
    assert len(result) == 10


def test_step_input_without_keep_retains_every_distinct_seed(ctx):
    """`keep` is opt-in -- omitting it must leave the original
    keep-everything behavior untouched, since every existing schedule in
    the repo (and every saved one) relies on it."""
    result = sched.run_steps([], ctx, [{"kind": "input", "count": 60}])
    assert len(result) == 60, "seeds collided or were culled without a keep"


def test_step_input_keep_does_not_refresh_change_vectors(ctx, monkeypatch):
    """The reason `keep` lives on "input" instead of being a separate
    "select" step after it: seed_population() has just computed exact
    change vectors for every seed, and a "select" step would immediately
    recompute all of them (refresh-first convention). Over the largest
    population a run ever holds, that redundant refresh is the single most
    expensive thing `keep` exists to avoid -- so assert it doesn't happen,
    rather than trusting the reading."""
    calls = []
    real_refresh = sched.refresh_change_vectors
    monkeypatch.setattr(sched, "refresh_change_vectors",
                        lambda *a, **k: (calls.append(1), real_refresh(*a, **k))[1])

    sched.run_steps([], ctx, [{"kind": "input", "count": 20, "keep": 5}])
    assert not calls, "input's keep refreshed change vectors seed_population had already computed"

    # ...and the contrast that makes the point: the separate select step
    # this replaces does refresh, which is exactly the cost being skipped.
    sched.run_steps([], ctx, [{"kind": "input", "count": 20}, {"kind": "select", "target_size": 5}])
    assert calls


def test_step_input_keep_is_clamped_to_sequence_space(analysis_objects, weights):
    # "MC": only 2 distinct sequences exist at all, so keeping "the best
    # 1000" can only ever mean keeping both -- same ceiling the "select"
    # step's target_size is clamped to.
    tiny_ctx = sched.ScheduleContext(aa_seq="MC", weights=weights, analysis_objects=analysis_objects)
    result = sched.run_steps([], tiny_ctx, [{"kind": "input", "count": 50, "keep": 1000}])
    assert 0 < len(result) <= 2


def test_step_input_keep_below_one_raises(ctx):
    with pytest.raises(ValueError, match="keep must be >= 1"):
        sched.run_steps([], ctx, [{"kind": "input", "count": 10, "keep": 0}])


def test_step_growth_reproduces_and_dedups_within_the_step(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 1}])
    result = sched.run_steps(seeded, ctx, [{"kind": "growth", "rate": 5, "mutation_chance": 0.3}])
    codons_seen = [tuple(p.codons) for p in result]
    assert len(codons_seen) == len(set(codons_seen)), "growth step produced duplicate genotypes as separate entries"


def test_step_growth_clamps_rate_to_remaining_sequence_space(monkeypatch, analysis_objects, weights):
    # "MC" again: space=2, and after seeding one individual only 1 distinct
    # genotype is left uncovered -- a requested rate of 1000 must clamp down
    # to that 1 remaining slot, not pass through unchanged (directed_fraction
    # forced to 0.0 so every replicate takes the random-mutation path this
    # spy watches).
    tiny_ctx = sched.ScheduleContext(aa_seq="MC", weights=weights, analysis_objects=analysis_objects)
    seeded = sched.run_steps([], tiny_ctx, [{"kind": "input", "count": 1}])
    real_replicate = sched.replicate_and_mutate_random

    def _spy_replicate(sol, aa_seq, nreplicates=10, mutation_rate=0.05):
        _spy_replicate.seen = nreplicates
        return real_replicate(sol, aa_seq, nreplicates=nreplicates, mutation_rate=mutation_rate)

    monkeypatch.setattr(sched, "replicate_and_mutate_random", _spy_replicate)
    sched.run_steps(seeded, tiny_ctx, [{"kind": "growth", "rate": 1000, "directed_fraction": 0.0}])
    assert _spy_replicate.seen == 1


def test_step_kill_off_reduces_total_replicate_count(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 20}])
    total_before = sum(p.number for p in seeded)
    result = sched.run_steps(seeded, ctx, [{"kind": "kill_off", "percent_cut": 50}])
    assert sum(p.number for p in result) < total_before


def test_step_kill_off_by_term_reduces_total_replicate_count(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 20}])
    total_before = sum(p.number for p in seeded)
    result = sched.run_steps(seeded, ctx, [{"kind": "kill_off_by_term", "term": "CodonPairBias", "percent_cut": 50}])
    assert sum(p.number for p in result) < total_before


def test_step_kill_off_by_term_requires_term_param(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 3}])
    with pytest.raises(KeyError):
        sched.run_steps(seeded, ctx, [{"kind": "kill_off_by_term", "percent_cut": 50}])


def test_step_kill_off_inline_protect_shields_without_setting_protected(ctx):
    """The `protect` option on a "kill_off" step is a one-cull-only
    exemption, not the permanent "protect" step's flag -- the top 20% by
    distance from optimal must survive a 100% cut here, but come out with .protected
    still False (a later plain kill_off can still remove them)."""
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 20}])
    result = sched.run_steps(seeded, ctx, [
        {"kind": "kill_off", "percent_cut": 100, "protect": [["distance_from_optimal", 0.2]]},
    ])
    assert len(result) > 0
    assert all(not p.protected for p in result)


def test_step_kill_off_by_term_inline_protect_shields_without_setting_protected(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 20}])
    result = sched.run_steps(seeded, ctx, [
        {"kind": "kill_off_by_term", "term": "CodonPairBias", "percent_cut": 100, "protect": [["distance_from_optimal", 0.2]]},
    ])
    assert len(result) > 0
    assert all(not p.protected for p in result)


def test_step_protect_shields_everyone_when_top_fraction_is_one(ctx):
    """A protect step with top_fraction=1.0 marks the whole population
    protected -- a subsequent kill_off must then remove nothing at all."""
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 10}])
    total_before = sum(p.number for p in seeded)
    protected = sched.run_steps(seeded, ctx, [{"kind": "protect", "criteria": [["distance_from_optimal", 1.0]]}])
    assert all(p.protected for p in protected)

    result = sched.run_steps(protected, ctx, [{"kind": "kill_off", "percent_cut": 100}])
    assert sum(p.number for p in result) == total_before
    assert len(result) == len(protected)


def test_step_protect_requires_criteria_param(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 3}])
    with pytest.raises(KeyError):
        sched.run_steps(seeded, ctx, [{"kind": "protect"}])


def test_step_natural_range_cutoff_cuts_nothing_at_an_astronomical_threshold(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 5}])
    result = sched.run_steps(seeded, ctx, [{"kind": "natural_range_cutoff", "threshold": 1e9}])
    assert len(result) == len(seeded)


def test_step_natural_range_cutoff_requires_threshold_param(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 3}])
    with pytest.raises(KeyError):
        sched.run_steps(seeded, ctx, [{"kind": "natural_range_cutoff"}])


def test_step_select_caps_population_size(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 30}])
    result = sched.run_steps(seeded, ctx, [{"kind": "select", "target_size": 5}])
    assert len(result) == 5


def test_step_select_clamps_target_size_to_sequence_space(monkeypatch, analysis_objects, weights):
    # "MC": only 2 distinct sequences exist for this aa_seq at all, so a
    # requested target_size of 1000 must clamp down to 2, not pass through
    # unchanged -- checked by spying on select_survivors directly, since a
    # tiny population can't itself exceed 2 individuals to prove the clamp
    # engaged rather than select_survivors' own len(pop) behavior.
    tiny_ctx = sched.ScheduleContext(aa_seq="MC", weights=weights, analysis_objects=analysis_objects)
    seeded = sched.run_steps([], tiny_ctx, [{"kind": "input", "count": 1}])
    real_select_survivors = sched.select_survivors

    def _spy_select_survivors(pop, weights, target_size):
        _spy_select_survivors.seen = target_size
        return real_select_survivors(pop, weights, target_size)

    monkeypatch.setattr(sched, "select_survivors", _spy_select_survivors)
    sched.run_steps(seeded, tiny_ctx, [{"kind": "select", "target_size": 1000}])
    assert _spy_select_survivors.seen == 2


def test_step_select_auto_sizes_target_size_when_omitted(analysis_objects, weights):
    """No `target_size` in params -- ga.suggest_population_size() picks a
    hardware-aware default. "MC" (space of exactly 2) keeps this fast and
    deterministic regardless of test-machine RAM, same reasoning as the
    matching "input" test above."""
    tiny_ctx = sched.ScheduleContext(aa_seq="MC", weights=weights, analysis_objects=analysis_objects)
    seeded = sched.run_steps([], tiny_ctx, [{"kind": "input", "count": 2}])
    result = sched.run_steps(seeded, tiny_ctx, [{"kind": "select"}])
    assert len(result) <= 2


def test_step_flatten_collapses_everyone_to_one_copy(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 15}])
    result = sched.run_steps(seeded, ctx, [{"kind": "flatten", "recursion_limit": 1}])
    assert all(p.number == 1 for p in result)


def _stale_all_change_vecs(pop):
    n = len(pop[0].codons)
    stale = {'RareCodons': [0.0] * n, 'CodonUsage': [0.0] * n,
             'CodonPairBias': [0.0] * n, 'GC': [0.0] * n, 'Kmer': [0.0] * n}
    for p in pop:
        p.change_vecs = dict(stale)
    return pop


def test_step_kill_off_refreshes_change_vectors_first(ctx, analysis_objects):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 10}])
    _stale_all_change_vecs(seeded)
    result = sched.run_steps(seeded, ctx, [{"kind": "kill_off", "percent_cut": 0}])
    for p in result:
        assert p.change_vecs == calculate_change_vector(p.codons, analysis_objects)


def test_step_select_refreshes_change_vectors_first(ctx, analysis_objects):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 10}])
    _stale_all_change_vecs(seeded)
    result = sched.run_steps(seeded, ctx, [{"kind": "select", "target_size": 10}])
    for p in result:
        assert p.change_vecs == calculate_change_vector(p.codons, analysis_objects)


def test_step_flatten_refreshes_change_vectors_first(ctx, analysis_objects):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 10}])
    _stale_all_change_vecs(seeded)
    # recursion_limit=0 so flatten does no cash-in redistribution of its
    # own, isolating "did it refresh the input population" from "did
    # flatten's own new individuals get exact vecs (they always do)".
    result = sched.run_steps(seeded, ctx, [{"kind": "flatten", "recursion_limit": 0}])
    for p in result:
        assert p.change_vecs == calculate_change_vector(p.codons, analysis_objects)


def test_step_growth_diffs_new_individuals_against_their_parent(ctx, monkeypatch):
    """Growth's whole performance point is that new genotypes are diffed
    from their parent (see ga.merge_replicate(parent=...)), not fully
    recomputed. Confirm the code path is actually exercised, rather than
    inferring it indirectly -- whether the excerpt approximation ends up
    numerically different from a full recompute depends on sequence length
    vs. margin (see diff_change_vector's fallback threshold), so asserting
    on the *values* would be fragile; asserting the function was called
    is not."""
    import genewriter.ga as ga_module

    calls = []
    original = ga_module.diff_change_vector

    def _spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ga_module, "diff_change_vector", _spy)

    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 1}])
    sched.run_steps(seeded, ctx, [{"kind": "growth", "rate": 8, "mutation_chance": 0.5}])

    assert calls, "growth step never called diff_change_vector for a new genotype"


def test_step_growth_with_xp_and_lookahead_uses_batched_growth(aa_seq, analysis_objects, weights, monkeypatch):
    """ctx.xp set + lookahead=True (the default) must route "growth"
    through ga.directed_evolution_batch(), not the per-individual
    directed_evolution() -- see Handoff.md sec 6 on why this is the
    real-scale bottleneck the xp wiring is meant to fix."""
    import numpy as np

    # schedule.py did `from .ga import directed_evolution_batch`, which
    # binds the name into schedule's own module namespace -- patching
    # genewriter.ga's attribute would not affect schedule's already-bound
    # reference, so this patches `sched` (the module _step_growth actually
    # looks the name up in) instead.
    calls = []
    original = sched.directed_evolution_batch

    def _spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def _boom(*args, **kwargs):
        raise AssertionError("directed_evolution (per-individual) should not be called when ctx.xp is set and lookahead=True")

    monkeypatch.setattr(sched, "directed_evolution_batch", _spy)
    monkeypatch.setattr(sched, "directed_evolution", _boom)

    xp_ctx = sched.ScheduleContext(aa_seq=aa_seq, weights=weights, analysis_objects=analysis_objects, xp=np)
    seeded = sched.run_steps([], xp_ctx, [{"kind": "input", "count": 6}])
    sched.run_steps(seeded, xp_ctx, [{"kind": "growth", "rate": 4, "mutation_chance": 0.3}])

    assert calls, "growth step with ctx.xp set never used the batched growth path"


def test_step_directed_growth_never_calls_random_mutation(ctx, monkeypatch):
    """The whole point of "directed_growth" vs. "growth": 100% directed,
    0% random-mutation replicates. Confirm replicate_and_mutate_random is
    never even called, not just that its output happens not to show up."""
    def _boom(*args, **kwargs):
        raise AssertionError("directed_growth must never call replicate_and_mutate_random")

    monkeypatch.setattr(sched, "replicate_and_mutate_random", _boom)

    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 3}])
    result = sched.run_steps(seeded, ctx, [{"kind": "directed_growth", "rate": 5}])
    assert len(result) >= 1


def test_step_directed_growth_reproduces_and_dedups_within_the_step(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 1}])
    result = sched.run_steps(seeded, ctx, [{"kind": "directed_growth", "rate": 5}])
    codons_seen = [tuple(p.codons) for p in result]
    assert len(codons_seen) == len(set(codons_seen)), "directed_growth step produced duplicate genotypes as separate entries"


def test_step_directed_growth_diffs_new_individuals_against_their_parent(ctx, monkeypatch):
    """Same diffed-storage contract as "growth" (see
    test_step_growth_diffs_new_individuals_against_their_parent) -- the
    per-individual (xp=None) path still stores new genotypes via
    ga.merge_replicate()'s diff_change_vector() approximation."""
    import genewriter.ga as ga_module

    calls = []
    original = ga_module.diff_change_vector

    def _spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ga_module, "diff_change_vector", _spy)

    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 1}])
    sched.run_steps(seeded, ctx, [{"kind": "directed_growth", "rate": 8}])

    assert calls, "directed_growth step never called diff_change_vector for a new genotype"


def test_step_directed_growth_with_xp_and_lookahead_uses_batched_growth(aa_seq, analysis_objects, weights, monkeypatch):
    """Same xp-batching contract as "growth" (see
    test_step_growth_with_xp_and_lookahead_uses_batched_growth) -- must
    route through directed_evolution_batch(), not per-individual
    directed_evolution(), when ctx.xp is set and lookahead=True."""
    import numpy as np

    calls = []
    original = sched.directed_evolution_batch

    def _spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def _boom(*args, **kwargs):
        raise AssertionError("directed_evolution (per-individual) should not be called when ctx.xp is set and lookahead=True")

    monkeypatch.setattr(sched, "directed_evolution_batch", _spy)
    monkeypatch.setattr(sched, "directed_evolution", _boom)

    xp_ctx = sched.ScheduleContext(aa_seq=aa_seq, weights=weights, analysis_objects=analysis_objects, xp=np)
    seeded = sched.run_steps([], xp_ctx, [{"kind": "input", "count": 6}])
    sched.run_steps(seeded, xp_ctx, [{"kind": "directed_growth", "rate": 4}])

    assert calls, "directed_growth step with ctx.xp set never used the batched growth path"


def test_step_directed_growth_after_flatten_does_not_crash(ctx):
    """The strategic placement this step was actually designed for -- see
    its docstring and the schedule module docstring."""
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 6}])
    result = sched.run_steps(seeded, ctx, [
        {"kind": "flatten", "recursion_limit": 1},
        {"kind": "directed_growth", "rate": 3},
    ])
    assert len(result) >= 1


def test_step_save_writes_a_checkpoint(tmp_path, aa_seq, analysis_objects, weights):
    ctx_with_save = sched.ScheduleContext(
        aa_seq=aa_seq, weights=weights, analysis_objects=analysis_objects,
        save_dir=str(tmp_path), run_name="test_run",
    )
    seeded = sched.run_steps([], ctx_with_save, [{"kind": "input", "count": 3}])
    sched.run_steps(seeded, ctx_with_save, [{"kind": "save"}])

    run_dir = os.path.join(tmp_path, "test_run")
    assert os.path.isdir(run_dir)
    assert any(f.startswith("gen") for f in os.listdir(run_dir))


def test_step_repeat_runs_the_nested_schedule_the_requested_number_of_times(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 4}])
    result = sched.run_steps(seeded, ctx, [
        {"kind": "repeat", "times": 3, "steps": [
            {"kind": "growth", "rate": 2, "mutation_chance": 0.2},
            {"kind": "select", "target_size": 6},
        ]},
    ])
    assert len(result) <= 6


def test_unknown_step_kind_raises_a_clear_error(ctx):
    with pytest.raises(ValueError, match="nonexistent_kind"):
        sched.run_steps([], ctx, [{"kind": "nonexistent_kind"}])


def test_missing_kind_key_raises_a_clear_error(ctx):
    with pytest.raises(ValueError, match="kind"):
        sched.run_steps([], ctx, [{"rate": 4}])


def test_run_schedule_end_to_end_matches_the_spec_example_shape(aa_seq, analysis_objects, weights):
    """Directly exercises the shape from the request: input, growth at one
    rate, kill_off, growth at a different rate, flatten -- as one declarative,
    JSON-able schedule rather than hand-written Python calls."""
    schedule = [
        {"kind": "input", "count": 6},
        {"kind": "growth", "rate": 4, "mutation_chance": 0.1},
        {"kind": "kill_off", "percent_cut": 30},
        {"kind": "growth", "rate": 3, "mutation_chance": 0.1},
        {"kind": "flatten", "recursion_limit": 1},
        {"kind": "select", "target_size": 10},
    ]
    # Schedule must round-trip through JSON untouched -- that's the whole point.
    schedule = json.loads(json.dumps(schedule))

    result = sched.run_schedule(aa_seq, weights, analysis_objects, schedule)

    assert len(result) > 0
    assert len(result) <= 10
    for p in result:
        assert isinstance(p, Proposed_Solution)
        assert len(p.codons) == len(aa_seq)


def test_run_schedule_auto_computes_chunk_size_when_omitted_and_xp_is_set(monkeypatch, aa_seq, analysis_objects, weights):
    """chunk_size=None used to mean "one unchunked batch forever" -- now
    run_schedule() replaces it with gpu_change_vector.suggest_chunk_size()'s
    VRAM-aware default whenever xp is set. Spy on the real function (still
    calling through to it) to confirm it's actually invoked with this run's
    aa_seq length, rather than asserting on a specific numeric chunk_size
    (which depends on the test machine's free VRAM)."""
    import numpy as np

    real_suggest_chunk_size = sched.suggest_chunk_size

    def _spy_suggest_chunk_size(xp, aa_seq_len, **kwargs):
        _spy_suggest_chunk_size.seen = aa_seq_len
        return real_suggest_chunk_size(xp, aa_seq_len, **kwargs)

    monkeypatch.setattr(sched, "suggest_chunk_size", _spy_suggest_chunk_size)
    sched.run_schedule(aa_seq, weights, analysis_objects, [{"kind": "input", "count": 2}], xp=np)
    assert _spy_suggest_chunk_size.seen == len(aa_seq)


def test_run_schedule_with_chunk_size_matches_without(aa_seq, analysis_objects, weights):
    """chunk_size (see ScheduleContext.chunk_size) only bounds how many
    individuals any single xp-batched call processes at once -- it must not
    change the schedule's result given the same RNG seed."""
    import random

    import numpy as np

    schedule = [
        {"kind": "input", "count": 7},
        {"kind": "growth", "rate": 3, "mutation_chance": 0.1},
        {"kind": "kill_off", "percent_cut": 30},
        {"kind": "select", "target_size": 10},
        {"kind": "flatten", "recursion_limit": 1},
        {"kind": "directed_growth", "rate": 2},
    ]

    random.seed(4242)
    pop_unchunked = sched.run_schedule(aa_seq, weights, analysis_objects, schedule, xp=np)

    random.seed(4242)
    pop_chunked = sched.run_schedule(aa_seq, weights, analysis_objects, schedule, xp=np, chunk_size=3)

    assert {tuple(p.codons) for p in pop_unchunked} == {tuple(p.codons) for p in pop_chunked}
    by_codons_unchunked = {tuple(p.codons): p.number for p in pop_unchunked}
    by_codons_chunked = {tuple(p.codons): p.number for p in pop_chunked}
    assert by_codons_unchunked == by_codons_chunked


# ---------------------------------------------------------------------------
# RAM-aware expansion steps. available_ram_bytes() and the measured
# per-individual footprint are both pinned in each test so the arithmetic is
# exact; the real machine's free RAM never enters into it.
#
# Note which module each patch targets: schedule.py imports
# estimate_bytes_per_individual by name, so ctx.bytes_per_individual()
# resolves it through sched's namespace, while clamp_growth_rate_to_ram()
# reaches available_ram_bytes through ga's.
# ---------------------------------------------------------------------------


def _pin_ram(monkeypatch, total_bytes):
    monkeypatch.setattr(ga, "available_ram_bytes", lambda *a, **k: total_bytes)


def _pin_footprint(monkeypatch, bytes_per_individual=1000):
    monkeypatch.setattr(sched, "estimate_bytes_per_individual", lambda *a, **k: bytes_per_individual)
    return bytes_per_individual


def _budget_for(pop_size, rate, bytes_per_individual, ram_fraction=0.5):
    return int(pop_size * rate * bytes_per_individual / ram_fraction)


def _ram_warnings(recwarn):
    return [w for w in recwarn if issubclass(w.category, ga.PopulationRamWarning)]


def test_growth_step_reduces_its_rate_to_what_ram_allows(ctx, monkeypatch):
    """The user-facing shape of the whole feature: a schedule asking for
    rate 5 on a machine with room for 4.5 runs at 4 and says so. Asserted
    on the resulting population too, not just the warning -- 10 individuals
    at rate 4 cannot exceed 50 distinct, where rate 5 could reach 60."""
    bpi = _pin_footprint(monkeypatch)
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 10}])
    _pin_ram(monkeypatch, _budget_for(pop_size=len(seeded), rate=4.5, bytes_per_individual=bpi))

    with pytest.warns(ga.PopulationRamWarning, match=r"\[growth\].*Reducing rate 5 -> 4"):
        result = sched.run_steps(seeded, ctx, [{"kind": "growth", "rate": 5, "mutation_chance": 0.3}])

    assert len(result) <= len(seeded) * (1 + 4)


def test_directed_growth_step_reduces_its_rate_too(ctx, monkeypatch):
    """Both growth kinds route through _do_growth but pass their own
    step_label -- and this is the one likelier to hit the ceiling, since
    every individual reproduces rather than a directed_fraction subset."""
    bpi = _pin_footprint(monkeypatch)
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 10}])
    _pin_ram(monkeypatch, _budget_for(pop_size=len(seeded), rate=2.5, bytes_per_individual=bpi))

    with pytest.warns(ga.PopulationRamWarning, match=r"\[directed_growth\].*Reducing rate 8 -> 2"):
        sched.run_steps(seeded, ctx, [{"kind": "directed_growth", "rate": 8}])


def test_growth_step_is_a_no_op_when_ram_has_no_headroom(ctx, monkeypatch):
    """Degrade to doing nothing rather than to an OOM kill. The population
    comes back unchanged, and step_count still advances so a saved run's
    generation numbering does not silently shift under it."""
    bpi = _pin_footprint(monkeypatch)
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 10}])
    _pin_ram(monkeypatch, _budget_for(pop_size=len(seeded), rate=0.5, bytes_per_individual=bpi))
    before_steps = ctx.step_count

    with pytest.warns(ga.PopulationRamWarning, match="SKIPPED"):
        result = sched.run_steps(seeded, ctx, [{"kind": "growth", "rate": 5, "mutation_chance": 0.3}])

    assert {tuple(p.codons) for p in result} == {tuple(p.codons) for p in seeded}
    assert ctx.step_count == before_steps + 1


def test_growth_step_ram_fraction_none_opts_out(ctx, monkeypatch, recwarn):
    """Per-step escape hatch, and it must be distinguishable from "not
    specified": the key present with a None value disables the clamp even
    though ctx.ram_fraction is set."""
    _pin_footprint(monkeypatch)
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 10}])
    _pin_ram(monkeypatch, 1)

    sched.run_steps(seeded, ctx, [{"kind": "growth", "rate": 5, "mutation_chance": 0.3,
                                   "ram_fraction": None}])
    assert not _ram_warnings(recwarn)


def test_context_ram_fraction_none_opts_the_whole_run_out(analysis_objects, weights, aa_seq, monkeypatch, recwarn):
    _pin_footprint(monkeypatch)
    _pin_ram(monkeypatch, 1)
    loose_ctx = sched.ScheduleContext(aa_seq=aa_seq, weights=weights,
                                      analysis_objects=analysis_objects, ram_fraction=None)
    sched.run_steps([], loose_ctx, [{"kind": "input", "count": 5},
                                    {"kind": "growth", "rate": 3, "mutation_chance": 0.3}])
    assert not _ram_warnings(recwarn)


def test_step_ram_fraction_overrides_the_context_default(ctx, monkeypatch, recwarn):
    """A tighter budget on one step than on the run as a whole, since the
    peak is usually one specific step rather than the whole schedule."""
    bpi = _pin_footprint(monkeypatch)
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 10}])
    # Exactly comfortable at the ctx default of 0.5; far too tight at 0.05.
    _pin_ram(monkeypatch, _budget_for(pop_size=len(seeded), rate=5, bytes_per_individual=bpi))

    sched.run_steps(seeded, ctx, [{"kind": "growth", "rate": 5, "mutation_chance": 0.3}])
    assert not _ram_warnings(recwarn), "clamped at a budget that exactly fits"

    sched.run_steps(seeded, ctx, [{"kind": "growth", "rate": 5, "mutation_chance": 0.3,
                                   "ram_fraction": 0.05}])
    assert _ram_warnings(recwarn), "the step's own ram_fraction did not override ctx's"


def test_context_measures_the_per_individual_footprint_only_once(ctx, monkeypatch):
    """A schedule with several growth steps must not pay for a throwaway
    seed-and-score on each -- the number depends on aa_seq and the
    registered terms, neither of which moves during a run."""
    calls = []
    real = sched.estimate_bytes_per_individual
    monkeypatch.setattr(sched, "estimate_bytes_per_individual",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    sched.run_steps([], ctx, [{"kind": "input", "count": 5},
                              {"kind": "growth", "rate": 2, "mutation_chance": 0.3},
                              {"kind": "growth", "rate": 2, "mutation_chance": 0.3},
                              {"kind": "directed_growth", "rate": 2}])
    assert len(calls) == 1, f"measured {len(calls)} times across 3 growth steps"


def test_flatten_step_warns_but_still_runs_when_it_overruns_ram(ctx, monkeypatch):
    """flatten expands too, but has no rate to turn down -- its ceiling is
    the replicate mass already accumulated. So it reports and proceeds,
    rather than silently doing less than it was asked to."""
    bpi = _pin_footprint(monkeypatch)
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 5}])
    for p in seeded:
        p.number = 40
    _pin_ram(monkeypatch, 10 * bpi)  # room for ~5 of the up-to-200 new genotypes

    with pytest.warns(ga.PopulationRamWarning, match=r"\[flatten\]"):
        result = sched.run_steps(seeded, ctx, [{"kind": "flatten", "recursion_limit": 1}])

    assert len(result) >= len(seeded), "flatten was silently prevented from running"


def test_flatten_step_is_silent_when_the_projection_fits(ctx, monkeypatch, recwarn):
    bpi = _pin_footprint(monkeypatch)
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 5}])
    _pin_ram(monkeypatch, 10 ** 6 * bpi)

    sched.run_steps(seeded, ctx, [{"kind": "flatten", "recursion_limit": 1}])
    assert not _ram_warnings(recwarn)
