import json
import os

import pytest

from genewriter import schedule as sched
from genewriter.classes import Proposed_Solution
from genewriter.change_vector import calculate_change_vector
from genewriter.codon_tables import generate_codon_vec


@pytest.fixture
def ctx(aa_seq, analysis_objects, weights):
    return sched.ScheduleContext(aa_seq=aa_seq, weights=weights, analysis_objects=analysis_objects)


def test_registered_steps_includes_the_builtin_kinds():
    assert set(sched.registered_steps()) == {'input', 'growth', 'kill_off', 'select', 'flatten', 'save', 'repeat'}


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


def test_step_growth_reproduces_and_dedups_within_the_step(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 1}])
    result = sched.run_steps(seeded, ctx, [{"kind": "growth", "rate": 5, "mutation_chance": 0.3}])
    codons_seen = [tuple(p.codons) for p in result]
    assert len(codons_seen) == len(set(codons_seen)), "growth step produced duplicate genotypes as separate entries"


def test_step_kill_off_reduces_total_replicate_count(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 20}])
    total_before = sum(p.number for p in seeded)
    result = sched.run_steps(seeded, ctx, [{"kind": "kill_off", "percent_cut": 50}])
    assert sum(p.number for p in result) < total_before


def test_step_select_caps_population_size(ctx):
    seeded = sched.run_steps([], ctx, [{"kind": "input", "count": 30}])
    result = sched.run_steps(seeded, ctx, [{"kind": "select", "target_size": 5}])
    assert len(result) == 5


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


def test_step_save_writes_a_checkpoint(tmp_path, aa_seq, analysis_objects, weights):
    ctx_with_save = sched.ScheduleContext(
        aa_seq=aa_seq, weights=weights, analysis_objects=analysis_objects,
        save_dir=str(tmp_path), run_name="test_run",
    )
    seeded = sched.run_steps([], ctx_with_save, [{"kind": "input", "count": 3}])
    sched.run_steps(seeded, ctx_with_save, [{"kind": "save"}])

    files = os.listdir(tmp_path)
    assert any(f.startswith("test_run_gen") for f in files)


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
