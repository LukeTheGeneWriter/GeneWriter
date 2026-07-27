"""Verifies the `xp` wiring added to ga.py/schedule.py -- seed_population(),
refresh_change_vectors(), directed_evolution_batch(), run_ga(), and
run_schedule()'s "input"/"growth"/"kill_off"/"select"/"flatten" steps --
actually runs on the GPU (xp=cupy) and produces results equivalent to the
numpy-batched (xp=numpy) path, not just that passing xp doesn't crash.

Note this compares xp=numpy against xp=cupy, NOT against the default
xp=None per-individual path: growth's own behavior genuinely changes when
xp is set (see ga.directed_evolution_batch()) -- it scores each candidate
exactly via a batched full recompute instead of diff_change_vector()'s
local-excerpt approximation, so xp=None and xp=<anything> are expected to
diverge (different scores can pick a different "best" synonymous
alternative). What must still hold is that xp=numpy and xp=cupy agree with
each other, since they run the identical algorithm on different backends.

Skipped automatically wherever cupy or a GPU isn't available, same pattern as
test_gpu_change_vector_cuda.py (whose battle-tested numpy-vs-cupy equivalence
for the underlying batch_calculate_change_vectors() math is the correctness
oracle this file leans on -- it isn't re-verified here).
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    cp.cuda.runtime.getDeviceCount()
    _has_gpu = True
except Exception:
    _has_gpu = False

pytestmark = pytest.mark.skipif(not _has_gpu, reason="no CUDA GPU available")

from genewriter import ga, schedule as sched  # noqa: E402
from genewriter.change_vector import calculate_change_vector  # noqa: E402
from genewriter.classes import Proposed_Solution  # noqa: E402


def _assert_change_vecs_close(a: dict, b: dict, tol=1e-6):
    assert set(a) == set(b)
    for term, values in a.items():
        other = b[term]
        assert len(values) == len(other)
        for x, y in zip(values, other):
            assert x == pytest.approx(y, abs=tol)


def test_seed_population_gpu_matches_cpu(aa_seq, analysis_objects):
    seeds = [ga.generate_seed(aa_seq) for _ in range(8)]
    cpu = ga.seed_population(seeds, analysis_objects, xp=np)
    gpu = ga.seed_population(seeds, analysis_objects, xp=cp)

    assert set(cpu) == set(gpu)
    for key in cpu:
        assert cpu[key].number == gpu[key].number
        _assert_change_vecs_close(cpu[key].change_vecs, gpu[key].change_vecs)


def test_refresh_change_vectors_gpu_matches_cpu(aa_seq, analysis_objects):
    pop_cpu = [Proposed_Solution(ga.generate_seed(aa_seq), 1, {}) for _ in range(6)]
    pop_gpu = [Proposed_Solution(list(p.codons), p.number, {}) for p in pop_cpu]

    ga.refresh_change_vectors(pop_cpu, analysis_objects, xp=np)
    ga.refresh_change_vectors(pop_gpu, analysis_objects, xp=cp)

    for a, b in zip(pop_cpu, pop_gpu):
        assert a.codons == b.codons
        _assert_change_vecs_close(a.change_vecs, b.change_vecs)


def test_directed_evolution_batch_gpu_matches_numpy(aa_seq, analysis_objects, weights):
    """directed_evolution_batch() itself, numpy vs cupy, isolated from the
    rest of run_ga -- same random-draw sequence (position selection doesn't
    touch xp at all), same candidates, so the two backends must pick the
    identical best alternative for every individual."""
    import random

    individuals = []
    for _ in range(5):
        codons = ga.generate_seed(aa_seq)
        individuals.append(Proposed_Solution(codons, 1, calculate_change_vector(codons, analysis_objects)))

    random.seed(42)
    reps_np = ga.directed_evolution_batch(individuals, weights, aa_seq, analysis_objects, xp=np)

    random.seed(42)
    reps_gpu = ga.directed_evolution_batch(individuals, weights, aa_seq, analysis_objects, xp=cp)

    for ind in individuals:
        assert reps_np[id(ind)] == reps_gpu[id(ind)]


def test_run_ga_numpy_xp_matches_gpu_xp_final_population(aa_seq, analysis_objects, weights):
    """With xp set, growth's batched lookahead scoring (see
    ga.directed_evolution_batch()) is the identical algorithm on both
    backends -- only the array module differs -- so xp=numpy and xp=cupy
    must land on the same final population given the same RNG seed. This
    is deliberately NOT compared against the default xp=None path: growth's
    own scoring genuinely changes when xp is set (batched exact recompute
    vs. diff_change_vector's excerpt approximation), so xp=None is expected
    to diverge -- see this file's module docstring."""
    import random

    seeds = [ga.generate_seed(aa_seq) for _ in range(4)]

    random.seed(1234)
    pop_np = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=4, target_size=8, xp=np)

    random.seed(1234)
    pop_gpu = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=4, target_size=8, xp=cp)

    assert {tuple(p.codons) for p in pop_np} == {tuple(p.codons) for p in pop_gpu}
    by_codons_np = {tuple(p.codons): p for p in pop_np}
    by_codons_gpu = {tuple(p.codons): p for p in pop_gpu}
    for key in by_codons_np:
        assert by_codons_np[key].number == by_codons_gpu[key].number


def test_merge_replicates_batch_gpu_matches_cpu(aa_seq, analysis_objects):
    new_solutions = [ga.generate_seed(aa_seq) for _ in range(6)]
    # Duplicate one entry so the dedup-within-batch path is exercised too.
    new_solutions.append(new_solutions[0])

    pop_index_cpu = {}
    ga.merge_replicates_batch(pop_index_cpu, new_solutions, analysis_objects, None, xp=np)
    pop_index_gpu = {}
    ga.merge_replicates_batch(pop_index_gpu, new_solutions, analysis_objects, None, xp=cp)

    assert set(pop_index_cpu) == set(pop_index_gpu)
    for key in pop_index_cpu:
        assert pop_index_cpu[key].number == pop_index_gpu[key].number
        _assert_change_vecs_close(pop_index_cpu[key].change_vecs, pop_index_gpu[key].change_vecs)


def test_directed_evolution_batch_vecs_out_gpu_matches_cpu(aa_seq, analysis_objects, weights):
    import random

    individuals = []
    for _ in range(5):
        codons = ga.generate_seed(aa_seq)
        individuals.append(Proposed_Solution(codons, 1, calculate_change_vector(codons, analysis_objects)))

    random.seed(99)
    vecs_out_cpu = {}
    reps_cpu = ga.directed_evolution_batch(individuals, weights, aa_seq, analysis_objects, xp=np, vecs_out=vecs_out_cpu)

    random.seed(99)
    vecs_out_gpu = {}
    reps_gpu = ga.directed_evolution_batch(individuals, weights, aa_seq, analysis_objects, xp=cp, vecs_out=vecs_out_gpu)

    for ind in individuals:
        assert reps_cpu[id(ind)] == reps_gpu[id(ind)]
    assert set(vecs_out_cpu) == set(vecs_out_gpu)
    for key in vecs_out_cpu:
        _assert_change_vecs_close(vecs_out_cpu[key], vecs_out_gpu[key])


def test_flatten_round_gpu_matches_cpu(aa_seq, analysis_objects):
    pop_cpu = [Proposed_Solution(ga.generate_seed(aa_seq), 5, {}) for _ in range(4)]
    for p in pop_cpu:
        p.change_vecs = calculate_change_vector(p.codons, analysis_objects)
    pop_gpu = [Proposed_Solution(list(p.codons), p.number, dict(p.change_vecs)) for p in pop_cpu]

    import random
    random.seed(77)
    result_cpu = ga._flatten_round(pop_cpu, aa_seq, analysis_objects, xp=np)
    random.seed(77)
    result_gpu = ga._flatten_round(pop_gpu, aa_seq, analysis_objects, xp=cp)

    cpu_by_codons = {tuple(p.codons): p.number for p in result_cpu}
    gpu_by_codons = {tuple(p.codons): p.number for p in result_gpu}
    assert cpu_by_codons == gpu_by_codons


def test_run_schedule_numpy_xp_matches_gpu_xp(aa_seq, analysis_objects, weights):
    import random

    schedule = [
        {"kind": "input", "count": 10},
        {"kind": "growth", "rate": 3, "mutation_chance": 0.1},
        {"kind": "kill_off", "percent_cut": 30},
        {"kind": "select", "target_size": 10},
        {"kind": "flatten", "recursion_limit": 1},
        {"kind": "directed_growth", "rate": 3},  # strategic placement right after flatten -- see its docstring
    ]

    random.seed(5678)
    pop_np = sched.run_schedule(aa_seq, weights, analysis_objects, schedule, xp=np)

    random.seed(5678)
    pop_gpu = sched.run_schedule(aa_seq, weights, analysis_objects, schedule, xp=cp)

    assert {tuple(p.codons) for p in pop_np} == {tuple(p.codons) for p in pop_gpu}


def test_run_schedule_gpu_result_matches_exact_recompute(aa_seq, analysis_objects, weights):
    """End-to-end sanity check that the GPU path isn't silently returning
    stale/wrong scores: every individual surviving a GPU-routed schedule
    should have change_vecs exactly matching a fresh calculate_change_vector()
    call, same invariant test_step_select_refreshes_change_vectors_first
    checks for the CPU path in test_schedule.py."""
    schedule = [{"kind": "input", "count": 12}, {"kind": "select", "target_size": 12}]
    result = sched.run_schedule(aa_seq, weights, analysis_objects, schedule, xp=cp)
    for p in result:
        _assert_change_vecs_close(p.change_vecs, calculate_change_vector(p.codons, analysis_objects))
