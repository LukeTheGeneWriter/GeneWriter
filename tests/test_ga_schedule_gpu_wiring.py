"""Verifies the `xp` wiring added to ga.py/schedule.py -- seed_population(),
refresh_change_vectors(), run_ga(), and run_schedule()'s "input"/"kill_off"/
"select"/"flatten" steps -- actually runs on the GPU (xp=cupy) and produces
results equivalent to the CPU path, not just that passing xp doesn't crash.

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


def test_run_ga_gpu_matches_per_individual_final_population(aa_seq, analysis_objects, weights):
    """Growth/selection both consume random.random()/random.choices() the
    same number of times regardless of xp (only seeding and the periodic
    exact refresh route through the batched math), so with the module RNG
    reseeded identically before each call, xp=cupy and the default
    per-individual path should walk the exact same sequence of generations
    and land on the same final population."""
    import random

    seeds = [ga.generate_seed(aa_seq) for _ in range(4)]

    random.seed(1234)
    pop_default = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=4, target_size=8)

    random.seed(1234)
    pop_gpu = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=4, target_size=8, xp=cp)

    assert {tuple(p.codons) for p in pop_default} == {tuple(p.codons) for p in pop_gpu}
    by_codons_default = {tuple(p.codons): p for p in pop_default}
    by_codons_gpu = {tuple(p.codons): p for p in pop_gpu}
    for key in by_codons_default:
        assert by_codons_default[key].number == by_codons_gpu[key].number


def test_run_schedule_with_gpu_xp_matches_cpu(aa_seq, analysis_objects, weights):
    import random

    schedule = [
        {"kind": "input", "count": 10},
        {"kind": "growth", "rate": 3, "mutation_chance": 0.1},
        {"kind": "kill_off", "percent_cut": 30},
        {"kind": "select", "target_size": 10},
        {"kind": "flatten", "recursion_limit": 1},
    ]

    random.seed(5678)
    pop_default = sched.run_schedule(aa_seq, weights, analysis_objects, schedule)

    random.seed(5678)
    pop_gpu = sched.run_schedule(aa_seq, weights, analysis_objects, schedule, xp=cp)

    assert {tuple(p.codons) for p in pop_default} == {tuple(p.codons) for p in pop_gpu}


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
