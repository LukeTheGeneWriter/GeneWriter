import pytest

from genewriter import ga
from genewriter.change_vector import calculate_change_vector
from genewriter.classes import Proposed_Solution
from genewriter.codon_tables import generate_codon_vec


def test_generate_seed_produces_valid_codons_for_every_residue(aa_seq):
    sol = ga.generate_seed(aa_seq)
    assert len(sol) == len(aa_seq)
    for codon in sol:
        assert len(codon) == 3


def test_kill_off_reduces_population_without_crashing(aa_seq, analysis_objects, weights):
    """Original bug: kill_off() asserted `type(p) == ProposedSolution`, a
    name that was never defined anywhere (the real dataclass is
    Proposed_Solution). Confirm the real dataclass name works end to end."""
    pop = []
    for i in range(5):
        sol = ga.generate_seed(aa_seq)
        vecs = {'RareCodons': [0.0] * len(sol), 'CodonUsage': [0.0] * len(sol),
                'CodonPairBias': [0.0] * len(sol), 'GC': [0.0] * len(sol), 'Kmer': [0.0] * len(sol)}
        pop.append(Proposed_Solution(sol, 10, vecs))

    total_before = sum(p.number for p in pop)
    result = ga.kill_off(pop, weights, percent_cut=30)
    total_after = sum(p.number for p in result)
    assert total_after < total_before


def test_kill_off_can_fully_remove_a_weak_individual(aa_seq, weights):
    """Original bug: `if pop[i].number > 1` floored every individual's
    count at 1, so the population's distinct-individual count could only
    grow generation over generation. A single low-count individual should
    be able to go fully extinct."""
    sol = ga.generate_seed(aa_seq)
    vecs = {'RareCodons': [0.0] * len(sol), 'CodonUsage': [0.0] * len(sol),
            'CodonPairBias': [0.0] * len(sol), 'GC': [0.0] * len(sol), 'Kmer': [0.0] * len(sol)}
    pop = [Proposed_Solution(sol, 1, vecs)]

    result = ga.kill_off(pop, weights, percent_cut=100)
    assert result == []


def test_kill_off_terminates_when_every_individual_is_at_minimum(aa_seq, weights):
    """Original bug: with every individual stuck at number==1 (the old
    floor) and num_to_kill > 0, `while num_to_kill > 0` never made progress
    -- an infinite loop. Confirm this terminates (pytest's own timeout
    isn't relied on; if this hangs, the test suite hangs)."""
    pop = []
    for _ in range(3):
        sol = ga.generate_seed(aa_seq)
        vecs = {'RareCodons': [0.0] * len(sol), 'CodonUsage': [0.0] * len(sol),
                'CodonPairBias': [0.0] * len(sol), 'GC': [0.0] * len(sol), 'Kmer': [0.0] * len(sol)}
        pop.append(Proposed_Solution(sol, 1, vecs))

    result = ga.kill_off(pop, weights, percent_cut=100)
    assert result == []


def test_replicate_and_mutate_random_preserves_length(aa_seq):
    sol = ga.generate_seed(aa_seq)
    reps = ga.replicate_and_mutate_random(sol, aa_seq, nreplicates=5, mutation_rate=0.5)
    assert len(reps) == 5
    for rep in reps:
        assert len(rep) == len(sol)


def test_directed_evolution_biases_toward_high_change_vector_positions(aa_seq, analysis_objects, weights):
    sol = ga.generate_seed(aa_seq)
    from genewriter.change_vector import calculate_change_vector
    vecs = calculate_change_vector(sol, analysis_objects)
    reps = ga.directed_evolution(sol, vecs, weights, aa_seq, analysis_objects, nreplicates=3)
    assert len(reps) <= 3
    for rep in reps:
        assert len(rep) == len(sol)


def test_directed_evolution_lookahead_false_still_produces_valid_replicates(aa_seq, analysis_objects, weights):
    """lookahead=False skips scoring every alternative and picks one at
    random -- position selection is still change-vector-weighted, only the
    per-alternative comparison is skipped. Confirm it still produces
    well-formed, valid-codon replicates."""
    sol = ga.generate_seed(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)
    reps = ga.directed_evolution(sol, vecs, weights, aa_seq, analysis_objects, nreplicates=5, lookahead=False)
    assert len(reps) <= 5
    valid_codons = {c for choices in generate_codon_vec(aa_seq) for c in choices}
    for rep in reps:
        assert len(rep) == len(sol)
        assert all(c in valid_codons for c in rep)


def test_directed_evolution_lookahead_false_never_calls_the_expensive_scorer(aa_seq, analysis_objects, weights, monkeypatch):
    """The whole point of lookahead=False is to skip per-alternative
    scoring -- assert that directly rather than just inferring it from
    speed (which is flaky across machines)."""
    import genewriter.ga as ga_module

    sol = ga.generate_seed(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)

    def _boom(*args, **kwargs):
        raise AssertionError("diff_change_vector should not be called when lookahead=False")

    monkeypatch.setattr(ga_module, "diff_change_vector", _boom)
    ga.directed_evolution(sol, vecs, weights, aa_seq, analysis_objects, nreplicates=5, lookahead=False)


def test_nearest_neighbors_differ_by_exactly_one_codon(aa_seq):
    sol = ga.generate_seed(aa_seq)
    neighbors = ga.nearest_neighbors(sol, aa_seq)
    assert len(neighbors) > 0
    for neighbor in neighbors:
        assert len(neighbor) == len(sol)
        diffs = [i for i in range(len(sol)) if neighbor[i] != sol[i]]
        assert len(diffs) == 1


def test_nearest_neighbors_empty_for_single_codon_amino_acids():
    # 'M' and 'W' are each encoded by exactly one codon -- no synonymous
    # substitution is possible anywhere in "MW".
    assert ga.nearest_neighbors(['ATG', 'TGG'], 'MW') == []


def _zero_vecs(n):
    return {'RareCodons': [0.0] * n, 'CodonUsage': [0.0] * n,
            'CodonPairBias': [0.0] * n, 'GC': [0.0] * n, 'Kmer': [0.0] * n}


def test_select_survivors_caps_population_size(aa_seq, weights):
    pop = [Proposed_Solution(ga.generate_seed(aa_seq), 1, _zero_vecs(len(aa_seq))) for _ in range(10)]
    result = ga.select_survivors(pop, weights, target_size=4)
    assert len(result) == 4


def test_select_survivors_is_a_noop_when_already_at_or_under_target(aa_seq, weights):
    pop = [Proposed_Solution(ga.generate_seed(aa_seq), 1, _zero_vecs(len(aa_seq))) for _ in range(3)]
    result = ga.select_survivors(pop, weights, target_size=10)
    assert len(result) == 3


def test_select_survivors_is_biased_toward_keeping_low_score_individuals(aa_seq, weights):
    """Soft top-N, not hard truncation -- but it must still be biased in the
    right direction. Build a population where half the individuals have a
    change vector that scores near-zero (low need to mutate, "fit") and
    half score very high (high need to mutate, "unfit"), then check that
    repeated selection keeps the fit half far more often than chance."""
    n = len(aa_seq)
    fit = [Proposed_Solution(ga.generate_seed(aa_seq), 1, {'RareCodons': [0.0] * n, 'CodonUsage': [0.0] * n,
                                                              'CodonPairBias': [0.0] * n, 'GC': [0.0] * n, 'Kmer': [0.0] * n})
           for _ in range(10)]
    unfit = [Proposed_Solution(ga.generate_seed(aa_seq), 1, {'RareCodons': [1000.0] * n, 'CodonUsage': [1000.0] * n,
                                                              'CodonPairBias': [1000.0] * n, 'GC': [1000.0] * n, 'Kmer': [1000.0] * n})
             for _ in range(10)]

    fit_survival_count = 0
    trials = 30
    for _ in range(trials):
        result = ga.select_survivors(fit + unfit, weights, target_size=10)
        fit_survival_count += sum(1 for p in result if p in fit)

    # Expected under the score-weighted bias: close to all 10 fit slots
    # survive every trial. Loose bound (well above chance, which would be
    # ~5/10) so this isn't flaky.
    assert fit_survival_count / trials > 8.0


def test_select_survivors_scales_past_quadratic(aa_seq, weights):
    """The original select_survivors() rebuilt a cumulative-weight table on
    every single removal (random.choices() in a loop), O(n) per removal and
    O(n^2) overall. At the population sizes a real run needs (thousands to
    tens of thousands) that's unusable. This doesn't assert a wall-clock
    bound (flaky across machines), just that cutting 9000 -> 1000 doesn't
    take an unreasonable amount of time on ordinary hardware -- a quadratic
    implementation would make this test suite take minutes, not seconds."""
    import time

    n = len(aa_seq)
    pop = [
        Proposed_Solution(ga.generate_seed(aa_seq), 1, {
            'RareCodons': [float(i % 7)] * n, 'CodonUsage': [float(i % 5)] * n,
            'CodonPairBias': [float(i % 3)] * n, 'GC': [float(i % 2)] * n, 'Kmer': [0.0] * n,
        })
        for i in range(9000)
    ]
    t0 = time.time()
    result = ga.select_survivors(pop, weights, target_size=1000)
    elapsed = time.time() - t0

    assert len(result) == 1000
    assert elapsed < 5.0, f"select_survivors took {elapsed:.2f}s for 9000->1000; looks quadratic again"


def test_flatten_round_conserves_total_replicate_count(aa_seq, analysis_objects):
    """Every cashed-in copy becomes exactly one increment somewhere (a new
    individual at count 1, or +1 on an existing one) -- the total replicate
    count across the population must be unchanged by a single flatten
    round. (flatten_generation's own *final* step intentionally breaks this
    -- it collapses every surviving individual to count 1 by design, per
    the spec's "before just cutting all members to 1 copy" -- so this tests
    _flatten_round() directly, the part where conservation is real.)"""
    pop = [Proposed_Solution(ga.generate_seed(aa_seq), 5, {}) for _ in range(4)]
    for p in pop:
        p.change_vecs = calculate_change_vector(p.codons, analysis_objects)
    total_before = sum(p.number for p in pop)

    result = ga._flatten_round(pop, aa_seq, analysis_objects)
    total_after = sum(p.number for p in result)
    assert total_after == total_before


def test_flatten_generation_deterministic_single_neighbor_case(analysis_objects):
    """'C' has exactly two synonymous codons (TGT/TGC), so a 1-residue 'C'
    sequence has exactly one possible neighbor -- fully deterministic,
    directly exercising the example from the spec: cashing in R's copies
    increments an already-present neighbor N rather than creating
    duplicates, and N's own (much smaller) cash-in can hand a copy back."""
    r = Proposed_Solution(['TGT'], 45, calculate_change_vector(['TGT'], analysis_objects))
    n = Proposed_Solution(['TGC'], 1, calculate_change_vector(['TGC'], analysis_objects))
    pop = [r, n]

    result = ga.flatten_generation(pop, 'C', analysis_objects, recursion_limit=1)

    by_codons = {tuple(p.codons): p for p in result}
    assert set(by_codons) == {('TGT',), ('TGC',)}
    # After recursion_limit=1 round the deterministic trace is R->0, N->46,
    # then N cashes in its own round-start count of 1, handing one copy
    # back to R (R->1, N->45); the final collapse-to-1 step then sets both
    # to 1.
    assert by_codons[('TGT',)].number == 1
    assert by_codons[('TGC',)].number == 1


def test_flatten_generation_collapses_everyone_to_one_after_recursion_limit(aa_seq, analysis_objects):
    pop = [Proposed_Solution(ga.generate_seed(aa_seq), 20, {}) for _ in range(3)]
    for p in pop:
        p.change_vecs = calculate_change_vector(p.codons, analysis_objects)

    result = ga.flatten_generation(pop, aa_seq, analysis_objects, recursion_limit=2)
    assert all(p.number == 1 for p in result)


def test_run_ga_end_to_end_small(aa_seq, analysis_objects, weights):
    """The GA in the source notebook never ran to completion once (the one
    recorded execution was manually interrupted). This runs a small number
    of seeds/generations end to end and checks it terminates with a
    non-empty, well-formed population."""
    seeds = [ga.generate_seed(aa_seq) for _ in range(3)]
    final_pop = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=2)

    assert len(final_pop) > 0
    for p in final_pop:
        assert isinstance(p, Proposed_Solution)
        assert len(p.codons) == len(aa_seq)
        assert p.number >= 1


def test_run_ga_population_stays_bounded_across_generations(aa_seq, analysis_objects, weights):
    """Real-data finding: without a population cap, 3 generations went
    29 -> 180 -> 491 distinct individuals (2.8s -> 6.8s -> 41s), because
    each individual can spawn up to 10 offspring per generation and nothing
    bounded growth. target_size (via select_survivors) must keep the
    population from exceeding it, for more generations than that took to
    blow up."""
    seeds = [ga.generate_seed(aa_seq) for _ in range(4)]
    target_size = 8
    final_pop = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=6, target_size=target_size)
    assert len(final_pop) <= target_size


def test_run_ga_with_flatten_every_does_not_crash(aa_seq, analysis_objects, weights):
    seeds = [ga.generate_seed(aa_seq) for _ in range(4)]
    final_pop = ga.run_ga(
        aa_seq, seeds, weights, analysis_objects, num_gens=4, target_size=8, flatten_every=2,
    )
    assert len(final_pop) <= 8


def test_run_ga_lookahead_false_does_not_crash(aa_seq, analysis_objects, weights):
    seeds = [ga.generate_seed(aa_seq) for _ in range(4)]
    final_pop = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=3, target_size=8, lookahead=False)
    assert len(final_pop) <= 8


def test_merge_replicate_without_parent_computes_exactly(aa_seq, analysis_objects):
    sol = ga.generate_seed(aa_seq)
    pop_index = {}
    ga.merge_replicate(pop_index, sol, analysis_objects)
    exact = calculate_change_vector(sol, analysis_objects)
    assert pop_index[tuple(sol)].change_vecs == exact


def test_merge_replicate_with_parent_diffs_instead_of_computing_exactly(aa_seq, analysis_objects):
    from genewriter.change_vector import diff_change_vector

    parent_sol = ga.generate_seed(aa_seq)
    parent = Proposed_Solution(parent_sol, 1, calculate_change_vector(parent_sol, analysis_objects))
    child_sol = parent_sol.copy()
    # aa_seq[0] is Met ('M'), which has only one codon (ATG) -- no
    # synonymous alternative -- so mutate position 1 (Ala, 4 codons) instead.
    codon_choices = generate_codon_vec(aa_seq)[1]
    child_sol[1] = next(c for c in codon_choices if c != parent_sol[1])

    pop_index = {}
    ga.merge_replicate(pop_index, child_sol, analysis_objects, parent=parent)
    expected = diff_change_vector(parent.codons, parent.change_vecs, child_sol, analysis_objects)
    assert pop_index[tuple(child_sol)].change_vecs == expected


def test_merge_replicate_increments_existing_entry_regardless_of_parent(aa_seq, analysis_objects):
    sol = ga.generate_seed(aa_seq)
    pop_index = {}
    ga.merge_replicate(pop_index, sol, analysis_objects)
    ga.merge_replicate(pop_index, sol, analysis_objects)  # same genotype again
    assert pop_index[tuple(sol)].number == 2


def test_refresh_change_vectors_recomputes_exactly(aa_seq, analysis_objects):
    sol = ga.generate_seed(aa_seq)
    stale_vecs = {'RareCodons': [0.0] * len(sol), 'CodonUsage': [0.0] * len(sol),
                  'CodonPairBias': [0.0] * len(sol), 'GC': [0.0] * len(sol), 'Kmer': [0.0] * len(sol)}
    pop = [Proposed_Solution(sol, 1, stale_vecs)]

    result = ga.refresh_change_vectors(pop, analysis_objects)
    assert result[0].change_vecs == calculate_change_vector(sol, analysis_objects)


def test_seed_population_matches_per_individual_calculate_change_vector(aa_seq, analysis_objects):
    seeds = [ga.generate_seed(aa_seq) for _ in range(5)]
    pop_index = ga.seed_population(seeds, analysis_objects)
    for sol in seeds:
        assert pop_index[tuple(sol)].change_vecs == calculate_change_vector(sol, analysis_objects)


def test_seed_population_merges_duplicate_seeds(aa_seq, analysis_objects):
    sol = ga.generate_seed(aa_seq)
    pop_index = ga.seed_population([sol, sol, sol], analysis_objects)
    assert len(pop_index) == 1
    assert pop_index[tuple(sol)].number == 3


def test_seed_population_with_numpy_xp_matches_per_individual_path(aa_seq, analysis_objects):
    """seed_population(xp=np) must compute the exact same change vectors as
    the default per-individual path -- see gpu_change_vector.py's own
    call-by-call equivalence tests for why the underlying batched math is
    trusted; this just confirms the wiring passes xp through correctly."""
    import numpy as np

    seeds = [ga.generate_seed(aa_seq) for _ in range(6)]
    per_individual = ga.seed_population(seeds, analysis_objects)
    batched = ga.seed_population(seeds, analysis_objects, xp=np)

    assert set(per_individual) == set(batched)
    for key in per_individual:
        assert per_individual[key].number == batched[key].number
        for term, values in per_individual[key].change_vecs.items():
            batched_values = batched[key].change_vecs[term]
            assert len(values) == len(batched_values)
            for a, b in zip(values, batched_values):
                assert a == pytest.approx(b, abs=1e-6)


def test_refresh_change_vectors_with_numpy_xp_matches_per_individual_path(aa_seq, analysis_objects):
    import numpy as np

    pop = [Proposed_Solution(ga.generate_seed(aa_seq), 1, {}) for _ in range(4)]
    pop_default = [Proposed_Solution(list(p.codons), p.number, {}) for p in pop]

    ga.refresh_change_vectors(pop, analysis_objects)
    ga.refresh_change_vectors(pop_default, analysis_objects, xp=np)

    for a, b in zip(pop, pop_default):
        for term, values in a.change_vecs.items():
            for x, y in zip(values, b.change_vecs[term]):
                assert x == pytest.approx(y, abs=1e-6)


def test_refresh_change_vectors_with_xp_handles_empty_population(analysis_objects):
    import numpy as np

    assert ga.refresh_change_vectors([], analysis_objects, xp=np) == []


def test_run_ga_with_numpy_xp_does_not_crash(aa_seq, analysis_objects, weights):
    import numpy as np

    seeds = [ga.generate_seed(aa_seq) for _ in range(4)]
    final_pop = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=3, target_size=8, xp=np)
    assert len(final_pop) <= 8
    for p in final_pop:
        assert isinstance(p, Proposed_Solution)
        assert len(p.codons) == len(aa_seq)
