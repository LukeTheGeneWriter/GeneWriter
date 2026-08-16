import numpy as np
import pytest

from genewriter import ga
from genewriter.change_vector import calculate_change_vector, score_changevec
from genewriter.classes import Proposed_Solution
from genewriter.codon_tables import generate_codon_vec


def _assert_change_vecs_close(a: dict, b: dict, tol=1e-6):
    """Batched (numpy) and per-individual computation of the same term can
    differ by float rounding (different reduction order -- see
    tests/test_gpu_change_vector.py, which uses the same tolerance-based
    comparison rather than exact equality for this reason), so a plain ==
    is too strict here."""
    assert set(a) == set(b)
    for term, values in a.items():
        other = b[term]
        assert len(values) == len(other)
        for x, y in zip(values, other):
            assert x == pytest.approx(y, abs=tol)


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


def test_kill_off_outside_natural_range_drops_only_the_outlier_genotype(analysis_objects):
    """Hard cutoff, unlike kill_off(): a genotype whose own aggregate
    RareCodons rate is 9 std devs from the natural per-gene baseline mean
    (forced via the overridden usagePerGene below) must be dropped
    entirely, in full (not proportionally reduced like kill_off()),
    regardless of its replicate count -- while a genotype within range
    survives untouched."""
    # Real continuous Gaussian samples, not exact-alternating 2-point
    # arrays: the normal-scores transform (distribution_fit.py) picks
    # among several candidate distribution families by AIC, and a discrete
    # two-point mixture like [0.0, 0.2]*100 isn't well-described by any of
    # them -- AIC ends up choosing whichever shape fits that pathological
    # case "least badly" (e.g. lognorm), which can quietly change which
    # genotypes cross `threshold` here for reasons that have nothing to do
    # with what this test means to check. A real Gaussian sample at the
    # same (mean, std) keeps that intent intact: norm reliably wins, so
    # "N std from the baseline mean" means what the comments below say.
    rng = np.random.default_rng(0)
    analysis_objects.rare_codon.usagePerGene = rng.normal(0.1, 0.1, 5000).tolist()  # mean 0.1, std 0.1
    analysis_objects.gc.gcPerGene = rng.normal(0.5, 0.2, 5000).tolist()  # mean 0.5, std 0.2
    analysis_objects.codon_usage.codonFreqsLit = {c: 20.0 for c in analysis_objects.codon_usage.codonFreqsLit}
    analysis_objects.codon_usage.codonUsageScoreByGene = rng.normal(20.0, 2.0, 5000).tolist()
    analysis_objects.codon_pair_bias.cpb_lit = {p: 50000.0 for p in analysis_objects.codon_pair_bias.cpb_lit}
    analysis_objects.codon_pair_bias.cpbPerGene = rng.normal(50000.0, 5000.0, 5000).tolist()

    normal = ['CTT'] * 10  # Leu, not rare, GC fraction 1/3 -- within range on every axis
    outlier = ['GCG'] * 10  # Ala, every codon rare -- fraction 1.0 is ~9 raw std from the
                             # baseline mean, though distribution_fit.py caps how extreme a
                             # transform can report (_CDF_EPS) rather than reporting the raw
                             # value -- comfortably still above `threshold` either way.
    pop = [Proposed_Solution(normal, 3, {}), Proposed_Solution(outlier, 5, {})]

    survivors = ga.kill_off_outside_natural_range(pop, analysis_objects, threshold=3.0)

    assert len(survivors) == 1
    assert survivors[0].codons == normal
    assert survivors[0].number == 3  # untouched, not proportionally reduced


def test_kill_off_outside_natural_range_keeps_everyone_when_threshold_is_high(analysis_objects):
    analysis_objects.rare_codon.usagePerGene = [0.0, 0.2] * 100

    pop = [Proposed_Solution(['GCG'] * 10, 1, {}), Proposed_Solution(['CTT'] * 10, 1, {})]
    survivors = ga.kill_off_outside_natural_range(pop, analysis_objects, threshold=1e9)

    assert len(survivors) == 2


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


def _make_individual(aa_seq, analysis_objects):
    codons = ga.generate_seed(aa_seq)
    return Proposed_Solution(codons, 1, calculate_change_vector(codons, analysis_objects))


def test_directed_evolution_batch_produces_valid_replicates(aa_seq, analysis_objects, weights):
    individuals = [_make_individual(aa_seq, analysis_objects) for _ in range(4)]
    results = ga.directed_evolution_batch(individuals, weights, aa_seq, analysis_objects, nreplicates=5, xp=np)

    valid_codons = {c for choices in generate_codon_vec(aa_seq) for c in choices}
    for ind in individuals:
        reps = results[id(ind)]
        assert len(reps) <= 5
        for rep in reps:
            assert len(rep) == len(ind.codons)
            assert all(c in valid_codons for c in rep)
            diffs = [i for i in range(len(rep)) if rep[i] != ind.codons[i]]
            assert len(diffs) == 1, "expected exactly one synonymous substitution per replicate"


def test_directed_evolution_batch_picks_the_true_best_alternative(aa_seq, analysis_objects, weights):
    """Unlike directed_evolution()'s diff_change_vector-based local-excerpt
    approximation, the batched path scores each candidate with a full exact
    recompute -- verify each returned replicate's substitution really is
    the lowest-scoring synonymous alternative at that position, checked
    independently via calculate_change_vector() rather than trusting the
    same code path under test."""
    ind = _make_individual(aa_seq, analysis_objects)
    codons = ind.codons

    results = ga.directed_evolution_batch([ind], weights, aa_seq, analysis_objects, nreplicates=8, xp=np)
    reps = results[id(ind)]
    assert reps, "expected at least one replicate for this test to be meaningful"

    codon_vec = generate_codon_vec(aa_seq)
    for rep in reps:
        diffs = [i for i in range(len(rep)) if rep[i] != codons[i]]
        assert len(diffs) == 1
        position = diffs[0]
        chosen_codon = rep[position]

        alternatives = [c for c in codon_vec[position] if c != codons[position]]
        scores = {}
        for alt in alternatives:
            candidate = codons.copy()
            candidate[position] = alt
            scores[alt] = score_changevec(calculate_change_vector(candidate, analysis_objects), weights)
        assert scores[chosen_codon] == min(scores.values())


def test_directed_evolution_batch_memoizes_repeated_position_draws(aa_seq, analysis_objects, weights):
    """Two replicate slots that land on the same position must pick the
    same alternative -- the whole point of memoizing by (individual,
    position) instead of rescoring per replicate slot."""
    codons = ga.generate_seed(aa_seq)
    n = len(codons)
    # aa_seq[0] is Met (single codon, never drawable as a *useful* mutation
    # target); make position 1 overwhelmingly dominate the weighted draw so
    # repeated replicate slots are virtually guaranteed to land on it.
    vecs = {'RareCodons': [0.0] * n, 'CodonUsage': [0.0] * n,
            'CodonPairBias': [0.0] * n, 'GC': [0.0] * n, 'Kmer': [0.0] * n}
    vecs['RareCodons'][1] = 1e6
    ind = Proposed_Solution(codons, 1, vecs)

    results = ga.directed_evolution_batch([ind], weights, aa_seq, analysis_objects, nreplicates=10, xp=np)
    reps = results[id(ind)]
    position_1_reps = [rep for rep in reps if rep[1] != codons[1]]
    assert len(position_1_reps) >= 2, "expected the dominant position to be drawn more than once"
    assert len({rep[1] for rep in position_1_reps}) == 1


def test_directed_evolution_batch_skips_positions_without_synonymous_alternatives(analysis_objects, weights):
    # 'M' and 'W' are each encoded by exactly one codon -- no synonymous
    # substitution is possible anywhere, same case as
    # test_nearest_neighbors_empty_for_single_codon_amino_acids.
    codons = ['ATG', 'TGG']
    ind = Proposed_Solution(codons, 1, calculate_change_vector(codons, analysis_objects))

    results = ga.directed_evolution_batch([ind], weights, 'MW', analysis_objects, nreplicates=5, xp=np)
    assert results[id(ind)] == []


def test_run_ga_with_xp_and_lookahead_uses_batched_growth(aa_seq, analysis_objects, weights, monkeypatch):
    """xp set + lookahead=True (the default) must route growth through
    directed_evolution_batch(), not directed_evolution() -- the whole point
    of threading xp into run_ga's growth loop (see Handoff.md sec 6)."""
    import genewriter.ga as ga_module

    calls = []
    original = ga_module.directed_evolution_batch

    def _spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def _boom(*args, **kwargs):
        raise AssertionError("directed_evolution (per-individual) should not be called when xp is set and lookahead=True")

    monkeypatch.setattr(ga_module, "directed_evolution_batch", _spy)
    monkeypatch.setattr(ga_module, "directed_evolution", _boom)

    seeds = [ga.generate_seed(aa_seq) for _ in range(4)]
    ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=2, target_size=8, xp=np)
    assert calls, "run_ga with xp set and lookahead=True never used the batched growth path"


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


def test_merge_replicate_exact_stores_the_given_vecs_without_recomputing(aa_seq, analysis_objects, monkeypatch):
    """merge_replicate_exact() must never call calculate_change_vector() or
    diff_change_vector() -- the whole point is that its caller already has
    the exact vecs in hand (see directed_evolution_batch(vecs_out=...))."""
    import genewriter.ga as ga_module

    def _boom(*args, **kwargs):
        raise AssertionError("merge_replicate_exact should not compute anything")

    monkeypatch.setattr(ga_module, "calculate_change_vector", _boom)
    monkeypatch.setattr(ga_module, "diff_change_vector", _boom)

    sol = ga.generate_seed(aa_seq)
    vecs = {'RareCodons': [0.0] * len(sol), 'CodonUsage': [0.0] * len(sol),
            'CodonPairBias': [0.0] * len(sol), 'GC': [0.0] * len(sol), 'Kmer': [0.0] * len(sol)}
    pop_index = {}
    ga.merge_replicate_exact(pop_index, sol, vecs)
    assert pop_index[tuple(sol)].change_vecs == vecs
    assert pop_index[tuple(sol)].number == 1


def test_merge_replicate_exact_increments_existing_entry(aa_seq, analysis_objects):
    sol = ga.generate_seed(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)
    pop_index = {}
    ga.merge_replicate_exact(pop_index, sol, vecs)
    ga.merge_replicate_exact(pop_index, sol, vecs)  # same genotype again
    assert pop_index[tuple(sol)].number == 2
    assert pop_index[tuple(sol)].change_vecs == vecs


def test_merge_replicates_batch_matches_calculate_change_vector(aa_seq, analysis_objects):
    """Every genuinely-new genotype's batched vecs must exactly match a
    direct calculate_change_vector() call -- same correctness oracle
    tests/test_gpu_change_vector.py uses for batch_calculate_change_vectors
    itself."""
    new_solutions = [ga.generate_seed(aa_seq) for _ in range(6)]

    pop_index = {}
    ga.merge_replicates_batch(pop_index, new_solutions, analysis_objects, None, xp=np)

    assert len(pop_index) <= len(new_solutions)
    for sol in new_solutions:
        key = tuple(sol)
        assert key in pop_index
        expected = calculate_change_vector(sol, analysis_objects)
        _assert_change_vecs_close(pop_index[key].change_vecs, expected)


def test_merge_replicates_batch_dedups_within_batch_and_against_existing_pop(aa_seq, analysis_objects):
    sol = ga.generate_seed(aa_seq)
    other = ga.generate_seed(aa_seq)
    while other == sol:
        other = ga.generate_seed(aa_seq)

    # sol already present in pop_index (count 3) before the batch call.
    pop_index = {}
    ga.merge_replicate(pop_index, sol, analysis_objects)
    ga.merge_replicate(pop_index, sol, analysis_objects)
    ga.merge_replicate(pop_index, sol, analysis_objects)
    assert pop_index[tuple(sol)].number == 3

    # sol drawn twice more, other drawn once, all in the same batch call.
    new_solutions = [sol, sol, other]
    ga.merge_replicates_batch(pop_index, new_solutions, analysis_objects, None, xp=np)

    assert pop_index[tuple(sol)].number == 5  # 3 existing + 2 new draws
    assert pop_index[tuple(other)].number == 1


def test_merge_replicates_batch_noop_for_empty_input(aa_seq, analysis_objects):
    pop_index = {}
    ga.merge_replicates_batch(pop_index, [], analysis_objects, None, xp=np)
    assert pop_index == {}


def test_directed_evolution_batch_vecs_out_matches_calculate_change_vector(aa_seq, analysis_objects, weights):
    """vecs_out should be populated with each returned replicate's exact
    change vector -- the same value a direct calculate_change_vector() call
    on that replicate's codons would give, not an approximation."""
    individuals = [_make_individual(aa_seq, analysis_objects) for _ in range(4)]
    vecs_out = {}
    results = ga.directed_evolution_batch(
        individuals, weights, aa_seq, analysis_objects, nreplicates=5, xp=np, vecs_out=vecs_out,
    )

    any_reps = False
    for ind in individuals:
        for rep in results[id(ind)]:
            any_reps = True
            assert tuple(rep) in vecs_out
            expected = calculate_change_vector(rep, analysis_objects)
            _assert_change_vecs_close(vecs_out[tuple(rep)], expected)
    assert any_reps, "expected at least one replicate for this test to be meaningful"


def test_directed_evolution_batch_without_vecs_out_is_unchanged(aa_seq, analysis_objects, weights):
    """Omitting vecs_out (every pre-existing call site) must behave exactly
    as before this parameter was added."""
    individuals = [_make_individual(aa_seq, analysis_objects) for _ in range(4)]
    results = ga.directed_evolution_batch(individuals, weights, aa_seq, analysis_objects, nreplicates=5, xp=np)
    for ind in individuals:
        for rep in results[id(ind)]:
            assert len(rep) == len(ind.codons)


def test_run_ga_growth_batching_matches_reference_merge_path(aa_seq, analysis_objects, weights):
    """Old-path-vs-new-path equivalence check: with xp set, growth's storage
    now goes through merge_replicate_exact()/merge_replicates_batch()
    instead of one merge_replicate() (diff_change_vector) call per
    replicate. Both must produce the identical final population (same
    genotypes, same counts) given the same RNG seed -- the dedup-by-genotype
    semantics must be preserved exactly across the refactor, same
    correctness bar as test_ga_schedule_gpu_wiring.py's numpy-vs-cupy
    checks."""
    import random

    from genewriter.ga import directed_evolution_batch, replicate_and_mutate_random

    def _reference_run_ga_one_generation(seeds):
        """Old growth storage: one merge_replicate() call per replicate,
        kept verbatim here (not imported) so a future rewrite of run_ga's
        internals can't silently drift this reference along with it."""
        pop_index = ga.seed_population(seeds, analysis_objects, xp=np)
        pop = list(pop_index.values())
        pop_index = {tuple(p.codons): p for p in pop}

        is_random = [random.random() < 0.5 for _ in pop]
        random_individuals = [p for p, r in zip(pop, is_random) if r]
        directed_individuals = [p for p, r in zip(pop, is_random) if not r]

        for p in random_individuals:
            for rep in replicate_and_mutate_random(p.codons, aa_seq):
                ga.merge_replicate(pop_index, rep, analysis_objects, None, parent=p)

        batch_reps = directed_evolution_batch(directed_individuals, weights, aa_seq, analysis_objects, xp=np)
        for p in directed_individuals:
            for rep in batch_reps[id(p)]:
                ga.merge_replicate(pop_index, rep, analysis_objects, None, parent=p)

        return list(pop_index.values())

    seeds = [ga.generate_seed(aa_seq) for _ in range(6)]

    random.seed(4242)
    reference_pop = _reference_run_ga_one_generation(seeds)

    random.seed(4242)
    new_pop = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=1, target_size=len(reference_pop) + 100, xp=np)

    ref_by_codons = {tuple(p.codons): p.number for p in reference_pop}
    new_by_codons = {tuple(p.codons): p.number for p in new_pop}
    assert ref_by_codons == new_by_codons


def test_flatten_round_with_xp_matches_calculate_change_vector(aa_seq, analysis_objects):
    """New-neighbor vecs computed via the batched xp path must exactly
    match calculate_change_vector() -- same oracle as the growth-batching
    tests above."""
    pop = [Proposed_Solution(ga.generate_seed(aa_seq), 5, {}) for _ in range(4)]
    for p in pop:
        p.change_vecs = calculate_change_vector(p.codons, analysis_objects)

    result = ga._flatten_round(pop, aa_seq, analysis_objects, xp=np)
    for p in result:
        expected = calculate_change_vector(p.codons, analysis_objects)
        _assert_change_vecs_close(p.change_vecs, expected)


def test_flatten_round_with_xp_conserves_total_replicate_count(aa_seq, analysis_objects):
    pop = [Proposed_Solution(ga.generate_seed(aa_seq), 5, {}) for _ in range(4)]
    for p in pop:
        p.change_vecs = calculate_change_vector(p.codons, analysis_objects)
    total_before = sum(p.number for p in pop)

    result = ga._flatten_round(pop, aa_seq, analysis_objects, xp=np)
    total_after = sum(p.number for p in result)
    assert total_after == total_before


def test_run_ga_with_flatten_every_and_xp_does_not_crash(aa_seq, analysis_objects, weights):
    seeds = [ga.generate_seed(aa_seq) for _ in range(4)]
    final_pop = ga.run_ga(
        aa_seq, seeds, weights, analysis_objects, num_gens=4, target_size=8, flatten_every=2, xp=np,
    )
    assert len(final_pop) <= 8


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


def test_seed_population_chunk_size_matches_unchunked(aa_seq, analysis_objects):
    """chunk_size is purely a memory-bounding implementation detail of how
    the underlying batch_calculate_change_vectors() call is split (see its
    docstring) -- seed_population(chunk_size=...) must produce an identical
    pop_index to chunk_size=None, not just a compatible one, for a
    chunk_size that doesn't evenly divide the number of unique seeds."""
    seeds = [ga.generate_seed(aa_seq) for _ in range(7)]
    unchunked = ga.seed_population(seeds, analysis_objects, xp=np)
    chunked = ga.seed_population(seeds, analysis_objects, xp=np, chunk_size=3)

    assert set(unchunked) == set(chunked)
    for key in unchunked:
        assert unchunked[key].number == chunked[key].number
        assert unchunked[key].change_vecs == chunked[key].change_vecs


def test_refresh_change_vectors_chunk_size_matches_unchunked(aa_seq, analysis_objects):
    pop = [Proposed_Solution(ga.generate_seed(aa_seq), 1, {}) for _ in range(7)]
    pop_chunked = [Proposed_Solution(list(p.codons), p.number, {}) for p in pop]

    ga.refresh_change_vectors(pop, analysis_objects, xp=np)
    ga.refresh_change_vectors(pop_chunked, analysis_objects, xp=np, chunk_size=3)

    for a, b in zip(pop, pop_chunked):
        assert a.codons == b.codons
        assert a.change_vecs == b.change_vecs


def test_run_ga_with_chunk_size_matches_without(aa_seq, analysis_objects, weights):
    """chunk_size must not change run_ga's result given the same RNG seed --
    it only bounds how many individuals any single xp-batched call processes
    at once, not any random draw or selection logic."""
    import random

    seeds = [ga.generate_seed(aa_seq) for _ in range(4)]

    random.seed(2026)
    pop_unchunked = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=3, target_size=8, xp=np)

    random.seed(2026)
    pop_chunked = ga.run_ga(aa_seq, seeds, weights, analysis_objects, num_gens=3, target_size=8, xp=np, chunk_size=3)

    assert {tuple(p.codons) for p in pop_unchunked} == {tuple(p.codons) for p in pop_chunked}
    by_codons_unchunked = {tuple(p.codons): p for p in pop_unchunked}
    by_codons_chunked = {tuple(p.codons): p for p in pop_chunked}
    for key in by_codons_unchunked:
        assert by_codons_unchunked[key].number == by_codons_chunked[key].number
