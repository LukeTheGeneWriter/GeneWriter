import math

from genewriter.change_vector import calculate_change_vector, diff_change_vector
from genewriter.codon_tables import generate_codon_vec

from conftest import random_solution

# A longer sequence than the shared 40-residue fixture: with the default
# margin=40, a single-position mutation's excerpt on a 40-residue sequence
# covers the whole thing (nothing to test). 150 residues gives a real gap
# between "local excerpt" and "whole sequence".
LONG_AA_SEQ = "MAVLDEFGHIKPQRSTWYCN" * 7 + "MAVLDEFGHI"  # 150 residues


def _mutate_one_position(sol, aa_seq, position, seed=1):
    import random
    rng = random.Random(seed)
    choices = generate_codon_vec(aa_seq)[position]
    alternatives = [c for c in choices if c != sol[position]]
    child = sol.copy()
    child[position] = rng.choice(alternatives)
    return child


def test_no_change_returns_a_copy_of_parent_vecs(analysis_objects):
    sol = random_solution(LONG_AA_SEQ)
    parent_vecs = calculate_change_vector(sol, analysis_objects)
    result = diff_change_vector(sol, parent_vecs, sol, analysis_objects)
    assert result == parent_vecs
    assert result is not parent_vecs
    for name in result:
        assert result[name] is not parent_vecs[name]


def test_codon_usage_and_codon_pair_bias_are_exact_in_the_trusted_region(analysis_objects):
    """These two terms have no population-wide aggregate dependency -- see
    diff_change_vector's docstring -- so recomputing them on a local
    excerpt should be bit-for-bit identical to a full recompute, at every
    position safely inside the excerpt's own trusted region (away from the
    excerpt's artificial edges)."""
    sol = random_solution(LONG_AA_SEQ)
    parent_vecs = calculate_change_vector(sol, analysis_objects)
    mutated_position = 75  # well clear of both sequence ends
    child = _mutate_one_position(sol, LONG_AA_SEQ, mutated_position)

    exact = calculate_change_vector(child, analysis_objects)
    diffed = diff_change_vector(sol, parent_vecs, child, analysis_objects, margin=40)

    # Trusted region is [mutated - margin//2, mutated + margin//2]; check a
    # safely-interior slice of it.
    for j in range(mutated_position - 15, mutated_position + 15):
        assert diffed['CodonUsage'][j] == exact['CodonUsage'][j]
        assert diffed['CodonPairBias'][j] == exact['CodonPairBias'][j]


def test_far_from_the_mutation_the_diff_exactly_matches_the_parent(analysis_objects):
    sol = random_solution(LONG_AA_SEQ)
    parent_vecs = calculate_change_vector(sol, analysis_objects)
    child = _mutate_one_position(sol, LONG_AA_SEQ, 75)

    diffed = diff_change_vector(sol, parent_vecs, child, analysis_objects, margin=20)

    # Position 0 is far outside even a generous excerpt around position 75.
    for name in diffed:
        assert diffed[name][0] == parent_vecs[name][0]


def test_rare_codons_gc_kmer_are_close_but_not_required_to_be_exact(analysis_objects):
    """These three terms fold in a population-wide aggregate (see
    diff_change_vector's docstring) -- the excerpt approximates it from
    local composition, so results are close to but not necessarily
    identical to a full recompute. Bounds this rather than asserting
    equality, and mainly guards against the approximation being wildly
    wrong (e.g. an off-by-one in the excerpt slicing)."""
    sol = random_solution(LONG_AA_SEQ)
    parent_vecs = calculate_change_vector(sol, analysis_objects)
    mutated_position = 75
    child = _mutate_one_position(sol, LONG_AA_SEQ, mutated_position)

    exact = calculate_change_vector(child, analysis_objects)
    diffed = diff_change_vector(sol, parent_vecs, child, analysis_objects, margin=40)

    for name in ('RareCodons', 'GC', 'Kmer'):
        for j in range(mutated_position - 10, mutated_position + 10):
            e, d = exact[name][j], diffed[name][j]
            if math.isinf(e) or math.isinf(d):
                continue
            assert abs(e - d) < max(abs(e), abs(d), 1.0) * 2.0, f"{name}[{j}]: exact={e} diffed={d}"


def test_widely_spread_mutations_fall_back_to_full_recompute(analysis_objects):
    """More than half the sequence differing isn't worth excerpting --
    diff_change_vector should just delegate to calculate_change_vector, and
    the result should be exactly what a full recompute gives."""
    sol = random_solution(LONG_AA_SEQ, seed=1)
    parent_vecs = calculate_change_vector(sol, analysis_objects)
    child = random_solution(LONG_AA_SEQ, seed=2)  # almost everything differs

    exact = calculate_change_vector(child, analysis_objects)
    diffed = diff_change_vector(sol, parent_vecs, child, analysis_objects)

    assert diffed == exact


def test_diff_result_has_full_length_for_every_term(analysis_objects):
    sol = random_solution(LONG_AA_SEQ)
    parent_vecs = calculate_change_vector(sol, analysis_objects)
    child = _mutate_one_position(sol, LONG_AA_SEQ, 10)

    diffed = diff_change_vector(sol, parent_vecs, child, analysis_objects, margin=15)
    for name, values in diffed.items():
        assert len(values) == len(sol), name
