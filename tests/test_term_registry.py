import pytest

from genewriter import change_vector
from genewriter.change_vector import calculate_change_vector, register_term, registered_terms, score_changevec

from conftest import random_solution


@pytest.fixture
def temp_term():
    """Register a throwaway term for one test, then remove it -- the
    registry is module-global, so a leaked test term would silently show up
    in every other test's calculate_change_vector() output afterward."""
    registered_names = []

    def _register(name, fn):
        register_term(name)(fn)
        registered_names.append(name)
        return fn

    yield _register

    for name in registered_names:
        change_vector._TERM_REGISTRY.pop(name, None)


def test_registered_terms_includes_the_six_builtins():
    names = set(registered_terms())
    assert names == {'RareCodons', 'CodonUsage', 'CodonPairBias', 'GC', 'Kmer', 'Uracil'}


def test_custom_term_is_picked_up_by_calculate_change_vector(aa_seq, analysis_objects, temp_term):
    temp_term('EndsInC', lambda sol, ao, locvec: [1.0 if c[2] == 'C' else 0.0 for c in sol])

    sol = random_solution(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)

    assert 'EndsInC' in vecs
    assert vecs['EndsInC'] == [1.0 if c[2] == 'C' else 0.0 for c in sol]


def test_custom_term_receives_the_full_analysis_objects_bundle(aa_seq, analysis_objects, temp_term):
    """A custom term should be free to pull whatever sub-analysis it needs,
    not just a fixed one -- confirm the full bundle (not some subset) is
    what's actually passed through."""
    seen = {}

    def spy_term(sol, ao, locvec):
        seen['analysis_objects'] = ao
        seen['locvec'] = locvec
        return [0.0] * len(sol)

    temp_term('Spy', spy_term)
    sol = random_solution(aa_seq)
    calculate_change_vector(sol, analysis_objects, locvec=['I'] * len(sol))

    assert seen['analysis_objects'] is analysis_objects
    assert seen['locvec'] == ['I'] * len(sol)


def test_registering_a_duplicate_name_raises():
    with pytest.raises(ValueError):
        register_term('RareCodons')(lambda sol, ao, locvec: [0.0] * len(sol))


def test_score_changevec_raises_a_clear_error_for_a_missing_weight(aa_seq, analysis_objects, temp_term):
    temp_term('Unweighted', lambda sol, ao, locvec: [1.0] * len(sol))
    sol = random_solution(aa_seq)
    vecs = calculate_change_vector(sol, analysis_objects)
    weights_missing_one = {'RareCodons': 1.0, 'CodonUsage': 1.0, 'CodonPairBias': 1.0, 'GC': 1.0, 'Kmer': 1.0}

    with pytest.raises(ValueError, match='Unweighted'):
        score_changevec(vecs, weights_missing_one)
