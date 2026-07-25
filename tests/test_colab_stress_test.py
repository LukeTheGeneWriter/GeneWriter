"""Tests for scripts/colab_stress_test.py -- the first test coverage of a
script file in this repo. Loaded via importlib.util.spec_from_file_location
rather than adding scripts/ to pythonpath (pyproject.toml only puts `src`
there for pytest -- adding scripts/ risks shadowing genewriter imports or
picking up an unrelated __pycache__)."""

import importlib.util
import os

import pytest

from conftest import make_synthetic_gene, make_synthetic_isoform

_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "colab_stress_test.py")


@pytest.fixture(scope="module")
def colab_script():
    spec = importlib.util.spec_from_file_location("colab_stress_test", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_cfg(colab_script, **overrides):
    cfg = dict(colab_script.CONFIG)
    cfg.update(overrides)
    return cfg


def test_pick_target_custom_mode_bypasses_gene_lookup(colab_script):
    cfg = _base_cfg(colab_script, TARGET_MODE="custom", CUSTOM_AA_SEQ="mklker")
    aa_seq, locvec = colab_script.pick_target(cfg, genes=[])  # no genes loaded at all
    assert aa_seq == "MKLKER"
    assert locvec == ["I"] * 6


def test_pick_target_custom_mode_rejects_invalid_aa_seq(colab_script):
    cfg = _base_cfg(colab_script, TARGET_MODE="custom", CUSTOM_AA_SEQ="MKLXER")
    with pytest.raises(ValueError):
        colab_script.pick_target(cfg, genes=[])


def test_pick_target_custom_mode_rejects_star(colab_script):
    cfg = _base_cfg(colab_script, TARGET_MODE="custom", CUSTOM_AA_SEQ="MKLKER*")
    with pytest.raises(ValueError, match=r"\*"):
        colab_script.pick_target(cfg, genes=[])


def test_pick_target_custom_mode_requires_custom_aa_seq(colab_script):
    cfg = _base_cfg(colab_script, TARGET_MODE="custom", CUSTOM_AA_SEQ=None)
    with pytest.raises(ValueError, match="CUSTOM_AA_SEQ"):
        colab_script.pick_target(cfg, genes=[])


def test_pick_target_custom_mode_validates_locvec_length(colab_script):
    cfg = _base_cfg(colab_script, TARGET_MODE="custom", CUSTOM_AA_SEQ="MKLKER", CUSTOM_LOCVEC=["I", "I"])
    with pytest.raises(ValueError, match="CUSTOM_LOCVEC"):
        colab_script.pick_target(cfg, genes=[])


def test_pick_target_custom_mode_uses_custom_locvec_when_given(colab_script):
    custom_locvec = ["F", "I", "I", "I", "I", "T"]
    cfg = _base_cfg(colab_script, TARGET_MODE="custom", CUSTOM_AA_SEQ="MKLKER", CUSTOM_LOCVEC=custom_locvec)
    aa_seq, locvec = colab_script.pick_target(cfg, genes=[])
    assert locvec == custom_locvec


def test_pick_target_unknown_target_mode_raises(colab_script):
    cfg = _base_cfg(colab_script, TARGET_MODE="bogus")
    with pytest.raises(ValueError, match="TARGET_MODE"):
        colab_script.pick_target(cfg, genes=[])


def test_pick_target_gene_mode_unchanged(colab_script):
    # Regression guard: TARGET_MODE="gene" (the default) still exercises
    # the existing gene-lookup path, using a synthetic gene (never
    # Gene_Obj_Samples/, which still has the codons<->aaSeq alignment bug).
    from genewriter.codon_tables import codon_choices_for_aa

    aa_seq = "MKLKER"
    loc_tags = ['F', 'I', 'I', 'I', 'I', 'T']
    iso = make_synthetic_isoform(aa_seq, lambda aa, i: codon_choices_for_aa(aa)[0], loc_tags)
    genes = [make_synthetic_gene(42, [iso])]

    cfg = _base_cfg(colab_script, TARGET_MODE="gene", GENE_ID=42, ISOFORM_INDEX=0)
    result_aa_seq, result_locvec = colab_script.pick_target(cfg, genes)
    assert result_aa_seq == aa_seq
    assert result_locvec == loc_tags


def test_build_seed_fn_uniform_returns_generate_seed(colab_script):
    from genewriter.ga import generate_seed

    cfg = _base_cfg(colab_script, SEED_STRATEGY="uniform")
    seed_fn = colab_script._build_seed_fn(cfg, genes=[])
    assert seed_fn is generate_seed


def test_build_seed_fn_ngram_returns_working_partial(colab_script, monkeypatch):
    from genewriter.classes import CodonNgramModel

    stub_model = CodonNgramModel(organism="test", transcriptome="test", context_orders=(1,), context_counts={'1': {}})

    def _fake_build(genes, organism, context_orders):
        return stub_model

    monkeypatch.setattr(
        "genewriter.codon_ngram.build_codon_ngram_model", _fake_build,
    )

    cfg = _base_cfg(colab_script, SEED_STRATEGY="ngram", NGRAM_MODEL_PATH=None)
    seed_fn = colab_script._build_seed_fn(cfg, genes=[])
    result = seed_fn("MKL")
    assert len(result) == 3


def test_build_seed_fn_unknown_strategy_raises(colab_script):
    cfg = _base_cfg(colab_script, SEED_STRATEGY="bogus")
    with pytest.raises(ValueError, match="SEED_STRATEGY"):
        colab_script._build_seed_fn(cfg, genes=[])
