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
    aa_seq, locvec, natural_codons = colab_script.pick_target(cfg, genes=[])  # no genes loaded at all
    assert aa_seq == "MKLKER"
    assert locvec == ["I"] * 6
    assert natural_codons is None  # no real gene to embed alongside the GA population


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
    aa_seq, locvec, natural_codons = colab_script.pick_target(cfg, genes=[])
    assert locvec == custom_locvec
    assert natural_codons is None


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
    result_aa_seq, result_locvec, result_natural_codons = colab_script.pick_target(cfg, genes)
    assert result_aa_seq == aa_seq
    assert result_locvec == loc_tags
    assert result_natural_codons == [codon for codon, _loc in iso.codons]


def test_pick_target_gene_mode_omits_natural_codons_on_length_mismatch(colab_script):
    """Malformed data (isoform's codon count doesn't match its protein
    length) must not silently hand back a natural_codons that's the wrong
    length for plot_population_trajectory to embed alongside a real,
    correctly-shaped GA population -- None (and a printed warning) instead
    of a mismatched list."""
    from genewriter.codon_tables import codon_choices_for_aa

    aa_seq = "MKLKER"
    loc_tags = ['F', 'I', 'I', 'I', 'I', 'T']
    iso = make_synthetic_isoform(aa_seq, lambda aa, i: codon_choices_for_aa(aa)[0], loc_tags)
    iso.codons = iso.codons[:-1]  # now shorter than aa_seq
    genes = [make_synthetic_gene(42, [iso])]

    cfg = _base_cfg(colab_script, TARGET_MODE="gene", GENE_ID=42, ISOFORM_INDEX=0)
    _aa_seq, _locvec, natural_codons = colab_script.pick_target(cfg, genes)
    assert natural_codons is None


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


def test_build_seed_fn_ngram_loads_existing_model_without_rebuilding(colab_script, tmp_path, monkeypatch):
    from genewriter.classes import CodonNgramModel
    from genewriter.codon_ngram import save_codon_ngram_model

    stub_model = CodonNgramModel(organism="test", transcriptome="test", context_orders=(1,), context_counts={'1': {}})
    model_path = str(tmp_path / "model.json")
    save_codon_ngram_model(stub_model, model_path)

    def _fail_if_called(genes, organism, context_orders):
        raise AssertionError("build_codon_ngram_model should not be called when the model file already exists")

    monkeypatch.setattr("genewriter.codon_ngram.build_codon_ngram_model", _fail_if_called)

    cfg = _base_cfg(colab_script, SEED_STRATEGY="ngram", NGRAM_MODEL_PATH=model_path)
    seed_fn = colab_script._build_seed_fn(cfg, genes=[])
    assert len(seed_fn("MKL")) == 3


def test_build_seed_fn_ngram_builds_and_saves_when_path_is_missing(colab_script, tmp_path, monkeypatch):
    from genewriter.classes import CodonNgramModel

    stub_model = CodonNgramModel(organism="test", transcriptome="test", context_orders=(1,), context_counts={'1': {}})
    monkeypatch.setattr("genewriter.codon_ngram.build_codon_ngram_model", lambda genes, organism, context_orders: stub_model)

    model_path = str(tmp_path / "not_there_yet.json")
    assert not os.path.exists(model_path)

    cfg = _base_cfg(colab_script, SEED_STRATEGY="ngram", NGRAM_MODEL_PATH=model_path)
    seed_fn = colab_script._build_seed_fn(cfg, genes=[])

    assert os.path.isfile(model_path)  # built model was saved for next time
    assert len(seed_fn("MKL")) == 3


def test_resolve_weights_returns_config_weights_unchanged_by_default(colab_script, analysis_objects):
    cfg = _base_cfg(colab_script, USE_INTOLERANCE_WEIGHTS=False)
    assert colab_script._resolve_weights(cfg, analysis_objects) is cfg["WEIGHTS"]


def test_resolve_weights_calibrates_when_enabled(colab_script, analysis_objects, monkeypatch):
    calibrated = {"RareCodons": 2.0, "CodonUsage": 0.5, "CodonPairBias": 1.0, "GC": 1.5, "Kmer": 1.0, "Uracil": 0.0}
    monkeypatch.setattr(
        "genewriter.weight_calibration.compute_intolerance_weights",
        lambda analysis_objects, fallback_weights: calibrated,
    )
    cfg = _base_cfg(colab_script, USE_INTOLERANCE_WEIGHTS=True)
    assert colab_script._resolve_weights(cfg, analysis_objects) == calibrated


def test_config_schedule_uses_only_registered_step_kinds(colab_script):
    """The shipped SCHEDULE is meant to be pasted into Colab and run as-is,
    so a step kind that was renamed (or never existed) should fail here, not
    30 minutes into a real run after the gene corpus finished loading."""
    from genewriter.schedule import registered_steps

    kinds = set(registered_steps())

    def walk(steps):
        for step in steps:
            assert step["kind"] in kinds, f"unregistered step kind {step['kind']!r}"
            if step["kind"] == "repeat":
                walk(step["steps"])

    walk(colab_script.CONFIG["SCHEDULE"])


def test_config_caps_the_population_before_the_first_growth_step(colab_script):
    """Regression guard for a real Colab finding: seeding auto-sizes to as
    many individuals as RAM holds, which is right for coverage and far too
    many to hand straight to a growth step that multiplies it. Both run
    modes must cap the population between seeding and the first growth --
    "input"'s `keep` for the schedule, `target_size` for run_ga (which
    applies it right after seeding too). Either one left unset is the exact
    shape that had to be patched by hand mid-run."""
    schedule = colab_script.CONFIG["SCHEDULE"]
    assert schedule[0]["kind"] == "input"
    assert schedule[0].get("keep"), "SCHEDULE's 'input' step has no keep -- seeding is uncapped into growth"
    assert colab_script.CONFIG["RUN_GA_OPTIONS"]["target_size"], (
        "RUN_GA_OPTIONS['target_size'] is None -- run_ga would take the auto-sized seed count "
        "as its steady-state population size"
    )


def test_config_budgets_host_ram_for_the_expansion_steps(colab_script):
    """RAM_FRACTION is what stops a growth step from allocating
    len(pop) * rate new solutions past what the VM holds -- the crash this
    script is most likely to hit at real scale, and the one CHUNK_SIZE does
    NOT cover (that bounds the GPU side; this bounds the host side, where
    the population itself lives). A driver shipped without it silently
    reverts to the unbounded behavior."""
    ram_fraction = colab_script.CONFIG["RAM_FRACTION"]
    assert ram_fraction is None or 0 < ram_fraction <= 1, (
        f"RAM_FRACTION={ram_fraction!r} is not a usable share of free RAM"
    )


def test_config_ram_fraction_reaches_both_run_modes(colab_script):
    """Both call sites have to thread it: an unwired knob reads as working
    (no error, sensible-looking value in CONFIG) while the run it is
    supposed to protect keeps the library default instead of the one the
    config says."""
    import inspect

    source = inspect.getsource(colab_script.run_pipeline)
    assert source.count('cfg["RAM_FRACTION"]') == 2, (
        "RAM_FRACTION must be passed to both run_ga and run_schedule"
    )
