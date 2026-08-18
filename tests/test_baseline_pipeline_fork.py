"""Exercises baseline_pipeline.run_pipeline() for real, through actual
forked child processes -- skip-guarded to the platforms this pipeline's
whole design requires (POSIX with the 'fork' start method; see
baseline_pipeline.py's own docstring on why 'spawn' isn't an acceptable
substitute). Runnable today under this project's existing WSL dev workflow;
skipped on native Windows Python, which has no 'fork' start method at all.
"""
import dataclasses
import json
import multiprocessing as mp
import os
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == 'win32' or 'fork' not in mp.get_all_start_methods(),
    reason="requires the POSIX 'fork' multiprocessing start method",
)

from genewriter import baseline_pipeline, standards_io  # noqa: E402
# Aliased on import: pytest's default collection would otherwise try to treat
# TestSpec as a test class purely because of its name (it has an __init__,
# which is what triggers the (harmless but noisy) PytestCollectionWarning).
from genewriter.baseline_pipeline import TestSpec as PipelineTestSpec  # noqa: E402
from genewriter.baseline_pipeline import default_test_specs, run_pipeline  # noqa: E402

from conftest import make_synthetic_gene, make_synthetic_isoform, random_solution  # noqa: E402

_AA_SEQ = "MAVLDEFGHIKPQRSTWYCN" * 2


class _BrokenModule:
    """A fake test module whose compute_and_write_shard always raises --
    used to confirm one test's failure doesn't stop the other tests in the
    same wave/chunk from completing."""
    SHARD_EXT = '.json'

    @staticmethod
    def compute_and_write_shard(genes, shard_path, organism="human", **kwargs):
        raise RuntimeError("intentional test failure")


def _write_gene_files(gene_dir: str, n: int) -> None:
    os.makedirs(gene_dir, exist_ok=True)
    for i in range(n):
        codons = random_solution(_AA_SEQ, seed=i)
        iso = make_synthetic_isoform(_AA_SEQ, lambda aa, j, c=codons: c[j], ['I'] * len(_AA_SEQ))
        gene = make_synthetic_gene(i, [iso])
        with open(os.path.join(gene_dir, f'{i}.json'), 'w') as f:
            json.dump(dataclasses.asdict(gene), f)


def test_run_pipeline_writes_shards_for_every_test(tmp_path):
    gene_dir = str(tmp_path / 'genes')
    standards_dir = str(tmp_path / 'Standards')
    _write_gene_files(gene_dir, 5)

    specs = default_test_specs(standards_dir, k_values=(2, 3), use_gpu_for_kmer=False)
    problems = run_pipeline(gene_dir, standards_dir, chunk_size=5, tests=specs)

    assert problems == {}
    for spec in specs:
        shard_path = os.path.join(spec.shard_dir, f'chunk_0000{spec.module.SHARD_EXT}')
        assert os.path.exists(shard_path)

    baseline_pipeline.finalize_all(standards_dir, specs)
    loaded = standards_io.load_standards(standards_dir)
    assert loaded.gc.totalCodons > 0


def test_run_pipeline_resumes_without_rereading_completed_chunks(tmp_path):
    gene_dir = str(tmp_path / 'genes')
    standards_dir = str(tmp_path / 'Standards')
    _write_gene_files(gene_dir, 3)
    specs = default_test_specs(standards_dir, k_values=(2,), use_gpu_for_kmer=False)

    run_pipeline(gene_dir, standards_dir, chunk_size=3, tests=specs)
    shard_path = os.path.join(specs[0].shard_dir, 'chunk_0000.json')
    mtime_before = os.path.getmtime(shard_path)

    problems = run_pipeline(gene_dir, standards_dir, chunk_size=3, tests=specs, resume=True)

    assert problems == {}
    assert os.path.getmtime(shard_path) == mtime_before  # not rewritten -- the chunk was skipped entirely


def test_run_pipeline_raises_loudly_on_empty_gene_dir(tmp_path):
    # Real bug hit live: an empty/wrong gene_dir made chunk_paths() return
    # [], so run_pipeline()'s main loop silently never executed and
    # returned problems == {} -- indistinguishable from a genuinely clean
    # run of a real corpus ("All chunks completed with no failures", having
    # touched zero genes). This must raise instead of returning a vacuous
    # success.
    gene_dir = str(tmp_path / 'empty_genes')
    os.makedirs(gene_dir, exist_ok=True)
    standards_dir = str(tmp_path / 'Standards')
    specs = default_test_specs(standards_dir, k_values=(2,), use_gpu_for_kmer=False)

    with pytest.raises(RuntimeError, match="No gene JSON files found"):
        run_pipeline(gene_dir, standards_dir, chunk_size=3, tests=specs)


def test_run_pipeline_raises_loudly_on_nonexistent_gene_dir(tmp_path):
    gene_dir = str(tmp_path / 'does_not_exist')
    standards_dir = str(tmp_path / 'Standards')
    specs = default_test_specs(standards_dir, k_values=(2,), use_gpu_for_kmer=False)

    with pytest.raises(RuntimeError, match="No gene JSON files found"):
        run_pipeline(gene_dir, standards_dir, chunk_size=3, tests=specs)


def test_run_pipeline_one_test_failure_does_not_stop_the_others(tmp_path):
    gene_dir = str(tmp_path / 'genes')
    standards_dir = str(tmp_path / 'Standards')
    _write_gene_files(gene_dir, 3)

    specs = default_test_specs(standards_dir, k_values=(2,), use_gpu_for_kmer=False)
    broken = PipelineTestSpec('broken', _BrokenModule, os.path.join(standards_dir, '_partial', 'broken'), wave=1)

    problems = run_pipeline(gene_dir, standards_dir, chunk_size=3, tests=specs + [broken])

    assert problems == {0: {'wave1_failures': ['broken'], 'wave2_failures': []}}
    # The four real (non-broken) wave-1 tests' shards still landed on disk.
    for spec in specs:
        if spec.wave != 1:
            continue
        shard_path = os.path.join(spec.shard_dir, f'chunk_0000{spec.module.SHARD_EXT}')
        assert os.path.exists(shard_path)
