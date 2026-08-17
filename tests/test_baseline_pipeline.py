"""Tests for the parts of baseline_pipeline.py that don't require real
forking: default_test_specs()'s shape, the atomic-write/shard-loading
helpers' crash-safety, and finalize_all() (which never forks -- it just
calls each test module's finalize() sequentially). See
test_baseline_pipeline_fork.py for run_pipeline() itself, exercised through
real forked child processes.
"""
import json
import os

from genewriter import standards_io
from genewriter.baseline_pipeline import default_test_specs, finalize_all
from genewriter.baseline_shard_util import atomic_write_json, load_json_shards

from conftest import make_synthetic_gene, make_synthetic_isoform, random_solution

_AA_SEQ = "MAVLDEFGHIKPQRSTWYCN" * 2


def _uniform_tag_genes(n: int) -> list:
    genes = []
    for i in range(n):
        codons = random_solution(_AA_SEQ, seed=i)
        iso = make_synthetic_isoform(_AA_SEQ, lambda aa, j, c=codons: c[j], ['I'] * len(_AA_SEQ))
        genes.append(make_synthetic_gene(i, [iso]))
    return genes


def test_default_test_specs_shape(tmp_path):
    specs = default_test_specs(str(tmp_path))

    names = {s.name for s in specs}
    assert names == {'rare_codon', 'codon_usage', 'codon_pair_bias', 'gc', 'kmer'}

    waves = {s.name: s.wave for s in specs}
    assert waves['kmer'] == 2
    for name in ('rare_codon', 'codon_usage', 'codon_pair_bias', 'gc'):
        assert waves[name] == 1

    for s in specs:
        assert s.shard_dir == os.path.join(str(tmp_path), '_partial', s.name)


def test_atomic_write_json_leaves_no_tmp_file(tmp_path):
    path = str(tmp_path / 'chunk_0000.json')

    atomic_write_json(path, {'a': 1})

    assert os.path.exists(path)
    assert not os.path.exists(path + '.tmp')
    with open(path) as f:
        assert json.load(f) == {'a': 1}


def test_load_json_shards_ignores_stray_tmp_files(tmp_path):
    atomic_write_json(str(tmp_path / 'chunk_0000.json'), {'a': 1})
    # A leftover .tmp file (as if a previous write were interrupted) must
    # not be picked up as a completed shard by a naive glob.
    with open(str(tmp_path / 'chunk_0001.json.tmp'), 'w') as f:
        f.write('{not valid json')

    shards = load_json_shards(str(tmp_path))

    assert shards == [{'a': 1}]


def test_finalize_all_writes_standard_files_and_round_trips(tmp_path):
    standards_dir = str(tmp_path / 'Standards')
    specs = default_test_specs(standards_dir, k_values=(2, 3), use_gpu_for_kmer=False)
    genes = _uniform_tag_genes(4)

    # Write one shard per test directly -- finalize_all() itself never
    # forks, so this doesn't need run_pipeline()'s multiprocessing at all.
    for spec in specs:
        os.makedirs(spec.shard_dir, exist_ok=True)
        shard_path = os.path.join(spec.shard_dir, f'chunk_0000{spec.module.SHARD_EXT}')
        spec.module.compute_and_write_shard(genes, shard_path, **spec.kwargs)

    written = finalize_all(standards_dir, specs)

    assert set(written) == {'rare_codon', 'codon_usage', 'codon_pair_bias', 'gc', 'kmer'}
    for path in written.values():
        assert os.path.exists(path)

    # The real end-to-end contract check: standards_io.load_standards() must
    # accept exactly what finalize_all() wrote.
    loaded = standards_io.load_standards(standards_dir)
    assert loaded.rare_codon.totalCodons == loaded.gc.totalCodons  # both count every codon in the same 4 genes
    assert loaded.rare_codon.totalCodons > 0
