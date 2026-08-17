import dataclasses
import json
import os

from genewriter.gene_io import chunk_paths, iter_gene_chunks, load_gene_chunk, load_genes

from conftest import make_synthetic_gene, make_synthetic_isoform


def _write_gene(tmp_path, gene_id):
    iso = make_synthetic_isoform("MKL", lambda aa, i: 'ATG', ['I', 'I', 'I'])
    gene = make_synthetic_gene(gene_id, [iso])
    path = os.path.join(tmp_path, f'{gene_id}.json')
    with open(path, 'w') as f:
        json.dump(dataclasses.asdict(gene), f)
    return path


def test_chunk_paths_matches_load_genes_total_count(tmp_path):
    for i in range(7):
        _write_gene(tmp_path, i)

    chunks = chunk_paths(str(tmp_path), chunk_size=3)
    total = sum(len(c) for c in chunks)

    assert total == len(load_genes(str(tmp_path)))
    assert total == 7


def test_chunk_paths_sizes_and_remainder(tmp_path):
    for i in range(7):
        _write_gene(tmp_path, i)

    chunks = chunk_paths(str(tmp_path), chunk_size=3)

    assert [len(c) for c in chunks] == [3, 3, 1]


def test_chunk_size_larger_than_corpus_yields_one_chunk(tmp_path):
    for i in range(4):
        _write_gene(tmp_path, i)

    chunks = chunk_paths(str(tmp_path), chunk_size=100)

    assert len(chunks) == 1
    assert len(chunks[0]) == 4


def test_chunk_paths_matches_load_genes_sort_order(tmp_path):
    for i in range(5):
        _write_gene(tmp_path, i)

    chunks = chunk_paths(str(tmp_path), chunk_size=2)
    flat = [p for chunk in chunks for p in chunk]

    assert flat == sorted(flat)


def test_load_gene_chunk_returns_natural_genes(tmp_path):
    paths = [_write_gene(tmp_path, i) for i in range(3)]

    genes = load_gene_chunk(paths)

    assert [g.geneID for g in genes] == [0, 1, 2]


def test_iter_gene_chunks_yields_index_and_genes(tmp_path):
    for i in range(5):
        _write_gene(tmp_path, i)

    seen = list(iter_gene_chunks(str(tmp_path), chunk_size=2))

    assert [idx for idx, _ in seen] == [0, 1, 2]
    assert [len(genes) for _, genes in seen] == [2, 2, 1]
