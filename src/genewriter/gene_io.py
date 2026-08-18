"""Load NaturalGene objects from the JSON files saveNaturalGeneObj() writes
(GeneDataSourcing.ipynb et al) -- the RefGenes/NHGeneBodySupp/<geneID>.json
files. Equivalent to the loadGeneBody() function duplicated verbatim across
Rare_Codons.ipynb, Codon_Usage.ipynb, Codon_Pair_Bias.ipynb, GC_Analysis.ipynb
and Kmer_Analysis.ipynb.

load_genes()/load_gene_chunk() load concurrently via a thread pool
(_load_paths(), max_workers -- default _DEFAULT_MAX_WORKERS) rather than one
file at a time. This matters specifically because the real cost here (~1-2s
per gene, confirmed live against Drive-mounted Colab storage) is network/FUSE
round-trip *latency* per file open, not local CPU work -- Python's GIL
prevents true parallel *compute* across threads, but a thread blocked on a
file read releases the GIL while it waits, so many reads can be in flight on
the network at once even on a single CPU core. This is the classic
"I/O-bound work parallelizes fine with threads despite the GIL" case, unlike
CPU-bound work (e.g. the codon-window math elsewhere in this codebase),
which genuinely needs multiple processes/cores and is what
baseline_pipeline.py's own multiprocessing exists for instead.
`concurrent.futures.ThreadPoolExecutor.map()` is used specifically because it
preserves input order in its results even though the underlying reads
complete out of order -- callers (e.g. baseline_pipeline.py's
resume-by-chunk-index logic doesn't care, but tests/test_gene_io.py's own
`test_load_gene_chunk_returns_natural_genes` explicitly asserts gene order
is preserved) shouldn't see any behavior change from this beyond speed.
"""

import concurrent.futures
import glob
import json
import os

from .classes import IsoformGeneBody, NaturalGene, ProteinObj

# A starting point, not a measured optimum -- Drive's API can start
# rate-limiting/erroring under too much concurrent load, so this is
# deliberately moderate rather than maximized. Tune down if you see
# rate-limit errors, up if a real run shows headroom.
_DEFAULT_MAX_WORKERS = 8


def load_gene(path: str) -> NaturalGene:
    with open(path) as f:
        data = json.load(f)

    isoforms = []
    for iso in data['isoforms']:
        protein = ProteinObj(**iso['associatedProtein'])
        isoforms.append(IsoformGeneBody(
            isoformNumber=iso['isoformNumber'],
            associatedProtein=protein,
            fullSequence=iso['fullSequence'],
            mRNASeq=iso['mRNASeq'],
            codons=iso['codons'],
            relativeAbundance=iso['relativeAbundance'],
            geneBody=iso['geneBody'],
        ))

    return NaturalGene(
        isoforms=isoforms,
        geneID=data['geneID'],
        geneName=data['geneName'],
        organism=data['organism'],
        DNASequence=data['DNASequence'],
        spliceAIDonor=data['spliceAIDonor'],
        spliceAIReceptor=data['spliceAIReceptor'],
        energetics=data['energetics'],
        chromosome=data['chromosome'],
    )


def _load_paths(paths: list, max_workers: int = _DEFAULT_MAX_WORKERS) -> list:
    """load_gene() for every path, concurrently via a thread pool -- see
    module docstring for why threads (not processes) are the right tool
    here. max_workers <= 1 falls back to a plain sequential loop (also
    what an empty `paths` short-circuits to) -- no pool overhead for a
    trivially small or explicitly-serial case."""
    if not paths:
        return []
    if max_workers <= 1:
        return [load_gene(p) for p in paths]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(load_gene, paths))


def load_genes(directory: str, max_workers: int = _DEFAULT_MAX_WORKERS) -> list:
    paths = sorted(glob.glob(os.path.join(directory, '*.json')))
    return _load_paths(paths, max_workers=max_workers)


def chunk_paths(directory: str, chunk_size: int = 750) -> list:
    """The same sorted glob() load_genes() uses, partitioned into fixed-size
    chunks of paths -- no gene JSON is touched here, just path partitioning,
    so a caller (baseline_pipeline.run_pipeline) can decide per chunk_index
    whether a chunk needs loading at all (e.g. every test already has a shard
    for it) before reading a single gene file."""
    paths = sorted(glob.glob(os.path.join(directory, '*.json')))
    return [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]


def load_gene_chunk(paths: list, max_workers: int = _DEFAULT_MAX_WORKERS) -> list:
    """load_gene() for every path in one chunk (one chunk_paths() entry),
    concurrently -- see _load_paths()/module docstring."""
    return _load_paths(paths, max_workers=max_workers)


def iter_gene_chunks(directory: str, chunk_size: int = 750, max_workers: int = _DEFAULT_MAX_WORKERS):
    """chunk_paths() + load_gene_chunk() fused: yields (chunk_index, genes)
    pairs, one chunk resident in memory at a time -- unlike load_genes(),
    never the whole corpus at once."""
    for i, paths in enumerate(chunk_paths(directory, chunk_size)):
        yield i, load_gene_chunk(paths, max_workers=max_workers)


def protein_coding_isoforms(genes: list):
    """Yield (gene, isoform) for isoforms with a non-empty codon stream,
    skipping computationally-predicted-only transcripts the way every
    analysis notebook does (`if 'X' in str(iso.isoformNumber): continue`)."""
    for gene in genes:
        for iso in gene.isoforms:
            if 'X' in str(iso.isoformNumber):
                continue
            if not iso.codons:
                continue
            yield gene, iso
