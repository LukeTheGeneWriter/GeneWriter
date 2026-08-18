"""Orchestrates the 5 baseline_<name>.py tests over the full gene corpus:
chunked loading, two-wave forked-process dispatch per chunk, and the final
shard-merge step. See Baseline_Pipeline.ipynb for the thin Colab driver that
configures and calls this module -- this is where the actual logic lives, so
it's importable/testable outside a notebook (matching gene_io.py/baseline.py's
existing shape).

Per chunk of genes (gene_io.chunk_paths/load_gene_chunk):
  1. Load the chunk once, into this (parent) process's memory.
  2. Wave 1: fork one child process per non-kmer TestSpec, concurrently. Each
     child already has the chunk's genes for free via copy-on-write (no
     re-read, no IPC serialization of the gene list) -- it just calls its
     module's compute_and_write_shard() and writes its own shard file
     straight to disk. The parent's only signal per child is
     Process.exitcode after join().
  3. Wave 2 (kmer): only after wave 1 finishes -- both to avoid CPU/GPU
     contention with wave 1's 4 concurrent processes, and so a kmer crash
     can't take the other four down (they've already written their shards by
     this point regardless of what happens in wave 2).
  4. Free the chunk, move to the next one.
After every chunk: finalize_all() merges each test's shards into the single
flat Standards/<Name>Analysis.json standards_io.py already expects.

Resumable at chunk granularity: if every registered test already has a shard
for a given chunk index, that chunk's gene JSONs are never even read.

FORK-SAFETY, load-bearing: this module never imports cupy, directly or
indirectly, in the parent process. baseline_kmer.py's GPU backend selection
only imports cupy inside a function that executes already inside the forked
wave-2 child (see that module's own docstring) -- so the parent stays free of
any CUDA context for the whole run, and every chunk's fork() call forks from
a CUDA-context-free parent. Forking a process that has already initialized
CUDA is a well-known crash hazard; this is why run_pipeline() explicitly
requires the 'fork' start method rather than silently falling back to
'spawn' (which has no CUDA/fork interaction issue, but would re-serialize
the whole in-memory chunk to every child over a pipe, defeating the
zero-copy design this pipeline exists to provide).
"""

import dataclasses
import json
import multiprocessing as mp
import os

from . import baseline_codon_pair_bias, baseline_codon_usage, baseline_gc, baseline_kmer, baseline_rare_codon
from .gene_io import chunk_paths, load_gene_chunk


@dataclasses.dataclass
class TestSpec:
    name: str
    module: object  # exposes NAME/FILENAME/SHARD_EXT/compute_and_write_shard()/finalize()
    shard_dir: str
    wave: int = 1  # 1 = concurrent with the other wave-1 tests; 2 = its own isolated wave, run after wave 1
    kwargs: dict = dataclasses.field(default_factory=dict)


def default_test_specs(standards_dir: str, k_values=range(2, 11), use_gpu_for_kmer: bool = True) -> list:
    """The literal "list of tests and their save directories" -- add or
    remove a bioinformatic test by editing this list (or building your own
    list of TestSpecs and passing it to run_pipeline()/finalize_all()
    directly instead of using this default set)."""
    partial_dir = os.path.join(standards_dir, '_partial')
    return [
        TestSpec('rare_codon', baseline_rare_codon, os.path.join(partial_dir, 'rare_codon')),
        TestSpec('codon_usage', baseline_codon_usage, os.path.join(partial_dir, 'codon_usage')),
        TestSpec('codon_pair_bias', baseline_codon_pair_bias, os.path.join(partial_dir, 'codon_pair_bias')),
        TestSpec('gc', baseline_gc, os.path.join(partial_dir, 'gc')),
        TestSpec('kmer', baseline_kmer, os.path.join(partial_dir, 'kmer'), wave=2,
                 kwargs={'k_values': k_values, 'use_gpu': use_gpu_for_kmer}),
    ]


def _shard_path(spec: TestSpec, chunk_index: int) -> str:
    return os.path.join(spec.shard_dir, f'chunk_{chunk_index:04d}{spec.module.SHARD_EXT}')


def _shard_worker(module, genes: list, shard_path: str, organism: str, kwargs: dict) -> None:
    # Any exception here propagates out of the child's target function --
    # multiprocessing's default behavior is to print the traceback and exit
    # non-zero, which is the parent's only signal it needs (see _run_wave):
    # the worker already wrote its result straight to disk, so there's no
    # payload to marshal back through a Pool/Queue.
    module.compute_and_write_shard(genes, shard_path, organism=organism, **kwargs)


def _run_wave(ctx, specs: list, genes: list, chunk_index: int, organism: str) -> list:
    procs = []
    for spec in specs:
        shard_path = _shard_path(spec, chunk_index)
        if os.path.exists(shard_path):
            continue  # this test already has this chunk's shard -- resumable
        p = ctx.Process(target=_shard_worker, args=(spec.module, genes, shard_path, organism, spec.kwargs))
        p.start()
        procs.append((spec, p))

    failures = []
    for spec, p in procs:
        p.join()
        if p.exitcode != 0:
            failures.append(spec.name)
    return failures


def run_pipeline(gene_dir: str, standards_dir: str, chunk_size: int = 750, organism: str = "human",
                  tests: list = None, resume: bool = True) -> dict:
    """Runs every registered test over every chunk of genes under gene_dir.
    Returns {chunk_index: {'wave1_failures': [...], 'wave2_failures': [...]}}
    for any chunk where at least one test failed -- a single test's crash on
    a single chunk does not stop the run or affect any other test/chunk.
    Does not call finalize_all() -- that's a separate step, run once after
    every chunk you care about has succeeded (see finalize_all())."""
    if 'fork' not in mp.get_all_start_methods():
        raise RuntimeError(
            "baseline_pipeline.run_pipeline() requires the 'fork' multiprocessing start "
            "method -- its whole design relies on a forked child sharing the parent's "
            "already-loaded gene chunk via copy-on-write, at zero serialization cost. "
            "Native Windows Python has no 'fork' start method at all; run this under "
            "WSL, Colab, or another POSIX/Linux environment instead."
        )
    ctx = mp.get_context('fork')

    chunks = chunk_paths(gene_dir, chunk_size)
    if not chunks:
        # Real bug hit live: chunk_paths() just glob()s gene_dir for *.json
        # -- if gene_dir is wrong (a stale relative path after a cwd change,
        # a typo, genes landed somewhere else, etc.), this loop below simply
        # never executes, and an empty `problems` dict looks IDENTICAL to a
        # genuinely clean run of a real corpus -- "All chunks completed with
        # no failures" printed with zero chunks ever having been touched.
        # That silent false-positive is worse than a loud failure here, so
        # a gene_dir with no matching gene JSONs at all is treated as a hard
        # configuration error, not a vacuous success.
        raise RuntimeError(
            f"No gene JSON files found in {gene_dir!r} (glob '*.json') -- nothing to run. "
            f"Check that gene_dir is correct and that the current working directory "
            f"({os.getcwd()!r}) is what you expect (e.g. after a Drive mount/%cd), "
            f"rather than silently doing nothing."
        )

    tests = tests if tests is not None else default_test_specs(standards_dir)
    for spec in tests:
        os.makedirs(spec.shard_dir, exist_ok=True)
    wave1 = [t for t in tests if t.wave == 1]
    wave2 = [t for t in tests if t.wave == 2]

    problems = {}
    for chunk_index, paths in enumerate(chunks):
        if resume and all(os.path.exists(_shard_path(t, chunk_index)) for t in tests):
            continue  # every test already has this chunk done -- skip the disk read entirely

        genes = load_gene_chunk(paths)
        wave1_failures = _run_wave(ctx, wave1, genes, chunk_index, organism)
        wave2_failures = _run_wave(ctx, wave2, genes, chunk_index, organism)  # strictly after wave 1
        if wave1_failures or wave2_failures:
            problems[chunk_index] = {'wave1_failures': wave1_failures, 'wave2_failures': wave2_failures}
        del genes

    return problems


def finalize_all(standards_dir: str, tests: list = None, organism: str = "human") -> dict:
    """Merges every test's chunk shards into its final flat
    Standards/<Name>Analysis.json (the exact file standards_io.load_standards()
    expects). Returns {test_name: path_written}. Call once, after
    run_pipeline() has (successfully, for the chunks you care about) produced
    shards for every test."""
    tests = tests if tests is not None else default_test_specs(standards_dir)
    written = {}
    for spec in tests:
        analysis = spec.module.finalize(spec.shard_dir, organism=organism, **spec.kwargs)
        out_path = os.path.join(standards_dir, spec.module.FILENAME)
        with open(out_path, 'w') as f:
            json.dump(dataclasses.asdict(analysis), f)
        written[spec.name] = out_path
    return written
