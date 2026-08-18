"""Orchestrates the 5 baseline_<name>.py tests over the full gene corpus:
chunked loading, two-wave forked-process dispatch per chunk, and the final
shard-merge step. See Baseline_Pipeline.ipynb for the thin Colab driver that
configures and calls this module -- this is where the actual logic lives, so
it's importable/testable outside a notebook (matching gene_io.py/baseline.py's
existing shape).

Per chunk of genes (gene_io.chunk_paths/load_gene_chunk):
  1. Load the chunk once, into this (parent) process's memory --
     load_gene_chunk() itself loads concurrently via a thread pool
     (gene_io.py's `load_workers`/_DEFAULT_MAX_WORKERS), since the real cost
     there is Drive/FUSE round-trip *latency* per file, not local CPU work --
     see that module's own docstring for why threads help despite the GIL
     and Colab's single CPU core (this is a genuinely different bottleneck
     from the CPU/GPU-bound compute steps below, which is why this pipeline
     uses multiprocessing for those and threading for this).
  2. Wave 1: fork one child process per wave-1 TestSpec, concurrently. Each
     child already has the chunk's genes for free via copy-on-write (no
     re-read, no IPC serialization of the gene list) -- it just calls its
     module's compute_and_write_shard() and writes its own shard file
     straight to disk. The parent's only signal per child is
     Process.exitcode after join(). As of 2026-08-19, wave 1 is empty by
     default (default_test_specs() moved every test to wave 2 once they all
     became GPU-capable, see point 3) -- kept for any future genuinely
     CPU-only test, not removed.
  3. Wave 2: all 5 tests run SEQUENTIALLY inside ONE shared forked process
     (_run_sequential_group()), not as separate concurrent processes.
     Deliberate: once every test can use the GPU, N concurrent processes
     means N simultaneous CUDA contexts fighting over one physical GPU's
     VRAM -- likely more contention risk than real gain, since CUDA
     serializes kernel execution on one device across contexts anyway. One
     shared process means one CUDA context, one VRAM budget, no contention.
     Each spec's compute_and_write_shard() call is individually try/excepted
     inside the shared child so one test's failure doesn't stop its
     siblings in the same group -- this module's own docstring below still
     promises "a single test's crash does not stop any other test/chunk",
     which a bare shared process would otherwise silently break.
  4. Free the chunk, move to the next one.
After every chunk: finalize_all() merges each test's shards into the single
flat Standards/<Name>Analysis.json standards_io.py already expects.

Resumable at chunk granularity: if every registered test already has a shard
for a given chunk index, that chunk's gene JSONs are never even read.

FORK-SAFETY, load-bearing: this module never imports cupy, directly or
indirectly, in the parent process. Every baseline_<name>.py module's GPU
backend selection (gpu_corpus_batch.select_backend(), shared by all 5 as of
2026-08-19) only imports cupy inside a function that executes already
inside the forked wave-2 child (see that function's own docstring) -- so
the parent stays free of any CUDA context for the whole run, and every
chunk's fork() call forks from a CUDA-context-free parent. Forking a
process that has already initialized CUDA is a well-known crash hazard;
this is why run_pipeline() explicitly requires the 'fork' start method
rather than silently falling back to 'spawn' (which has no CUDA/fork
interaction issue, but would re-serialize the whole in-memory chunk to
every child over a pipe, defeating the zero-copy design this pipeline
exists to provide).
"""

import dataclasses
import json
import multiprocessing as mp
import os
import time

from . import baseline_codon_pair_bias, baseline_codon_usage, baseline_gc, baseline_kmer, baseline_rare_codon, gene_io
from .gene_io import chunk_paths, load_gene_chunk


@dataclasses.dataclass
class TestSpec:
    name: str
    module: object  # exposes NAME/FILENAME/SHARD_EXT/compute_and_write_shard()/finalize()
    shard_dir: str
    wave: int = 1  # 1 = concurrent with the other wave-1 tests; 2 = its own isolated wave, run after wave 1
    kwargs: dict = dataclasses.field(default_factory=dict)


def default_test_specs(standards_dir: str, k_values=range(2, 11), use_gpu_for_kmer: bool = True,
                        use_gpu_for_baselines: bool = True) -> list:
    """The literal "list of tests and their save directories" -- add or
    remove a bioinformatic test by editing this list (or building your own
    list of TestSpecs and passing it to run_pipeline()/finalize_all()
    directly instead of using this default set).

    All 5 tests are wave=2 as of 2026-08-19 (run sequentially in one shared
    process, see this module's docstring) -- every test is now GPU-capable
    (rare_codon/codon_usage/codon_pair_bias/gc via gpu_<name>_count.py,
    matching kmer's existing gpu_kmer_count.py). use_gpu_for_baselines is a
    separate parameter from use_gpu_for_kmer (not merged into one flag) so
    an existing caller that only wants to toggle kmer's GPU use (e.g.
    tests/test_baseline_pipeline_fork.py) isn't affected. Even with
    use_gpu_for_baselines=False, the 4 non-kmer tests still use their new
    xp-injected counting modules on numpy -- already much faster than the
    old pure-Python baseline.py loops, same precedent as kmer's own numpy
    fallback -- there is no toggle back to calling baseline.py directly."""
    partial_dir = os.path.join(standards_dir, '_partial')
    return [
        TestSpec('rare_codon', baseline_rare_codon, os.path.join(partial_dir, 'rare_codon'), wave=2,
                 kwargs={'use_gpu': use_gpu_for_baselines}),
        TestSpec('codon_usage', baseline_codon_usage, os.path.join(partial_dir, 'codon_usage'), wave=2,
                 kwargs={'use_gpu': use_gpu_for_baselines}),
        TestSpec('codon_pair_bias', baseline_codon_pair_bias, os.path.join(partial_dir, 'codon_pair_bias'), wave=2,
                 kwargs={'use_gpu': use_gpu_for_baselines}),
        TestSpec('gc', baseline_gc, os.path.join(partial_dir, 'gc'), wave=2,
                 kwargs={'use_gpu': use_gpu_for_baselines}),
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


def _run_wave(ctx, specs: list, genes: list, chunk_index: int, organism: str, verbose: bool = True) -> list:
    procs = []
    for spec in specs:
        shard_path = _shard_path(spec, chunk_index)
        if os.path.exists(shard_path):
            if verbose:
                print(f"[run_pipeline]   chunk {chunk_index}: {spec.name} (wave 1) already has a shard -- skipping.", flush=True)
            continue  # this test already has this chunk's shard -- resumable
        if verbose:
            print(f"[run_pipeline]   chunk {chunk_index}: {spec.name} (wave 1) starting...", flush=True)
        p = ctx.Process(target=_shard_worker, args=(spec.module, genes, shard_path, organism, spec.kwargs))
        p.start()
        procs.append((spec, p, time.perf_counter()))

    failures = []
    for spec, p, t0 in procs:
        p.join()
        if p.exitcode != 0:
            failures.append(spec.name)
        if verbose:
            status = 'FAILED' if p.exitcode != 0 else 'done'
            print(f"[run_pipeline]   chunk {chunk_index}: {spec.name} (wave 1) {status} in {time.perf_counter() - t0:.1f}s.", flush=True)
    return failures


def _sequential_group_worker(specs: list, genes: list, chunk_index: int, organism: str, result_queue, verbose: bool = True) -> None:
    failures = []
    for spec in specs:
        shard_path = _shard_path(spec, chunk_index)
        if os.path.exists(shard_path):
            if verbose:
                print(f"[run_pipeline]   chunk {chunk_index}: {spec.name} already has a shard -- skipping.", flush=True)
            continue  # this test already has this chunk's shard -- resumable
        if verbose:
            print(f"[run_pipeline]   chunk {chunk_index}: {spec.name} starting...", flush=True)
        t0 = time.perf_counter()
        try:
            spec.module.compute_and_write_shard(genes, shard_path, organism=organism, **spec.kwargs)
            if verbose:
                print(f"[run_pipeline]   chunk {chunk_index}: {spec.name} done in {time.perf_counter() - t0:.1f}s.", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            failures.append(spec.name)
            if verbose:
                print(f"[run_pipeline]   chunk {chunk_index}: {spec.name} FAILED after {time.perf_counter() - t0:.1f}s (see traceback above).", flush=True)
        try:
            # cupy's memory pool doesn't return freed blocks to the driver
            # by default -- without this, gpu_corpus_batch.vram_aware_batch_size()'s
            # free-VRAM query for the NEXT spec in this group would see
            # artificially less free memory than a fresh process would
            # (this group shares one CUDA context across all its specs, by
            # design -- see this module's own docstring), silently
            # shrinking later specs' batch sizes. A real, non-obvious perf
            # consequence of sharing one process, not a correctness bug --
            # only matters if cupy was actually imported by some spec above.
            #
            # Broad except, not just ImportError: real bug hit live -- a
            # spec whose OWN select_backend() call already caught a CUDA
            # failure and fell back to numpy still leaves cupy import-able
            # (the package is installed) but with a broken/uninitialized
            # context, and get_default_memory_pool() touching that context
            # raises its own CUDARuntimeError. This cleanup is a best-effort
            # optimization, not a correctness requirement -- any failure
            # here must never take down the whole group the way an
            # uncaught one did the first time this was tested.
            import cupy as _cupy_for_pool_cleanup
            _cupy_for_pool_cleanup.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    result_queue.put(failures)


def _run_sequential_group(ctx, specs: list, genes: list, chunk_index: int, organism: str, verbose: bool = True) -> list:
    """Forks ONE process for the whole `specs` group, running each spec's
    compute_and_write_shard() in turn inside it, instead of one process per
    spec (_run_wave()'s shape) -- see this module's own docstring for why
    (GPU-contention avoidance once every test is GPU-capable).

    Each spec is individually try/excepted inside the shared child
    (_sequential_group_worker()) so one test's ordinary failure doesn't stop
    its siblings in the same group from running -- this module's own
    "a single test's crash does not stop any other test/chunk" promise,
    which a bare shared process would otherwise silently break.
    """
    if not specs:
        return []
    result_queue = ctx.Queue()
    p = ctx.Process(target=_sequential_group_worker, args=(specs, genes, chunk_index, organism, result_queue, verbose))
    p.start()
    # queue.put()-then-join() is only a deadlock risk for payloads large
    # enough to fill the OS pipe buffer (typically 64KB+) -- `failures` is
    # always a short list of test-name strings, nowhere close, so join()
    # first (simpler, no polling loop) is safe here.
    p.join()
    if p.exitcode != 0:
        # An uncatchable crash (segfault, CUDA driver fault) never reaches
        # _sequential_group_worker's own try/except, and may have happened
        # before result_queue.put() ever ran -- fall back to checking which
        # specs in the group are still missing an on-disk shard. This is the
        # one place OS-level process isolation genuinely can't be fully
        # replicated by an in-process try/except.
        return [spec.name for spec in specs if not os.path.exists(_shard_path(spec, chunk_index))]
    try:
        return result_queue.get_nowait()
    except Exception:
        # Exited 0 but the queue is empty -- shouldn't happen given the
        # worker always puts before returning, but fall back the same way
        # rather than assume.
        return [spec.name for spec in specs if not os.path.exists(_shard_path(spec, chunk_index))]


def run_pipeline(gene_dir: str, standards_dir: str, chunk_size: int = 750, organism: str = "human",
                  tests: list = None, resume: bool = True, verbose: bool = True,
                  load_workers: int = gene_io._DEFAULT_MAX_WORKERS) -> dict:
    """Runs every registered test over every chunk of genes under gene_dir.
    Returns {chunk_index: {'wave1_failures': [...], 'wave2_failures': [...]}}
    for any chunk where at least one test failed -- a single test's crash on
    a single chunk does not stop the run or affect any other test/chunk.
    Does not call finalize_all() -- that's a separate step, run once after
    every chunk you care about has succeeded (see finalize_all()).

    verbose: default True -- prints per-chunk-load and per-spec-dispatch
    progress (with timing), flushed immediately so it shows up in real time
    in a Colab cell (default Python stdout buffering can otherwise delay
    output for a non-interactive process). Real bug hit live: this whole
    function used to print nothing at all, anywhere, for the entire run --
    a chunk's gene-loading step alone can take 20+ minutes against
    Drive-mounted storage, and with zero output there was no way to tell
    "working normally, just I/O-bound" from "hung" from "silently GPU-
    falling-back" without interrupting and guessing. Set False to match the
    old silent behavior (e.g. if a caller wants to do its own logging, or
    for quieter test output).

    load_workers: threads used to load each chunk's gene JSONs concurrently
    (gene_io.load_gene_chunk()) -- the real lever for that 20+ minute load
    time, since it's network/FUSE round-trip *latency* per file against
    Drive-mounted storage, not local CPU work (see gene_io.py's module
    docstring for why threads help here despite the GIL, and Colab's single
    CPU core). Tune down if you see Drive API rate-limit errors, up if a
    real run shows headroom -- see gene_io._DEFAULT_MAX_WORKERS."""
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

    if verbose:
        print(f"[run_pipeline] {len(chunks)} chunk(s) found in {gene_dir!r}.", flush=True)

    problems = {}
    for chunk_index, paths in enumerate(chunks):
        if resume and all(os.path.exists(_shard_path(t, chunk_index)) for t in tests):
            if verbose:
                print(f"[run_pipeline] chunk {chunk_index + 1}/{len(chunks)}: already complete -- skipping.", flush=True)
            continue  # every test already has this chunk done -- skip the disk read entirely

        if verbose:
            print(f"[run_pipeline] chunk {chunk_index + 1}/{len(chunks)}: loading {len(paths)} gene JSON(s) "
                  f"({load_workers} concurrent worker(s))"
                  f"{' -- Drive-mounted storage can still take a while even threaded, especially for the first chunk' if chunk_index == 0 else ''}...",
                  flush=True)
        t_load = time.perf_counter()
        genes = load_gene_chunk(paths, max_workers=load_workers)
        if verbose:
            print(f"[run_pipeline] chunk {chunk_index + 1}/{len(chunks)}: loaded in {time.perf_counter() - t_load:.1f}s -- running tests...", flush=True)

        t_chunk = time.perf_counter()
        wave1_failures = _run_wave(ctx, wave1, genes, chunk_index, organism, verbose=verbose)
        wave2_failures = _run_sequential_group(ctx, wave2, genes, chunk_index, organism, verbose=verbose)  # strictly after wave 1
        if verbose:
            print(f"[run_pipeline] chunk {chunk_index + 1}/{len(chunks)}: all tests done in {time.perf_counter() - t_chunk:.1f}s.", flush=True)
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
