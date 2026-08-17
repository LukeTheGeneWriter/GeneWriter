"""KmerAnalysis baseline test -- unlike the other four baseline_<name>.py
modules, its shard is raw per-(k, bucket) occurrence counts (.npz), not a
KmerAnalysis-shaped dict: fold_enrich is a *derived* stat
(observed / expected) only correct once computed from counts summed across
the *whole* corpus, so finalize() sums first and normalizes once, at the very
end. See gpu_kmer_count.py for the counting itself.

FORK-SAFETY, load-bearing -- see baseline_pipeline.py's own docstring for the
full picture: this module (and gpu_kmer_count.py, which it calls) never
imports cupy at module scope. _select_backend() imports cupy *inside* the
function body, which only ever executes already inside the forked wave-2
child process baseline_pipeline.run_pipeline() spawns per chunk. This keeps
the *parent* process itself free of any CUDA context for the whole run, so
every subsequent chunk's fork() call forks from a CUDA-context-free parent --
forking a process that has already initialized CUDA is a well-known crash
hazard. Do not move `import cupy` to this file's top level; that would
silently reintroduce it.
"""

import glob
import os

import numpy as np

from . import gpu_kmer_count
from .classes import KmerAnalysis
from .codon_tables import decode_base4_string
from .standards_io import STANDARD_FILES

NAME = 'kmer'
FILENAME = STANDARD_FILES[NAME][0]
SHARD_EXT = '.npz'

_WINDOW_BUCKET_NAMES = ('ExonL50', 'Exon', 'ExonR50')


def _select_backend(use_gpu: bool):
    """cupy import lives here, and only here -- see module docstring."""
    if not use_gpu:
        return np
    try:
        import cupy as xp
        xp.cuda.Device(0).use()
        return xp
    except Exception as e:
        print(f"[baseline_kmer] GPU unavailable ({e!r}) -- falling back to numpy for this chunk.")
        return np


def _atomic_write_npz(path: str, **arrays) -> None:
    tmp_path = path + '.tmp'
    # np.savez appends '.npz' to any *string* filename that doesn't already
    # end with it -- tmp_path ends in '.tmp', not '.npz', so passing it as a
    # string would silently become 'tmp_path.npz' and break the os.replace
    # below. Passing an open file object instead skips that suffixing logic
    # entirely (it only applies when the argument has no .write attribute).
    with open(tmp_path, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp_path, path)


def compute_and_write_shard(genes: list, shard_path: str, organism: str = "human",
                             k_values=range(2, 11), use_gpu: bool = True) -> None:
    xp = _select_backend(use_gpu)
    raw, loc_totals = gpu_kmer_count.count_kmers_for_chunk(xp, genes, k_values)
    payload = {}
    for k in k_values:
        for name in _WINDOW_BUCKET_NAMES:
            arr = raw[k][name]
            payload[f'{k}__{name}'] = arr if xp is np else xp.asnumpy(arr)
            payload[f'loc_total__{k}__{name}'] = np.asarray(loc_totals[k][name])
    _atomic_write_npz(shard_path, **payload)


def finalize(shard_dir: str, organism: str = "human", k_values=range(2, 11), **_unused_kwargs) -> KmerAnalysis:
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, 'chunk_*.npz')))
    if not shard_paths:
        raise RuntimeError(f"No shards found in {shard_dir} -- run baseline_pipeline.run_pipeline() first")

    total_counts = {k: {name: np.zeros(4 ** k, dtype=np.int64) for name in _WINDOW_BUCKET_NAMES} for k in k_values}
    total_loc = {k: {name: 0 for name in _WINDOW_BUCKET_NAMES} for k in k_values}
    for path in shard_paths:
        data = np.load(path)
        for k in k_values:
            for name in _WINDOW_BUCKET_NAMES:
                total_counts[k][name] += data[f'{k}__{name}']
                total_loc[k][name] += int(data[f'loc_total__{k}__{name}'])

    kmer_dict = {}
    l_dict = {}
    for k in k_values:
        expected = {name: (total / (4 ** k) if total else 0.0) for name, total in total_loc[k].items()}
        observed_mask = np.zeros(4 ** k, dtype=bool)
        for name in _WINDOW_BUCKET_NAMES:
            observed_mask |= total_counts[k][name] > 0

        scored = {}
        for code in np.nonzero(observed_mask)[0].tolist():
            seq = decode_base4_string(code, k)
            scored[seq] = {}
            for name in _WINDOW_BUCKET_NAMES:
                obs = int(total_counts[k][name][code])
                fold = obs / expected[name] if expected[name] else 1.0
                # p (chi-square significance) fixed at 1.0, matching
                # baseline.compute_kmer_analysis's existing simplification --
                # confirmed unread by anything downstream (only fold_enrich
                # is ever consulted, see gpu_change_vector._build_kmer_score_table).
                scored[seq][name] = {'p': 1.0, 'fold_enrich': fold}
        kmer_dict[str(k)] = scored
        l_dict[str(k)] = total_loc[k]

    return KmerAnalysis(
        organism=organism,
        transcriptome="corpus-wide",
        kmer_dict=kmer_dict,
        l_dict=l_dict,
        # Unused by anything downstream (grep-confirmed against src/genewriter)
        # -- the legacy notebook's second full-corpus pass that populated this
        # field is exactly the redundant reread this pipeline exists to remove.
        kmer_score_obs={},
    )
