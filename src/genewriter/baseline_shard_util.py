"""Shared arithmetic/IO helpers for the per-test baseline_<name>.py modules'
finalize() steps.

Deliberately *not* a generic "merge any dict by inspecting its shape"
reducer: several *Analysis dataclass fields are plain `dict`/`list` but need
different merge rules that aren't distinguishable from the value's Python
type alone -- e.g. RareCodonAnalysis.rareCodonsByLocation's value is a
2-element `[n_rare, n_total]` list needing an *elementwise* sum, while
CodonAnalysis.usagePerGene is a growing per-gene list needing *concatenation*,
and both are plain `list`s. Each baseline_<name>.py names its own field's
merge rule explicitly using the primitives below, rather than this module
guessing.
"""

import glob
import json
import os


def atomic_write_json(path: str, obj) -> None:
    """Write obj as JSON to path, crash-safely: writes to path + '.tmp' then
    os.replace()s it into place (atomic on POSIX). Without this, a shard
    writer killed mid-json.dump would leave a truncated-but-present file that
    a naive `os.path.exists(shard_path)` resume check would wrongly treat as
    a completed shard."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp_path, path)


def load_json_shards(shard_dir: str, pattern: str = 'chunk_*.json') -> list:
    """Every shard file in shard_dir, loaded and sorted by chunk index (the
    glob pattern's zero-padded chunk_%04d naming sorts correctly as plain
    strings). Raises RuntimeError with a clear message if none exist -- a
    test's finalize() called before its pipeline run is a caller mistake,
    not a silently-empty result."""
    paths = sorted(glob.glob(os.path.join(shard_dir, pattern)))
    if not paths:
        raise RuntimeError(f"No shards found in {shard_dir} -- run baseline_pipeline.run_pipeline() first")
    shards = []
    for p in paths:
        with open(p) as f:
            shards.append(json.load(f))
    return shards


def sum_scalar_by_key(dicts) -> dict:
    """dicts: iterable of {key: number}. Union of keys across every dict;
    a key missing from some dict contributes 0 for that dict."""
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def sum_pairwise_by_key(dicts) -> dict:
    """dicts: iterable of {key: [n0, n1, ...]}, all values for a given key
    the same length across every dict. Union of keys, elementwise sum."""
    out = {}
    for d in dicts:
        for k, v in d.items():
            if k not in out:
                out[k] = list(v)
            else:
                out[k] = [a + b for a, b in zip(out[k], v)]
    return out


def sum_nested_dict_by_key(dicts) -> dict:
    """dicts: iterable of {key1: {key2: number}}. Union of keys at both
    levels (e.g. CodonAnalysis.codonFreqsByLocation: codon -> bucket -> count)."""
    out = {}
    for d in dicts:
        for k1, inner in d.items():
            bucket = out.setdefault(k1, {})
            for k2, v in inner.items():
                bucket[k2] = bucket.get(k2, 0) + v
    return out


def concat_lists(lists) -> list:
    """lists: iterable of lists -- flat concatenation in order."""
    out = []
    for lst in lists:
        out.extend(lst)
    return out


def concat_dict_of_lists(dicts) -> dict:
    """dicts: iterable of {key: [items...]}. Union of keys, concatenated
    lists per key (e.g. GCAnalysis.windows: bucket -> per-window GC values)."""
    out = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out
