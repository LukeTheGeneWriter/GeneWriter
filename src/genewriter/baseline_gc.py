"""GCAnalysis baseline test -- see baseline_rare_codon.py's module docstring
for the shared shape/rationale, including the 2026-08-19 switch from
baseline.compute_gc_analysis() to gpu_gc_count.count_gc_for_chunk()
(xp-injected, byte-for-byte verified against the same genes -- including the
majority-vote algorithm, gpu_gc_count.py deliberately uses
gpu_corpus_batch.majority_raw_tag_bucket(), NOT
gpu_change_vector.majority_window_bucket(), see
memory/majority_vote_bucket_discrepancy.md).
"""

import dataclasses

from .baseline_shard_util import atomic_write_json, concat_dict_of_lists, concat_lists, load_json_shards
from .classes import GCAnalysis
from .gpu_corpus_batch import select_backend
from .gpu_gc_count import count_gc_for_chunk
from .standards_io import STANDARD_FILES

NAME = 'gc'
FILENAME = STANDARD_FILES[NAME][0]
SHARD_EXT = '.json'


def compute_and_write_shard(genes: list, shard_path: str, organism: str = "human", winsize: int = 21,
                             use_gpu: bool = True, vram_fraction: float = 0.5) -> None:
    xp = select_backend(use_gpu, NAME)
    analysis = count_gc_for_chunk(xp, genes, organism, winsize, vram_fraction=vram_fraction)
    atomic_write_json(shard_path, dataclasses.asdict(analysis))


def finalize(shard_dir: str, organism: str = "human", **_unused_kwargs) -> GCAnalysis:
    shards = load_json_shards(shard_dir)
    return GCAnalysis(
        organism=shards[0]['organism'],
        transcriptome=shards[0]['transcriptome'],
        totalCodons=sum(s['totalCodons'] for s in shards),
        windows=concat_dict_of_lists(s['windows'] for s in shards),
        taggedGC1=concat_dict_of_lists(s['taggedGC1'] for s in shards),
        taggedGC2=concat_dict_of_lists(s['taggedGC2'] for s in shards),
        taggedGC3=concat_dict_of_lists(s['taggedGC3'] for s in shards),
        windowsize=shards[0]['windowsize'],
        gcPerGene=concat_lists(s['gcPerGene'] for s in shards),
    )
