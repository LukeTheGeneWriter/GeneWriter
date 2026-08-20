"""CodonAnalysis (codon usage) baseline test -- see baseline_rare_codon.py's
module docstring for the shared shape/rationale, including the 2026-08-19
switch from baseline.compute_codon_usage_analysis() to
gpu_codon_usage_count.count_codon_usage_for_chunk() (xp-injected,
byte-for-byte verified against the same genes).
"""

import dataclasses

from .baseline_shard_util import (
    atomic_write_json,
    concat_lists,
    concat_nested_dict_of_lists,
    load_json_shards,
    sum_nested_dict_by_key,
    sum_scalar_by_key,
)
from .classes import CodonAnalysis
from .gpu_codon_usage_count import count_codon_usage_for_chunk
from .gpu_corpus_batch import select_backend
from .standards_io import STANDARD_FILES

NAME = 'codon_usage'
FILENAME = STANDARD_FILES[NAME][0]
SHARD_EXT = '.json'


def compute_and_write_shard(genes: list, shard_path: str, organism: str = "human", winsize: int = 15,
                             use_gpu: bool = True, vram_fraction: float = 0.5) -> None:
    xp = select_backend(use_gpu, NAME)
    analysis = count_codon_usage_for_chunk(xp, genes, organism, winsize, vram_fraction=vram_fraction)
    atomic_write_json(shard_path, dataclasses.asdict(analysis))


def finalize(shard_dir: str, organism: str = "human", **_unused_kwargs) -> CodonAnalysis:
    shards = load_json_shards(shard_dir)
    return CodonAnalysis(
        organism=shards[0]['organism'],
        transcriptome=shards[0]['transcriptome'],
        AAFreqs=sum_scalar_by_key(s['AAFreqs'] for s in shards),
        totalCodons=sum(s['totalCodons'] for s in shards),
        codonFreqsByLocation=sum_nested_dict_by_key(s['codonFreqsByLocation'] for s in shards),
        codonFreqsLit=shards[0]['codonFreqsLit'],  # literal constant, identical in every shard
        codonUsageScoreByGene=concat_lists(s['codonUsageScoreByGene'] for s in shards),
        windowsize=shards[0]['windowsize'],
        windowscores=concat_lists(s['windowscores'] for s in shards),
        windowdistancesfromoptimal=concat_lists(s['windowdistancesfromoptimal'] for s in shards),
        codonUsagePercentByAA=concat_nested_dict_of_lists(s['codonUsagePercentByAA'] for s in shards),
    )
