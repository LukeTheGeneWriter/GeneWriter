"""CodonAnalysis (codon usage) baseline test -- see baseline_rare_codon.py's
module docstring for the shared shape/rationale (one file per test,
compute_and_write_shard()/finalize() is the fixed interface, the underlying
statistics logic in baseline.py is reused unchanged).
"""

import dataclasses

from .baseline import compute_codon_usage_analysis
from .baseline_shard_util import atomic_write_json, concat_lists, load_json_shards, sum_nested_dict_by_key, sum_scalar_by_key
from .classes import CodonAnalysis
from .standards_io import STANDARD_FILES

NAME = 'codon_usage'
FILENAME = STANDARD_FILES[NAME][0]
SHARD_EXT = '.json'


def compute_and_write_shard(genes: list, shard_path: str, organism: str = "human", winsize: int = 15) -> None:
    analysis = compute_codon_usage_analysis(genes, organism, winsize)
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
    )
