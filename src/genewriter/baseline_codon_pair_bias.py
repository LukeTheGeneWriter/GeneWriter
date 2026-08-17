"""CodonPairBiasAnalysis baseline test -- see baseline_rare_codon.py's module
docstring for the shared shape/rationale.
"""

import dataclasses

from .baseline import compute_codon_pair_bias_analysis
from .baseline_shard_util import atomic_write_json, concat_lists, load_json_shards
from .classes import CodonPairBiasAnalysis
from .standards_io import STANDARD_FILES

NAME = 'codon_pair_bias'
FILENAME = STANDARD_FILES[NAME][0]
SHARD_EXT = '.json'


def compute_and_write_shard(genes: list, shard_path: str, organism: str = "human", winsize: int = 15) -> None:
    analysis = compute_codon_pair_bias_analysis(genes, organism, winsize)
    atomic_write_json(shard_path, dataclasses.asdict(analysis))


def finalize(shard_dir: str, organism: str = "human", **_unused_kwargs) -> CodonPairBiasAnalysis:
    shards = load_json_shards(shard_dir)
    return CodonPairBiasAnalysis(
        organism=shards[0]['organism'],
        transcriptome=shards[0]['transcriptome'],
        totalCodonPairs=sum(s['totalCodonPairs'] for s in shards),
        cpbPerGene=concat_lists(s['cpbPerGene'] for s in shards),
        cpb_lit=shards[0]['cpb_lit'],  # literal constant, identical in every shard
        cpbPerWindow=concat_lists(s['cpbPerWindow'] for s in shards),
        windowsize=shards[0]['windowsize'],
    )
