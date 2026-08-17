"""RareCodonAnalysis baseline test -- one file per test, per the pipeline's
plugin design (see baseline_pipeline.py), so this can be edited/tuned
independently of the other four. compute_and_write_shard()/finalize() are the
fixed interface baseline_pipeline.TestSpec dispatches through.

The actual statistics logic is untouched, existing code:
baseline.compute_rare_codon_analysis() already just accumulates over whatever
`genes` list it's given, so it works correctly as-is whether called on the
whole corpus (as tests/test_baseline.py does) or on one chunk (as this module
does) -- this file only adds the shard-write/shard-merge plumbing around it.
"""

import dataclasses

from .baseline import compute_rare_codon_analysis
from .baseline_shard_util import atomic_write_json, concat_lists, load_json_shards, sum_pairwise_by_key, sum_scalar_by_key
from .classes import RareCodonAnalysis
from .standards_io import STANDARD_FILES

NAME = 'rare_codon'
FILENAME = STANDARD_FILES[NAME][0]
SHARD_EXT = '.json'


def compute_and_write_shard(genes: list, shard_path: str, organism: str = "human", winsize: int = 15) -> None:
    analysis = compute_rare_codon_analysis(genes, organism, winsize)
    atomic_write_json(shard_path, dataclasses.asdict(analysis))


def finalize(shard_dir: str, organism: str = "human", **_unused_kwargs) -> RareCodonAnalysis:
    shards = load_json_shards(shard_dir)
    # rare_codon_windows' keys are ints in compute_rare_codon_analysis()'s
    # own direct output (and after standards_io.load_standards() normalizes
    # them back) -- each shard's keys came back as strings from its own JSON
    # round-trip, so they're restored to int here too, keeping finalize()'s
    # return value consistent with compute_rare_codon_analysis()'s regardless
    # of whether it went through the shard/merge path.
    windows = {int(k): v for k, v in sum_scalar_by_key(s['rare_codon_windows'] for s in shards).items()}
    return RareCodonAnalysis(
        organism=shards[0]['organism'],
        transcriptome=shards[0]['transcriptome'],
        totalCodons=sum(s['totalCodons'] for s in shards),
        rareCodonsByLocation=sum_pairwise_by_key(s['rareCodonsByLocation'] for s in shards),
        rare_codon_windows=windows,
        codonFreqsLit=shards[0]['codonFreqsLit'],  # literal constant, identical in every shard
        usagePerGene=concat_lists(s['usagePerGene'] for s in shards),
    )
