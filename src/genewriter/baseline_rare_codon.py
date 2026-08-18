"""RareCodonAnalysis baseline test -- one file per test, per the pipeline's
plugin design (see baseline_pipeline.py), so this can be edited/tuned
independently of the other four. compute_and_write_shard()/finalize() are the
fixed interface baseline_pipeline.TestSpec dispatches through.

compute_and_write_shard() calls gpu_rare_codon_count.count_rare_codons_for_chunk()
(xp-injected, numpy or cupy -- see gpu_corpus_batch.select_backend()) as of
2026-08-19, not baseline.compute_rare_codon_analysis() directly -- verified
to produce byte-for-byte identical output on the same genes
(tests/test_baseline_rare_codon.py's test_chunked_finalize_matches_monolithic).
baseline.compute_rare_codon_analysis() itself is untouched, still used by
tests/test_baseline.py/local dev/baseline.compute_baselines(), and as that
correctness-oracle ground truth -- not called from this pipeline anymore.

FORK-SAFETY: cupy import lives inside gpu_corpus_batch.select_backend(),
called only from inside the forked wave-2 child -- see baseline_pipeline.py's
own docstring for why this matters.
"""

import dataclasses

from .baseline_shard_util import atomic_write_json, concat_lists, load_json_shards, sum_pairwise_by_key, sum_scalar_by_key
from .classes import RareCodonAnalysis
from .gpu_corpus_batch import select_backend
from .gpu_rare_codon_count import count_rare_codons_for_chunk
from .standards_io import STANDARD_FILES

NAME = 'rare_codon'
FILENAME = STANDARD_FILES[NAME][0]
SHARD_EXT = '.json'


def compute_and_write_shard(genes: list, shard_path: str, organism: str = "human", winsize: int = 15,
                             use_gpu: bool = True, vram_fraction: float = 0.5) -> None:
    xp = select_backend(use_gpu, NAME)
    analysis = count_rare_codons_for_chunk(xp, genes, organism, winsize, vram_fraction=vram_fraction)
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
