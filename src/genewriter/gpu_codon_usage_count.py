"""CodonAnalysis (codon usage) baseline counting -- the mirror-image of
gpu_change_vector.batch_codon_usage_term()'s scoring code. See
gpu_rare_codon_count.py's module docstring for the shared shape/rationale
(codon granularity, VRAM-aware batching via gpu_corpus_batch.py, the full
never-drop-a-window convention).

Not a literal reuse of batch_codon_usage_term()'s array ops: that function
solves a different problem (windowed scores smeared back onto per-position
GA-candidate scores, with its own `span = windowsize - 1` convention for
that purpose) -- this module instead mirrors
baseline.compute_codon_usage_analysis's own window loop directly (span ==
winsize, one aggregate value appended per window, not smeared).
"""

import numpy as np

from .baseline import accumulate_codon_usage_percent_by_aa
from .classes import CodonAnalysis
from .codon_tables import CODON_FREQS_LIT, CODON_LIST, TAG_TO_BUCKET, encode_codons
from .gene_io import protein_coding_isoforms
from .gpu_change_vector import BEST_FREQ_BY_INDEX, _sliding_window_view
from .gpu_corpus_batch import concat_isoform_batch, segment_mean, valid_window_mask, vram_aware_batch_size

_CODON_FREQ_TABLE = np.asarray([CODON_FREQS_LIT[c] for c in CODON_LIST], dtype=float)
_BEST_FREQ_TABLE = np.asarray(BEST_FREQ_BY_INDEX, dtype=float)
_BUCKET_NAMES_4WAY = ('ExonL50', 'ExonR50', 'Exon', 'Splice')
_BUCKET_INDEX_4WAY = {name: i for i, name in enumerate(_BUCKET_NAMES_4WAY)}
_TAG_TO_BUCKET_INDEX = {tag: _BUCKET_INDEX_4WAY[bucket] for tag, bucket in TAG_TO_BUCKET.items()}

_VRAM_SAFETY_FACTOR = 2
_DEFAULT_CPU_BATCH_CODONS = 5_000_000
_MIN_BATCH_CODONS = 100_000


def encode_isoform_codons(xp, iso) -> tuple:
    """Same shape as gpu_rare_codon_count.encode_isoform_codons -- codon
    granularity, untripled loc_string."""
    codon_idx = np.asarray(encode_codons([cod for cod, _loc in iso.codons]), dtype=np.int64)
    loc_string = ''.join(loc for _cod, loc in iso.codons)
    return (codon_idx if xp is np else xp.asarray(codon_idx)), loc_string


def _estimate_bytes_per_codon(winsize: int) -> int:
    """Same reasoning as gpu_rare_codon_count._estimate_bytes_per_codon --
    dominant term is the sliding-window views over the (two) per-codon
    float arrays, each 8*winsize bytes/position, doubled for margin."""
    return _VRAM_SAFETY_FACTOR * (2 * 8 * winsize + 8 + 8)


def count_codon_usage_for_batch(xp, codon_idx: 'np.ndarray', loc_string: str,
                                 starts: list, lengths: list, winsize: int) -> dict:
    """One VRAM sub-batch. Returns a dict:
      codon_freqs_by_location: dict[str, dict[str, int]] -- codon -> 4-way
        bucket -> count, per-codon (not windowed).
      usage_score_by_gene: list[float], one per isoform.
      window_scores: list[float], one per valid window (full convention).
      window_dist_from_optimal: list[float], same length/order as window_scores.
      total_codons: int.
    """
    M = int(codon_idx.shape[0])
    freq_table = xp.asarray(_CODON_FREQ_TABLE)
    best_freq_table = xp.asarray(_BEST_FREQ_TABLE)
    cuvec = freq_table[codon_idx]  # (M,)
    best_freq = best_freq_table[codon_idx]  # (M,)

    n_isoforms = len(lengths)
    if n_isoforms:
        segment_ids = np.repeat(np.arange(n_isoforms), lengths)
        means = segment_mean(xp, cuvec, segment_ids, n_isoforms)
        means = means if xp is np else xp.asnumpy(means)
        usage_score_by_gene = means.tolist()
    else:
        usage_score_by_gene = []

    window_scores: list = []
    window_dist_from_optimal: list = []
    max_start = M - winsize + 1
    if max_start > 0:
        valid = valid_window_mask(M, starts, lengths, winsize)
        if valid.any():
            valid_arr = valid if xp is np else xp.asarray(valid)
            win_scores_all = _sliding_window_view(xp, cuvec, winsize, axis=0)[:max_start].mean(axis=1)
            win_best_all = _sliding_window_view(xp, best_freq, winsize, axis=0)[:max_start].mean(axis=1)
            win_scores_valid = win_scores_all[valid_arr]
            win_dist_valid = xp.abs(win_best_all[valid_arr] - win_scores_valid)
            win_scores_valid = win_scores_valid if xp is np else xp.asnumpy(win_scores_valid)
            win_dist_valid = win_dist_valid if xp is np else xp.asnumpy(win_dist_valid)
            window_scores = win_scores_valid.tolist()
            window_dist_from_optimal = win_dist_valid.tolist()

    codon_idx_np = codon_idx if xp is np else xp.asnumpy(codon_idx)
    bucket_of_codon = np.fromiter(
        (_TAG_TO_BUCKET_INDEX[c] for c in loc_string), dtype=np.int64, count=len(loc_string),
    ) if loc_string else np.zeros(0, dtype=np.int64)
    flat_index = codon_idx_np * 4 + bucket_of_codon
    counts_flat = np.bincount(flat_index, minlength=64 * 4).reshape(64, 4)
    codon_freqs_by_location = {
        CODON_LIST[i]: {name: int(counts_flat[i, b]) for b, name in enumerate(_BUCKET_NAMES_4WAY)}
        for i in range(64)
    }

    return {
        'codon_freqs_by_location': codon_freqs_by_location,
        'usage_score_by_gene': usage_score_by_gene,
        'window_scores': window_scores,
        'window_dist_from_optimal': window_dist_from_optimal,
        'total_codons': M,
    }


def count_codon_usage_for_chunk(xp, genes: list, organism: str = "human",
                                 winsize: int = 15, vram_fraction: float = 0.5) -> CodonAnalysis:
    """One in-memory chunk of NaturalGene objects -- same accumulation shape
    as gpu_rare_codon_count.count_rare_codons_for_chunk. AAFreqs stays a
    plain Python dict accumulation over aaSeq strings (cheap, string-based,
    not GPU-shaped work) -- codonUsagePercentByAA's accumulation
    (accumulate_codon_usage_percent_by_aa(), imported from baseline.py) is
    the same shape: a per-gene Python-level pass over that gene's own
    codons, not GPU-batched. Deliberate, not an oversight -- this stat
    scales with total gene count * amino acid count (tens of thousands),
    not with total corpus nucleotide/codon count the way the VRAM-batched
    window/frequency stats below do (millions), so it's nowhere near the
    scale that motivated GPU-batching those in the first place; a plain
    Python loop here is fast enough, and reuses the exact same per-isoform
    iteration this function already does for aa_freqs."""
    aa_freqs: dict = {}
    codon_freqs_by_location = {c: {name: 0 for name in _BUCKET_NAMES_4WAY} for c in CODON_LIST}
    usage_score_by_gene: list = []
    window_scores: list = []
    window_dist_from_optimal: list = []
    codon_usage_percent_by_aa: dict = {}
    total_codons = 0

    max_batch_codons = vram_aware_batch_size(
        xp, _estimate_bytes_per_codon(winsize), vram_fraction=vram_fraction,
        default_cpu_batch=_DEFAULT_CPU_BATCH_CODONS, min_batch=_MIN_BATCH_CODONS,
    )

    def _flush(batch):
        if not batch:
            return
        codon_idx, loc_string, starts, lengths = concat_isoform_batch(xp, batch)
        result = count_codon_usage_for_batch(xp, codon_idx, loc_string, starts, lengths, winsize)
        usage_score_by_gene.extend(result['usage_score_by_gene'])
        window_scores.extend(result['window_scores'])
        window_dist_from_optimal.extend(result['window_dist_from_optimal'])
        for codon, by_bucket in result['codon_freqs_by_location'].items():
            for name, n in by_bucket.items():
                codon_freqs_by_location[codon][name] += n
        nonlocal total_codons
        total_codons += result['total_codons']

    batch = []
    batch_codons = 0
    for _gene, iso in protein_coding_isoforms(genes):
        for aa in iso.associatedProtein.aaSeq:
            aa_freqs[aa] = aa_freqs.get(aa, 0) + 1
        accumulate_codon_usage_percent_by_aa(iso.codons, codon_usage_percent_by_aa)
        encoded = encode_isoform_codons(xp, iso)
        n = int(encoded[0].shape[0])
        if batch and batch_codons + n > max_batch_codons:
            _flush(batch)
            batch = []
            batch_codons = 0
        batch.append(encoded)
        batch_codons += n
    _flush(batch)

    return CodonAnalysis(
        organism=organism,
        transcriptome="local-sample",  # matches baseline.compute_codon_usage_analysis exactly -- see gpu_rare_codon_count.py's matching comment
        AAFreqs=aa_freqs,
        totalCodons=total_codons,
        codonFreqsByLocation=codon_freqs_by_location,
        codonFreqsLit=dict(CODON_FREQS_LIT),
        codonUsageScoreByGene=usage_score_by_gene,
        windowsize=winsize,
        windowscores=window_scores,
        windowdistancesfromoptimal=window_dist_from_optimal,
        codonUsagePercentByAA=codon_usage_percent_by_aa,
    )
