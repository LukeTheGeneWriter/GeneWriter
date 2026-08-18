"""CodonPairBiasAnalysis baseline counting -- the mirror-image of
gpu_change_vector.batch_codon_pair_bias_term()'s scoring code. See
gpu_rare_codon_count.py's module docstring for the shared shape/rationale.

Operates on adjacent-CODON-PAIR scores, not per-codon ones -- an isoform of
N codons has N-1 pairs. Isoforms with fewer than 2 codons contribute no
pairs at all and are skipped entirely (matching
baseline.compute_codon_pair_bias_analysis's own `if len(just_codons) < 2:
continue`), not included as a zero/placeholder entry in cpbPerGene.

cpbPerWindow's windowing happens over the PAIR-score sequence, not the
codon sequence: a `winsize`-codon window contains `winsize - 1` pairs, so
the boundary mask (gpu_corpus_batch.valid_window_mask) is built with the
isoform spans expressed in PAIR-index terms (length - 1 pairs per isoform,
same flat offset as the codon-level start) and span=`winsize - 1`.
"""

import numpy as np

from .baseline import _load_cpb_lit
from .classes import CodonPairBiasAnalysis
from .codon_tables import CODON_LIST, CODON_TO_INDEX, encode_codons
from .gene_io import protein_coding_isoforms
from .gpu_change_vector import _sliding_window_view
from .gpu_corpus_batch import concat_isoform_batch, segment_mean, valid_window_mask, vram_aware_batch_size

_VRAM_SAFETY_FACTOR = 2
_DEFAULT_CPU_BATCH_CODONS = 5_000_000
_MIN_BATCH_CODONS = 100_000


def _build_cpb_table(cpb_lit: dict) -> np.ndarray:
    num_codons = len(CODON_LIST)
    table = np.zeros((num_codons, num_codons), dtype=float)
    for pair, score in cpb_lit.items():
        i, j = CODON_TO_INDEX[pair[:3]], CODON_TO_INDEX[pair[3:]]
        table[i, j] = score
    return table


def encode_isoform_codons(xp, iso) -> tuple:
    """Same shape as gpu_rare_codon_count.encode_isoform_codons -- codon
    granularity, untripled loc_string. Callers of this module must skip any
    isoform with fewer than 2 codons before encoding it (see module
    docstring) -- this function itself doesn't filter."""
    codon_idx = np.asarray(encode_codons([cod for cod, _loc in iso.codons]), dtype=np.int64)
    loc_string = ''.join(loc for _cod, loc in iso.codons)
    return (codon_idx if xp is np else xp.asarray(codon_idx)), loc_string


def _estimate_bytes_per_codon(winsize: int) -> int:
    return _VRAM_SAFETY_FACTOR * (8 * max(winsize - 1, 1) + 8 + 8)


def count_codon_pair_bias_for_batch(xp, codon_idx: 'np.ndarray', starts: list, lengths: list,
                                     winsize: int, cpb_table: 'np.ndarray') -> dict:
    """One VRAM sub-batch (every isoform already filtered to >=2 codons by
    the caller). Returns a dict:
      per_gene: list[float], one per isoform -- mean pair score.
      per_window: list[float], one per valid (winsize-1)-pair window.
      total_pairs: int -- count of valid (non-cross-isoform) pairs.
    """
    M = int(codon_idx.shape[0])
    table = xp.asarray(cpb_table)
    pair_scores = table[codon_idx[:-1], codon_idx[1:]]  # (M-1,)

    pair_lengths = [max(length - 1, 0) for length in lengths]
    n_isoforms = len(lengths)

    pair_valid = valid_window_mask(M, starts, lengths, span=2)  # (M-1,) codon-pair validity
    pair_valid_arr = pair_valid if xp is np else xp.asarray(pair_valid)
    valid_pair_scores = pair_scores[pair_valid_arr]
    total_pairs = int(pair_valid.sum())

    if n_isoforms and total_pairs:
        segment_ids = np.repeat(np.arange(n_isoforms), pair_lengths)
        means = segment_mean(xp, valid_pair_scores, segment_ids, n_isoforms)
        means = means if xp is np else xp.asnumpy(means)
        # Isoforms with 0 pairs (shouldn't occur -- caller filters length<2
        # isoforms out entirely) would otherwise appear as a spurious 0.0
        # entry; guard defensively anyway rather than assume the filter
        # upstream is airtight.
        per_gene = [m for m, pl in zip(means.tolist(), pair_lengths) if pl > 0]
    else:
        per_gene = []

    per_window: list = []
    span = max(winsize - 1, 1)
    max_start = (M - 1) - span + 1
    if max_start > 0 and n_isoforms:
        window_valid = valid_window_mask(M - 1, starts, pair_lengths, span=span)
        if window_valid.any():
            window_valid_arr = window_valid if xp is np else xp.asarray(window_valid)
            win_means_all = _sliding_window_view(xp, pair_scores, span, axis=0)[:max_start].mean(axis=1)
            win_means_valid = win_means_all[window_valid_arr]
            win_means_valid = win_means_valid if xp is np else xp.asnumpy(win_means_valid)
            per_window = win_means_valid.tolist()

    return {'per_gene': per_gene, 'per_window': per_window, 'total_pairs': total_pairs}


def count_codon_pair_bias_for_chunk(xp, genes: list, organism: str = "human",
                                     winsize: int = 15, vram_fraction: float = 0.5) -> CodonPairBiasAnalysis:
    """One in-memory chunk of NaturalGene objects. Isoforms with fewer than
    2 codons are skipped entirely before ever being batched (matching
    baseline.compute_codon_pair_bias_analysis's own skip)."""
    cpb_lit = _load_cpb_lit()
    cpb_table = _build_cpb_table(cpb_lit)

    per_gene: list = []
    per_window: list = []
    total_pairs = 0

    max_batch_codons = vram_aware_batch_size(
        xp, _estimate_bytes_per_codon(winsize), vram_fraction=vram_fraction,
        default_cpu_batch=_DEFAULT_CPU_BATCH_CODONS, min_batch=_MIN_BATCH_CODONS,
    )

    def _flush(batch):
        if not batch:
            return
        codon_idx, _loc_string, starts, lengths = concat_isoform_batch(xp, batch)
        result = count_codon_pair_bias_for_batch(xp, codon_idx, starts, lengths, winsize, cpb_table)
        per_gene.extend(result['per_gene'])
        per_window.extend(result['per_window'])
        nonlocal total_pairs
        total_pairs += result['total_pairs']

    batch = []
    batch_codons = 0
    for _gene, iso in protein_coding_isoforms(genes):
        if len(iso.codons) < 2:
            continue
        encoded = encode_isoform_codons(xp, iso)
        n = int(encoded[0].shape[0])
        if batch and batch_codons + n > max_batch_codons:
            _flush(batch)
            batch = []
            batch_codons = 0
        batch.append(encoded)
        batch_codons += n
    _flush(batch)

    return CodonPairBiasAnalysis(
        organism=organism,
        transcriptome="local-sample",  # matches baseline.compute_codon_pair_bias_analysis exactly -- see gpu_rare_codon_count.py's matching comment
        totalCodonPairs=total_pairs,
        cpbPerGene=per_gene,
        cpb_lit=cpb_lit,
        cpbPerWindow=per_window,
        windowsize=winsize,
    )
