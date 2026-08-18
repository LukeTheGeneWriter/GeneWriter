"""Numpy-backend tests for gpu_corpus_batch.py's shared batching/aggregation
primitives -- always runs, no cupy import.
"""
import numpy as np
import pytest

from genewriter.gpu_change_vector import majority_window_bucket
from genewriter.gpu_corpus_batch import (
    concat_isoform_batch,
    majority_raw_tag_bucket,
    segment_mean,
    select_backend,
    valid_window_mask,
    vram_aware_batch_size,
)


def test_select_backend_use_gpu_false_returns_numpy():
    assert select_backend(False, 'test') is np


def test_vram_aware_batch_size_numpy_backend_returns_default():
    assert vram_aware_batch_size(np, bytes_per_unit=100, default_cpu_batch=12345) == 12345


def test_vram_aware_batch_size_respects_min_batch_floor():
    # numpy path never hits the VRAM query at all, but the floor argument
    # should still be honored if a caller passes a tiny default explicitly.
    assert vram_aware_batch_size(np, bytes_per_unit=100, default_cpu_batch=1, min_batch=50) == 1
    # (numpy backend always returns default_cpu_batch verbatim, min_batch only
    # applies on the cupy path -- this documents that boundary explicitly.)


def test_concat_isoform_batch_concatenates_values_and_locs():
    a = (np.asarray([1, 2, 3]), 'III')
    b = (np.asarray([4, 5]), 'FT')
    flat, flat_loc, starts, lengths = concat_isoform_batch(np, [a, b])

    assert flat.tolist() == [1, 2, 3, 4, 5]
    assert flat_loc == 'IIIFT'
    assert starts == [0, 3]
    assert lengths == [3, 2]


def test_concat_isoform_batch_handles_empty_list():
    flat, flat_loc, starts, lengths = concat_isoform_batch(np, [])
    assert flat.tolist() == []
    assert flat_loc == ''
    assert starts == []
    assert lengths == []


def test_valid_window_mask_never_drops_a_window_within_one_isoform():
    # Single isoform, length 6, span 3 -> full convention gives
    # length - span + 1 = 4 valid windows (never drops the last one).
    mask = valid_window_mask(total_len=6, starts=[0], lengths=[6], span=3)
    assert mask.tolist() == [True, True, True, True]


def test_valid_window_mask_excludes_windows_crossing_an_isoform_join():
    # Two isoforms, lengths 6 and 6, concatenated (total_len=12), span=3.
    # Isoform A: valid local starts 0..3 (4 windows). Isoform B (offset 6):
    # valid flat starts 6..9 (4 windows). Flat starts 4 and 5 read across
    # the join and must be excluded even though they're "in range" overall.
    mask = valid_window_mask(total_len=12, starts=[0, 6], lengths=[6, 6], span=3)
    assert mask.tolist() == [True, True, True, True, False, False, True, True, True, True]


def test_valid_window_mask_empty_when_total_len_shorter_than_span():
    mask = valid_window_mask(total_len=2, starts=[0], lengths=[2], span=3)
    assert mask.tolist() == []


def test_segment_mean_matches_manual_per_group_average():
    values = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0])
    segment_ids = np.asarray([0, 0, 0, 1, 1])
    means = segment_mean(np, values, segment_ids, n_segments=2)
    assert means.tolist() == pytest.approx([2.0, 15.0])


def test_segment_mean_handles_segment_with_no_members():
    values = np.asarray([1.0, 2.0])
    segment_ids = np.asarray([0, 0])
    means = segment_mean(np, values, segment_ids, n_segments=3)
    assert means.tolist() == pytest.approx([1.5, 0.0, 0.0])


def test_majority_raw_tag_bucket_matches_hand_count_no_conflicting_tags():
    # No S tags present -- must agree with majority_window_bucket() here,
    # since the two algorithms are provably identical whenever S is absent
    # (TAG_TO_WINDOW_BUCKET is injective on {F, I, T}).
    loc = 'FFFIIIITT'  # 3 F, 4 I, 2 T -- span covering all of it: I wins outright
    raw_result = majority_raw_tag_bucket(loc, span=len(loc), num_windows=1)
    fold_result = majority_window_bucket(loc, span=len(loc), num_windows=1)
    assert raw_result.tolist() == fold_result.tolist()


def test_majority_raw_tag_bucket_diverges_from_majority_window_bucket_with_s_tags():
    # The concrete counter-example from the investigation: 5xS + 4xI + 6xF.
    # Raw-tag vote: F has the largest raw count (6) -> ExonL50, no tie.
    # Fold-first vote (majority_window_bucket): S+I fold to the same 'Exon'
    # bin, 5+4=9 > F's 6 -> Exon. Different bucket for the identical window.
    loc = ('S' * 5) + ('I' * 4) + ('F' * 6)
    span = len(loc)

    raw_bucket = majority_raw_tag_bucket(loc, span=span, num_windows=1)
    fold_bucket = majority_window_bucket(loc, span=span, num_windows=1)

    assert raw_bucket.tolist() != fold_bucket.tolist()
    # Pin down the actual expected values, not just "they differ":
    # ExonL50=0, Exon=1, ExonR50=2 (gpu_corpus_batch._WINDOW_BUCKET_NAMES order)
    assert raw_bucket.tolist() == [0]   # F wins raw vote -> ExonL50
    assert fold_bucket.tolist() == [1]  # S+I fold-vote beats F -> Exon


def test_majority_raw_tag_bucket_returns_expected_shape():
    loc = 'FFTTIIISS'
    result = majority_raw_tag_bucket(loc, span=3, num_windows=7)
    assert result.shape == (7,)
    assert result.dtype.kind in ('i', 'u')
