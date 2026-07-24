import dataclasses
import json
import os

import pytest

from genewriter import standards_io


def test_load_standards_round_trips_through_json_and_normalizes_int_keys(
    tmp_path, rare_codon_analysis, codon_usage_analysis, codon_pair_bias_analysis, gc_analysis, kmer_analysis
):
    """json.dump/json.load turns RareCodonAnalysis.rare_codon_windows' int
    keys into strings -- exactly what saveRareCodonAnalysis()/the Standards
    notebooks do in production. Confirm load_standards() hands back int
    keys, matching what change_vector.py's _rare_codon_term expects."""
    objects = {
        'RareCodonAnalysis.json': rare_codon_analysis,
        'CodonUsageAnalysis.json': codon_usage_analysis,
        'CodonPairBiasAnalysis.json': codon_pair_bias_analysis,
        'GCAnalysis.json': gc_analysis,
        'KmerAnalysis.json': kmer_analysis,
    }
    for filename, obj in objects.items():
        with open(tmp_path / filename, 'w') as f:
            json.dump(dataclasses.asdict(obj), f)

    loaded = standards_io.load_standards(str(tmp_path))

    assert all(isinstance(k, int) for k in loaded.rare_codon.rare_codon_windows)
    assert loaded.codon_usage.windowsize == codon_usage_analysis.windowsize
    assert loaded.gc.taggedGC1.keys() == gc_analysis.taggedGC1.keys()


def test_load_standards_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="RareCodonAnalysis.json"):
        standards_io.load_standards(str(tmp_path))
