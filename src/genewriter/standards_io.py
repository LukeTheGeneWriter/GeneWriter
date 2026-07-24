"""Load the precomputed genome-wide baseline analyses ("Standards") from disk.

Ported from standards_check() in GeneRider_Cloud.ipynb. The original reads
from a Google-Drive-mounted `Standards/` folder and, if a file is missing,
triggers %run of the corresponding analysis notebook (Rare_Codons.ipynb,
Codon_Usage.ipynb, Codon_Pair_Bias.ipynb, GC_Analysis.ipynb,
Kmer_Analysis.ipynb) to regenerate it -- each of those takes hours to run
over the ~18,864-gene reference set. This version only loads from a local
directory and raises FileNotFoundError with a clear message if a baseline is
missing, rather than silently kicking off an hours-long recompute.

json.dump/json.load round-trips every dict key through str(), so integer
keys (RareCodonAnalysis.rare_codon_windows is keyed by rare-codon count,
0..window size) come back as strings. Normalized back to int here so
downstream code (change_vector.py) doesn't have to guess the key type.
"""

import json
import os

from .classes import CodonAnalysis, CodonPairBiasAnalysis, GCAnalysis, KmerAnalysis, RareCodonAnalysis
from .change_vector import AnalysisObjects

_FILES = {
    'rare_codon': ('RareCodonAnalysis.json', RareCodonAnalysis),
    'codon_usage': ('CodonUsageAnalysis.json', CodonAnalysis),
    'codon_pair_bias': ('CodonPairBiasAnalysis.json', CodonPairBiasAnalysis),
    'gc': ('GCAnalysis.json', GCAnalysis),
    'kmer': ('KmerAnalysis.json', KmerAnalysis),
}


def _load_one(standards_dir: str, filename: str, cls):
    path = os.path.join(standards_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing baseline analysis: {path}. Run the corresponding "
            f"Standards notebook (or copy an existing {filename}) before "
            f"calling load_standards()."
        )
    with open(path) as f:
        data = json.load(f)
    return cls(**data)


def load_standards(standards_dir: str) -> AnalysisObjects:
    kwargs = {key: _load_one(standards_dir, filename, cls) for key, (filename, cls) in _FILES.items()}

    kwargs['rare_codon'].rare_codon_windows = {
        int(k): v for k, v in kwargs['rare_codon'].rare_codon_windows.items()
    }

    return AnalysisObjects(**kwargs)
