"""Amino-acid subsequence ("motif") frequency index over a loaded
proteome, and a report/query tool for "how common is this motif overall".

Unlike codon_ngram.py's context model, this index is NOT restricted to
interior ('I'-tagged) positions -- it counts every occurrence across the
WHOLE amino acid sequence of every protein_coding_isoforms() isoform,
because the intent here is an intuitive, unrestricted "how common is this
motif in the human proteome" count, not a codon-choice model. It also only
ever reads iso.associatedProtein.aaSeq, never iso.codons -- so, unlike
codon_ngram.py, this module has NO dependency on iso.codons/aaSeq being
positionally aligned, and can be built and tested against real gene JSON
data as-is (its counts just aren't a scientifically meaningful proteome-
wide reference from a handful of local sample genes -- same "smoke test"
caveat baseline.py already carries for anything built from Gene_Obj_Samples/).

k=2 motifs are deliberately excluded (see DEFAULT_MOTIF_K_VALUES) --
already covered by the existing CodonPairBias change-vector term
(adjacent-codon bias), which this module doesn't replace or overlap with.

Why a dict/Counter-based index rather than a suffix array or an aligner:
query lengths are only ever within the fixed k range this index was built
for, not arbitrary-length -- a suffix array's main advantage (arbitrary-
length substring queries after one build) buys nothing here, at real extra
implementation complexity. A real aligner (BLAST/Smith-Waterman, even
GPU-accelerated) is the wrong tool for a different reason: those are built
for approximate/gapped matching, and this feature wants exact substring
counts.
"""

from collections import Counter

from .classes import AAMotifIndex
from .codon_tables import STANDARD_AMINO_ACIDS, validate_aa_sequence
from .gene_io import protein_coding_isoforms

# k=2 excluded on purpose -- see module docstring.
DEFAULT_MOTIF_K_VALUES = tuple(range(3, 11))  # 3..10 inclusive


def build_aa_motif_index(
    genes: list,
    organism: str = "human",
    k_values: tuple = DEFAULT_MOTIF_K_VALUES,
    progress_every: int = None,
) -> AAMotifIndex:
    """Count every length-k amino-acid substring occurrence (for each k in
    k_values) across every protein_coding_isoforms() isoform's
    associatedProtein.aaSeq, unrestricted by location tag.

    Raises ValueError if 2 is in k_values (see module docstring).
    A residue outside STANDARD_AMINO_ACIDS (e.g. 'X' for an
    ambiguous/unresolved position in real NCBI protein data) simply can't
    form part of any motif key containing it -- windows spanning such a
    residue are skipped, not treated as a crash condition, since real
    proteome data legitimately contains these.

    progress_every: if set, print elapsed-time progress every this-many
    isoforms scanned -- diagnostic only, no effect on the returned index.
    Real-scale cost against the full ~18,864-gene set is unmeasured (see
    Handoff.md); this is here so a slow build is visible rather than guessed
    at, matching this codebase's established instrumentation pattern
    (seed_population, batch_calculate_change_vectors).
    """
    k_values = tuple(sorted(k_values))
    if 2 in k_values:
        raise ValueError("k=2 motifs are excluded -- already covered by the CodonPairBias change-vector term")
    if not k_values:
        raise ValueError("k_values must not be empty")

    if progress_every:
        import time as _time
        t0 = _time.perf_counter()

    max_k = max(k_values)
    counts = {k: Counter() for k in k_values}
    totals = {k: 0 for k in k_values}

    n_isoforms = 0
    for _gene, iso in protein_coding_isoforms(genes):
        seq = iso.associatedProtein.aaSeq
        for i in range(len(seq)):
            window = seq[i:i + max_k]
            if len(window) < min(k_values):
                break
            for k in k_values:
                if len(window) < k:
                    continue
                motif = window[:k]
                if any(ch not in STANDARD_AMINO_ACIDS for ch in motif):
                    continue
                counts[k][motif] += 1
                totals[k] += 1

        n_isoforms += 1
        if progress_every and n_isoforms % progress_every == 0:
            elapsed = _time.perf_counter() - t0
            print(f"  [build_aa_motif_index] {n_isoforms} isoforms scanned "
                  f"({n_isoforms / elapsed:.1f} isoforms/s)", flush=True)

    return AAMotifIndex(
        organism=organism,
        transcriptome="local-sample",
        k_values=k_values,
        motif_counts={str(k): dict(counts[k]) for k in k_values},
        total_windows={str(k): totals[k] for k in k_values},
    )


def query_motif_count(index: AAMotifIndex, substring: str) -> int:
    """Raw occurrence count for `substring` (validated + uppercased via
    codon_tables.validate_aa_sequence()). Raises ValueError if
    len(substring) is not one of index.k_values."""
    substring = validate_aa_sequence(substring)
    k = len(substring)
    if k not in index.k_values:
        raise ValueError(
            f"Query length {k} is not in this index's k_values {index.k_values} -- "
            f"build_aa_motif_index() was not asked to track motifs of this length."
        )
    return index.motif_counts[str(k)].get(substring, 0)


def report_motif_occurrences(index: AAMotifIndex, substring: str, proteome_label: str = None) -> str:
    """Prints and returns:
        Found {count} occurrences of substring "{substring}" in the {label} proteome
    proteome_label defaults to index.organism."""
    substring = validate_aa_sequence(substring)
    count = query_motif_count(index, substring)
    label = proteome_label if proteome_label is not None else index.organism
    message = f'Found {count} occurrences of substring "{substring}" in the {label} proteome'
    print(message)
    return message
