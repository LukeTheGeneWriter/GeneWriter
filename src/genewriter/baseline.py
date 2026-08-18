"""Compute the *Analysis baseline dataclasses directly from a set of loaded
NaturalGene objects.

Fixed, consolidated ports of rc_analysis() (Rare_Codons.ipynb), cu_analysis()
(Codon_Usage.ipynb), cpb_analysis() (Codon_Pair_Bias.ipynb), gc_analysis()
(GC_Analysis.ipynb) and a k=2-only stand-in for kmer_analysis()
(Kmer_Analysis.ipynb -- the real one iterates k from 2 to 10 over the full
~18,864-gene reference set and its own logged run took over 10 hours just for
k=9, so it is not something to run against an arbitrary local gene set on
demand; a real KmerAnalysis baseline should still come from the precomputed
Standards/KmerAnalysis.json).

These are useful for local development and for validating the change-vector
math against real gene-body data (see gene_io.py), but a handful of sample
genes is not a substitute for the real ~18,864-gene genome-wide baseline the
Standards notebooks compute -- treat anything built here as a smoke test, not
a scientifically meaningful reference distribution.
"""

import json
import os

from .classes import CodonAnalysis, CodonPairBiasAnalysis, GCAnalysis, KmerAnalysis, RareCodonAnalysis
from .codon_tables import CODON_FREQS_LIT, RARE_CODONS_LIT, TAG_TO_BUCKET, TAG_TO_WINDOW_BUCKET
from .change_vector import AnalysisObjects, dist_from_optimal
from .gene_io import protein_coding_isoforms

_CPB_LIT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'cpb_lit.json')


def _load_cpb_lit() -> dict:
    with open(_CPB_LIT_PATH) as f:
        return json.load(f)


def compute_rare_codon_analysis(genes: list, organism: str = "human", winsize: int = 15) -> RareCodonAnalysis:
    rare_codon_windows = {i: 0 for i in range(0, winsize + 1)}
    rare_by_location = {'ExonL50': [0, 0], 'Exon': [0, 0], 'ExonR50': [0, 0], 'Splice': [0, 0]}
    usage_per_gene = []
    total_codons = 0

    for _gene, iso in protein_coding_isoforms(genes):
        codons = iso.codons
        total_codons += len(codons)
        n_rare = sum(1 for cod, _loc in codons if cod in RARE_CODONS_LIT)
        usage_per_gene.append(n_rare / len(codons))

        if len(codons) >= winsize:
            for i in range(0, len(codons) - winsize + 1):
                window = codons[i:i + winsize]
                n = sum(1 for cod, _loc in window if cod in RARE_CODONS_LIT)
                rare_codon_windows[n] = rare_codon_windows.get(n, 0) + 1

        for cod, loc in codons:
            bucket = TAG_TO_BUCKET.get(loc)
            if bucket is None:
                continue
            rare_by_location[bucket][1] += 1
            if cod in RARE_CODONS_LIT:
                rare_by_location[bucket][0] += 1

    return RareCodonAnalysis(
        organism=organism,
        transcriptome="local-sample",
        totalCodons=total_codons,
        rareCodonsByLocation=rare_by_location,
        rare_codon_windows=rare_codon_windows,
        codonFreqsLit=dict(CODON_FREQS_LIT),
        usagePerGene=usage_per_gene,
    )


def compute_codon_usage_analysis(genes: list, organism: str = "human", winsize: int = 15) -> CodonAnalysis:
    aa_freqs = {}
    codon_freqs_by_location = {c: {'ExonL50': 0, 'Exon': 0, 'ExonR50': 0, 'Splice': 0} for c in CODON_FREQS_LIT}
    usage_score_by_gene = []
    window_scores = []
    window_dist_from_optimal = []
    total_codons = 0

    for _gene, iso in protein_coding_isoforms(genes):
        for aa in iso.associatedProtein.aaSeq:
            aa_freqs[aa] = aa_freqs.get(aa, 0) + 1

        codons = iso.codons
        total_codons += len(codons)
        score = sum(CODON_FREQS_LIT[cod] for cod, _loc in codons) / len(codons)
        usage_score_by_gene.append(score)

        for cod, loc in codons:
            bucket = TAG_TO_BUCKET.get(loc)
            if bucket:
                codon_freqs_by_location[cod][bucket] += 1

        just_codons = [cod for cod, _loc in codons]
        # Full window count (len - winsize + 1), matching
        # compute_rare_codon_analysis's own loop above -- never drop the
        # mathematically valid last window. Used to stop one window short
        # (`max(len - winsize, 0)`) until 2026-08-19, an inherited,
        # unintended asymmetry with rare_codon's own convention that had no
        # deliberate reason behind it -- fixed here, and correspondingly in
        # gpu_codon_usage_count.py's counting-side equivalent.
        for i in range(0, len(just_codons) - winsize + 1):
            window = just_codons[i:i + winsize]
            window_score = sum(CODON_FREQS_LIT[c] for c in window) / winsize
            window_scores.append(window_score)
            window_dist_from_optimal.append(abs(dist_from_optimal(window) - window_score))

    return CodonAnalysis(
        organism=organism,
        transcriptome="local-sample",
        AAFreqs=aa_freqs,
        totalCodons=total_codons,
        codonFreqsByLocation=codon_freqs_by_location,
        codonFreqsLit=dict(CODON_FREQS_LIT),
        codonUsageScoreByGene=usage_score_by_gene,
        windowsize=winsize,
        windowscores=window_scores,
        windowdistancesfromoptimal=window_dist_from_optimal,
    )


def compute_codon_pair_bias_analysis(genes: list, organism: str = "human", winsize: int = 15) -> CodonPairBiasAnalysis:
    cpb_lit = _load_cpb_lit()
    per_gene = []
    per_window = []
    total_pairs = 0

    for _gene, iso in protein_coding_isoforms(genes):
        just_codons = [cod for cod, _loc in iso.codons]
        if len(just_codons) < 2:
            continue
        pairs = [just_codons[i] + just_codons[i + 1] for i in range(len(just_codons) - 1)]
        total_pairs += len(pairs)
        per_gene.append(sum(cpb_lit[p] for p in pairs) / len(pairs))

        # Full window count -- see compute_codon_usage_analysis's matching
        # comment above; same fix, same reasoning.
        for i in range(0, len(just_codons) - winsize + 1):
            window = just_codons[i:i + winsize]
            window_pairs = [window[j] + window[j + 1] for j in range(len(window) - 1)]
            if window_pairs:
                per_window.append(sum(cpb_lit[p] for p in window_pairs) / len(window_pairs))

    return CodonPairBiasAnalysis(
        organism=organism,
        transcriptome="local-sample",
        totalCodonPairs=total_pairs,
        cpbPerGene=per_gene,
        cpb_lit=cpb_lit,
        cpbPerWindow=per_window,
        windowsize=winsize,
    )


def compute_gc_analysis(genes: list, organism: str = "human", winsize: int = 21) -> GCAnalysis:
    tagged_gc1 = {'ExonL50': [], 'Exon': [], 'ExonR50': [], 'Splice': []}
    tagged_gc2 = {'ExonL50': [], 'Exon': [], 'ExonR50': [], 'Splice': []}
    tagged_gc3 = {'ExonL50': [], 'Exon': [], 'ExonR50': [], 'Splice': []}
    windows = {'ExonL50': [], 'Exon': [], 'ExonR50': []}
    gc_per_gene = []
    total_codons = 0

    for _gene, iso in protein_coding_isoforms(genes):
        if not iso.codons:
            continue
        gc_bases = sum(1 for cod, _loc in iso.codons for base in cod if base in 'GC')
        gc_per_gene.append(gc_bases / (3 * len(iso.codons)))

        for cod, loc in iso.codons:
            total_codons += 1
            bucket = TAG_TO_BUCKET.get(loc)
            if bucket is None:
                continue
            tagged_gc1[bucket].append(1 if cod[0] in 'GC' else 0)
            tagged_gc2[bucket].append(1 if cod[1] in 'GC' else 0)
            tagged_gc3[bucket].append(1 if cod[2] in 'GC' else 0)

        continuous = ''.join(cod for cod, _loc in iso.codons)
        loc_string = ''.join(loc * 3 for _cod, loc in iso.codons)
        # Full window count -- see compute_codon_usage_analysis's matching
        # comment above; same fix, same reasoning.
        for i in range(0, len(continuous) - winsize + 1):
            window = continuous[i:i + winsize]
            loc_window = loc_string[i:i + winsize]
            # Raw-tag-vote-then-fold: vote among the 4 raw tags (F/T/I/S)
            # first, fold only the winner through TAG_TO_WINDOW_BUCKET after.
            # NOT equivalent to gpu_change_vector.majority_window_bucket()'s
            # fold-then-vote (folds every nt's tag first, so S and I merge
            # into the same 'Exon' vote-bin before counting) -- the two
            # disagree whenever a window mixes S and I against an F/T
            # plurality. This function's raw-tag-vote-then-fold convention is
            # what gpu_corpus_batch.majority_raw_tag_bucket() reproduces for
            # the GPU-batched counting path; do not swap this loop's logic
            # for majority_window_bucket()'s without reading
            # memory/majority_vote_bucket_discrepancy.md first.
            majority = max(set(loc_window), key=loc_window.count)
            bucket = TAG_TO_WINDOW_BUCKET[majority]
            windows[bucket].append(sum(1 for ch in window if ch in 'GC') / winsize)

    return GCAnalysis(
        organism=organism,
        transcriptome="local-sample",
        totalCodons=total_codons,
        windows=windows,
        taggedGC1=tagged_gc1,
        taggedGC2=tagged_gc2,
        taggedGC3=tagged_gc3,
        windowsize=winsize,
        gcPerGene=gc_per_gene,
    )


def compute_kmer_analysis(genes: list, organism: str = "human", k_values=(2, 3)) -> KmerAnalysis:
    """Only k=2,3 by default -- see module docstring on why this isn't meant
    to substitute for the real Standards/KmerAnalysis.json baseline."""
    kmer_dict = {}
    l_dict = {}

    for k in k_values:
        counts = {}
        loc_totals = {'ExonL50': 0, 'Exon': 0, 'ExonR50': 0}
        for _gene, iso in protein_coding_isoforms(genes):
            continuous = ''.join(cod for cod, _loc in iso.codons)
            loc_string = ''.join(loc * 3 for _cod, loc in iso.codons)
            # Window count deliberately NOT changed to the "full" convention
            # compute_codon_usage_analysis/compute_codon_pair_bias_analysis/
            # compute_gc_analysis got 2026-08-19 -- kmer's own "one short"
            # convention is left as-is, a separate, not-yet-decided
            # follow-up (see memory/majority_vote_bucket_discrepancy.md).
            for i in range(0, max(len(continuous) - k, 0)):
                window = continuous[i:i + k]
                loc_window = loc_string[i:i + k]
                # Raw-tag-vote-then-fold -- same convention as
                # compute_gc_analysis's identical line above, NOT equivalent
                # to gpu_change_vector.majority_window_bucket()'s
                # fold-then-vote (which gpu_kmer_count.py's counting side
                # already uses, disagreeing with this function -- see
                # memory/majority_vote_bucket_discrepancy.md).
                majority = max(set(loc_window), key=loc_window.count)
                bucket = TAG_TO_WINDOW_BUCKET[majority]
                loc_totals[bucket] += 1
                entry = counts.setdefault(window, {'ExonL50': 0, 'Exon': 0, 'ExonR50': 0})
                entry[bucket] += 1

        # Fold enrichment vs. a uniform 1/4^k expectation per bucket.
        expected = {loc: total / (4 ** k) if total else 0.0 for loc, total in loc_totals.items()}
        scored = {}
        for seq, obs in counts.items():
            scored[seq] = {}
            for loc in ('ExonL50', 'Exon', 'ExonR50'):
                fold = obs[loc] / expected[loc] if expected[loc] else 1.0
                scored[seq][loc] = {'p': 1.0, 'fold_enrich': fold}
        kmer_dict[str(k)] = scored
        l_dict[str(k)] = loc_totals

    return KmerAnalysis(
        organism=organism,
        transcriptome="local-sample",
        kmer_dict=kmer_dict,
        l_dict=l_dict,
        kmer_score_obs={},
    )


def compute_baselines(genes: list, organism: str = "human") -> AnalysisObjects:
    return AnalysisObjects(
        rare_codon=compute_rare_codon_analysis(genes, organism),
        codon_usage=compute_codon_usage_analysis(genes, organism),
        codon_pair_bias=compute_codon_pair_bias_analysis(genes, organism),
        gc=compute_gc_analysis(genes, organism),
        kmer=compute_kmer_analysis(genes, organism),
    )
