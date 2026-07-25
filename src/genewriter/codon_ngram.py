"""Codon-choice model conditioned on local amino-acid context, learned from
natural genes' interior ('I'-tagged) codon positions -- an alternative to
ga.generate_seed()'s pure-uniform-random per-residue synonymous choice for
GA initial-population seeding.

DATA-QUALITY DEPENDENCY: training here reads each isoform's `codons` list
(codon + location tag pairs) and translates each codon to its own amino
acid directly (via codon_tables.CODON_TO_AA) to build the context key --
it does NOT cross-reference iso.associatedProtein.aaSeq. This sidesteps
the codons<->aaSeq positional-alignment bug that used to exist in
GeneDataSourcing.ipynb's check_cds_against_protein() (now fixed there) --
training here can't crash or silently misalign from that specific bug,
because it only ever needs `codons` to be internally self-consistent
(each entry's own codon/tag pair), which `locate_codons()` already
guarantees regardless of whether the atg_start it was given was the
*biologically correct* one.

That said, this does NOT make the model scientifically meaningful against
data generated before the notebook fix: if atg_start was wrong, `codons`
walks the wrong reading frame -- self-consistent as codon/context pairs,
but not real translated protein sequence, so the model would learn
context->codon associations from frameshifted noise. The practical
failure mode is graceful (a model trained on frame-shifted noise mostly
just fails to match anything when queried against real amino-acid
sequences later, falling back to uniform-random -- see
sample_codon_ngram_seed()), not actively wrong, but it is not a
substitute for regenerating gene data with the notebook fix in place.
Do not treat a model built from currently-available gene data as a
validated result.

Restriction to 'I'-tagged windows is the DEFAULT (restrict_to_bucket='I')
and applies ONLY to this module, not to aa_motif_index.py's report tool.
A caller can pass restrict_to_bucket=None to train on every codon
regardless of location tag -- kept general (not hardcoded to 'I') for a
forward-looking reason: a future feature lets a caller mark custom
per-position location-tag overrides (e.g. "insert an artificial intron
here"), and this module's API shouldn't preclude conditioning
training/sampling on something other than a flat all-'I' assumption later.

Back-off: to predict the codon at position i, sample_codon_ngram_seed()
tries model.context_orders in descending order (longest first). For a
given k, the context is aa_seq[i-k+1:i+1] (the k-1 preceding residues plus
the current one); it is usable only if it was observed at least
min_observations times in training. If no k (down to k=1, "condition on
the current residue's own identity only") meets the threshold, this falls
back to ga.generate_seed()'s own behavior: uniform random among
codon_tables.codon_choices_for_aa(aa). This guarantees
sample_codon_ngram_seed() always returns a valid codon for every residue,
even against an empty/undertrained model.
"""

import dataclasses
import json
import random

from .classes import CodonNgramModel
from .codon_tables import CODON_TO_AA, STANDARD_AMINO_ACIDS, codon_choices_for_aa
from .gene_io import protein_coding_isoforms

DEFAULT_NGRAM_CONTEXT_ORDERS = (6, 5, 4, 3, 2, 1)  # descending, longest first
DEFAULT_MIN_OBSERVATIONS = 5


def build_codon_ngram_model(
    genes: list,
    organism: str = "human",
    context_orders: tuple = DEFAULT_NGRAM_CONTEXT_ORDERS,
    restrict_to_bucket: str = 'I',
    progress_every: int = None,
) -> CodonNgramModel:
    """Train context_counts from protein_coding_isoforms(genes). See module
    docstring for the alignment/data-quality caveat and the interior
    restriction.

    context_orders must be given strictly descending (longest first) --
    raises ValueError otherwise, since sample_codon_ngram_seed() relies on
    iterating it in that order without re-sorting every call.

    A window contributes to order k only if every codon in that k-length
    span (a) translates to a STANDARD_AMINO_ACIDS entry (real proteome data
    occasionally has non-standard codes) and (b) -- when restrict_to_bucket
    is not None -- is tagged restrict_to_bucket. Tracked via a running
    "consecutive valid positions" counter rather than rescanning the last k
    positions once per k, so this is one pass per isoform, not O(isoform
    length * len(context_orders)).

    progress_every: if set, print elapsed-time progress every this-many
    isoforms scanned -- diagnostic only, see aa_motif_index.build_aa_motif_index
    for the identical pattern and its rationale.
    """
    context_orders = tuple(context_orders)
    if list(context_orders) != sorted(context_orders, reverse=True):
        raise ValueError(f"context_orders must be strictly descending (longest first), got {context_orders}")
    if len(set(context_orders)) != len(context_orders):
        raise ValueError(f"context_orders must not contain duplicates, got {context_orders}")
    if not context_orders:
        raise ValueError("context_orders must not be empty")

    if progress_every:
        import time as _time
        t0 = _time.perf_counter()

    context_counts = {str(k): {} for k in context_orders}

    n_isoforms = 0
    for _gene, iso in protein_coding_isoforms(genes):
        codon_pairs = iso.codons
        n = len(codon_pairs)
        aas = [CODON_TO_AA.get(codon.upper()) for codon, _loc in codon_pairs]

        run_length = 0
        for i in range(n):
            codon, loc = codon_pairs[i]
            aa = aas[i]
            is_valid = (
                aa is not None and aa in STANDARD_AMINO_ACIDS
                and (restrict_to_bucket is None or loc == restrict_to_bucket)
            )
            if not is_valid:
                run_length = 0
                continue
            run_length += 1

            codon = codon.upper()
            for k in context_orders:
                if k > run_length:
                    continue
                context = ''.join(aas[i - k + 1:i + 1])
                bucket = context_counts[str(k)].setdefault(context, {})
                bucket[codon] = bucket.get(codon, 0) + 1

        n_isoforms += 1
        if progress_every and n_isoforms % progress_every == 0:
            elapsed = _time.perf_counter() - t0
            print(f"  [build_codon_ngram_model] {n_isoforms} isoforms scanned "
                  f"({n_isoforms / elapsed:.1f} isoforms/s)", flush=True)

    return CodonNgramModel(
        organism=organism,
        transcriptome="local-sample",
        context_orders=context_orders,
        context_counts=context_counts,
    )


def _select_context(aa_seq: str, i: int, model: CodonNgramModel, min_observations: int) -> tuple:
    """Pure, RNG-free back-off decision -- separated from sampling so the
    back-off logic itself is directly unit-testable.

    Returns (k_used, counts_dict) for the longest k in model.context_orders
    whose context aa_seq[i-k+1:i+1] was observed with total count >=
    min_observations, or (None, None) if no such k exists (caller falls
    back to uniform-random)."""
    for k in model.context_orders:
        lo = i - k + 1
        if lo < 0:
            continue
        context = aa_seq[lo:i + 1]
        counts = model.context_counts.get(str(k), {}).get(context)
        if counts and sum(counts.values()) >= min_observations:
            return k, counts
    return None, None


def _sample_from_counts(counts: dict, rng: random.Random) -> str:
    keys = list(counts.keys())
    weights = list(counts.values())
    return rng.choices(keys, weights=weights)[0]


def sample_codon_ngram_seed(
    aa_seq: str,
    model: CodonNgramModel,
    locvec: list = None,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    rng: random.Random = None,
) -> list:
    """Drop-in alternative to ga.generate_seed(aa_seq) for GA initial-
    population seeding: samples each position's codon from `model`'s
    learned context-conditioned distribution (with back-off, see module
    docstring) instead of uniform-random among synonyms.

    aa_seq: uppercase amino acid string -- not validated here (callers
        building a custom target should call codon_tables.
        validate_aa_sequence() first, same as any other aa_seq consumer).
    locvec: optional per-position 'F'/'T'/'I'/'S' tags for the sequence
        being generated (distinct from the training data's own tags,
        already baked into `model`). Accepted but unused today -- reserved
        so this signature doesn't need to change later when per-position
        location-tag overrides (e.g. marking an artificial-intron insertion
        point) need to gate which positions this sampler may draw
        context-conditioned codons for at all.
    rng: random.Random instance for reproducible sampling (tests pass a
        seeded one); defaults to a fresh, unseeded random.Random().

    Returns: list of codon strings, one per residue of aa_seq -- exactly
    ga.generate_seed()'s return shape.
    """
    if rng is None:
        rng = random.Random()
    aa_seq = aa_seq.upper()

    result = []
    for i, aa in enumerate(aa_seq):
        _, counts = _select_context(aa_seq, i, model, min_observations)
        if counts:
            result.append(_sample_from_counts(counts, rng))
        else:
            result.append(rng.choice(codon_choices_for_aa(aa)))
    return result


def save_codon_ngram_model(model: CodonNgramModel, path: str) -> None:
    """JSON dump, mirroring standards_io.py's dataclass<->JSON pattern --
    simpler here, context_counts is string-keyed all the way down (no
    int-key normalization needed on load, unlike
    RareCodonAnalysis.rare_codon_windows)."""
    with open(path, 'w') as f:
        json.dump(dataclasses.asdict(model), f)


def load_codon_ngram_model(path: str) -> CodonNgramModel:
    with open(path) as f:
        data = json.load(f)
    data['context_orders'] = tuple(data['context_orders'])
    return CodonNgramModel(**data)
