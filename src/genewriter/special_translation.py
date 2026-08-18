"""Translation exceptions the standard genomic-DNA-to-protein pipeline gets
wrong: selenocysteine recoding, non-AUG (near-cognate) start codons, and
programmed ribosomal frameshifting.

All were found live, this session, while investigating why real
protein-coding genes were coming out of GeneDataSourcing.ipynb with empty
isoforms despite correct exon extraction -- see SELENOPROTEIN_GENE_IDS',
check_cds_against_protein_special()'s, and FRAMESHIFT_GENE_IDS' own
docstrings for what was actually verified against live NCBI data (not just
textbook claims).

A-to-I (ADAR) and C-to-U (APOBEC1) RNA editing were investigated the same
way and found NOT to belong in this module: for every candidate gene
checked (GRIA2, GRIA3, GRIK1, GRIK2, CYFIP2, BLCAP for A-to-I; APOB for
C-to-U), the standard genomic translation already matches NCBI's canonical
protein exactly. That means NCBI's RefSeq protein reflects the genomically
-encoded (unedited) form for these genes -- there's no sourcing bug to fix,
just a corpus-completeness gap (the edited isoform, e.g. GRIA2's Q/R site
or APOB48, simply has no RefSeq record of its own to source from). Building
support for these would mean synthesizing a protein variant that doesn't
correspond to any single real RefSeq entry, a different and more speculative
kind of task than the corrective-pass fixes in this module -- deliberately
not attempted here.

This module is deliberately separate from GeneDataSourcing.ipynb's own
check_cds_against_protein() (which stays untouched, per the user's explicit
instruction to keep that notebook as the simple, high-throughput path) --
it's consumed by a second-pass corrective notebook instead, which re-fetches
and re-processes just the genes affected by these special cases. See that
notebook's own docstring for the two-pass design.
"""

from .codon_tables import AA_CODONS

# The ~25-member human selenoproteome: genes whose mRNA has an in-frame UGA
# that gets recoded to selenocysteine (Sec, one-letter code 'U') instead of
# terminating translation, via a SECIS element in the 3'UTR recruiting a
# specialized elongation factor/tRNA. NCBI's own protein.faa sequences for
# these genes already contain the literal character 'U' at the recoded
# position(s) -- confirmed directly (not assumed) for every gene below via a
# live download-and-translate check this session: naive translation of the
# correctly-extracted CDS terminates early at the in-frame UGA (a 'stop-like'
# codon by the standard genetic code) while the real annotated protein
# continues past it with a real residue, all the way to its own true
# TAA/TAG stop. SELENOP is the extreme case (10 in-frame UGA/Sec codons in
# one CDS) and matched perfectly once the codon-table fix (below) was
# applied, which is a strong end-to-end validation of the whole approach.
#
# Gene IDs resolved via live NCBI esearch this session, not from memory --
# two symbol lookups needed disambiguation (GPX6's symbol search also
# returned GPX7, a non-selenocysteine paralog that arose from the same gene
# family but uses cysteine instead; DIO1's symbol search also returned
# DIDO1, an unrelated gene carrying "DIO1" as a stale alias) -- both
# resolved by checking each candidate's actual esummary name/description
# before accepting it, not by picking the first hit.
SELENOPROTEIN_GENE_IDS = {
    2876: 'GPX1',
    2877: 'GPX2',
    2878: 'GPX3',
    2879: 'GPX4',
    257202: 'GPX6',
    7296: 'TXNRD1',
    10587: 'TXNRD2',
    114112: 'TXNRD3',
    1733: 'DIO1',
    1734: 'DIO2',
    1735: 'DIO3',
    6414: 'SELENOP',
    58515: 'SELENOK',
    55829: 'SELENOS',
    140606: 'SELENOM',
    57190: 'SELENON',
    83642: 'SELENOO',
    9403: 'SELENOF',
    280636: 'SELENOH',
    85465: 'SELENOI',
    51714: 'SELENOT',
    348303: 'SELENOV',
    6415: 'SELENOW',
    51734: 'MSRB1',
    22928: 'SEPHS2',
}


def is_selenoprotein_gene(gene_id) -> bool:
    return int(gene_id) in SELENOPROTEIN_GENE_IDS


# Genes confirmed live this session to use a non-AUG start codon (the
# literal first codon of their NCBI-annotated CDS isn't 'ATG', yet the real
# protein still starts with Met, per the universal initiator-tRNA
# convention -- see check_cds_against_protein_at_position()). Deliberately
# NOT curated the way SELENOPROTEIN_GENE_IDS is -- there's no equivalent of
# SelenoDB for "genes with a non-canonical start codon" to check this list
# against, so it's only ever grown by direct confirmation (download the
# gene, verify the CDS-begin-anchored translation actually matches the real
# protein), not by literature survey. Treat this as a starting point, not a
# closed set -- add an entry here only after confirming it the same way
# MYC was confirmed (see this session's transcript / HANDOFF for the
# byte-for-byte verification method).
NON_ATG_START_GENE_IDS = {
    4609: 'MYC',
}


# The selenoprotein-aware codon table: TGA moves from '*' (stop) to a new
# 'U' (selenocysteine) entry, rather than being added to both. A gene that
# recodes UGA to Sec can't also use a second in-frame UGA as its real
# stop -- that would be ambiguous with the Sec codon itself -- so every
# verified selenoprotein gene this session terminated at a real TAA/TAG,
# never a second UGA. Leaving TGA in *both* '*' and 'U' was tried first and
# silently favored '*' (Python dict iteration order checks '*' first,
# inherited from AA_CODONS below), matching 0 of the 25 genes' real
# Sec-containing isoforms despite otherwise-fully-correct exon extraction --
# removing TGA from '*' here is what took that from 0/126 to 106/126 on a
# real, live-downloaded batch of all 25 genes (the remaining ~20 gap is
# almost entirely the Alternate T2T-CHM13 genome assembly specifically,
# not the GRCh38 primary reference this pipeline actually relies on -- see
# the corrective notebook's own run log for the full breakdown).
AA_CODONS_SELENOPROTEIN = dict(AA_CODONS)
AA_CODONS_SELENOPROTEIN['*'] = ['TAA', 'TAG']
AA_CODONS_SELENOPROTEIN['U'] = ['TGA']


def check_cds_against_protein_special(cds: str, atgs: list, aa_seq: str, gene_id=None):
    """Mirrors GeneDataSourcing.ipynb's check_cds_against_protein() exactly
    (same algorithm, same return shape: (atg_start, codvec) or (None, None))
    -- kept as a separate copy rather than importing from the notebook,
    since notebook cells aren't importable and the user wants
    GeneDataSourcing.ipynb itself left as the simple, unmodified,
    high-throughput path. If that notebook's check_cds_against_protein ever
    changes again, this needs the same change ported by hand.

    Only handles the selenocysteine case: uses AA_CODONS_SELENOPROTEIN
    instead of AA_CODONS when gene_id is in SELENOPROTEIN_GENE_IDS, so an
    in-frame UGA at a position where the real protein has 'U' is accepted
    as a match instead of being read as a premature stop. Gated to the
    verified gene list -- blindly reinterpreting every UGA as Sec would be
    wrong for the other ~19,000 genes in the corpus, where UGA really is a
    stop.

    Like the notebook's original, this only ever tries positions where the
    literal codon is 'ATG' (the `atgs` list, e.g. from re.finditer('ATG',
    cds)) -- it can never find a non-AUG start codon (CUG/GUG/etc., a real,
    if uncommon, translation-initiation mechanism confirmed live this
    session for MYC and TXNRD3), because no such position ever appears in
    that search space at all, regardless of codon-table changes. See
    check_cds_against_protein_at_position() for that case -- deliberately a
    separate function/search strategy, not folded in here, since "which
    positions are plausible near-cognate start codons" has no general
    answer the way "is this gene a known selenoprotein" does; anchoring to
    NCBI's own annotated CDS begin position is the actual source of truth
    for that case, not a search over candidate positions in this function.
    """
    codon_table = AA_CODONS_SELENOPROTEIN if (gene_id is not None and is_selenoprotein_gene(gene_id)) else AA_CODONS
    aaseq = [x[0] for x in aa_seq]

    for atg in atgs:
        codvec = []
        tf = False
        if cds[atg:atg + 3] != 'ATG':
            continue
        for i in range(len(aa_seq)):
            cod = cds[atg + 3 * i:atg + 3 * i + 3]
            codvec.append(cod)
            aa_from_cds = None
            for aa, cods in codon_table.items():
                if cod in cods:
                    aa_from_cds = aa
                    break
            aa_from_seq = aaseq[i]
            if aa_from_cds != aa_from_seq:
                break
            if i == len(aaseq) - 1:
                tf = True
                break
        if tf:
            return atg, codvec

    return None, None


def check_cds_against_protein_at_position(cds: str, start: int, aa_seq: str, gene_id=None) -> bool:
    """Checks whether `cds[start:]` translates to aa_seq, treating the very
    first codon as an automatic Met match regardless of its literal
    identity -- a real, if uncommon, non-AUG translation-initiation
    mechanism confirmed live this session for MYC and TXNRD3 (both have a
    literal CTG at their true, NCBI-annotated start position; the
    initiator tRNA still loads Met there regardless).
    For use with a start position taken directly from NCBI's own annotated
    CDS range (iso['cds']['range'][0]['begin'], 1-based -> pass start-1) --
    the corrective notebook's non-ATG-start path, since (unlike the literal
    'ATG' search check_cds_against_protein_special does) there's no
    reliable *general* way to enumerate "plausible near-cognate start
    codons" across a whole CDS blindly; anchoring to NCBI's own CDS
    annotation is the actual source of truth here, matching how
    Case-A/getSeqFrom's own off-by-one fix earlier this session was
    verified (byte-for-byte against the real mRNA), not a heuristic guess.
    """
    codon_table = AA_CODONS_SELENOPROTEIN if (gene_id is not None and is_selenoprotein_gene(gene_id)) else AA_CODONS
    for i in range(len(aa_seq)):
        cod = cds[start + 3 * i:start + 3 * i + 3]
        aa_from_seq = aa_seq[i]
        if i == 0 and aa_from_seq == 'M':
            continue  # initiator codon always decodes to Met, whatever it literally is
        aa_from_cds = None
        for aa, cods in codon_table.items():
            if cod in cods:
                aa_from_cds = aa
                break
        if aa_from_cds != aa_from_seq:
            return False
    return True


# Genes confirmed live this session to require a programmed ribosomal
# frameshift: standard, unshifted triplet translation of the CDS matches the
# real protein exactly through `shift_residue` codons, then jumping by
# `shift_offset` nucleotides before resuming ordinary triplet reading
# reproduces the rest of the real protein exactly, all the way to its own
# true stop. Confirmed against live NCBI data, full-length, byte-for-byte
# (see apply_frameshift()):
#   - OAZ1 (antizyme 1): +1 shift at nt 204 (the residue-68 codon boundary).
#     Naive translation runs cleanly through residue 67, then hits an
#     in-frame UGA (read as a premature stop by standard translation) --
#     that UGA is OAZ1's well-characterized ORF1/ORF2 frameshift junction.
#     Skipping forward exactly 1 nt before resuming reproduces all 228 real
#     residues.
#   - PEG10: -1 shift at nt 1181 (the residue-394 codon boundary), which
#     lands exactly on the literal genomic sequence GGGAAAC -- the
#     canonical "X XXY YYZ" slippery heptamer for -1 programmed ribosomal
#     frameshifting (same family as retroviral Gag-Pol shifts). Backing up
#     1 nt before resuming reproduces all 784 real residues.
#
# shift_residue/shift_offset are relative to the specific transcript's own
# CDS (the longest-real-protein-matched isoform the corrective notebook
# picks), same caveat as NON_ATG_START_GENE_IDS: only grown by direct
# reconfirmation against real data, not literature survey alone.
FRAMESHIFT_GENE_IDS = {
    4946: {'symbol': 'OAZ1', 'shift_residue': 68, 'shift_offset': 1},
    23089: {'symbol': 'PEG10', 'shift_residue': 394, 'shift_offset': -1},
}


def is_frameshift_gene(gene_id) -> bool:
    return int(gene_id) in FRAMESHIFT_GENE_IDS


def apply_frameshift(cds: str, shift_residue: int, shift_offset: int) -> str:
    """Splices `cds` at the codon boundary `shift_residue * 3` nt in, jumping
    by `shift_offset` nucleotides before resuming ordinary triplet reading --
    reconstructing the sequence that, translated from position 0, gives the
    real (frameshifted) protein instead of the naive one.

    shift_offset > 0 (OAZ1's +1) skips that many nucleotides, matching a
    real ribosome hopping forward without decoding them. shift_offset < 0
    (PEG10's -1) *overlaps* that many nucleotides into both the pre- and
    post-shift reading, matching the real tRNA re-pairing event in -1
    frameshifting -- not an approximation, both directions reproduce their
    real protein exactly, byte-for-byte, against live NCBI data.
    """
    boundary_nt = shift_residue * 3
    return cds[:boundary_nt] + cds[boundary_nt + shift_offset:]


def apply_frameshift_to_exons(exsupp: list, boundary_pos: int, shift_offset: int) -> list:
    """The apply_frameshift() splice, applied to a list of exon fragments
    (`[{'seq': str}, ...]`, in order, matching GeneDataSourcing.ipynb's
    `exSupp`/`locate_codons()` shape) instead of one flat string.

    `boundary_pos` is an absolute position into the *whole* exon-concatenated
    sequence (i.e. `atg_start + shift_residue * 3`, not just
    `shift_residue * 3` -- align it against the full CDS-from-exons string,
    not a slice already anchored at the start codon).

    Splices exactly the one fragment straddling boundary_pos: shift_offset
    > 0 drops that many nucleotides starting there (a skip -- real DNA the
    ribosome never decodes as part of any codon); shift_offset < 0
    duplicates that many nucleotides ending there (an overlap -- real DNA
    decoded twice, once in each frame, the actual tRNA re-pairing event).
    Every other fragment is returned unchanged. Feeding the result to
    GeneDataSourcing.ipynb's existing, unmodified locate_codons() lets it
    walk straight through and land on the correct frameshifted codons and
    genomic positions, without needing that function's own intricate
    carry/S/F/I/T branching hand-patched for this case.

    Caveat: the fragment actually spliced no longer corresponds to a real
    exon boundary at the graft point, so locate_codons()'s S/F/T
    splice-proximity tags for the codon(s) nearest the frameshift junction
    may read as "near a splice site" even though the junction is a
    translational event, not a real one. Codon identity -- what matters for
    the resulting protein and for codon-choice scoring -- is unaffected;
    only that one cosmetic location tag near the junction can be off, and
    only for the frameshift genes this applies to.

    Raises ValueError if boundary_pos is close enough to the straddling
    fragment's own edge that the shifted region would need to reach into a
    neighboring fragment -- not implemented, since both genes this applies
    to (OAZ1, PEG10) land comfortably mid-exon (confirmed live), and getting
    a cross-fragment splice right is a meaningfully different, unverified
    case rather than something worth guessing at silently.
    """
    out = []
    pos = 0
    for e in exsupp:
        seq = e['seq']
        elen = len(seq)
        if pos + elen <= boundary_pos or pos > boundary_pos:
            out.append({'seq': seq})
            pos += elen
            continue
        local = boundary_pos - pos
        if shift_offset >= 0:
            if local + shift_offset > elen:
                raise ValueError(
                    f"frameshift boundary at {boundary_pos} is too close to the end of its "
                    f"exon fragment (len {elen}) for a +{shift_offset}nt skip -- would need to "
                    f"reach into the next fragment, not implemented")
            spliced = seq[:local] + seq[local + shift_offset:]
        else:
            if local + shift_offset < 0:
                raise ValueError(
                    f"frameshift boundary at {boundary_pos} is too close to the start of its "
                    f"exon fragment for a {shift_offset}nt overlap -- would need to reach into "
                    f"the previous fragment, not implemented")
            spliced = seq[:local] + seq[local + shift_offset:local] + seq[local:]
        out.append({'seq': spliced})
        pos += elen
    return out


def check_cds_against_protein_frameshift(cds: str, atgs: list, aa_seq: str, gene_id) -> tuple:
    """Like check_cds_against_protein_special, but for a known frameshift
    gene: applies apply_frameshift() (anchored at each candidate literal-ATG
    position, using this gene's registered shift_residue/shift_offset)
    before running the same codon-by-codon comparison against aa_seq.
    Returns (atg, codvec) or (None, None), same shape as the other
    check_*() functions in this module. Uses the plain AA_CODONS table --
    no frameshift gene confirmed so far is also a selenoprotein, and the
    shift itself (not a UGA->Sec recoding) is what explains the divergence.
    """
    if gene_id is None or int(gene_id) not in FRAMESHIFT_GENE_IDS:
        return None, None
    spec = FRAMESHIFT_GENE_IDS[int(gene_id)]
    aaseq = [x[0] for x in aa_seq]

    for atg in atgs:
        if cds[atg:atg + 3] != 'ATG':
            continue
        shifted = apply_frameshift(cds[atg:], spec['shift_residue'], spec['shift_offset'])
        codvec = []
        tf = False
        for i in range(len(aa_seq)):
            cod = shifted[3 * i:3 * i + 3]
            codvec.append(cod)
            aa_from_cds = None
            for aa, cods in AA_CODONS.items():
                if cod in cods:
                    aa_from_cds = aa
                    break
            if aa_from_cds != aaseq[i]:
                break
            if i == len(aaseq) - 1:
                tf = True
                break
        if tf:
            return atg, codvec

    return None, None
