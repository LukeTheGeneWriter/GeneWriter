"""Load NaturalGene objects from the JSON files saveNaturalGeneObj() writes
(GeneDataSourcing.ipynb et al) -- the RefGenes/NHGeneBodySupp/<geneID>.json
files. Equivalent to the loadGeneBody() function duplicated verbatim across
Rare_Codons.ipynb, Codon_Usage.ipynb, Codon_Pair_Bias.ipynb, GC_Analysis.ipynb
and Kmer_Analysis.ipynb.
"""

import glob
import json
import os

from .classes import IsoformGeneBody, NaturalGene, ProteinObj


def load_gene(path: str) -> NaturalGene:
    with open(path) as f:
        data = json.load(f)

    isoforms = []
    for iso in data['isoforms']:
        protein = ProteinObj(**iso['associatedProtein'])
        isoforms.append(IsoformGeneBody(
            isoformNumber=iso['isoformNumber'],
            associatedProtein=protein,
            fullSequence=iso['fullSequence'],
            mRNASeq=iso['mRNASeq'],
            codons=iso['codons'],
            relativeAbundance=iso['relativeAbundance'],
            geneBody=iso['geneBody'],
        ))

    return NaturalGene(
        isoforms=isoforms,
        geneID=data['geneID'],
        geneName=data['geneName'],
        organism=data['organism'],
        DNASequence=data['DNASequence'],
        spliceAIDonor=data['spliceAIDonor'],
        spliceAIReceptor=data['spliceAIReceptor'],
        energetics=data['energetics'],
        chromosome=data['chromosome'],
    )


def load_genes(directory: str) -> list:
    paths = sorted(glob.glob(os.path.join(directory, '*.json')))
    return [load_gene(p) for p in paths]


def protein_coding_isoforms(genes: list):
    """Yield (gene, isoform) for isoforms with a non-empty codon stream,
    skipping computationally-predicted-only transcripts the way every
    analysis notebook does (`if 'X' in str(iso.isoformNumber): continue`)."""
    for gene in genes:
        for iso in gene.isoforms:
            if 'X' in str(iso.isoformNumber):
                continue
            if not iso.codons:
                continue
            yield gene, iso
