GeneWriter — Session Handoff

Purpose: context transfer from a chat session into Claude Code, where on-device paths work. Date: 2026-07-23 Author of record: Luke (LukeTheGeneWriter)

0. Read this first (Claude Code)

This document is design context, not a spec to implement top to bottom. Nothing here has been validated against the actual code, because the notebooks were never visible in the originating session — only described. Profile before optimizing. Read the notebooks before refactoring.

Where this document and the code disagree, the code wins. Flag the disagreement rather than silently conforming to either.

1. Locations
Thing	Where	Status
Colab notebooks	C:\Users\lukef\OneDrive\Desktop\GeneWriter	Local, not yet reviewed
Target repo	https://github.com/LukeTheGeneWriter/GeneWriter	Returned 404 externally — private or renamed
CUDA folder	https://github.com/LukeTheGeneWriter/CUDA_RNAFold	Public, 64 commits, ViennaRNA autotools tree
Open question on CUDA_RNAFold

GitHub's language breakdown reports C 72.3% / C++ 6.9% / Shell / Makefile / PostScript / TeX — no CUDA. Linguist does detect .cu. So either kernels aren't committed, they're gitignored, or they live inside .c/.cpp. First thing to check on-device: git status, .gitignore, and whether kernel work exists only in the working tree.

2. What the pipeline does

A genetic algorithm searching synonymous codon space for CDS that express well in a target cell.

Current understood flow:

GA over codon space — fitness spans codon usage, GC content, and related sequence metrics
Clustering of surviving CDS
RNA folding to eliminate undesirable secondary structure
SpliceAI post-processing to eliminate CDS with cryptic/interwoven splice signals
Optional intron insertion for RNA half-life and to carry intronic elements

Tiered-cascade ordering (cheap filters first) is already correct and should be preserved.

3. Licensing — OWNER HANDLING THIS

Luke is handling licensing and publicity himself. Do not restructure license files.

Recorded for context only: CUDA_RNAFold carries an MIT LICENSE alongside the inherited ViennaRNA COPYING. ViennaRNA's terms permit research, educational, and commercial use and modification provided derived works are not redistributed for a fee beyond media costs, and that credit is given to the authors and the Institute for Theoretical Chemistry, University of Vienna; commercial inclusion requires contacting the authors. MIT's sublicense/sell grant conflicts with that.

Not legal advice. Owner's call, owner's action item.

4. Repo structure (proposed)
GeneWriter/
├── src/genewriter/
│   ├── genetic_code.py        # codon tables, aa maps, usage tables per organism
│   ├── encoding.py            # seq <-> uint8 codon-index arrays (SoA layout)
│   ├── objectives/            # ONE module per fitness term; Python reference impl
│   │   ├── cai.py
│   │   ├── gc.py              # global + windowed
│   │   ├── cpg.py
│   │   ├── codon_pair_bias.py
│   │   ├── uracil.py
│   │   ├── motifs.py          # restriction sites, homopolymers, repeats
│   │   ├── folding.py         # wraps CUDA_RNAFold
│   │   └── splice.py          # wraps SpliceAI
│   ├── change_vector.py       # position-wise mutation propensity (see §5)
│   ├── ga/                    # selection, crossover, mutation, NSGA-II
│   ├── cascade.py             # tiered filter orchestration
│   ├── cluster.py
│   ├── cache/                 # tiered cache (see §7)
│   └── kernels/               # bindings to native code
├── native/
│   ├── src/*.c  *.cu
│   └── CMakeLists.txt
├── tests/
│   ├── golden/                # frozen input -> output vectors
│   └── test_parity.py         # python vs C vs CUDA agreement
├── notebooks/                 # thin; import from src only
└── benchmarks/

Core discipline: objectives/ holds the Python reference implementation and remains the source of truth. Native implementations must reproduce it on the golden vectors. Never delete the Python version once C exists — it is the oracle.

5. Change vector (main new design work)

Position-wise mutation propensity: each codon gets a score; higher score means higher probability of being selected for mutation. Terms include rare codons, codon overuse, local GC, CpG dinucleotides, codon pair bias, uracil content.

Mutation count per offspring is stochastic (1–4) to escape local minima.

Design decisions to carry over

Normalize before weighting. Terms have incompatible scales; raw summation lets the widest-range term dominate. Z-score or rank-normalize each term across positions within the sequence, then apply weights.

Validate against realized Δfitness. Highest-value experiment available. Sample positions, actually mutate them, measure fitness change, correlate against predicted score — per term. Turns weight tuning into measurement. Fit once per target organism, cache the weights.

Neighborhood-scoped incremental updates. Codon pair bias spans adjacent pairs; local GC spans a window. Mutating codon i dirties roughly [i-w, i+w]. Global recompute is wasteful; stale is wrong. Dirty-window scatter-add — good CUDA kernel (coalesced, bounded, low divergence).

Gumbel-top-k for the multi-mutation draw. Independent softmax draws produce duplicate positions. Gumbel noise on log-weights + top-k is exactly weighted sampling without replacement — branchless, parallel-friendly.

Adaptive k. Skew toward 1 (geometric / truncated Poisson). Tie to acceptance rate: falling acceptance → raise k to escape; high acceptance → lower k to refine.

Rare-codon survival. Stochastic survival alone is weak — high-signal positions get resampled and eventually hit. Stronger: temperature on the softmax (near-uniform early, sharp late); tabu list on recently-mutated positions; explicit freeze masks.

Biological caveat — do not skip

Rare codons are not uniformly bad. The 5′ ramp and deliberate translational pause sites support co-translational folding. A naive "rare = change me" signal destroys them.

Exclude roughly the first 30–50 codons from the rare-codon penalty
Treat the 5′ region under a separate objective — it wants weak local structure for initiation, which runs directly against global MFE minimization
Unresolved

Is the change vector a heuristic proposal inside a selection-based GA, or is there accept/reject on fitness? If the latter, this is Metropolis–Hastings with a non-uniform proposal, and without the q(x|x')/q(x'|x) correction the stationary distribution is biased toward whatever the change vector prefers rather than toward fitness. Determine which from the code before extending it.

6. Fitness function — open concerns

Multi-objective handling. If objectives are currently collapsed into a single weighted scalar, consider NSGA-II and a Pareto front instead. CAI, GC, MFE, and splice score genuinely conflict, and a weighted sum forces choosing λ before the tradeoff surface is known.

MFE may be the wrong target. MFE describes one structure; equilibrium has many. mRNA generally wants conformational flexibility during ribosomal unwinding — MFE minimization produces steep energy landscapes. Consider ensemble free energy or local accessibility near the start codon instead of a global MFE threshold. Related: lower MFE correlates with reduced translation initiation efficiency, so a hard "most stable wins" filter can select against expression.

Two sequences, not one. Once introns are inserted:

splice signals are a property of the pre-mRNA (unspliced)
folding and codon metrics are properties of the mature mRNA (spliced)

Easy to conflate when both live in one seq variable. Verify the current code doesn't.

SpliceAI context. SpliceAI is context-dependent and was trained with large genomic flanks. Scoring a bare CDS produces numbers that don't describe the real construct. Embed the CDS in the intended vector/genomic context with the full input window populated. If the current code pads with Ns or scores naked CDS, that is a correctness bug and takes priority over any performance work.

7. Caching (2 TB cloud available)

Tier it. In-process LRU (hot) → local SSD (warm) → cloud (cold/archival + cross-run reuse). A cloud round-trip is tens of ms; GPU folding of a 1 kb sequence may be single-digit ms — the cache can be slower than recomputing. Cloud is for provenance and cross-session reuse, not inner loop.

Config hash in the key. Energy model version, parameter set, temperature, dangling-ends mode, SpliceAI model version. Without it, results from before and after a parameter change mix silently.

128-bit hash (blake3 / xxh128). 64-bit begins colliding around a few billion entries, which is inside the plausible range here.

Shard columnar (Parquet) by hash prefix. Not one object per sequence — 2 TB of tiny objects is expensive to store and painful to list.

Cache the expensive things only: MFE + structure, SpliceAI scores. Codon arithmetic is cheaper to recompute than to fetch.

8. Performance strategy

Expected bottleneck is folding, not codon arithmetic. CAI/GC/CpG/homopolymer scans over a few thousand sequences are microseconds; folding is O(n³) per sequence.

Coarse-grained beats fine-grained here. A GA has a population of independent sequences — one sequence per thread block, batched across the generation, keeps SMs busy without fighting the anti-diagonal dependency structure of the Zuker recurrence. Much easier than intra-sequence wavefront parallelization. (Existing CUDA_RNAFold work may already do this — check before rebuilding.)

Check LinearFold-style O(n) beam search for the screening pass, with exact folding reserved for survivors. An algorithmic complexity reduction on the first filter dwarfs a 10–20× constant factor.

Don't rewrite SpliceAI. Already a GPU model. Wins are batching, resident weights, fp16.

Clustering (if k-mer or edit-distance based) is a pairwise distance matrix — friendliest possible CUDA target, good first kernel.

Suggested order for the C/CUDA work
Scalar codon metrics → C. Low stakes, immediately testable, teaches the binding boundary and SoA layout without algorithmic difficulty.
Pairwise distances → CUDA. Tiled, shared memory, coalesced.
Batched folding → CUDA. The real prize, attempted once the test harness exists.
Two gotchas

Reduction order changes results. Floating-point addition isn't associative, so parallel reduction won't bit-match Python's sum(). In a GA this compounds: a 1e-12 difference flips a selection comparison and two "identical" runs diverge into different populations. Fix reduction order or use compensated summation if reproducibility matters. Parity tests assert on tolerance, not equality.

Memoize fitness on sequence hash. GAs regenerate duplicate offspring constantly. A hash map in front of the folding call often beats the kernel you were about to write.

9. Benchmark target

LinearDesign solves the CAI+MFE subproblem exactly — it encodes synonymous codon choices as a DFA, models folding with an SCFG, and applies lattice parsing over their weighted intersection, optimizing MFE − λ·|p|·log(CAI) in O(n³). It uses the same thermodynamic energy model as ViennaRNA, so MFE values are directly comparable.

It cannot express splice constraints, intron insertion, GC windows, or motif avoidance — so it does not replace the GA. But it provides a ground-truth optimum:

Run the GA with all objectives except CAI and MFE disabled. It should recover the LinearDesign optimum. If it doesn't, there is a convergence or fitness-scaling bug, and no amount of CUDA will fix that.

Worth wiring into CI as a regression test once the GA is modular enough to disable objectives.

10. Colab deployment

Eventual runtime target is Colab, called from CLI with precompiled binaries.

Fat binary, -gencode covering sm_70 / 75 / 80 / 89 plus PTX for forward compatibility. Colab allocates T4, L4, A100, occasionally V100 — don't discover an arch mismatch post-allocation.
Ship prebuilt wheel or tarball to Drive, extract at session start. Compiling on Colab burns paid GPU-hours.
Checkpoint GA state to Drive every N generations with clean resume. Colab sessions die routinely; this is not optional for long runs.
11. Immediate next steps
Confirm repo visibility and whether CUDA kernels are actually committed (git status, .gitignore)
Read the notebooks. Do not refactor yet.
Profile. Establish where time actually goes before touching anything.
Audit for the correctness issues in §6 — SpliceAI context and pre-/post-splicing sequence conflation both outrank performance work.
Extract objectives into modules, build golden test vectors from current Python behavior
Then, and only then, start the C port in the order given in §8
12. Notes on provenance

Sections 5–10 are design recommendations from a chat session, developed from a verbal description of the pipeline. None of it has been checked against the code. Treat every claim about what the current implementation does as a hypothesis to verify, not a fact.