GeneWriter — Session Handoff (v2)

Purpose: context transfer for whoever (human or Claude) picks this up next. Date: 2026-07-24. Supersedes the original Handoff.md (2026-07-23) — that one was design intent from a chat session with the notebooks never actually read; everything below has been read, run, and tested against the real code.

0. Read this first

Unlike v1, this document describes code that exists, runs, and is tested — not a proposal. Where it says something works, it means `pytest` passes on it. Where it says something isn't done, it isn't started. The original 14 notebooks are still in the repo root, unmodified, and are now considered historical/reference material — the live code is `src/genewriter/`.

1. Environment — read before doing anything else

**There is no usable Python on native Windows on this machine.** `python`/`python3` resolve to a Microsoft Store stub, not a real interpreter. All development and testing happens via **WSL (Ubuntu)**. From the repo root:

```
wsl -d Ubuntu -e bash -lc "cd '/mnt/c/Users/lukef/OneDrive/Desktop/GeneWriter' && python3 -m pytest -q"
```

`pyproject.toml` sets `pythonpath = ["src"]`, so no install step is needed. WSL's `python3` had no `pip`; it was bootstrapped via `curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --user --break-system-packages` (Debian's externally-managed-environment guard blocks a plain `pip install`; no `apt`/`venv`/`ensurepip`/`sudo` available non-interactively). `numpy`/`pytest`/`cupy-cuda12x` and friends are all installed the same way.

**A real CUDA toolchain is set up and verified working**, including a compiled-and-run `.cu` file on the actual GPU (NVIDIA RTX 3050 Laptop, 4GB, Ampere sm_86). GPU driver passthrough (`libcuda.so`) was already present via WSL2's paravirtualization — no setup needed, just `PATH=/usr/lib/wsl/lib:$PATH nvidia-smi` to see it (driver 566.14, CUDA 12.7). The rest (`nvcc`, `gcc`/`g++`, `cupy`) came from pip + user-space Miniforge (conda-forge), no root needed anywhere:

```
curl -sSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda install -y -n base -c conda-forge gcc gxx
$HOME/miniforge3/bin/conda install -y -n base -c nvidia -c conda-forge cuda-nvcc=12.9
# plus: pip install --user --break-system-packages cupy-cuda12x nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12
```
Add `$HOME/miniforge3/bin` to PATH for `nvcc`/`gcc`/`g++`. Note pip's own `nvidia-cuda-nvcc-cu12` wheel does *not* ship the real `nvcc` driver binary (only `ptxas` + NVVM) — that's what the conda-forge/nvidia-channel install is for.

Full detail on both of the above (exact failure modes hit, exact fix commands) is saved in this project's Claude memory (`dev_environment.md`) — check there before re-deriving any of it.

2. What's actually in the repo

```
GeneWriter/
├── *.ipynb                    # 14 original notebooks, historical/reference only now
├── Gene_Obj_Samples/           # gitignored: ~6 real NCBI gene JSONs, user-provided, for local dev/testing
├── pyproject.toml
├── src/genewriter/
│   ├── codon_tables.py         # codon<->AA, codon<->index encoding, precomputed per-codon lookup tables
│   ├── classes.py              # canonical dataclasses (from GeneClassesCloud.ipynb)
│   ├── change_vector.py        # the pluggable, vectorized change-vector scoring system
│   ├── gpu_change_vector.py    # population-batched change-vector computation (numpy or cupy backend)
│   ├── ga.py                   # GA primitives: growth, selection, flatten, kill_off, diffing
│   ├── schedule.py             # declarative JSON-able GA schedules (the pipeline-config system)
│   ├── gene_io.py              # load NaturalGene objects from RefGenes-style JSON
│   ├── baseline.py             # compute the 5 *Analysis baseline dataclasses from loaded genes
│   └── standards_io.py         # load precomputed Standards/*.json baselines
└── tests/                      # 138 tests, all passing
```

3. What works, is tested, and is real

- **Full GA pipeline**: seed generation → growth (random mutation + directed evolution) → selection (kill_off / select_survivors / flatten) → repeat, driven either by the simple `ga.run_ga()` convenience function or by a declarative `schedule.py` pipeline (JSON-able step list, matching an ML-training-config shape). Verified end-to-end against real sample gene data.
- **Change-vector scoring**: all 5 original terms (RareCodons, CodonUsage, CodonPairBias, GC, Kmer) ported, bug-fixed, and vectorized with numpy. Pluggable via `@register_term`/`@register_step` decorators — a new objective (uracil minimization, a future RNA-folding signal, etc.) just needs the right function signature, no core code changes.
- **Approximate incremental diffing** (`diff_change_vector`): a child's change vector is estimated from its parent's by recomputing only a local excerpt, instead of a full recompute — exact for 2 of 5 terms (no global dependency), approximate for the other 3 (which fold in a population-wide aggregate). `refresh_change_vectors` forces an exact recompute; wired in automatically before every `kill_off`/`select`/`flatten` step in `schedule.py` (population-decision steps), on a configurable cadence in `run_ga`.
- **Lookahead toggle**: `directed_evolution(..., lookahead=False)` skips the expensive "score every synonymous alternative" step.
- **Population-batched GPU computation** (`gpu_change_vector.py`): 4 of 5 terms (not Kmer — see §5) computed for an entire population in one batched pass, on CPU (numpy) or the real GPU (cupy) via the same code. Verified correct against the per-individual implementation and, separately, GPU-vs-CPU-backend agreement on real hardware.
- **Data pipeline**: load real NCBI gene JSON → compute baseline statistics → feed the GA. Verified against ~6 real sample genes in `Gene_Obj_Samples/` (not committed; ask the user for a copy, or point at their Colab Drive `RefGenes/NHGeneBodySupp/`).

Every claim above has a corresponding test in `tests/`. Read the module docstrings for the "why" behind each design choice and each historical bug fix — they're written to be self-sufficient; this document is a map, not a replacement.

4. Correctness process worth knowing about

Two real regressions were caught during this session, both by writing a direct comparison against a known-correct reference rather than by code review alone:
- `tests/test_windowed_average.py` — the shared windowing helper, checked against the original pure-Python loop across 550 randomized cases.
- `tests/test_change_vector_reference.py` — all 5 terms, checked against the pre-vectorization implementation (kept verbatim in the test file) across 40 randomized cases. This is what caught a real off-by-one in the RareCodons term that had shipped silently for two turns.
- `tests/test_gpu_change_vector.py` / `test_gpu_change_vector_cuda.py` — the batched implementation, checked against the per-individual one (numpy backend), then the GPU backend checked against the numpy backend on real hardware. This is what caught a GC-term polarity bug.

If you're extending any of these systems, extend the corresponding reference test rather than trusting a rewrite by inspection — it has a proven catch rate on this exact codebase.

5. What's NOT done — the honest gap list

- **Kmer batching**: not implemented. Its substring→score lookup is a hash-map operation over arbitrary-length k-mer strings, not a natural fit for array vectorization without re-encoding k-mers as base-4 integers first. `batch_calculate_change_vectors()` currently falls back to the per-individual (already-vectorized, but not batched/GPU) implementation for this one term, which measurably dilutes the batched pipeline's overall speedup (Kmer was ~34% of per-individual runtime in one real-gene benchmark).
- **The GPU batched path is not wired into the actual GA loop.** `run_ga()` and `schedule.py`'s steps still call the per-individual `calculate_change_vector`/`diff_change_vector`. `gpu_change_vector.py` is built, tested, and benchmarked standalone, but nothing in the GA machinery calls it yet. This is the natural next step if the GPU work is going to pay for itself in an actual run.
- **RNA folding, SpliceAI, intron insertion**: still entirely unimplemented, exactly as v1 of this handoff flagged. No code exists for any of these anywhere in the repo. Per the original pipeline description (still accurate) these come after the GA/codon-optimization stage.
- **Real-scale (tens of thousands of individuals) has not been run end-to-end.** The GPU batching was benchmarked up to population=10,000 on the 4 batchable terms in isolation (see §6) — a full schedule run (growth + selection + Kmer) at that scale has not been attempted; Kmer's unbatched cost would likely dominate.
- **Custom CUDA kernels**: not written. Everything GPU-side so far is CuPy (numpy-compatible array ops, JIT-compiled via NVRTC under the hood) — no hand-written `.cu` kernels in the actual pipeline (only in a standalone verification script proving the toolchain works). The biggest remaining CUDA opportunity is almost certainly RNA folding once it exists (O(n³) DP, the Handoff's own §8-9 already covers this) — bigger than anything left to gain on the already-fast change-vector terms.
- **Flagged-for-later items** (recorded in Claude memory as `future_work_items.md`, not started): n-gram/graph-based population seeding from real codon usage patterns (using the F/T/I/S location tags to restrict to intron-safe interior windows), uracil minimization as a new term, and a precomputed per-codon GC lookup — this last one is now partially done as a side effect of the GPU work (`codon_tables.GC_FLAGS_BY_INDEX`), but not yet wired into the per-individual `_gc_term`.

6. Performance numbers on record (so nobody re-derives these from scratch)

All measured on the real MS4A18 gene (398 residues) against the toy 6-gene sample baseline, this WSL environment, this RTX 3050.

- `calculate_change_vector()` (one individual, all 5 terms): 389ms → 5.7ms after numpy vectorization (~68x).
- `diff_change_vector()` (one child, default margin): 41ms → 2.8ms after vectorization (~15x on top of the diffing approximation itself).
- The 4 batchable terms only (RareCodons/CodonUsage/CodonPairBias/GC), population loop vs batched:
  - population=1,000: numpy 9.1x, GPU 6.7x (GPU loses to fixed overhead at this scale)
  - population=10,000: numpy 17.9x, **GPU 24.8x** (GPU wins once there's enough work to amortize transfer/launch overhead)

7. Everything below this line is unchanged from Handoff v1

Still accurate, still not double-checked against code (because the relevant code still doesn't exist): licensing (owner's call, don't touch license files), the CUDA_RNAFold repo open question (language-breakdown mismatch, worth checking `git status`/`.gitignore` there directly), the change-vector design discussion in v1 §5 (Metropolis-Hastings question still unresolved — is the GA's selection step doing proper accept/reject, or is the change vector just a heuristic proposal? worth settling before doing much more GA-tuning work), the fitness-function open concerns in v1 §6 (multi-objective/Pareto vs weighted-sum, MFE-as-wrong-target, pre-/post-splicing sequence conflation, SpliceAI context), the caching design in v1 §7 (not built yet — nothing to cache until folding/SpliceAI exist), the Colab deployment notes in v1 §10, and the LinearDesign benchmark-target idea in v1 §9. Read v1 (in git history, this commit's parent) for the full text of all of these if picking that work up.
