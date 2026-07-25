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
- **Population-batched GPU computation** (`gpu_change_vector.py`): all 5 terms, including Kmer (as of this session — see §5), computed for an entire population in one batched pass, on CPU (numpy) or the real GPU (cupy) via the same code. Verified correct against the per-individual implementation and, separately, GPU-vs-CPU-backend agreement on real hardware.
- **GPU batched path is wired into the actual GA loop** (`ga.py`'s `seed_population()`/`refresh_change_vectors(..., xp=...)`/`directed_evolution_batch()`, threaded through `run_ga(..., xp=...)` and `schedule.run_schedule(..., xp=...)` / `ScheduleContext.xp`): every exact, whole-population computation — initial seeding (`run_ga`'s seed list, schedule.py's `"input"` step), every exact refresh (`"kill_off"`/`"select"`/`"flatten"`, and `run_ga`'s periodic `refresh_every`), and (when `lookahead=True`) `"growth"`'s best-alternative scoring — now routes through `batch_calculate_change_vectors()` when `xp=numpy`/`xp=cupy` is passed, instead of a Python loop calling `calculate_change_vector()`/`diff_change_vector()` once per individual. `xp=None` (default) keeps the original per-individual path untouched — this is opt-in, not a behavior change to existing callers. `"growth"`'s new-genotype *storage* (`merge_replicate()`'s per-replicate diffing) is a separate thing and is NOT batched — see §5. Verified end-to-end on real GPU hardware, `xp=numpy` vs `xp=cupy` (not vs `xp=None` — growth's own scoring genuinely changes when xp is set, see §5): identical final populations given the same RNG seed (`tests/test_ga_schedule_gpu_wiring.py`).
- **Data pipeline**: load real NCBI gene JSON → compute baseline statistics → feed the GA. Verified against ~6 real sample genes in `Gene_Obj_Samples/` (not committed; ask the user for a copy, or point at their Colab Drive `RefGenes/NHGeneBodySupp/`).

Every claim above has a corresponding test in `tests/`. Read the module docstrings for the "why" behind each design choice and each historical bug fix — they're written to be self-sufficient; this document is a map, not a replacement.

4. Correctness process worth knowing about

Two real regressions were caught during this session, both by writing a direct comparison against a known-correct reference rather than by code review alone:
- `tests/test_windowed_average.py` — the shared windowing helper, checked against the original pure-Python loop across 550 randomized cases.
- `tests/test_change_vector_reference.py` — all 5 terms, checked against the pre-vectorization implementation (kept verbatim in the test file) across 40 randomized cases. This is what caught a real off-by-one in the RareCodons term that had shipped silently for two turns.
- `tests/test_gpu_change_vector.py` / `test_gpu_change_vector_cuda.py` — the batched implementation, checked against the per-individual one (numpy backend), then the GPU backend checked against the numpy backend on real hardware. This is what caught a GC-term polarity bug.
- `tests/test_ga_schedule_gpu_wiring.py` — the `xp` wiring in `ga.py`/`schedule.py` (see §3), checked by running `run_ga`/`run_schedule` twice with the module RNG reseeded identically (once `xp=None`, once `xp=cupy`) and asserting the final populations are the same genotypes with the same counts, plus a direct `seed_population`/`refresh_change_vectors` numpy-vs-cupy comparison. Same real-GPU-hardware pattern as the cuda file above, same skip-if-no-GPU guard.

If you're extending any of these systems, extend the corresponding reference test rather than trusting a rewrite by inspection — it has a proven catch rate on this exact codebase.

5. What's NOT done — the honest gap list

- **`merge_replicate()`'s per-replicate storage diffing is not batched, and turns out to be the real remaining growth bottleneck.** Found by profiling `directed_evolution_batch()` after Kmer batching (below) landed and growth's *scoring* got fast but growth's *total* time barely moved: on real gene data, batched scoring took 1.45s for ~10,000 candidates, but the `merge_replicate()` loop that stores each accepted replicate's change_vecs (one `diff_change_vector()` Python call each, completely untouched by any of this session's batching) took 12.25s for that same ~10,000 replicates — the actual dominant cost. This is *not* the same code path Kmer batching fixed, and is *not* touched by `xp` at all right now. Batching it the way `seed_population()`/`refresh_change_vectors()` already batch their own storage would be the natural next step — not started. (Also affects `_flatten_round()`'s new-neighbor creation, which has the identical one-diff-call-per-new-genotype shape and wasn't touched either.)
- **Kmer batching: DONE this session** (`gpu_change_vector.batch_kmer_term()`), closing out what used to be the single biggest gap in the batched-computation work. Every k-mer window is re-encoded as a base-4 integer (A/C/G/T → 0..3, `codon_tables.NT_TO_BASE4`/`NT_BASE4_BY_CODON_INDEX`) so the baseline's fold_enrich scores become one flat `(num_buckets, 4**k)` array per k (`_build_kmer_score_table()`), and the substring→score lookup becomes one array-indexed gather across the whole batch instead of a per-window, per-individual Python dict lookup. `batch_calculate_change_vectors()` no longer has *any* per-individual fallback — all 5 terms are genuinely batched. Verified against the per-individual `_kmer_term` (numpy) and numpy-vs-cupy agreement on real hardware (`tests/test_gpu_change_vector.py`/`test_gpu_change_vector_cuda.py`, plus a dedicated multi-k test since the shared test fixture only exercises k=2 — real Standards baselines go up to k=9-10). Table memory is fine at that range (~25MB at k=10) and was flagged as a real limit only well past k=15, not a practical concern today.
- **`"growth"`'s lookahead scoring is batched too** (`ga.directed_evolution_batch()`, wired into `run_ga`/`schedule.py`'s "growth" step whenever `xp` is set and `lookahead=True`) — this was the real per-generation bottleneck originally flagged by the progress instrumentation below (~20-27 individuals/sec, tens of times slower than the batched exact-refresh path, almost certainly the real explanation for multi-hour real-scale runs before this). Growth's new-genotype *storage* is a separate, still-unbatched thing (see the `merge_replicate` bullet above) — only *which alternative gets chosen* is the now-batched part. Net measured speedup for growth end-to-end: a modest ~3x (see §6) — real, but capped by `merge_replicate`'s storage diffing now, not by Kmer anymore.
- **Progress instrumentation added** (`ga.run_ga(progress=, progress_every=)`, `schedule.run_schedule(progress=, progress_every=)`, `gpu_change_vector.batch_calculate_change_vectors(progress_every=)`): diagnostic-only prints (per-generation/per-step timing, plus per-term timing marks within `batch_calculate_change_vectors`, plus every-N-individuals throughput within the remaining per-individual growth/refresh/seeding loops), all default off/silent so existing behavior and tests are unaffected. Added specifically because a long real-scale run gives *zero* feedback otherwise — this is what surfaced the growth bottleneck above (twice — once for lookahead scoring, once for `merge_replicate` after that was fixed); use it before assuming any future slow run is a bug rather than a known cost.
- **RNA folding, SpliceAI, intron insertion**: still entirely unimplemented, exactly as v1 of this handoff flagged. No code exists for any of these anywhere in the repo. Per the original pipeline description (still accurate) these come after the GA/codon-optimization stage.
- **Real-scale (tens of thousands of individuals) has not been run end-to-end.** The fully-batched exact-refresh path was benchmarked up to population=10,000 (see §6), and growth's batching up to population=2,000 — a full schedule run at real scale (18,864 genes, genome-wide Standards) has not been attempted end-to-end by this session, though the user has since run the loader itself against the real data (see the Colab-crash follow-up below) and confirmed load alone takes ~30 minutes, independent of anything covered here.
- **Custom CUDA kernels**: not written. Everything GPU-side so far is CuPy (numpy-compatible array ops, JIT-compiled via NVRTC under the hood) — no hand-written `.cu` kernels in the actual pipeline (only in a standalone verification script proving the toolchain works). The biggest remaining CUDA opportunity is almost certainly RNA folding once it exists (O(n³) DP, the Handoff's own §8-9 already covers this) — bigger than anything left to gain on the already-fast change-vector terms.
- **Flagged-for-later items** (recorded in Claude memory as `future_work_items.md`, not started): n-gram/graph-based population seeding from real codon usage patterns (using the F/T/I/S location tags to restrict to intron-safe interior windows), uracil minimization as a new term, and a precomputed per-codon GC lookup — this last one is now partially done as a side effect of the GPU work (`codon_tables.GC_FLAGS_BY_INDEX`), but not yet wired into the per-individual `_gc_term`.
- **`scripts/colab_stress_test.py`'s `pick_target()` error messages were improved** after a real Colab run (18,864 genes + genome-wide Standards, ~30 min load) crashed with an opaque `ValueError` on an unmatched `GENE_ID`. It now distinguishes "no gene with this ID was loaded at all" from "the gene loaded but every isoform was filtered out by `protein_coding_isoforms()`" (computationally-predicted-only, or empty codon list), and the script's `main()` now documents a load-once/iterate-cheaply cell-splitting pattern for Colab, since re-running `main()` from scratch to try a different `GENE_ID` means re-paying that ~30-minute load.

6. Performance numbers on record (so nobody re-derives these from scratch)

All measured on real gene data (a 59-residue sample gene unless noted; the 398-residue MS4A18 numbers below predate this session), this WSL environment, this RTX 3050.

- `calculate_change_vector()` (one individual, all 5 terms, 398-residue gene): 389ms → 5.7ms after numpy vectorization (~68x).
- `diff_change_vector()` (one child, default margin, 398-residue gene): 41ms → 2.8ms after vectorization (~15x on top of the diffing approximation itself).
- **Full `batch_calculate_change_vectors()` (all 5 terms, now including Kmer) vs. the true per-individual `calculate_change_vector()` loop**, 59-residue gene:
  - population=1,000: per-individual 2.64s → numpy 0.10s (**25.5x**) → GPU 0.08s (**34.9x**)
  - population=10,000: per-individual 27.09s → numpy 1.00s (**27.2x**) → GPU 0.56s (**48.8x**)
  - Superseded numbers from before Kmer batching (4 batchable terms only, Kmer excluded from the comparison entirely): population=1,000 numpy 9.1x/GPU 6.7x; population=10,000 numpy 17.9x/**GPU 24.8x**. Kept for context on how much Kmer batching moved things, not as current guidance — the full-function numbers above are what to use now.
- **"growth" (`directed_evolution`'s lookahead) was found to be the real per-generation bottleneck at real scale, and is now batchable too** (`ga.directed_evolution_batch()`) — found via the `progress`/`progress_every` instrumentation (see §5) after a real Colab run and a local repro both silently ran for over an hour with no visibility into where the time was going.
  - Before any growth work: lookahead scoring ran at **~20-27 individuals/sec** per-individual — tens of times slower than the batched exact-refresh path.
  - After batching scoring alone (`directed_evolution_batch()`, before Kmer was also batched): population=200, 12.83s → 4.67s; population=2,000, 125.17s → 45.94s (numpy) / 45.33s (cupy) — **~2.7-2.8x**, with GPU giving essentially no win over numpy (1.0x), because Kmer's then-still-per-individual fallback loop now ran on FULL sequences (growth's batched candidates aren't `diff_change_vector`'s bounded ~80-codon excerpts) and dominated regardless of backend.
  - **After Kmer batching also landed, growth's own scoring got fast (1.45s for ~10,000 candidates on real hardware) — but growth's total time barely moved (population=2,000: still ~36-38s, ~3x over the original 114-125s), because a *different*, previously-invisible cost took over: `merge_replicate()`'s per-replicate storage diffing (12.25s for that same ~10,000 replicates), which nothing in this session touches.** GPU still measures ~1.0x vs numpy for growth, for a new reason now — the bottleneck is a Python loop, not batched math, so the array backend is irrelevant to it. This is genuinely a different bug/gap than the one Kmer batching closed, found only by re-profiling after the fix that was expected to help didn't help as much as predicted — see the `merge_replicate` bullet in §5 for the real next step.
  - `lookahead=False` remains a valid alternative mitigation (skips scoring entirely, see `directed_evolution`'s docstring) when growth's current ~3x isn't enough and picking the single best alternative per mutation isn't essential.

7. IN PROGRESS as of 2026-07-25 — read this first if picking this session back up

A plan is **approved** (by the user, via plan-mode) but **implementation has barely started** — do not assume anything below is built. Full plan, with exact function signatures, is at `C:\Users\lukef\.claude\plans\jazzy-puzzling-kite.md` — read that file in full before continuing; this section is a pointer + status, not a substitute for it.

**What the plan covers**: (a) a real bug fix in `GeneDataSourcing.ipynb` — `check_cds_against_protein()`'s per-position amino-acid mismatch check does `continue` instead of `break`/reject, so a wrong ATG start codon gets accepted whenever the *last* residue happens to translate correctly regardless of every earlier mismatch. Confirmed universal across every local sample gene: `iso.codons`' location tags do **not** correspond position-for-position with `iso.associatedProtein.aaSeq` for any real gene data on disk today (local `Gene_Obj_Samples/` or the user's real Colab `RefGenes/NHGeneBodySupp/` set) — this is a data problem, not something fixable in `src/genewriter/`. The user chose to fix the notebook's root cause rather than work around it in code; regenerating the actual gene JSON data is explicitly **out of scope for this plan** (a separate, hours-long, user-driven step, later). (b) A new AA-motif frequency index (`aa_motif_index.py`, fixed k=3..10, unrestricted by location tag) with a report function producing `Found {n} occurrences of substring "{s}" in the {label} proteome` — no alignment dependency, buildable/testable today even before the notebook fix. (c) A new codon n-gram seeding model (`codon_ngram.py`) conditioned on local amino-acid context, restricted to interior (`'I'`-tagged) codon windows by default, with n-gram-style back-off for undertrained contexts — an alternative to `ga.generate_seed()`'s pure-uniform-random choice, for smarter GA initial-population seeding. This one **does** depend on the codons↔aaSeq alignment fix (and is only scientifically meaningful post-regeneration) — must be tested against hand-built synthetic fixtures only, never `Gene_Obj_Samples/`. (d) Wiring: `schedule.ScheduleContext.seed_fn` (new, opt-in, defaults to `generate_seed` — same pattern as `xp`), and `scripts/colab_stress_test.py` CONFIG additions (`TARGET_MODE`/`CUSTOM_AA_SEQ`/`CUSTOM_LOCVEC` for a custom-protein-sequence target bypassing gene lookup entirely, `SEED_STRATEGY`/`NGRAM_*` for the new seeding model).

**Actual progress so far**: none persisted to disk. The notebook fix was drafted and previewed in an in-memory Python session but **never written** — `git status`/`git diff` on `GeneDataSourcing.ipynb` confirm it is byte-identical to before this plan started. Nothing else in the plan (new dataclasses, new modules, schedule.py wiring, colab script wiring, any tests) has been started at all.

**Resume here, in this order** (matches the plan file's own section order and the live todo list at the time this was written):
1. Fix `GeneDataSourcing.ipynb` cell 3, `check_cds_against_protein`: line with `continue` right after `if aa_from_cds != aa_from_seq:` → `break`. Edit the notebook's JSON directly via a small Python script (load with `json.load`, mutate `nb['cells'][3]['source']` list, `json.dump` back with the same formatting) — do **not** hand-edit the raw JSON text, the source lines carry embedded `\n`/escaping that's easy to corrupt via naive string replacement. Add a short comment line explaining the fix and that already-generated data predates it.
2. `src/genewriter/codon_tables.py`: add `STANDARD_AMINO_ACIDS` + `validate_aa_sequence()`.
3. `src/genewriter/classes.py`: add `AAMotifIndex` and `CodonNgramModel` dataclasses.
4. `src/genewriter/aa_motif_index.py` (new file): `build_aa_motif_index`, `query_motif_count`, `report_motif_occurrences`.
5. `src/genewriter/codon_ngram.py` (new file): `build_codon_ngram_model`, `_select_context`, `sample_codon_ngram_seed`, `save_codon_ngram_model`/`load_codon_ngram_model`.
6. `src/genewriter/schedule.py`: `ScheduleContext.seed_fn`, `_step_input`, `run_schedule(seed_fn=...)`.
7. `tests/conftest.py`: `make_synthetic_isoform()` helper — the only sanctioned way tests get codons↔aaSeq alignment; never trust a loaded JSON file for this until the notebook fix + data regeneration both happen.
8. Tests for everything above (see the plan file's full test list — `test_codon_tables.py`, new `test_aa_motif_index.py`, new `test_codon_ngram.py`, `test_schedule.py` additions).
9. `scripts/colab_stress_test.py`: `TARGET_MODE`/`CUSTOM_AA_SEQ`/`CUSTOM_LOCVEC`/`SEED_STRATEGY`/`NGRAM_*` CONFIG, `pick_target()` branch, new `_build_seed_fn()` helper, wire into `run_pipeline()`.
10. New `tests/test_colab_stress_test.py` (first test coverage of a script file in this repo — load via `importlib.util.spec_from_file_location`, don't add `scripts/` to `pythonpath`).
11. Full `pytest -q` run (163 passing as of this note — should only grow).
12. Manual smoke test with a synthetic multi-gene fixture (deliberately repeated motif + concentrated interior codon context) — confirm the report tool's exact output string and that seeding reliably reproduces the trained choice. Do **not** validate against `Gene_Obj_Samples/` as if it were meaningful.

**One thing to double check on resume, not yet verified**: whether the notebook's `for atg in atgs: ... if tf: return atg, codvec` structure correctly falls through to the next candidate ATG after the fixed `break` (it should — `break` only exits the inner `for i in range(len(aa_seq))` loop, not the outer `for atg in atgs`), but this hasn't been traced end-to-end against the surrounding cell code beyond the snippet already read.

8. Everything below this line is unchanged from Handoff v1

Still accurate, still not double-checked against code (because the relevant code still doesn't exist): licensing (owner's call, don't touch license files), the CUDA_RNAFold repo open question (language-breakdown mismatch, worth checking `git status`/`.gitignore` there directly), the change-vector design discussion in v1 §5 (Metropolis-Hastings question still unresolved — is the GA's selection step doing proper accept/reject, or is the change vector just a heuristic proposal? worth settling before doing much more GA-tuning work), the fitness-function open concerns in v1 §6 (multi-objective/Pareto vs weighted-sum, MFE-as-wrong-target, pre-/post-splicing sequence conflation, SpliceAI context), the caching design in v1 §7 (not built yet — nothing to cache until folding/SpliceAI exist), the Colab deployment notes in v1 §10, and the LinearDesign benchmark-target idea in v1 §9. Read v1 (in git history, this commit's parent) for the full text of all of these if picking that work up.
