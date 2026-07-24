"""GeneWriter — Colab stress-test / benchmark script.

Paste this whole file into a Colab cell (or `%run` it after cloning), edit
the CONFIG block below, and run. It does three things, each independently
toggleable in CONFIG:

  1. Loads real gene data + baseline ("analysis objects") data from paths
     you point it at (Drive-mounted or local to the Colab VM).
  2. Benchmarks gpu_change_vector.batch_calculate_change_vectors() -- the
     population-batched change-vector computation -- numpy vs cupy, across
     whatever population sizes you list, so you can extend the numbers
     already on record in Handoff.md (see its "Performance numbers" section)
     onto whatever GPU Colab hands you (T4 / L4 / A100 / etc).
  3. Optionally runs the *actual* GA pipeline end to end (ga.run_ga or
     schedule.run_schedule) against a real gene, so you can sanity-check a
     real run at whatever scale, not just the isolated batched-term math.

UPDATE: the GPU batched path IS now wired into the actual GA loop (ga.py's
seed_population()/refresh_change_vectors(xp=...), threaded through
run_ga(xp=...) and schedule.run_schedule(xp=...)/ScheduleContext.xp) --
step 3 below passes PIPELINE_XP through to whichever entry point it calls,
so USE_GPU=True now genuinely changes step 3's speed too, not just step 2's
benchmark. Two things worth knowing before reading a step-3 number:
  - Only the whole-population *exact* recomputes route through xp: initial
    seeding, and kill_off/select/flatten's pre-step refresh (run_ga's
    periodic refresh_every, too). "growth" itself still scores each new
    genotype by cheap per-parent diffing either way -- see ga.py's
    directed_evolution/replicate_and_mutate_random docstrings -- so a
    growth-heavy schedule with few refresh checkpoints will show a smaller
    GPU win than one that refreshes often (see SCHEDULE below).
  - Kmer still isn't batched (see gpu_change_vector.py), so it still runs
    per-individual inside every xp-routed call in step 3 too, same caveat
    as step 2's benchmark.
"""

import glob
import os
import subprocess
import sys
import time

# ============================================================================
# CONFIG -- every option currently exposed by the codebase, in one place.
# Edit this block; the rest of the script just reads from it.
# ============================================================================

CONFIG = dict(
    # ------------------------------------------------------------------
    # Repo / environment
    # ------------------------------------------------------------------
    # Where to git-clone (or find an existing clone of) GeneWriter on the
    # Colab VM. If REPO_DIR already exists and looks like the repo, cloning
    # is skipped and it's left as-is (so re-running a cell doesn't reclone).
    REPO_URL="https://github.com/LukeTheGeneWriter/GeneWriter.git",
    REPO_DIR="/content/GeneWriter",
    GIT_REF="main",  # branch, tag, or commit to check out after cloning

    # Install numpy + (if USE_GPU) a cupy wheel via pip. Turn off if you're
    # managing the environment yourself.
    INSTALL_DEPS=True,
    # cupy wheel to install -- Colab's runtime is CUDA 12.x regardless of
    # which GPU tier you attach (T4/L4/A100/H100), so cupy-cuda12x is the
    # right wheel across all of them. Change only if Colab's CUDA version
    # has moved on.
    CUPY_PACKAGE="cupy-cuda12x",

    # Try to use the GPU (cupy) for the step-2 benchmark. Auto-falls-back to
    # numpy-only if cupy import fails or no GPU is attached (e.g. you forgot
    # to set Runtime > Change runtime type > GPU) -- the script will print
    # which backend(s) it actually ran with.
    USE_GPU=True,

    # ------------------------------------------------------------------
    # Data locations -- gene objects (real genes) and analysis objects
    # (baselines the change-vector terms score against)
    # ------------------------------------------------------------------
    # Directory of real NCBI gene JSON files (what gene_io.load_genes()
    # reads -- one *.json per gene, RefGenes/NHGeneBodySupp/<geneID>.json
    # shape). This is the gitignored Gene_Obj_Samples/ folder -- point this
    # at a Drive-mounted copy, or upload the folder into the Colab VM first.
    GENE_OBJ_DIR="/content/drive/MyDrive/GeneWriter/Gene_Obj_Samples",

    # Mount Google Drive automatically if GENE_OBJ_DIR (or STANDARDS_DIR)
    # points under /content/drive. No-op outside Colab or if already mounted.
    AUTO_MOUNT_DRIVE=True,

    # Directory of precomputed genome-wide baseline JSON files (what
    # standards_io.load_standards() reads: RareCodonAnalysis.json,
    # CodonUsageAnalysis.json, CodonPairBiasAnalysis.json, GCAnalysis.json,
    # KmerAnalysis.json -- the real ~18,864-gene reference set, hours to
    # regenerate). Set to None to skip and fall back to
    # COMPUTE_LOCAL_BASELINES instead.
    STANDARDS_DIR=None,

    # If STANDARDS_DIR is None (or a file is missing from it), compute
    # baseline analysis objects on the fly from whatever's in GENE_OBJ_DIR
    # via baseline.compute_baselines(). NOTE: per baseline.py's own
    # docstring, a handful of local sample genes is a smoke-test baseline,
    # not a scientifically meaningful reference distribution -- use
    # STANDARDS_DIR for anything beyond a quick pipeline sanity check.
    COMPUTE_LOCAL_BASELINES=True,

    # Organism label attached to computed/loaded baselines (cosmetic --
    # stored on the AnalysisObjects dataclasses, not used in scoring math).
    ORGANISM="human",

    # ------------------------------------------------------------------
    # Which real gene/isoform to run against
    # ------------------------------------------------------------------
    # geneID (int, matches NaturalGene.geneID) to target, or None to just
    # use the first protein-coding isoform found in GENE_OBJ_DIR.
    GENE_ID=None,
    # Which isoform of that gene (index into its protein-coding isoforms,
    # in the order gene_io.protein_coding_isoforms() yields them). 0 = first.
    ISOFORM_INDEX=0,

    # ------------------------------------------------------------------
    # Change-vector term weights -- one entry per registered term. Must
    # cover every name in change_vector.registered_terms() (currently
    # RareCodons, CodonUsage, CodonPairBias, GC, Kmer) or run_ga/run_schedule
    # will raise. Positive weight = term counts normally; 0 = term computed
    # but ignored in scoring; negative = invert the term's pressure.
    # ------------------------------------------------------------------
    WEIGHTS=dict(RareCodons=1.0, CodonUsage=1.0, CodonPairBias=1.0, GC=1.0, Kmer=1.0),

    # ------------------------------------------------------------------
    # Step 2: batched change-vector GPU benchmark
    # ------------------------------------------------------------------
    RUN_BATCH_BENCHMARK=True,
    # Population sizes to benchmark at, one pass per size. Handoff.md's own
    # numbers cover 1,000 and 10,000 on an RTX 3050 (4GB) -- push higher
    # here on a bigger-memory Colab GPU (A100 40/80GB) to see where the
    # numpy/GPU crossover and any further speedup gains land.
    BENCHMARK_POP_SIZES=[1_000, 10_000, 50_000],
    # How the benchmark population's genotypes are generated: "random" seeds
    # every individual independently via ga.generate_seed() (fast, but no
    # realistic mutation clustering); "mutated" starts from one real seed and
    # applies a handful of random synonymous mutations per individual
    # (closer to what an actual GA generation's offspring look like).
    BENCHMARK_POP_MODE="mutated",
    BENCHMARK_MUTATION_RATE=0.05,  # only used when BENCHMARK_POP_MODE="mutated"

    # ------------------------------------------------------------------
    # Step 3: run the actual GA pipeline end to end (optional)
    # ------------------------------------------------------------------
    RUN_PIPELINE=True,
    # "run_ga"   -- ga.run_ga(): simple fixed-shape generation loop.
    # "schedule" -- schedule.run_schedule(): declarative step list, see
    #               SCHEDULE below. Prefer this for anything past a quick
    #               smoke test -- it lets you batch several cheap growth
    #               steps between exact-refresh checkpoints.
    RUN_MODE="run_ga",

    # Which array backend to route the pipeline's whole-population exact
    # recomputes through (seeding + kill_off/select/flatten refreshes -- see
    # module docstring): "gpu" (cupy, falls back to numpy if unavailable),
    # "numpy" (batched but CPU-only -- still faster than the per-individual
    # path per Handoff.md's own numbers), or "none" (original per-individual
    # path, xp=None, for an apples-to-apples before/after comparison).
    PIPELINE_XP_BACKEND="gpu",

    # Diagnostic-only progress logging for step 3 (no effect on the run's
    # results) -- turn this on before assuming a slow/stuck-looking run is a
    # bug. PROGRESS prints per-generation/per-step timing; PROGRESS_EVERY
    # additionally prints throughput every this-many individuals *within*
    # growth/refresh/seeding loops. This is what caught growth's lookahead
    # scoring being ~60-85x slower per individual than the batched refresh
    # path on real hardware (see Handoff.md sec 6) -- if a run looks hung,
    # turn this on before assuming xp="gpu" isn't working; growth doesn't
    # route through xp at all (see RUN_GA_OPTIONS.lookahead below), so a
    # slow growth phase looks identical whether xp is "gpu" or "none".
    PROGRESS=True,
    PROGRESS_EVERY=100,

    # --- options used when RUN_MODE == "run_ga" (ga.run_ga signature) ---
    RUN_GA_OPTIONS=dict(
        num_gens=10,                # generations to run
        target_size=None,           # cap population to this many distinct
                                     # individuals after each generation
                                     # (None = defaults to initial seed count)
        locvec=None,                # per-position 'F'/'T'/'I'/'S' exon tags;
                                     # None here means "use the real gene's
                                     # tags" -- filled in below from the
                                     # chosen isoform, not left as None
        flatten_every=None,         # run flatten_generation() every N gens
                                     # instead of plain reproduction (None =
                                     # off). Trades replicate concentration
                                     # for neighborhood breadth.
        flatten_recursion_limit=3,  # how many cash-in rounds flatten does
        lookahead=True,             # directed_evolution: score every
                                     # synonymous alt at the chosen position
                                     # (True) vs pick one at random (False,
                                     # cheaper). NOT "cheaper" as in a minor
                                     # optimization -- measured at
                                     # ~20-27 individuals/sec vs. ~350-1600/sec
                                     # for the batched refresh path on real
                                     # hardware (Handoff.md sec 6). This is
                                     # NOT affected by PIPELINE_XP_BACKEND --
                                     # growth is deliberately excluded from
                                     # xp/GPU batching (see ga.py's
                                     # directed_evolution docstring) -- if a
                                     # run seems stuck, this is almost
                                     # certainly why, not a GPU problem. Set
                                     # False here to confirm/mitigate.
        refresh_every=5,            # force an exact change-vector recompute
                                     # every N generations (0/None = never,
                                     # fastest but drift is unbounded; 1 =
                                     # always exact)
        save_dir="/content/genewriter_runs",  # per-generation population
                                               # checkpoints; None = don't save
        run_name="colab_stress_test",
    ),

    # --- schedule used when RUN_MODE == "schedule" (schedule.run_schedule)
    # Each dict's "kind" must be one of schedule.registered_steps():
    #   input      {"count": N}                       -- add N random seeds
    #   growth     {"rate", "mutation_chance",         -- reproduce every
    #               "directed_fraction", "lookahead"}     individual
    #   kill_off   {"percent_cut"}                     -- proportional cull
    #   select     {"target_size"}                     -- cap distinct pop
    #   flatten    {"recursion_limit"}                 -- breadth trade
    #   save       {}                                  -- checkpoint now
    #   repeat     {"times", "steps": [...]}            -- nested sub-schedule
    SCHEDULE=[
        {"kind": "input", "count": 2000},
        {"kind": "growth", "rate": 4, "mutation_chance": 0.1, "directed_fraction": 0.5, "lookahead": True},
        {"kind": "kill_off", "percent_cut": 30},
        {"kind": "repeat", "times": 5, "steps": [
            {"kind": "growth", "rate": 6, "mutation_chance": 0.05},
            {"kind": "select", "target_size": 3000},
        ]},
        {"kind": "flatten", "recursion_limit": 3},
        {"kind": "select", "target_size": 2000},
    ],
    SCHEDULE_SAVE_DIR="/content/genewriter_runs",
    SCHEDULE_RUN_NAME="colab_stress_test_schedule",
)

# ============================================================================
# Setup: clone repo, install deps, import GeneWriter
# ============================================================================


def _sh(cmd, **kw):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)


def setup_environment(cfg):
    if not os.path.isdir(os.path.join(cfg["REPO_DIR"], ".git")):
        _sh(f"git clone {cfg['REPO_URL']} {cfg['REPO_DIR']}")
    _sh(f"git -C {cfg['REPO_DIR']} fetch origin")
    _sh(f"git -C {cfg['REPO_DIR']} checkout {cfg['GIT_REF']}")
    _sh(f"git -C {cfg['REPO_DIR']} pull origin {cfg['GIT_REF']}")

    if cfg["INSTALL_DEPS"]:
        _sh(f"{sys.executable} -m pip install -q numpy pytest")
        if cfg["USE_GPU"]:
            _sh(f"{sys.executable} -m pip install -q {cfg['CUPY_PACKAGE']}")

    src_dir = os.path.join(cfg["REPO_DIR"], "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    if cfg["AUTO_MOUNT_DRIVE"]:
        needs_drive = any(
            str(cfg.get(k) or "").startswith("/content/drive")
            for k in ("GENE_OBJ_DIR", "STANDARDS_DIR")
        )
        if needs_drive and not os.path.isdir("/content/drive/MyDrive"):
            try:
                from google.colab import drive  # noqa: PLC0415
                drive.mount("/content/drive")
            except ImportError:
                print("Not running in Colab (or google.colab unavailable) -- skipping Drive mount.")


def detect_gpu_backend(cfg):
    """Returns (xp_module_or_None, label). None means GPU unavailable/off --
    callers fall back to numpy-only."""
    if not cfg["USE_GPU"]:
        return None, "GPU disabled in CONFIG"
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability  # forces a real device query
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        return cp, name
    except Exception as e:  # noqa: BLE001 -- diagnostic fallback, any failure just disables GPU
        return None, f"cupy unavailable ({e})"


# ============================================================================
# Data loading: gene objects (real genes) + analysis objects (baselines)
# ============================================================================


def load_gene_data(cfg):
    from genewriter import baseline, gene_io, standards_io

    if not os.path.isdir(cfg["GENE_OBJ_DIR"]):
        raise FileNotFoundError(
            f"GENE_OBJ_DIR does not exist: {cfg['GENE_OBJ_DIR']!r}. Point it at a "
            f"folder of real NCBI gene JSONs (Gene_Obj_Samples/-shaped)."
        )
    genes = gene_io.load_genes(cfg["GENE_OBJ_DIR"])
    if not genes:
        raise FileNotFoundError(f"No *.json gene files found in {cfg['GENE_OBJ_DIR']!r}.")
    print(f"Loaded {len(genes)} gene(s) from {cfg['GENE_OBJ_DIR']}")

    analysis_objects = None
    if cfg["STANDARDS_DIR"]:
        try:
            analysis_objects = standards_io.load_standards(cfg["STANDARDS_DIR"])
            print(f"Loaded genome-wide baselines from {cfg['STANDARDS_DIR']}")
        except FileNotFoundError as e:
            print(f"STANDARDS_DIR set but incomplete/missing ({e}); "
                  f"falling back to COMPUTE_LOCAL_BASELINES.")

    if analysis_objects is None:
        if not cfg["COMPUTE_LOCAL_BASELINES"]:
            raise RuntimeError(
                "No usable STANDARDS_DIR and COMPUTE_LOCAL_BASELINES is False -- "
                "no analysis objects to score against."
            )
        analysis_objects = baseline.compute_baselines(genes, organism=cfg["ORGANISM"])
        print(f"Computed local-sample baselines from {len(genes)} loaded gene(s) "
              f"(smoke-test quality only -- see baseline.py docstring).")

    return genes, analysis_objects


def pick_target(cfg, genes):
    """Returns (aa_seq, locvec) for the configured GENE_ID/ISOFORM_INDEX --
    locvec is the real 'F'/'T'/'I'/'S' location tag per residue, taken
    straight from the chosen isoform's codon list rather than defaulted to
    all-'I'."""
    from genewriter.gene_io import protein_coding_isoforms

    candidates = list(protein_coding_isoforms(genes))
    if cfg["GENE_ID"] is not None:
        candidates = [(g, iso) for g, iso in candidates if g.geneID == cfg["GENE_ID"]]
        if not candidates:
            raise ValueError(f"No protein-coding isoform found for geneID={cfg['GENE_ID']!r}")
    if not candidates:
        raise ValueError("No protein-coding isoforms found at all in the loaded genes.")

    idx = cfg["ISOFORM_INDEX"]
    if idx >= len(candidates):
        raise ValueError(f"ISOFORM_INDEX={idx} out of range (only {len(candidates)} available)")
    gene, iso = candidates[idx]

    aa_seq = iso.associatedProtein.aaSeq
    locvec = [loc for _codon, loc in iso.codons][:len(aa_seq)]
    if len(locvec) < len(aa_seq):
        locvec += ["I"] * (len(aa_seq) - len(locvec))

    print(f"Target: gene {gene.geneID} ({gene.geneName}), isoform {iso.isoformNumber}, "
          f"{len(aa_seq)} residues")
    return aa_seq, locvec


# ============================================================================
# Step 2: batched change-vector benchmark (numpy vs cupy)
# ============================================================================


def _make_benchmark_population(cfg, aa_seq, size):
    from genewriter.ga import generate_seed, replicate_and_mutate_random

    if cfg["BENCHMARK_POP_MODE"] == "random":
        return [generate_seed(aa_seq) for _ in range(size)]

    seed = generate_seed(aa_seq)
    pop = [seed]
    while len(pop) < size:
        pop.extend(replicate_and_mutate_random(
            seed, aa_seq, nreplicates=min(50, size - len(pop)),
            mutation_rate=cfg["BENCHMARK_MUTATION_RATE"],
        ))
    return pop[:size]


def run_batch_benchmark(cfg, aa_seq, locvec, analysis_objects, xp_gpu, gpu_label):
    import numpy as np
    from genewriter.gpu_change_vector import batch_calculate_change_vectors

    print("\n=== Step 2: batched change-vector benchmark ===")
    print(f"GPU backend: {gpu_label}")
    header = f"{'pop_size':>10} | {'numpy (s)':>10} | {'gpu (s)':>10} | {'speedup':>8}"
    print(header)
    print("-" * len(header))

    progress_every = cfg["PROGRESS_EVERY"] if cfg["PROGRESS"] else None

    for size in cfg["BENCHMARK_POP_SIZES"]:
        print(f"\nbuilding benchmark population of {size}...")
        t0 = time.perf_counter()
        pop_codons = _make_benchmark_population(cfg, aa_seq, size)
        print(f"  built {size} in {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        batch_calculate_change_vectors(pop_codons, analysis_objects, locvec, xp=np, progress_every=progress_every)
        numpy_time = time.perf_counter() - t0

        gpu_time_str = "n/a"
        speedup_str = "n/a"
        if xp_gpu is not None:
            batch_calculate_change_vectors(pop_codons, analysis_objects, locvec, xp=xp_gpu)  # warm up JIT/alloc
            t0 = time.perf_counter()
            batch_calculate_change_vectors(pop_codons, analysis_objects, locvec, xp=xp_gpu, progress_every=progress_every)
            gpu_time = time.perf_counter() - t0
            gpu_time_str = f"{gpu_time:.3f}"
            speedup_str = f"{numpy_time / gpu_time:.1f}x" if gpu_time > 0 else "inf"

        print(f"{size:>10} | {numpy_time:>10.3f} | {gpu_time_str:>10} | {speedup_str:>8}")

    print("\n(Kmer is not batched -- see gpu_change_vector.py's module docstring -- "
          "so it runs as a per-individual Python loop inside every row above, "
          "which dilutes the measured GPU speedup vs. Handoff.md's 4-term-only numbers.)")


# ============================================================================
# Step 3: run the actual GA pipeline end to end
# ============================================================================


def _resolve_pipeline_xp(cfg, xp_gpu):
    backend = cfg["PIPELINE_XP_BACKEND"]
    if backend == "none":
        return None
    if backend == "numpy":
        import numpy as np
        return np
    if backend == "gpu":
        if xp_gpu is None:
            print("PIPELINE_XP_BACKEND='gpu' but no GPU backend detected -- falling back to numpy.")
            import numpy as np
            return np
        return xp_gpu
    raise ValueError(f"Unknown PIPELINE_XP_BACKEND: {backend!r} (must be 'gpu', 'numpy', or 'none')")


def run_pipeline(cfg, aa_seq, locvec, analysis_objects, xp_gpu):
    from genewriter.change_vector import registered_terms, require_weights
    require_weights(registered_terms().keys(), cfg["WEIGHTS"])

    xp = _resolve_pipeline_xp(cfg, xp_gpu)
    xp_label = "none (per-individual)" if xp is None else getattr(xp, "__name__", str(xp))
    print(f"\n=== Step 3: running actual pipeline (mode={cfg['RUN_MODE']!r}, xp={xp_label}) ===")

    if cfg["RUN_MODE"] == "run_ga":
        from genewriter.ga import generate_seed, run_ga

        opts = dict(cfg["RUN_GA_OPTIONS"])
        opts["locvec"] = locvec  # override the placeholder None with the real tags
        opts["xp"] = xp
        opts["progress"] = cfg["PROGRESS"]
        opts["progress_every"] = cfg["PROGRESS_EVERY"]
        num_seeds = opts.pop("num_seeds", 200)
        seeds = [generate_seed(aa_seq) for _ in range(num_seeds)]

        t0 = time.perf_counter()
        final_pop = run_ga(aa_seq, seeds, cfg["WEIGHTS"], analysis_objects, **opts)
        elapsed = time.perf_counter() - t0

    elif cfg["RUN_MODE"] == "schedule":
        from genewriter.schedule import run_schedule

        t0 = time.perf_counter()
        final_pop = run_schedule(
            aa_seq, cfg["WEIGHTS"], analysis_objects, cfg["SCHEDULE"],
            locvec=locvec, save_dir=cfg["SCHEDULE_SAVE_DIR"], run_name=cfg["SCHEDULE_RUN_NAME"], xp=xp,
            progress=cfg["PROGRESS"], progress_every=cfg["PROGRESS_EVERY"],
        )
        elapsed = time.perf_counter() - t0

    else:
        raise ValueError(f"Unknown RUN_MODE: {cfg['RUN_MODE']!r} (must be 'run_ga' or 'schedule')")

    print(f"Final population: {len(final_pop)} distinct individuals in {elapsed:.1f}s")
    return final_pop


# ============================================================================
# Entry point
# ============================================================================


def main(cfg=CONFIG):
    setup_environment(cfg)
    xp_gpu, gpu_label = detect_gpu_backend(cfg)

    genes, analysis_objects = load_gene_data(cfg)
    aa_seq, locvec = pick_target(cfg, genes)

    if cfg["RUN_BATCH_BENCHMARK"]:
        run_batch_benchmark(cfg, aa_seq, locvec, analysis_objects, xp_gpu, gpu_label)

    if cfg["RUN_PIPELINE"]:
        run_pipeline(cfg, aa_seq, locvec, analysis_objects, xp_gpu)


if __name__ == "__main__":
    main()
