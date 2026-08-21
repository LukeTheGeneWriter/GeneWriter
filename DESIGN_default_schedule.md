# Design: a default GA schedule built on kicks, entropy, and greedy descent

Status: design only, nothing built. Written 2026-08-20 against
`schedule.py` @ 61c0a81. Companion to `HANDOFF_2026-08-20.md` §5 and §6,
which this turns into a concrete, runnable default.

---

## 1. The shape: population-level Iterated Local Search

The three ingredients named are not three separate features — they are the
three phases of one well-studied algorithm, **Iterated Local Search**
(ILS / basin hopping):

| ILS phase | our operator | status |
|---|---|---|
| local search to convergence | `directed_growth` (greedy 1-opt lookahead) | **exists** |
| convergence detection | entropy of `p(x)` (§6.2 of the handoff) | missing |
| perturbation ("kick") | k-opt saltation (§5 of the handoff) | missing |
| incumbent preservation | `protect` / `.protected` | **exists, but permanent — see §6.1** |
| acceptance | `select` / `kill_off` | **exists** |

The current default (`run_ga`, and every hand-written schedule in the repo)
has phase 1 and phase 5 only. It descends and it culls. It has no way to
detect that descent has stopped paying, and no move that can leave a 1-opt
basin without also discarding everything the run accumulated
(`HANDOFF_2026-08-20.md` §3.3). That is the whole gap this closes.

**The load-bearing design decision** is where the adaptivity lives. The
schedule DSL has no conditionals — only `repeat`. Rather than add an
`if`-step, put the adaptivity **in the kick step's targeting predicate**: a
kick step whose target set is "individuals that are good *and* stuck" is
automatically a no-op when nobody is stuck, and automatically ramps up as
the population converges. A fixed-cadence outer loop over a self-regulating
operator behaves like an adaptive schedule without needing control flow.
This is why the schedule below can be a flat list of `repeat` blocks.

---

## 2. New primitives needed

Three small ones, one medium, one blocker fix.

### 2.1 `ga.pressure_entropy()` — the stuck detector (free)

```python
def position_pressure(sol, changevecs, weights) -> list[float]:
    """p(x)[i] = SUM_t w_t * v_t(x)[i], per position, sanitized through
    _finite_nonneg. Already computed inside _ranked_positions(); this
    factors it out so it can be measured without also sorting."""

def pressure_entropy(sol, changevecs, weights) -> float:
    """Normalized Shannon entropy of the pressure profile, in [0, 1].

        q_i = p_i / SUM_j p_j
        H   = -SUM_i q_i ln q_i
        return H / ln(M)          # M = positions with degeneracy > 1

    LOW  (-> 0): one dominant hotspot. One directed mutation probably fixes
                 it. This individual is heading for the centre of its basin.
                 Leave it alone.
    HIGH (-> 1): dissatisfaction spread evenly across every mutable
                 position. No single mutation helps meaningfully.
                 **This is a stuck individual. Kick it.**
    """
```

Two details that matter and are easy to get wrong:

- **Normalize by `ln(M)`, not `ln(N)`**, where `M` = number of positions
  with more than one synonym. Immutable positions (Met, Trp) can never be
  fixed and mostly contribute 0 pressure anyway — including them in the
  denominator makes `H` depend on the protein's amino-acid composition, and
  the whole point is a number comparable across individuals and across
  targets.
- **`inf` collapses entropy to ~0.** `_finite_nonneg()` clamps a
  never-observed RareCodons window to `1e12`. One such position makes
  `q_i ≈ 1` and `H ≈ 0`, so the individual reads as "pinned by one hotspot"
  and is never a kick target. That is arguably the *correct* semantics — it
  genuinely does have one fixable catastrophic position — but it is an
  accident of the clamp, not a decision anyone made. Make it a deliberate
  one and say so in the docstring.

### 2.2 `Proposed_Solution.last_improvement` — the confirming detector (also free)

`HANDOFF_2026-08-20.md` §6.3 wants `Delta(x) = D(x) - min_{y in N1(x)} D(y)`
as ground truth for "certified 1-opt local optimum," and prices it at a
full 820-neighbour scan. There is a much cheaper 90% version:
`directed_evolution_batch()` **already computes exact `D` for every
candidate it scores**. Have it record, per individual,

```
last_improvement = D(parent) - min(D over the candidates it just scored)
```

on a new trailing-default field `Proposed_Solution.last_improvement: float = None`.

This is not certified — directed growth samples `rate` positions, not all
`N`, so `Delta_sampled >= Delta_true`. But `last_improvement <= 0` after a
`rate=8` step is strong evidence of a local optimum, it costs nothing, and
it is *empirical* rather than inferred. Use entropy as the cheap
population-wide prefilter and this as the confirming check on the
shortlist — exactly the two-tier structure §6.3 recommends, minus the
820-scan.

### 2.3 `@register_step("kick")` — the perturbation

```python
{"kind": "kick",
 "k": 2,                          # positions changed simultaneously
 "kicks_per_individual": 4,
 "target": {"quality_fraction": 0.50,    # must be best 50% by distance_from_optimal
            "entropy_fraction": 0.25,    # AND highest-entropy 25% OF THOSE
            "max_improvement": 0.0},     # AND last_improvement <= 0 (optional, §2.2)
 "position_weighting": "pressure",       # "pressure" | "uniform"
 "codon_choice": "random",               # "random" | "avoid_greedy"
 "blocked_positions": [],                # special_translation sites -- see §6.3
 "protect_colonists": True}
```

Semantics:

1. Rank the population by `distance_from_optimal`; keep the best
   `quality_fraction`. **This conjunction is the whole point** — §6.6's
   caution is that "kick the highest-entropy individuals" degenerates into
   "kick the worst individuals," which is a slower `kill_off` that burns
   the operator on things about to die. Filter for *quality first*, then
   rank for *stuckness within the survivors*. The targets are the
   individuals worth keeping that have stopped improving.
2. Within those, keep the top `entropy_fraction` by `pressure_entropy`.
3. For each target, emit `kicks_per_individual` offspring. Each draws `k`
   distinct positions without replacement, weighted ∝ `p(x)[i]`
   (`"pressure"`) — note that for a high-entropy individual `p` is nearly
   flat by construction, so this is close to uniform, which is fine and
   expected. At each drawn position pick a synonym uniformly at random
   (`"random"`), or uniformly among synonyms *excluding* the one greedy
   descent would pick (`"avoid_greedy"`, which guarantees the move is not
   one a hill climber could have made — more aggressive, more expensive,
   not the default).
4. **Score exactly**, via one `batch_calculate_change_vectors()` call over
   all colonists. Never diffed. A k-position change widens
   `diff_change_vector`'s excerpt and worsens its approximation, and these
   are precisely the offspring that need accurate scores — they look bad by
   construction and a pessimistic error kills them.
5. **Keep the parent.** ILS keeps the incumbent; the kick emits offspring
   alongside it, it does not replace it.
6. **Accept unconditionally.** Filtering kicks to only-improving ones
   rebuilds a hill climber that still cannot leave the basin. Colonize,
   then let re-descent and selection decide over the next cycle.
7. If `protect_colonists`, set `.protected` on every new genotype — the
   grace period. Without it, the very next `select` kills every colonist
   before it can descend into its new basin, and the operator does nothing
   at all. **This is not optional-in-practice; it is load-bearing.**

Cost, so this is not hand-waved. Targets = `0.50 × 0.25 × P = 0.125 P`.
At `P = 20,000` that's 2,500 targets × 4 kicks = **10,000 new exact change
vectors per kick step** — one `batch_calculate_change_vectors()` call of a
size `colab_stress_test.py` already runs routinely. Affordable every cycle.

### 2.4 `@register_step("deep_kick")` — exhaustive k=2, used once

`|N2| = C(P_mut, 2)` where `P_mut = SUM_i (d_i - 1)` ≈ 820 for a 400-codon
protein at mean degeneracy, so `|N2| ≈ 335,360` — and it scales as
`P_mut²/2`, so a 1000-codon target is ~6× that. Enumerable and exactly
scorable **per individual**, not per population.

So: **the default schedule samples, and reserves exhaustive k=2 for a
shortlist of ~10 finalists in the polish phase.** 10 × 335k = 3.35M scored
candidates, a few chunked GPU passes, once per run. Running it on 2,500
targets every cycle is 800M scores per cycle and is not a real option. This
is the one place `HANDOFF_2026-08-20.md` §5.1's "exhaustively enumerable"
needs qualifying: it's per individual, and the population is the problem.

The payoff is worth the once: an exact best double mutation is a move no
sequence of single steps can reach, found with certainty rather than
sampled.

### 2.5 `@register_step("report")` — build this first

There is currently no way to answer "is this schedule better than the old
one," which makes every number below unfalsifiable. A step that prints /
returns:

- median and p10 of `D/N`, plus the matching percentile against a
  `NaturalDistanceReference` (`natural_reference.py`, landed 61c0a81)
- per-term totals — the aggregate hides one catastrophic axis behind five
  good ones
- the entropy distribution (mean, p90) — this is the direct readout of
  whether the population is converging or stuck
- distinct-individual count, total replicate mass, protected count
- fraction of the population passing `is_within_natural_range(z=1.0)`

**Build `report` before `kick`.** `HANDOFF_2026-08-20.md` §7 item 3 says the
same thing for a different reason: plotting entropy against the existing
t-SNE is the cheap test of whether "stuck individuals cluster spatially" is
real at all, and that test should happen before the operator that depends
on it gets built.

---

## 3. The default schedule: two macros, alternating

The right factoring is **two macros that alternate**, not one flat cycle.
They are distinguished by what they do to the population's *size*:

| | EXPLORE | EXPLOIT |
|---|---|---|
| operators | `protect`, `flatten`, `kick` | `directed_growth`, `release_protection`, `kill_off_by_term`, `select` |
| population | **inflates** (P -> 2–3P) | **deflates** (back to P) |
| replicate mass | spent (flatten cashes it in) | rebuilt (offspring collide onto existing genotypes) |
| change vectors | exact (kick scores its colonists exactly) | diffed, then exact at the `select` checkpoint |
| what it optimizes | coverage of new basins | depth within the current basin |

That size asymmetry is the useful invariant, and it's what makes the pair
legible: **EXPLORE never selects; EXPLOIT always ends by selecting.** If
you find yourself wanting a `select` inside EXPLORE, the macro boundary is
in the wrong place.

### 3.1 The macro boundary is set by the grace period

The one hard coupling between the two macros: a kick colonist is *worse
than its parent by construction* — that is the entire point of accepting
kicks unconditionally. If a selection step runs before the colonist has
descended into its new basin, every colonist dies and the operator does
nothing at all.

So the grace period spans the macro boundary, and that fixes where the
boundary goes:

- EXPLORE **takes out** the protection (elite by criteria; colonists by
  origin — see §3.4) and hands the population over inflated.
- EXPLOIT **spends** the grace on its first descent steps, then
  `release_protection`, then selects.

The grace period is therefore measured in *descent steps*, not in cycles —
which is the semantically meaningful unit, and a better definition than the
one-cycle version in the earlier draft of this design.

### 3.2 The macros

```python
def explore(kick_k=2, kicks_per_individual=4, flatten=False,
            term_rotation=TERMS, quality_fraction=0.50, entropy_fraction=0.25):
    steps = [
        # Insurance FIRST, before anything destructive. flatten in
        # particular destroys replicate mass, and a just-flattened
        # individual is at its most vulnerable (handoff sec 2.4).
        {"kind": "protect", "criteria": [["distance_from_optimal", 0.05]]
                                        + [[t, 0.10] for t in term_rotation]},
    ]
    if flatten:
        steps.append({"kind": "flatten", "recursion_limit": 3})
    steps.append(
        {"kind": "kick", "k": kick_k,
         "kicks_per_individual": kicks_per_individual,
         "target": {"quality_fraction": quality_fraction,
                    "entropy_fraction": entropy_fraction}})
    return steps          # note: no select, deliberately


def exploit(depth=2, rate=8, target_size=None, cull_term=None):
    steps = [{"kind": "directed_growth", "rate": rate} for _ in range(depth)]
    steps.append({"kind": "release_protection"})       # grace expires here
    if cull_term:
        steps.append({"kind": "kill_off_by_term", "term": cull_term,
                      "percent_cut": 10})
    steps += [
        {"kind": "select", "target_size": target_size},
        {"kind": "report"},
    ]
    return steps
```

### 3.3 EXPLORE contains two different kinds of exploration

Worth keeping distinct rather than letting the macro name blur them:

- **`flatten` = diffusion.** Cashes replicate mass in for single-mutation
  neighbors and collapses every count to 1. It **fills the current
  neighborhood in**. Intensification by breadth.
- **`kick` = escape.** A simultaneous k-position move, targeted at
  individuals that are good and stuck. It **leaves the current
  neighborhood**.

They are complements, and they want different cadences. `flatten` only pays
once mass has actually accumulated — and mass accumulates only when
offspring collide onto genotypes already in the population, which is itself
a convergence signal. So: **kick every EXPLORE, flatten every ~4th.** Both
firing every time also stacks two population expansions on top of each
other, which is how you OOM (§6.6).

One non-obvious interaction: `flatten` collapses *every* individual's count
to 1, protected ones included — protection shields from
`kill_off`/`select`, not from flatten. The elite genotype survives as a
distinct individual, but loses its mass. That's fine, arguably desirable,
and should not be a surprise when it shows up in `report`.

### 3.4 EXPLORE uses two different protection mechanisms

Also worth being explicit about, because they are not interchangeable:

- **Criteria-based** (`protect` step, `mark_protected`): ranks by metric
  and shields the top fraction. Protects the *elite*.
- **Origin-based** (the `kick` step setting `.protected` on its own
  offspring): protects the *colonists*.

A criteria-based `protect` can never shield colonists — they rank near the
bottom by `distance_from_optimal` by construction. That's why the grace
period has to be built into the kick step itself and can't be expressed as
another `protect` step after it.

### 3.5 Should the alternation ratio anneal?

The instinct is yes — explore-heavy early, exploit-heavy late. **It should
not be scheduled by hand, because the targeting predicate already does it,
and it does it in the right shape.**

The intuitive "decay exploration over time" is actually wrong here. Trace
the run:

- **Early**: population is random seeds, gradient everywhere, descent is
  extremely productive, almost nothing is stuck. Kicks are *pointless* and
  the entropy predicate mostly no-ops. Effective explore intensity: low.
- **Middle**: descent exhausted for most of the population, entropy high,
  the predicate fires on a large fraction. Effective explore intensity:
  peak.
- **Late**: only the polish phase, where you want everything to settle.

So the natural curve is a **hump, not a monotone decay** — and a fixed 1:1
alternation with a self-regulating kick produces that hump for free, at
constant parameters. This is the same point as §1: the adaptivity lives in
the predicate, not in the control flow.

The one place it must be forced rather than emergent is the end: the polish
phase hard-disables kicks so the population settles into the bottom of
whatever basins it occupies. Otherwise the run finishes mid-colonization
and hands you sequences that are worse than their own parents.

### 3.6 The assembled default

```python
def default_schedule(
    cycles: int = 12,
    flatten_every: int = 4,
    depth: int = 2,              # directed_growth steps per EXPLOIT
    rate: int = 8,               # replicates per directed_growth
    kick_k: int = 2,
    population_size: int = None, # MUST be sized for the peak -- see sec 6.6
    finalists: int = 200,
    term_rotation: tuple = ("CodonPairBias", "GC", "Kmer", "RareCodons", "CodonUsage"),
) -> list:

    P = population_size

    # --- Phase A: seed, then descend to first convergence -----------------
    sched = [
        {"kind": "input", "count": P},
        *exploit(depth=3, rate=rate, target_size=P),
    ]

    # --- Phase B: the alternation ----------------------------------------
    for c in range(cycles):
        sched += explore(kick_k=kick_k,
                         flatten=(c % flatten_every == flatten_every - 1),
                         term_rotation=term_rotation)
        sched += exploit(depth=depth, rate=rate, target_size=P,
                         cull_term=term_rotation[c % len(term_rotation)])

    # --- Phase C: polish -- EXPLOIT only, no kicks ------------------------
    sched += [
        {"kind": "release_protection"},
        *exploit(depth=3, rate=rate, target_size=finalists * 10),
        {"kind": "natural_range_cutoff", "threshold": 3.0},
        {"kind": "select", "target_size": finalists},
        {"kind": "deep_kick", "top_n": 10},          # exhaustive k=2, once
        {"kind": "directed_growth", "rate": rate},
        {"kind": "natural_range_cutoff", "threshold": 2.5},
        {"kind": "select", "target_size": finalists},
        {"kind": "report"},
    ]
    return sched
```

Notes on the assembled form:

- **No random-mutation growth anywhere.** Deliberate reversal of the
  current default, and the strongest single claim here. At
  `mutation_chance=0.05` a random offspring differs from its parent at
  ~13.4 codons — one draw from a 10^28-genotype shell, indistinguishable
  from a fresh restart that happens to keep a bit of the parent (handoff
  §3.2). Every role it plays is played better by something in the macros:
  initial diversity by `input`, local diffusion by `flatten`, basin escape
  by `kick`. If you want it back, use `mutation_chance = 0.005–0.01` (≈1–3
  codons), not `0.05`.
- **`kill_off_by_term` rotates one term per EXPLOIT at 10%.**
  Lexicographic pressure — the classic multi-objective escape that *does*
  reach concave regions of the Pareto front, which fixed-weight linear
  scalarization provably cannot (handoff §3.4). Already built, currently
  unused by default, cheap. Rotation matters: pressuring one fixed term is
  just a weight change in disguise.
- **`select` is the workhorse; plain `kill_off` never appears.** `kill_off`
  draws with weight `D_i`, not `number_i * D_i`, so replicate mass acts as
  a survival shield rather than proportional exposure. Until that semantics
  question is settled (handoff §2.4), a default shouldn't lean on it.
  `select` operates on distinct individuals, which is also the currency
  that controls cost.
- **`natural_range_cutoff` only in polish**, tightened 3.0 -> 2.5 across two
  passes. It's a hard cutoff that drops an individual's entire replicate
  count, not a proportional cull. Run earlier and it clears the board:
  random seeds and fresh colonists are far outside natural range by
  construction.

---

## 4. Why this is expected to beat the current default

Three specific mechanisms, each falsifiable with `report`:

1. **The 1-opt basin is now escapable while retaining accumulated
   quality.** Today the only exit is a ~13-codon jump that discards
   everything. A simultaneous 2-codon change can cross a barrier where each
   single step is uphill and the pair is downhill — the exact class of move
   a 1-opt hill climber provably cannot make. (Biologically: compensatory
   mutation under epistasis.)
2. **The escape budget is spent on individuals that can use it.** An
   individual at the centre of its basin spends a k-step burst getting back
   to where it already was; it is surrounded by its own basin in every
   direction. An individual at the edge is already adjacent to unexplored
   territory, and the same budget carries it much further. Targeting is
   worth at least as much as the operator.
3. **Two of the three classic Pareto escapes are switched on by default**
   (rotating `kill_off_by_term` = lexicographic; `natural_range_cutoff` =
   ε-constraint; per-term `protect` = keeping axis champions alive). All
   three already exist in the codebase and none is in any current default.

**How we'd know it worked** — the acceptance criteria, in order of
directness:

- median `D/N` percentile against `NaturalDistanceReference`, vs. the same
  budget spent on the current default;
- specifically, does the output beat *the target gene's own natural CDS*?
  That is the one claim that is defensible in front of a customer;
- p90 entropy over cycles: should sawtooth (rise as the population
  converges, drop after each kick). A flat-high entropy trace means kicks
  aren't landing; a flat-low one means the targeting predicate is never
  firing and the whole schedule reduces to the old one;
- per-term totals, not just the aggregate;
- fraction passing `is_within_natural_range(z=1.0)` *and*
  `natural_range_cutoff(2.5)` — aggregate and per-axis together, since the
  aggregate gate misses a single blown axis and CPB/Kmer are the terms most
  likely to blow.

---

## 5. Build order

1. `report` step + entropy statistic. Free, and it makes everything else
   measurable. Plot entropy against the existing t-SNE first — that is the
   cheap test of whether the "edge of basin" framing is real before any
   operator depends on it.
2. `release_protection` (§6.1). Blocker; ~10 lines.
3. `kick` with sampled k=2, targeting by entropy ∧ quality.
4. `default_schedule()` builder + a run against the current default at
   equal compute budget.
5. `last_improvement` (§2.2) as the confirming detector, once there's a
   baseline to compare targeting variants against.
6. `deep_kick` (exhaustive k=2, polish phase only).

Deliberately **not** in this design: NSGA-II non-dominated sorting
(handoff §3.4) — highest ceiling, biggest change, and it would change what
"better" means mid-experiment. Scope it after this lands, not alongside it.

---

## 6. Blockers and open decisions found while designing this

### 6.1 `mark_protected` is permanent, and that breaks any ILS loop — BLOCKER

`ga.mark_protected()` is explicitly monotonic: it only ever sets
`.protected = True`, never clears it. `select_survivors()` and
`_kill_off_by_scores()` both skip protected individuals unconditionally,
and `select_survivors()` will exceed `target_size` rather than remove one.

A schedule that calls `protect` once per cycle — which every design above
needs, both for the incumbent and for the colonist grace period —
therefore **grows the immortal set every cycle until the entire population
is protected, selection becomes a no-op, and the population grows without
bound.** This is not a tuning problem; it makes the schedule structurally
non-viable. It has to be fixed before any of this runs.

Two options:

- **`{"kind": "release_protection"}`** — clears `.protected` on every
  individual. Minimal (~10 lines), keeps the flag boolean, and pairs
  naturally with a `protect` immediately after it at the top of each cycle.
  Grace period is then exactly one cycle. **Recommended.**
- **TTL**: `protected_until: int` compared against `ctx.step_count`, with
  `protect` taking a `duration`. Strictly better semantics (per-lineage
  grace periods of different lengths, no global lapse), but it touches
  `Proposed_Solution`, both cull paths, and every existing caller of
  `.protected`. Worth doing if grace periods longer than one cycle turn out
  to matter empirically — which `report` can tell us.

### 6.2 The DSL has no conditionals

Resolved by design rather than by feature: adaptivity lives in the kick's
targeting predicate, which no-ops when nothing is stuck. Worth stating
explicitly so nobody adds a `when` step for this reason. A real
`{"kind": "when", ...}` would still be useful for early stopping ("halt
once p50 is inside natural range"), which this design does not attempt.

### 6.3 Special-translation sites — now more urgent

Selenocysteine UGA/SECIS, non-AUG starts, frameshift junctions
(`special_translation.py`). A k-position burst is `k`× more likely to hit
one than a single directed step, and unlike directed growth it is *not*
guided away from them by the change vector. `blocked_positions` on `kick`
is the minimum viable guard; the real fix is a run-level position blocklist
that every operator respects. Already open backlog; kicks raise its
priority.

### 6.4 Change-vector drift under a longer growth chain

Two `directed_growth` steps between exact refreshes is a guess. The drift
from `diff_change_vector`'s excerpt approximation is bounded per step but
compounds, and nobody has measured how fast at real scale. `report` should
include a periodic exact-vs-diffed delta so this knob can be set from data
rather than from caution.

### 6.5 `kill_off`'s mass-as-shield semantics

Sidestepped here (the default leans on `select`), not resolved. Still a
live design question — see `HANDOFF_2026-08-20.md` §2.4.

### 6.6 Population sizing must account for the EXPLORE peak — practical trap

`ga.suggest_population_size()` targets `available_ram_bytes() * ram_fraction`
with `ram_fraction=0.5`, i.e. it sizes P to fill half of free RAM, and the
population lives entirely in host memory for the life of the run. Both the
`input` and `select` steps call it with defaults when `count`/`target_size`
is omitted.

But an EXPLORE macro deliberately inflates the population — `flatten`
multiplies distinct individuals and `kick` adds `0.125 P * kicks_per_individual`
colonists — so **the peak is 2–3x P, not P.** Taking the auto-sizing default
and then tripling it lands at ~1.5x available RAM: OOM, most likely in the
middle of a long Colab run.

Consequence for the API: **`default_schedule()` cannot rely on the steps'
auto-sizing.** It has to call `suggest_population_size()` itself and divide
by the expansion factor, then pass explicit `count`/`target_size` everywhere.
Either that, or `suggest_population_size()` grows a `peak_multiplier`
parameter so the intent is recorded rather than buried in a division. The
second is better — the number is a property of the schedule shape, and a
bare `// 3` in a builder is exactly the kind of thing that gets lost.

`report` should print peak distinct-individual count per cycle so the
expansion factor is measured rather than assumed.

### 6.7 Back-to-back steps each force their own exact refresh

`kill_off_by_term` and `select` both call `refresh_change_vectors()` before
doing their own work. Placing them adjacently — which the EXPLOIT macro does
— means two full exact recomputes over the whole population with nothing
changing in between. At real scale that refresh is one of the more expensive
things in the run.

Cheapest fix: a dirty flag on `ScheduleContext` (set by growth/kick, cleared
by any refresh) so the second call is a no-op. Alternatively a standalone
`{"kind": "refresh"}` step plus a `refresh: false` param on the steps that
currently force it — more explicit, more to get wrong. Not a blocker, but
it's a straight ~2x saving on the EXPLOIT tail and worth doing before any
timing comparison against the old default, or the comparison is unfair to
the new schedule.
