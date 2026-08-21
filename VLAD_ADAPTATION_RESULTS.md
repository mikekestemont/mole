# Vocabulary adaptation — LOAO result (2026-08-21)

Intermediary result. Reproduce with `scripts/run_loao_adapt.sh` (outputs in `outputs/loao_adapt/`).

**Question.** "All About VLAD" (Arandjelović & Zisserman, CVPR 2013) contribution #2, *vocabulary
adaptation*: given a frozen codebook fitted elsewhere, keep its Voronoi cells (assignment against the
original centres) but move each centre to the mean of a *target* corpus's descriptors in that cell — no
re-clustering. Does it help our charter writer-retrieval, leave-one-archive-out?

**Setup.** Base backbone `runs/pooled_bin_ft/checkpoint.pth` (vit_small@6ffcd327+step179100), VLAD K=100,
binarized + `--invert`, 224 px / overlap 0 / no zones, contrast foreground. Three arms per held-out
archive, each evaluated on its *own* gallery (`--cross-doc-only`, macro-mAP):

- **frozen** — codebook fit on the *other four* archives, applied to the held-out one as-is (the
  deployable "add a collection to a frozen index" baseline; held-out archive unseen in the vocabulary).
- **adapt** — that same frozen codebook, centres moved onto the held-out archive's descriptors
  (`mole codebook … --adapt-from <frozen> --adapt-min-assigned 50`). ~5–12 s, no k-means.
- **trans** — a full k-means refit on the held-out archive alone (the transductive upper bound).

## Results (macro-mAP)

| archive  | frozen | adapt      | trans  | adapt−frozen | adapt−trans |
|----------|--------|------------|--------|--------------|-------------|
| Antwerp  | 0.7807 | **0.8211** | 0.8171 | +0.0404      | +0.0040     |
| Brackley | 0.7576 | **0.7863** | 0.7758 | +0.0286      | +0.0104     |
| Flanders | 0.3970 | 0.4567     | **0.5109** | +0.0597  | **−0.0542** |
| Leroy    | 0.8117 | **0.8162** | 0.8090 | +0.0045      | +0.0072     |
| Utrecht  | 0.5609 | 0.6074     | **0.6238** | +0.0465  | **−0.0164** |

- **adapt vs frozen** (the deployable comparison): **mean Δmacro +0.0359, 95% CI [+0.0246, +0.0488],
  REAL, guardrail passed** (positive on all five; hand-weighted +0.0280). Largest where headroom is
  (Flanders +0.060, Utrecht +0.047), ~free where the base is already strong.
- **adapt vs full refit** (context): mean Δmacro **−0.0098, CI [−0.0189, −0.0007]**; guardrail *fails* on
  Flanders (−0.054), smaller shortfall on Utrecht (−0.016). On the other three, adapt ≥ refit.

## Reading

**Cells vs centres.** Adaptation moves centres, not cell boundaries. Where the pooled partition already
fits the collection (Antwerp / Brackley / Leroy), moving centres is *all* that's needed — and it beats an
archive-alone refit, because the frozen cells were conditioned on 12–16M descriptors from four archives
while the refit had only that one archive's 1.6–5M (a flakier 100-way partition). The two collections with
genuinely divergent scribes (Flanders, Utrecht) are the only ones where the refit wins: their rare hands
fall in the *wrong* cells, and only redrawing the cells recovers them. Adapt closes ~52 % of the Flanders
gap and ~74 % of the Utrecht gap; the remainder is a vocabulary (cell) problem, not a centre-bias one — so
a MAP-style soft-relevance factor on the centres would *not* help there.

**Scope / deployment.** This measured *per-archive* adaptation on each archive's own gallery = the honest
**within-collection** number. Each archive got its own adapted codebook, so those spaces are **not**
mutually comparable — a single *global cross-collection* index cannot use five different codebooks. For a
global index, adapt the one frozen universal codebook to the **union** of archives (one comparable space —
literally the paper's "images added after the vocabulary was learned" scenario); that recovers the global
frozen→transductive gap (~−0.032 historically) and is the natural next run.

**Relation to the earlier NetVLAD result.** Learning the aggregator by backprop on a frozen backbone
(Tier 2 / NetVLAD, closed negative in `36509be`) gave only **+0.006** on Antwerp; the closed-form
centre-move here gives **+0.040** on the same archive. NetVLAD learned a *fixed* aggregator from source
labels and never looked at the target collection; adaptation is unsupervised and transductive on the
target. Using target-domain statistics — the GMM-supervector / MAP intuition — is the lever, not a smarter
fixed aggregator.

## Intra-normalization A/B (contribution #1) — 2026-08-21, `scripts/run_intranorm_ab.sh`

Same backbone/geometry, one `.trans` (per-archive refit) codebook per archive, two embeds: plain VLAD
(Raven default) vs `--vlad-intra-norm` (per-cluster L2 before power-norm). macro-mAP:

| archive  | plain (A) | intra (B) | Δmacro   |
|----------|-----------|-----------|----------|
| Antwerp  | 0.8171    | 0.8152    | −0.0019  |
| Brackley | 0.7758    | 0.7669    | −0.0089  |
| Flanders | 0.5109    | **0.6033**| **+0.0924** |
| Leroy    | 0.8090    | 0.8120    | +0.0030  |
| Utrecht  | 0.6238    | 0.6409    | +0.0172  |

**mean Δmacro +0.0204, 95% CI [+0.0104, +0.0308], REAL, guardrail passed** (hand-weighted +0.0131).

**Reading.** Intra-normalization is not a wash — it is an archive-dependent lever driven by **burstiness**.
It costs a hair on the balanced archives (Antwerp/Brackley, within the −0.01 guardrail) and pays big on the
skewed ones: Flanders (**53 % one hand, KA_8**) +0.092, the largest single-lever move on that collection we
have found. The dominant hand's repeated strokes burst the VLAD vector; per-cluster L2 deflates that cluster
so the rare "different-mode" hands' residuals surface, lifting macro (which weights rare hands). Utrecht (86
diverse hands) +0.017 fits the same story. This **contradicts mole's current default** (intra-norm OFF for
Raven parity) — the §4.2 rule says turn it on, or make it an archive-level choice keyed on hand-/cluster-mass
skew (measurable unsupervised via VLAD cluster-occupancy concentration).

**Orthogonal to adaptation, and the real story is the stack.** These A-column values ARE the `.trans`
codebooks from the adaptation run, so intra-norm is a pure embed-time flag on top — no refit. Flanders across
everything: frozen+plain 0.397 → adapt+plain 0.457 → trans+plain 0.511 → **trans+intra 0.603** (+0.206 over
the naive baseline, hardest archive).

⚠️ **CAVEAT / open confirmation:** this A/B used the `.trans` codebooks; the deployable index uses the
**frozen/adapted** codebook. Burstiness is an encoding property, not a codebook one, so it should transfer —
but re-flipping `--vlad-intra-norm` on the frozen/adapt codebooks (Flanders suffices) is the confirmation,
and it directly tests whether **adapt + intra stack**.

## Intra-norm confirmation on the deployable codebooks (2026-08-21, `scripts/run_intranorm_codebooks.sh`)

The A/B above ran on `.trans` codebooks; the deployable index uses frozen/adapted. It transfers — with a
wrinkle. macro-mAP Δ (intra − plain) on each codebook:

| archive  | on FROZEN | on ADAPTED (the stack) |
|----------|-----------|------------------------|
| Antwerp  | −0.0081   | −0.0093                |
| Brackley | +0.0027   | −0.0100                |
| Flanders | **+0.1277** | **+0.1417**          |
| Leroy    | **−0.0123 ⚠** | −0.0090            |
| Utrecht  | +0.0597   | +0.0384                |
| **mean** | +0.0340 (CI [+0.018,+0.051]) | +0.0304 (CI [+0.015,+0.048]) |
| guardrail| **FAILED — Leroy −0.0123** | passed (all ≥ −0.01) |

**The intra-norm effect is real and transfers (Flanders +0.13 on the frozen codebook), but it is genuinely
burstiness-dependent, not universal:** big wins on skewed collections (Flanders, Utrecht), mild losses on the
balanced ones (Antwerp, Leroy, Brackley). On the **frozen** codebook the Leroy loss (−0.0123) trips the §4.2
guardrail — Leroy is 98 hands, macro≈micro, *no* dominant hand, so it has nothing to gain from burst
suppression and mildly loses from cluster equalisation. So intra-norm-ON is **not** a clean unconditional
default: it fails the guardrail in the global-frozen-index config.

### The deployable stack IS a clean win

frozen+plain → **adapt+intra**, the full recipe vs the naive baseline: **mean +0.0663, CI [+0.0439,+0.0907],
guardrail passed** (Antwerp +0.031 · Brackley +0.019 · **Flanders +0.2014** · Leroy −0.0045 · Utrecht +0.085;
hand-weighted +0.0465). Adaptation lifts Leroy enough (+0.005) that intra's −0.009 nets to −0.0045 — inside
the guardrail. So **adapt + intra together is guardrail-safe and strong even though intra-alone-on-frozen is
not.** Flanders goes 0.397 → 0.598 (+0.20) end to end.

## Decision / still open
- **Default flip (`intra-norm-default` branch):** justified for the adapted/transductive pipelines (guardrail
  clean), NOT clean for the global frozen index (Leroy −0.012). Options: (a) conditional default keyed on an
  unsupervised burstiness signal (VLAD cluster-mass concentration) — the principled fix; (b) keep opt-in +
  ship the recipe (intra ON for skewed archives, paired with adaptation); (c) flip anyway, accepting a small
  balanced-collection regression. DECISION PENDING.
- Deployable recommendation regardless: **adapted codebook + intra-norm** (the stack, +0.066, guardrail-safe).
- Union-adaptation is a NO-OP on the current (already-clustered) corpus; only relevant when a genuinely new
  archive is onboarded — then adapt the universal codebook to (corpus ∪ new) instead of re-clustering.
