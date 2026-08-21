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

## Still open
- `scripts/run_intranorm_ab.sh` — intra-normalization A/B (contribution #1), not yet run.
- Union-adaptation for the global single-index deployment (see *Scope* above), not yet run.
