# mole supervised module — implementation plan

*Drafted 2026-07-19 against `SUPERVISED_HANDOFF.md` (the canonical brief). Companion doc: this is the
executable plan; the handoff is the ground truth on findings and constraints.*

---

## 0. Recommended approach (summary)

Build **Tier 1 first**: a frozen-backbone, **masked-SupCon projection head** trained at the
window level on pooled labeled windows from all five archives. It is the only tier that is
structurally safe at N=2 documents (no per-hand parameters; one small shared head regularized by
~220 hands), it trains in minutes on cached features, it composes with every present and future
backbone, and it directly yields the attribution-suggestion product. Parametric UMAP is demoted to
visualization — the retrieval space should be optimized by a metric loss, not
neighborhood-preservation. **Tier 3 second** (two-stream SSL+SupCon hybrid, warm-started from the
pooled Phase-A checkpoint): the finetune-∝-data finding says the backbone is the big lever, and the
pooled infrastructure already exists. **Tier 2 (NetVLAD) is parked**: your own evidence says the
codebook is not the bottleneck (frozen codebooks travel; Flanders' own codebook is already optimal;
VLAD's multimodality edge survives freezing), and NetVLAD carries the re-embed cost without Tier 3's
expected gain — build it only on a defined trigger (see Phase 6). Before any training, one short
phase hardens the **measurement** (per-hand AP serialization, hand-held-out splits, cross-charter
relevance, Leroy confidence thresholding), because without it no supervised lift of the expected
size (+0.03–0.05 macro) will be readable against partial/auto-matched labels. The **negative rule is
enforced structurally, not by policy**: supervised batches contain only labeled windows and pair
masks have three states — positive (same hand, different document), negative (two confirmed
*different* labeled hands), ignore (same document, and anything touching an unlabeled image) — so a
(labeled, unlabeled) negative is unrepresentable in the loss.

---

## 1. Tier recommendation & sequencing

| order | tier | what | regime it serves | N=2-doc behavior |
|---|---|---|---|---|
| 1st | **Tier 1** — frozen backbone + masked-SupCon head | small projection (v0: linear 384→128; v1: MLP) on cached frozen window descriptors, trained on pooled labels from all 5 archives | everyone, immediately; the *only* option for starved hands | The hand contributes cross-doc window pairs (hundreds of pairs, effective sample size 2). It cannot be overfit *individually*: the head has no per-hand parameters and is shared across ~220 hands; early stopping is on **held-out hands**, so memorizing a 2-doc hand buys the optimizer nothing on the stopping metric. Worst case: the head ignores that hand (falls back to SSL geometry). |
| 2nd | **Tier 3** — SupCon-hybrid backbone finetune | pooled SSL loss on *all* images + λ·masked-SupCon on a supervised stream; warm-start from pooled Phase-A checkpoint | pooled corpus only (never a solo small archive) | A 2-doc hand is a tiny fraction of the SupCon term and zero of the SSL term; the SSL loss over the unlabeled majority anchors the backbone, so the supervised term refines rather than carries. λ is warmed up and kept small; held-out-hand eval is the tripwire for over-collapse. |
| parked | **Tier 2** — NetVLAD learnable codebook | soft-assignment differentiable VLAD replacing `embed/vlad.py` hard k-means | only if T1+T3 plateau AND the aggregation is shown to be the residual bottleneck | Codebook is global (no per-hand params), so N=2 is not the risk; the risk is cost/benefit — moving codebook ⇒ re-embed the corpus, against evidence that the codebook already generalizes. |

Trigger to un-park Tier 2 (Phase 6): after T1+T3, if VLAD-over-projected-descriptors still beats
mean-over-projected by ≥0.05 macro (multimodality still doing heavy lifting) *and* the frozen pooled
codebook loses ≥0.03 macro vs transductive on ≥2 archives — then learnable aggregation has headroom.
Otherwise NetVLAD is complexity without a target.

They compose: the Phase-5 deliverable is explicitly *Tier-3 backbone → re-cache features → retrain
the Tier-1 head on top*. The head is ~minutes to retrain, so it rides along with every backbone.

**Base checkpoint policy:** all supervised work pins ONE base backbone for comparability. Use
`runs/pooled_bin_ft` final checkpoint once its guardrail eval passes; until then, prototype against
`checkpoints/raven_checkpoint.pth` (everything below is checkpoint-agnostic by construction).

---

## 2. Module & API design

Existing scaffold signatures (`load_labeled_pairs`, `train_metric`, `train_probe`) are kept verbatim
and become the real entry points. New symbols are additive.

### 2.1 `supervised/datasets.py` — index, sampling, masks

```python
@dataclass
class SupItem:
    path: Path            # image file
    archive: str          # dataset folder name (from discover_datasets)
    hand: str             # NAMESPACED: f"{archive}/{hand_id}"  (§4 handoff rule)
    doc: str              # charter/document group id — sibling scans share it
    confidence: float | None   # labels.csv `confidence` (Leroy match_score lands here)

@dataclass
class SupervisedIndex:
    items: list[SupItem]                  # labeled images ONLY
    by_hand: dict[str, list[int]]
    docs_by_hand: dict[str, set[str]]
    unlabeled: list[tuple[str, Path]]     # (archive, path) — kept for `suggest`,
                                          # structurally absent from any sampler
    def retrievable_hands(self, min_docs: int = 2) -> list[str]: ...
    def split_hands(self, holdout_frac: float = 0.2, seed: int = 0,
                    stratify_by_archive: bool = True
                    ) -> tuple["SupervisedIndex", "SupervisedIndex"]: ...
    def stats(self) -> str: ...           # per-archive hands/docs/pair counts, N=2 census

def load_labeled_pairs(labels_root: str | Path,
                       min_confidence: float | None = None) -> SupervisedIndex:
    """FIXED interface, now real. Walks discover_datasets(labels_root) (works on the
    pooled symlink dir), namespaces hands by archive, resolves doc ids via
    doc_id_for(), applies the confidence floor (drops low-confidence rows to
    UNLABELED — they are then neither positives nor negatives)."""

def doc_id_for(filename: str, archive: str) -> str:
    """Per-archive sibling-scan grouping (regex table + optional per-dataset
    `doc_ids.csv` override). MUST be reviewed per archive — wrong grouping leaks
    same-page nuisance into 'cross-document' positives (open decision D3)."""
```

Feature cache (Tier-1 fuel; one GPU pass, then everything is CPU):

```python
def build_feature_cache(checkpoint: str | Path, index: SupervisedIndex,
                        out_dir: str | Path, *, window_size: int = 224,
                        overlap: float = 0.0, invert: bool = True,
                        fg_method: str = "contrast") -> Path:
    """Per labeled+unlabeled image: run the frozen backbone over the standard
    embed windowing, store ONE 384-d descriptor per window = mean of
    foreground (contrast) patch tokens. Writes cache.npy [N_windows, 384] float32
    + cache.index.json (window -> item, doc, hand, archive) + provenance
    (model_id, embed params). ~200k labeled windows × 384 × 4B ≈ 300 MB. Reuses
    the embed-path model loading/windowing verbatim — no second implementation."""
```

Sampler + masks (where the negative rule physically lives):

```python
class HandBatchSampler:
    """P×D×W batches over LABELED windows only: hands_per_batch=16 hands ×
    docs_per_hand=2 docs × windows_per_doc=4 windows (batch 128). Guarantees
    ≥2 distinct docs per sampled hand (hands with 1 doc are never anchors —
    they contribute only as negatives). same_archive_frac=0.5 forces at least
    half the hands in a batch to share one archive, so negatives are not
    dominated by the trivial cross-archive contrast (risk R1).
    D1 resolution (2026-07-19): archive disjointness is likely but NOT
    guaranteed → cross-archive negatives are high-confidence, not certified.
    Kept by default (config sup.cross_archive_negatives: true; residual
    false-negative rate negligible); within-archive negatives
    (labeler-certified) remain the core via same_archive_frac. The flag lets
    us ablate or disable them cleanly."""
    def __init__(self, index: SupervisedIndex, cache: FeatureCache,
                 hands_per_batch=16, docs_per_hand=2, windows_per_doc=4,
                 same_archive_frac=0.5, seed=0): ...

def pair_masks(hands: list[str], docs: list[str]
               ) -> tuple[Tensor, Tensor]:      # (pos_mask, neg_mask), bool [B,B]
    """pos = same hand AND different doc (cross-document positives ONLY).
    neg = different hand — both confirmed labeled by construction of the batch.
    Everything else (same doc; diagonal) is IGNORED: excluded from numerator
    AND denominator. Unlabeled windows cannot appear here at all."""
```

Unit tests (Phase 1 gate, `tests/test_sup_datasets.py`):
- no pair in `neg_mask` involves an unlabeled item (construct a poisoned batch attempt; assert it
  cannot be built);
- same-doc pairs are in neither mask;
- every anchor row used by the loss has ≥1 positive;
- namespacing: identical raw hand strings in two archives never become positives;
- `min_confidence` demotes rows to unlabeled (not to negatives).

### 2.2 `supervised/metric.py` — loss + both trainers

```python
def masked_supcon(z: Tensor, pos_mask: Tensor, neg_mask: Tensor,
                  temperature: float = 0.07) -> Tensor:
    """L_i = -1/|P(i)| · Σ_{p∈P(i)} log[ exp(z_i·z_p/τ) / Σ_{a∈P(i)∪N(i)} exp(z_i·z_a/τ) ]
    z L2-normalized. Denominator = positives + CONFIRMED negatives only (the
    departure from stock SupCon, which would count every non-positive as a
    negative). Anchors with |P(i)|=0 are dropped from the mean."""

def train_metric(config_path, base_checkpoint, labels_root, output_dir=None):
    """FIXED interface. config `sup.tier` dispatches:
    - "head"   → Tier 1: build/reuse feature cache from base_checkpoint; train the
                 projection head on cached descriptors (AdamW, cosine, ~30 epochs,
                 minutes on CPU/MPS); early-stop + model-select on HELD-OUT-HAND
                 retrieval macro-mAP (mean pooling of projected descriptors — the
                 fast proxy). Writes head.pt {state_dict, in_dim, out_dim, kind,
                 base_model_id, config_hash} + report.json + the split file used.
    - "hybrid" → Tier 3: two-stream loop (below), same checkpoint format as
                 selfsup training so `mole embed` consumes it unchanged."""
```

Config (`configs/sup_head.yaml`, new `sup:` section):

```yaml
sup:
  tier: head            # head | hybrid
  head: linear          # linear | mlp   (v0 = linear, see §2.4)
  out_dim: 128
  temperature: 0.07
  sampler: {hands_per_batch: 16, docs_per_hand: 2, windows_per_doc: 4, same_archive_frac: 0.5}
  cross_archive_negatives: true   # D1: high-confidence, not certified; flag to ablate
  min_confidence: null  # training-side label floor (Leroy; decision D2)
  holdout_frac: 0.2
  seed: 0
  lambda_sup: 0.1       # hybrid only; linear warmup over first 2 epochs
```

Tier-3 hybrid loop (inside `train_metric`, reusing `selfsup/train.py` machinery):
- **Stream A (SSL)**: the existing pooled PatchWindowDataset loader, untouched — every image,
  labeled + unlabeled, full DINO/iBOT/AttMask loss. No negatives exist here, so no rule to violate.
- **Stream B (supervised)**: `HandBatchSampler` yields labeled window *crops* (light augs: the
  binary-safe subset — blur + slight jitter; no morphology, no aggressive RRC — stroke width and
  scale are signal); forward through the *student*; window descriptor = fg-token mean; z = 2-layer
  MLP projection (SupCon-standard: loss on the projection, retrieval uses backbone features).
- Per step: `L = L_ssl(A) + λ·masked_supcon(B)`; λ warmup 0→λ_max. Cost ≈ +1 small forward/step.
- EMA teacher, resume, checkpointing, TB all inherited. Log `loss/supcon` separately.

### 2.3 `supervised/probe.py` — diagnostics + calibration

```python
def train_probe(embeddings_path, labels_root, output_dir=None) -> "ProbeReport":
    """FIXED interface. Cheap frozen-embedding diagnostics on PAGE embeddings:
    (a) kNN-classifier accuracy + logistic probe, hand-held-out CV;
    (b) per-hand accuracy table (the 'which hands are hopeless' view);
    (c) fits and saves the score→precision calibration model that
        `mole suggest` consumes (§3.3). Writes probe.json + calibration.pkl."""
```

### 2.4 Head application at embed time — `mole embed --head`

`mole embed <ckpt> <dir> <out.npy> --head runs/sup_head_v1/head.pt --pooling vlad|mean ...`

- **v0 (linear head)**: apply the 384→128 projection to every foreground **patch token**, then pool
  exactly as today. Legitimate because a linear map commutes with the fg-token mean used in
  training (train-on-window-mean ≡ apply-to-tokens-then-mean), and VLAD is refit *in the projected
  space* (new codebook artifact; `--codebook-from` works unchanged). Sidecar records
  `head_id = sha(head.pt)+base_model_id`; eval/viz untouched.
- **v1 (MLP head, only if v0 plateaus)**: nonlinearity breaks token/window commutation, so the unit
  of pooling becomes the **window descriptor**: embed = project each window's fg-token mean, then
  mean- or VLAD-pool the ~40–150 projected window vectors per page. Config-flagged
  (`sup.head: mlp` ⇒ embed auto-switches granularity from the head artifact metadata).
- Guard: `--head` whose `base_model_id` mismatches the checkpoint → hard error (a head is only
  valid on the backbone it was trained against).

### 2.5 CLI additions (one command per stage, CLI = Python API)

```
mole sup cache  <ckpt> <pool_root> <cache_dir>          # build_feature_cache
mole sup train  <cfg>  <ckpt> <pool_root> [--out DIR]   # train_metric (tier from cfg)
mole sup probe  <emb.npy> <root> [--out DIR]            # train_probe
mole suggest    <emb.npy> <root> [--head H] [--out DIR] # §3 (first-class command)
mole eval       ... --per-hand-out --min-confidence F --cross-doc-only --holdout-hands FILE
mole eval-compare A.eval.json B.eval.json               # paired per-hand bootstrap (§4)
```

---

## 3. Attribution suggestion path (`mole suggest` — the product)

### 3.1 Scoring
For each **unlabeled** document (page embedding in the head-projected space):
1. Cosine kNN against all labeled pages of the same archive. Cross-archive suggestion is off by
   default but available as `--cross-archive` (experimental): per D1 the archives are only
   *probably* scribe-disjoint, so a strong cross-archive match is a potential discovery worth
   surfacing for review, not a known-false hit.
2. Hand score `s(hand) = mean of top-2 cosines` among that hand's documents (max is
   scan-shortcut-prone if a sibling ever leaks; mean-top-2 wants agreement from 2 docs). Also
   record `margin = s(best) − s(second)`.

### 3.2 Calibration (what makes it trustworthy)
Leave-one-out on the **labeled** set of the same archive: each labeled doc is scored as if
unlabeled; we know whether its top-1 hand is correct. Fit isotonic regression score→P(top-1
correct), **per archive** (spaces differ). Every suggestion ships with `calibrated_p` = empirical
precision at that score. Below `--abstain-p` (default 0.5, reported not hidden): flagged
`possible-new-hand / abstain`.

### 3.3 Output (for human review, never auto-labeling)
- `suggestions.csv`: `archive, filename, rank, hand_id, raw_score, margin, calibrated_p,
  n_support_docs, exemplar_1..3` (nearest labeled filenames).
- `review.html` (same pattern as the prep QC sheets): unlabeled thumbnail left, top-3 candidate
  hands' nearest exemplar thumbnails right, scores + calibrated_p; sorted by calibrated_p.
- D5 resolution (2026-07-19): `mole suggest` NEVER writes to `labels.csv`. Suggestions live only
  in their own artifacts (`suggestions.csv` + `review.html`); if persistence is wanted later it
  will be a separate per-dataset `suggested_labels.csv` that the user merges manually. Any future
  human-accepted rows would carry `source=mole-suggest, confidence=calibrated_p` (the loader
  already supports both columns), but that merge is always a manual, user-driven step.

### 3.4 Credibility harness — masked-label recovery (`mole suggest --self-test`)
Hide a random 20% of labels (stratified per archive; only from hands with ≥3 docs, so the hand stays
represented), run the full suggest path, report **precision@1 / precision@5 vs coverage** and a
reliability diagram (is calibrated_p honest?). This is the product metric, and it is *immune to the
partial-label eval noise* — ground truth for hidden docs is known by construction. Run it per
archive; it is also the cleanest Tier-1-vs-baseline product comparison (same test, head on/off).

---

## 4. Evaluation methodology under partial / auto-matched labels

### 4.1 Fix the instrument first (Phase 0)
1. **Per-hand AP serialization** — `retrieval.py` computes then discards per-hand AP (handoff §10);
   write it into `eval.json` (`per_hand: {hand: {ap, n_docs}}`). Everything below consumes it.
2. **`--min-confidence`** on eval: uses `LabelTable.confidence`. Generic — bites wherever a
   dataset carries a `confidence` column. D2 resolution (2026-07-19): Leroy is used AS-IS for now
   (no labels.csv regeneration; user will revisit that dataset later) — until then Leroy's numbers
   stay flagged as optimistic (auto-matched selection bias) in every report.
3. **`--cross-doc-only`**: relevance = same hand AND different `doc_id` (sibling scans excluded from
   the relevant set and from top-k). Kills the scan shortcut; this is the honest metric for any
   *supervised* claim (a label-trained model must never get credit for re-finding a sibling scan).
4. **Frozen split files**: `splits/sup_holdout_v1.json` (seeded `split_hands`, committed) — every
   tier and every rerun uses the identical hand split.
5. **`mole eval-compare`**: paired per-hand ΔAP between two eval.json files; hand-level bootstrap
   (10k resamples) CI on Δmacro + sign test. A lift is "real" iff CI excludes 0.

### 4.2 Protocol for every label-trained model
- Train on **train-hands only** (80%); the 20% held-out hands are *unseen classes*.
- Report, per archive (own gallery, never pooled):
  (a) **held-out-hand macro-mAP** — queries = held-out-hand docs, gallery = full archive. The
  headline: measures whether supervision improved the *geometry*, not memorization, and doubles as
  the over-collapse tripwire (unseen classes suffer first);
  (b) train-hand macro (the overfit gap, sanity only);
  (c) `--cross-doc-only` variants of both;
  (d) suggest `--self-test` precision@1 (product metric).
- Decision rule per candidate: mean over the 5 archives of held-out Δmacro vs the pinned baseline,
  bootstrap CI excluding 0, **and no archive worse than −0.01**.

### 4.3 Success criteria
| tier | success | kill / rethink |
|---|---|---|
| Tier 1 v0 | mean held-out Δmacro ≥ +0.03 with CI>0; Flanders (most headroom, 0.385 base) ≥ +0.05; self-test precision@1 ≥ 0.6 at 50% coverage | Δ ≤ +0.01 everywhere → try v1 MLP@window; if that also flatlines, frozen features are the ceiling → weight shifts to Tier 3 |
| Tier 3 | mean held-out Δmacro ≥ +0.05 over the pooled-SSL baseline (same splits), guardrail: no archive regresses | supcon term collapses held-out macro while train-hand macro climbs → λ too big / over-collapse; halve λ, or stop at Tier 1 |
| Tier 2 (if triggered) | ≥ +0.02 over the T3-backbone + frozen-pooled-codebook stack | anything less: discard, keep hard k-means |

If Δs land inside the noise band across archives: escalate to a small **gold subset** — manually
verify the labels of the 10 worst per-hand-AP hands on Flanders + Antwerp (~30 min with the
review.html tooling) and re-measure on it. Build this only if needed.

---

## 5. Phased milestones (each independently testable, short-run)

**Phase 0 — Measurement hardening** *(CPU, ~1 session)*
Per-hand AP out; `--min-confidence`; `--cross-doc-only` (needs `doc_id_for` — pull that helper into
`data/`); split files; `mole eval-compare`. (Leroy stays as-is per D2.)
**Checkpoint:** re-run `mole eval` over the existing `outputs/*.npy` artifacts (no GPU); produce the
first per-hand tables + Leroy thresholded-vs-full comparison; eyeball Flanders per-hand APs.

**Phase 1 — `datasets.py` for real** *(CPU, ~1 session)*
`SupervisedIndex`, namespacing, `doc_id_for` + per-archive regex review, `HandBatchSampler`,
`pair_masks`, unit tests (§2.1 list). **Checkpoint:** `pytest tests/test_sup_datasets.py` green +
`index.stats()` printout reviewed (retrievable hands per archive, N=2 census, positive/negative
pair counts, same-archive fraction achieved).

**Phase 2 — Tier-1 v0 end-to-end** *(GPU: one cache pass ~ embed-time; train: minutes, CPU/MPS ok)*
`mole sup cache` on the pooled dir (pinned base ckpt) → `mole sup train` (linear head, d=128) →
`mole embed --head` on Antwerp + Flanders → §4.2 protocol. **Checkpoint:** held-out-hand Δmacro on
Antwerp (sanity: strong base, expect small +) and Flanders (headroom: the real test), via
`eval-compare`. Go/no-go against §4.3 before touching the other archives.

**Phase 3 — `mole suggest` + calibration + self-test** *(CPU)*
§3 complete, including review.html. **Checkpoint:** masked-label recovery numbers on all 5 archives
(head on vs off — the first product-level supervised-lift measurement); you hand-review the top-20
Flanders suggestions in review.html. This phase is valuable even if Phase 2's Δ is modest — it
ships the product on whatever the best current space is.

**Phase 4 — Tier-1 v1 (conditional)** *(CPU)*
Only if v0 plateaus: MLP head at window granularity; compare mean-of-projected-windows vs
VLAD-over-projected-windows. **Checkpoint:** same harness as Phase 2; also records the
Tier-2-trigger measurement (§1).

**Phase 5 — Tier-3 hybrid on the pooled corpus** *(GPU: ~2h probe, then ~1 day)*
Two-stream `train_metric(tier=hybrid)`, warm-start = pooled Phase-A final. First a **5-epoch probe
run** (λ=0.1): check `loss/supcon` descends, SSL losses undisturbed, quick guardrail eval on
Antwerp+Flanders mid-checkpoint. Then the full run (~15–20 epochs; plateau lesson says don't
overshoot). **Checkpoint:** the standard per-archive guardrail loop + §4.2 protocol; then re-run
Phase-2/3 (re-cache → retrain head → suggest self-test) on the new backbone — the composed stack is
the final deliverable.

**Phase 6 — NetVLAD (parked)**
Only on the §1 trigger. Design sketch is in the handoff; not planned further here on purpose.

Fits the working protocol: every GPU step is a command you run on the server; my side is code +
CPU-verifiable tests; sign-off between phases.

---

## 6. Risks, failure modes, open decisions

### Risks
- **R1 — archive-shortcut negatives.** Cross-archive negatives are plentiful and *easy* (different
  script/era); a head can lower the loss by separating archives while doing nothing for
  within-archive retrieval (the actual eval). Mitigation: `same_archive_frac=0.5` in the sampler +
  the eval is per-archive-own-gallery, so the shortcut earns zero credit; monitor the within-archive
  supcon loss separately.
- **R2 — over-collapse / hurting unseen hands.** Supervised shaping can tighten labeled hands at the
  expense of the open-world geometry unlabeled docs live in. Tripwire: held-out-hand macro (unseen
  classes) is the model-selection metric everywhere; Tier 3 additionally keeps the SSL term on all
  images.
- **R3 — measurement too noisy for +0.03.** With 95–278 queries per archive and partial labels, a
  small Δ can drown. Mitigation: paired per-hand bootstrap (same queries both sides), 5-archive
  mean, self-test precision (noise-free by construction), gold subset as escalation.
- **R4 — Leroy label noise as training positives.** A wrong auto-match becomes a false *positive*
  (pulls two different hands together) — the mirror image of the negative rule. Mitigation:
  training-side `min_confidence` (can be stricter than eval's), and the suggest loop is the
  long-term repair channel for Leroy.
- **R5 — doc-id grouping wrong.** If sibling scans are grouped as different docs, "cross-document"
  positives silently become same-page nuisance pairs (exactly what §6.2 of the handoff forbids), and
  `--cross-doc-only` under-excludes. Mitigation: per-archive regex table reviewed by you (D3), a
  printed sample of groupings in `index.stats()`, override file supported.
- **R6 — space versioning.** Even Tier 1 changes the embedding space: a head (+ its projected-space
  codebook) is a versioned artifact pair; mixing embeddings across head versions is invalid. The
  sidecar `head_id` + the existing model-id mixing warning cover detection; the frozen-index
  workflow treats a head bump like a codebook refit (batch re-embed — cheap, it's a linear map).
- **R7 — moving baseline.** Pooled Phase-A finishes mid-project. Pin the base checkpoint the moment
  its guardrail eval passes; all §4 comparisons are against that pin, not a moving best.

### Decisions — resolved with the user 2026-07-19 (D3 still open)
- **D1 — cross-archive disjointness: RESOLVED (likely, not guaranteed).** User: chance of a shared
  scribe is small but there is no strict guarantee. ⇒ cross-archive negatives are demoted from
  certified to high-confidence: kept by default behind `sup.cross_archive_negatives: true`
  (residual false-negative rate negligible; within-archive negatives are the certified core via
  `same_archive_frac`). Corollary: cross-archive *suggestions* become a potential-discovery
  feature (`mole suggest --cross-archive`, experimental), not a known-false case. Namespacing
  stays regardless (needed against raw-string collisions).
- **D2 — Leroy: RESOLVED (use as-is).** No labels.csv regeneration for now; user will return to
  the dataset later. `--min-confidence` is built generically and simply won't bite on Leroy yet.
  Leroy numbers remain flagged optimistic in every report.
- **D3 — doc/charter-id rules per archive: OPEN (the one remaining).** Working guesses: Antwerp
  `0-XXXX_DD_MM_YYYY-NN.png` → strip trailing `-NN`; Utrecht `1108.06.26a`-style → date+letter
  stem; Brackley → 1 image/charter already (manifest.csv `charter`); Flanders `134_2`-style →
  leading number; Leroy → `gysseling_nr` column (to verify). Plan: implement the guesses in
  Phase 1; `index.stats()` prints a per-archive sample of groupings for user sign-off BEFORE any
  training consumes them (R5 guard).
- **D4 — trainer location: RESOLVED (as recommended).** Tier-1 head trainer lives in `metric.py`
  under `sup.tier: head`; `probe.py` stays diagnostics + calibration. User follows the
  recommendation while reserving critical review.
- **D5 — labels are read-only: RESOLVED.** `mole suggest` never touches `labels.csv`; outputs are
  standalone artifacts only; any future persistence = a separate `suggested_labels.csv`, merged
  manually by the user.
- **D6 — head defaults: RESOLVED (locked as v0 defaults).** out_dim 128 (compression bottleneck =
  regularizer; 220+ hands still separable; VLAD 38,400-d → 12,800-d), temperature 0.07 (SupCon/
  DINO-family standard; lower = harder negative focus but unstable in low data), cosine metric
  (matches `mole eval` and the SupCon formulation). Whether `--head` becomes the *default* embed
  path is deferred until Tier 1 passes its success bar (recommendation: opt-in until then).
- **D7 — Tier-3 timing: RESOLVED (it waits).** Tier 3 runs only after the pooled Phase-A eval
  lands (warm-start AND baseline); ~2h probe run then ~1 GPU-day.

### Assumptions flagged (not verified)
- Labeled-window cache ≈ 200k windows / ≈300 MB (from ~1,574 labeled docs at 224/overlap-0) — order
  of magnitude, not measured.
- Windows per charter page at 224/overlap-0 lands in the ~40–150 range, enough for window-granularity
  VLAD in v1 — to be confirmed from the cache index in Phase 2.
- Light-aug choice for the Tier-3 supervised stream (no morphology / gentle crops, because stroke
  width and scale are writer signal) follows the TTA reasoning in the parked notes; it's a design
  judgment, not an experiment result — cheap to ablate later.
