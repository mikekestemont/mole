# MOLE — State of the Self-Supervised System & Brief for the Supervised Phase

*A self-contained handoff for an LLM tasked with drafting the implementation plan for
`mole`'s **supervised** module. It summarizes what the self-supervised (SSL) system is,
what it has achieved, the data and its caveats, and the design space + constraints for the
supervised bit. Written 2026-07-19.*

---

## 1. What mole is, and the goal

**mole** = continual **self-supervised** handwriting embeddings for premodern documents
(medieval charters), refactored from Tim Raven's AttMask fork, in the **DINO / iBOT /
AttMask** lineage. A ViT-S/16 backbone is pretrained with self-distillation + masked-image
modeling; page-level descriptors are then built by **VLAD-aggregating the backbone's patch
tokens**.

**The end goal is a continually-growing cross-archive search / attribution index**: embed
every charter once, retrieve same-scribe documents across collections, and *suggest
attributions for unlabeled documents*. "Writer identity" here = the scribal **hand**.

The task is **writer retrieval**, evaluated leave-one-out (relevance = same `hand_id`):
micro **mAP**, **macro-mAP** (per-hand averaged — the honest metric under class skew),
**Top-1 / Top-5**.

---

## 2. The SSL pipeline (all built & working)

CLI = Python API. One command per stage:

```
prep (Sauvola binarize + optional YOLO zones) → train (SSL / warm-start finetune)
   → embed (pool patch tokens → page vectors, VLAD/mean/…) → eval (retrieval mAP) → viz
```

- **`mole prep <dir> --binarize sauvola [--max-side N]`** → `<dir>-bin/` (black-on-white
  binarized copies + `labels.csv` carried over with extensions rewritten). YOLO text-zone
  detection exists but is **optional** (see §4).
- **`mole train <cfg> [--init-from raven.pth]`** → single-GPU-first, step-level resume,
  Ctrl-C checkpoint, TensorBoard + embedding projector. `--init-from` warm-starts from a
  bare backbone checkpoint (heads re-init fresh — see §6).
- **`mole embed <ckpt> <dir> <out.npy> [--invert] [--codebook-from C.npy] [--pooling …]
  --set window_size=224 --set overlap=0`** → page embeddings + `<out>.mapping.json` +
  (for VLAD) `<out>.codebook.npy`.
- **`mole eval <out.npy> <dir> --topk 1,5`** → mAP / macro / Top-k, with a within-vs-across
  digitization breakdown (a scan-shortcut confound detector). Writes `<out>.eval.json`.
- **`mole viz <out.npy> --color hand --out x.html`** → self-contained 2D UMAP/PCA scatter.

Poolings available in `embed/pooling.py`: `mean`, `meanstd`, `cov` (bilinear), `cls`,
`vlad` (default/strongest), `patches`. VLAD (`embed/vlad.py`) is **hard-assignment**
seeded k-means (MiniBatchKMeans), reproducible, saved as a codebook artifact.

---

## 3. Settled findings (the levers, empirically pinned)

1. **Token scale is the dominant quality lever.** At embed, a `window_size`-px window → 224
   → 14×14 tokens, so each token sees `window_size/14` px of the page. `window_size=224`
   (16 px/token) matches raven's native scale and **doubled** Brackley (macro 0.38 → 0.76).
   Always embed at `window_size=224, overlap=0`.
2. **Polarity matters.** Raven was trained **white-on-black**. Sauvola writes black-on-white,
   so `mole embed --invert` (finetune configs bake `invert:true` and it's inherited).
3. **Foreground = local contrast (std), not "darkness."** Polarity-invariant and
   background-agnostic (works on parchment/color/bitonal); `--foreground-method contrast`.
   This makes YOLO zones largely unnecessary once binarized.
4. **Finetuning helps in proportion to data volume** (see §5): big lift on data-rich
   collections, ~no-op on starved ones. This is the entire motivation for pooled training.
5. **Frozen-codebook VLAD is the incremental search descriptor.** Fit a codebook once,
   freeze it, encode each archive against it → cross-comparable, per-doc additions free.
   It **generalizes cross-archive for common hands** (micro/Top-1 barely move) but
   **degrades rare hands** (macro drops), and *worse the more diverse the target*
   (Brackley→Antwerp −0.041 · HWI→Utrecht −0.075 · Antwerp→Flanders −0.102). ⇒ the
   universal codebook must be fit on **pooled charters**, not one archive.
6. **`mean` pooling (codebook-free, 384-d) ties raven-raw VLAD** (~0.72 macro); VLAD's edge
   (+0.09 on a finetuned model) comes from **multimodality** (cluster structure), which no
   first-order descriptor captures.
7. **Page-resolution cap (`--max-side`) is a compute knob, not a quality one** (worth ~0.01
   macro). Token scale, set by `window_size`, is what matters.
8. **PCA whitening (`--whiten`) is available but DE-EMPHASIZED** — the project owner has
   had poor results with it. Do not make it a default lever; prefer the frozen codebook and
   the learnable projections in §7 for structure/dimensionality.
9. HWI reproduction is **certified**: macro 0.795 vs Raven's paper 0.826 (pipeline faithful).

---

## 4. Data assets (5 charter archives, all binarized)

| archive | dir (`data/…`) | images | hands | labels | label provenance |
|---|---|---|---|---|---|
| Antwerp | `antwerp-bin` | 470 | 13 | ~450 | manual clustering |
| Brackley | `brackley-2350` | 300 | 14 | ~100 | scholarly table |
| Utrecht | `utrecht-bin` | 841 | 86 | 297 | manual (xlsx) |
| Flanders | `flanders-set-bin` | 383 | 11 | 266 | manual clusters (KA_*) |
| Leroy | `leroy-bin` | 1398 | 98 | **461** | **auto-matched (confidence-scored)** |

**Label caveats — critical for the supervised design:**
- **Partial labels everywhere.** Only a fraction of each collection is labeled, and some
  *unlabeled* documents genuinely belong to identified hands. ⇒ **every macro-mAP is a
  LOWER BOUND**; unlabeled docs are never true negatives.
- **Leroy labels are auto-matched** (columns `match_score, margin, gysseling_nr`): 461 of
  1398 have a `hand_id`; the rest are blank. Two extra risks: (a) **label noise** (a
  low-`match_score` row may be wrong); (b) **selection bias** — the matcher kept the
  confidently-matchable (cleaner/distinctive) hands, so the labeled subset is likely
  *easier* and Leroy's 0.782 is probably optimistic.
- **Class imbalance** within collections (Flanders KA_8 = 53% of labels; Leroy skewed) ⇒
  macro-mAP is the honest metric.
- **Multi-image-per-document.** Many charters have several images sharing one hand; sibling
  images retrieve trivially → a **scan-shortcut** that inflates Top-1/micro.
- **Label spaces are disjoint across archives** (different regions/eras; needs one
  confirmation that no scribe spans two archives). If confirmed, cross-archive labeled pairs
  are **free, guaranteed-different negatives** (see §6). Hand IDs must be **namespaced by
  archive** (`antwerp/B`, `flanders/KA_8`) before any pooled eval or supervised sampling.

---

## 5. Current results (raven-raw vs solo finetune, macro-mAP)

All: binarized + `--invert`, `window_size=224, overlap=0`, VLAD-100 transductive, contrast fg.

| archive (hands) | raven-raw macro | solo finetune | Δ | notes |
|---|---|---|---|---|
| Antwerp (13) | 0.718 | **0.827** | +0.109 | data-rich → big lift |
| Utrecht (86) | 0.515 | **0.651** | +0.137 | most data, hardest space → biggest lift |
| Brackley (14) | 0.764 | 0.778 | +0.014 | starved (300 img) → ~no-op |
| Flanders (11) | 0.385 | — | — | hardest by macro; "different-mode" scribes + KA_8 skew |
| Leroy (98) | 0.782 | — | — | strong; labels auto-matched (optimistic) |

**Reading:** finetune gain tracks data volume (Utrecht/Antwerp big, Brackley nil). Flanders
is the hardest honest number — small "different-mode" hands scatter; its codebook is already
optimal (a frozen external codebook only *hurt* macro), so the lever there is the backbone,
not the codebook. Top-1 is high across the board (features find *a* same-hand doc easily);
macro is where the difficulty and the headroom live.

**In flight:** a **pooled multi-archive Phase-A finetune** (`configs/pooled_bin.yaml`,
`runs/pooled_bin_ft`) — one warm-started vit_small over all 5 binarized archives (~3,392
imgs → 429,850 windows; batches mix archives via global shuffle). Hypothesis: pooling lifts
the starved/hard collections (Brackley, Flanders, Utrecht's small hands) without hurting the
strong ones (Antwerp, Leroy). Evaluated **per-archive against its own gallery** (never a
pooled gallery — more distractors would drop mAP from difficulty alone). This is the last
substantive SSL experiment; the pooled backbone (or raven-raw) is the base the supervised
phase refines.

---

## 6. Constraints & design rules the supervised phase MUST respect

1. **Asymmetric knowledge (the core rule).** Under partial labels:
   - **Positives from labels are trustworthy** (two same-`hand_id` docs really share a hand,
     even with incomplete coverage).
   - **Negatives involving unlabeled docs are NOT** (open world; unlabeled ≠
     confirmed-different). Using them as negatives injects **false negatives** — the one
     thing metric learning can't absorb.
   - ⇒ **RULE: mine positives freely; restrict negatives to confirmed DIFFERENT labeled
     hands; never treat "unlabeled" as a negative.** Stronger than "ignore the unlabeled."
2. **Low-data overfitting is the central obstacle.** Many hands have 2–3 documents (Flanders
   KA_4 = 2). Document-level triplets are far too few. Mitigations: operate at the
   **window/patch level** (Multiple-Instance Learning: bag = document carries the label,
   patches inherit it — thousands of instances, already augmented by DINO multi-crop). But
   **patch-level buys pairs, not document diversity** — patches from one page are correlated
   (same ink/parchment), so the effective sample size ≈ #documents, and a 2-doc hand stays
   hard. ⇒ **mine positives ACROSS documents** (same-page pairs teach nuisance, not hand);
   binarization already strips page-appearance nuisance.
3. **mole's SSL is non-contrastive** (DINO/iBOT self-distillation, no negative sampling), so
   a **SSL + supervised hybrid** is unusually clean: keep every image (labeled + unlabeled)
   in the SSL loss (can't create false negatives there) and add a supervised term only over
   labeled pairs. The unlabeled majority keeps the backbone general so the tiny supervised
   signal only has to *refine*.
4. **Free positives from structure.** Same-charter images are guaranteed same-hand with no
   annotation — a strong label-independent positive that sidesteps partial labels.
5. **Cross-archive negatives** are free (if no scribe spans archives) — a large certified
   negative pool that directly fights low-data overfitting. Requires namespaced hand IDs.
6. **The eval ground truth is imperfect** (partial + Leroy auto-matched) → supervised gains
   will be measured against a soft target; small lifts will be ambiguous. Consider a
   manually-verified eval subset, or `match_score`-thresholding Leroy, before investing.

---

## 7. The supervised design space — a ladder of "how much you train"

Frame the supervised phase as a **spectrum from cheap/small-data-safe to expensive/data-
hungry**. All tiers obey the §6 negative rule. The right tier depends on the collection's
data volume.

### Tier 1 — Frozen backbone + **learnable label-guided projection** (cheapest; small-data)
Freeze the expensive SSL backbone (no backbone overfitting on 2-doc hands) and learn only a
small map from frozen features to a low-dim space, using the **partial labels as
(semi-)supervision** to shape it, then project the **unlabeled** points to **suggest novel
attributions**.
- Concretely: **semi-supervised / parametric UMAP** (an inductive NN encoder, so it
  generalizes to unlabeled/new points; `umap-learn` supports a supervised `target`), or a
  small SupCon-trained MLP/linear metric head with the same freeze-backbone philosophy.
- **Why it fits mole:** it's the lightweight answer for exactly the collections where full
  finetuning fails (Flanders/Brackley small hands), and it directly produces the product
  goal — *even the SSL features, plus a cheap label-guided projection, can rank attribution
  candidates for unlabeled documents.*
- **Caveats:** supervised UMAP can over-collapse classes into artificially clean clusters →
  treat outputs as **ranked candidates for human review**, not a classifier; calibrate/report
  uncertainty; UMAP optimizes neighborhood-preservation, not discrimination, so a metric
  head may transfer better for pure retrieval — parametric UMAP is one instantiation of the
  idea, not the only one.

### Tier 2 — Frozen (or lightly-tuned) backbone + **learnable codebook (NetVLAD)** (middle)
The current VLAD codebook is **hard-assignment k-means** (non-differentiable). **NetVLAD**
(Arandjelović 2016) replaces the hard argmin with a **soft, differentiable** assignment, so
the codebook centroids become **learnable parameters trainable by backprop**. This is the
"parameterizable VLAD" option.
- **Why it matters:** it lets the **aggregation** be refined by the supervised signal
  (NetVLAD's original recipe was a **weakly-supervised triplet loss** — a direct match to our
  partial-label SupCon plan) without necessarily retraining the whole backbone; and it is the
  natural **learnable codebook** for the growing search index.
- **The catch (shared with any moving codebook):** a changing codebook = a changing embedding
  space ⇒ previously-embedded corpus goes **stale** and must be **re-embedded** on update
  (a batch job — the normal search-index rebuild pattern). Budget for it.

### Tier 3 — Full backbone **SupCon-hybrid finetune** (most expensive; needs data)
Keep every image in the iBOT/AttMask SSL objective + add a **Supervised Contrastive** term
(Khosla 2020 — prefer over vanilla triplet: uses all in-batch positives/negatives, more
stable, no triplet-mining) over labeled **patch/window** pairs, with the §6 negative rule and
cross-document positives. Best on the **pooled** corpus (enough data + diversity); overkill /
overfits on a single small archive.

**Decision guide by regime:** small/starved collection → **Tier 1**; medium → **Tier 2**;
pooled/large → **Tier 3**. They compose (e.g. Tier-3 backbone + Tier-2 NetVLAD head; or
Tier-1 projection on top of any backbone).

---

## 8. Future directions / hints

- **NetVLAD joint training** — train backbone + soft-assignment codebook *jointly* under the
  supervised signal; the highest-value version of Tier 2 (but the biggest build + the
  re-embed cost).
- **Attribution-suggestion loop (active learning).** Tier-1 projection ranks unlabeled docs
  near a hand → human verifies → labels grow → projection improves. Especially valuable given
  Leroy's auto-matched labels (verify/repair) and the partial-label reality everywhere. Needs
  **calibrated confidence** so suggestions are trustworthy candidates.
- **The incremental index workflow** (SSL's real deliverable, still un-exercised end-to-end):
  fit ONE universal codebook on pooled charters, freeze it, and support *add-an-archive →
  encode-against-frozen-codebook → grows the index*; periodic offline refit + re-embed. Online
  k-means (`MiniBatchKMeans.partial_fit`) can keep the frozen codebook fresh incrementally.
- **vit_base** — the pooled corpus is finally large enough to justify testing a bigger backbone
  (kept at vit_small so far for clean ablation; treat as a deliberate, separate experiment).
- **Cleaner eval** — a manually-verified hand subset (or `match_score` threshold on Leroy) to
  measure supervised lifts against, since partial/auto-matched labels blur small gains.

---

## 9. References

- Metric learning: FaceNet triplet (Schroff 2015); **SupCon** (Khosla 2020) — preferred.
- Learnable VLAD: **NetVLAD** (Arandjelović 2016).
- Label-guided low-dim projection: **(Parametric) UMAP** (McInnes 2018; Sainburg 2021),
  supervised/semi-supervised UMAP via the `target` API.
- False-negative / sampling-bias problem (why the negative rule): Debiased Contrastive
  (Chuang 2020) + Robinson 2021 — *reference, not a plan* (needs a class prior unestimable in
  low data; prefer the conservative negative rule).
- Semi-supervised (SSL-on-all + sup-on-labeled): S4L (Zhai 2019), PAWS (Assran 2021).
- SSL backbone: DINO (Caron 2021) / iBOT (Zhou 2021) — non-contrastive self-distillation.

---

## 10. Where things live (code pointers)

- SSL: `src/mole/selfsup/` (`train.py`, `vit.py`, `attmask.py`, `loss.py`, `head.py`).
- Embed / VLAD: `src/mole/embed/` (`pooling.py`, `vlad.py` = hard-assignment k-means).
- Eval: `src/mole/eval/retrieval.py` (per-hand AP is computed then **discarded** — a small,
  useful addition would be to serialize the per-hand breakdown).
- **Supervised: `src/mole/supervised/` — SCAFFOLD ONLY** (`datasets.py`, `metric.py`,
  `probe.py` all raise `NotImplementedError`; interfaces fixed, implementation is this phase).
- Data / labels: `src/mole/data/datasets.py` (`load_labels`), `patches.py`.
- Configs: `configs/*.yaml` (`pooled_bin.yaml` = the live pooled run).
- `ARCHITECTURE.md` / `WORKFLOW.md` are partially stale (window_size, phase table, missing the
  binarized/finetune era) — trust this document over them for current state.
