# Cross-archive hand attribution — plan (2026-09-17)

**Question.** Do any scribes appear in more than one of the four Low-Countries charter
collections mole has been run on separately — Antwerp, Utrecht, Leroy (Gysseling
13th-c. Middle Dutch) and the comital chancery of Flanders? This is a *serendipity*
search: there is no ground truth, so the deliverable is a ranked, reviewable list of
candidate **document pairs**, each shown next to a yardstick that exists in every
archive (the within-archive same-hand / different-hand pairs).

**Decisions (2026-09-17, Mike):** the four archives only (no Brackley — no overlap
possible); no date-based controls or filters (dates are not uniformly available, and
Antwerp↔Utrecht is not ruled out); no cross-archive identifications exist yet; the
document pair is the unit — a page from a loose, heterogeneous hand is as eligible as
one from a tight hand, so within-archive cohesion is context, never a gate.
Built: `mole cross` (§6), validated on the July `universal_full` vectors (§3.1).

## 0. Priors: which pairs can even overlap

| archive (live folder) | pages / labeled | date range | region |
|---|---|---|---|
| `antwerp-bin` | 470 / 470 | 1300–1357 | Brabant (Antwerp) |
| `utrecht-charters` | 748 / 261 | 973–1250 (bulk 1200–1250) | Sticht Utrecht, recipient archive; hands named by issuing chancery |
| `leroy-sauvola` | 1304 / ~1100 (auto-matched) | 13th c., bulk 1250–1300 | Middle Dutch, mostly Flanders |
| `comital` | 313 / 209 | 12th–13th c. (**date column needed**) | comital chancery; shelfmarks RAGent, ADN Lille, RABrugge, SAGent, … |

Plausibility, from dates and geography alone:

* **comital ↔ Leroy** — same region, overlapping decades, *same physical repositories*
  (RAGent, RABrugge). Strongest expectation of shared hands — and the one pair where
  *identical charters* are genuinely possible, whatever the metadata says.
* **Utrecht ↔ comital** — overlap 1150–1250; Utrecht is a recipient archive, so an
  issuer's chancery scribe can sit in it. Plausible.
* **Utrecht ↔ Leroy**, **Antwerp ↔ Leroy** — thin decade of overlap each (1240s, 1290s–1300s).
* **Antwerp ↔ comital** — only if comital runs into the 14th c. (unknown until dated).
* **Antwerp ↔ Utrecht** — ≥50 years apart on the filename dates, but not ruled out
  (dating is not uniform); listed like every other pair.

**Found on the zero-th pass (§3.1):** the "Utrecht" set carries charters of the
Cistercian fonds **Ten Duinen – Ter Doest** (hands `TDTD`, `TDTDa..g`), and the
Flanders set holds pages from the same fonds (`…Archief Grootseminarie Brugge_Fonds
Ten Duinen-Ter Doest_…`). A known-provenance overlap between two of the four —
the natural first place to look for a confirmed cross-archive hand.

## 1. What the toolchain already does

* `mole embed <ckpt> data/<pool>` walks root + immediate subfolders → **one** `.npy`
  whose `mapping.json` paths carry the folder; `mole eval`, `mole review`, `mole viz`
  key labels and doc ids by that folder (`antwerp-bin/R`, `comital/KA_8`, …).
* `mole codebook <ckpt> <dirs…> --scale-target auto` pins the pooled script module in
  the codebook's provenance; every `mole embed --codebook-from` rescales each archive
  to it on the fly (`PageScaler`, factor = target/current). Needed here: comital is
  prep-normalized to 42.9 px, utrecht-charters to 48.4 px, antwerp-bin and
  leroy-sauvola not at all.
* `mole review` already scores **merges** (hand pairs whose cross-similarity matches
  their own cohesion) and **attributions** (unlabeled page → known hand) over the
  namespaced hands — cross-folder pairs fall out of both for free. Merges are parked
  (scored, not rendered); attributions are calibrated *within* the labeled pool.
* `mole eval` prints a within-/cross-dataset breakdown whenever labels span folders.
* `outputs/universal_full/` = the July pooled model + one codebook over all five
  archives — but on retired `utrecht-bin` / `leroy-bin` and without a scale target.

## 2. Design

### 2.1 One space
One checkpoint, one codebook (K=100, pinned scale), one `--vlad-intra-norm` setting
for the whole index. Intra-norm ON: it is a burstiness fix, which is also what an
archive-specific scan artifact is; it lifts the two skewed sets (+0.09 Flanders,
+0.06 Utrecht) and costs Leroy a little (`VLAD_ADAPTATION_RESULTS.md`).

### 2.2 Measure the domain gap before believing any match
Report, on the pooled `.npy`:
* **kNN archive purity** at k=1/5/10 vs chance (share of a page's neighbours from its
  own archive). Near 100 % ⇒ raw cosine ranks archives, not hands, and step 2.3 is
  mandatory.
* Cosine distributions: within-archive vs cross-archive, and per archive-pair.
* Per-archive within-archive macro-mAP under the shared codebook, next to the
  own-codebook numbers (`outputs/pooled_final`, per-collection runs) — the price of
  the shared space (July: −0.03 average, Flanders −0.09).

### 2.3 Domain normalization (cheap, CPU, on the existing vectors)
1. **Per-archive centering**: subtract each archive's mean VLAD vector, re-L2. The
   standard cross-collection fix for VLAD; removes the shared "this is how RAGent
   scans look" component. Measured on the July vectors: it equalises the *medians*
   (cross-archive pairs −0.034 → −0.003, level with within-archive different-hand
   pairs) but leaves the neighbour structure archive-bound (§3.1) — the signature
   is not an additive offset, which is what the scale-pinned codebook is for.
2. **CSLS scoring** for cross-archive pairs (Conneau et al. 2018, the hubness fix
   from cross-lingual embedding alignment):
   `csls(q, c) = 2·cos(q, c) − r_B(q) − r_A(c)`, where `r_B(q)` is q's mean cosine
   to its k nearest pages in the *other* archive. Pages that are everyone's
   neighbour (hubs — formulaic, layout-heavy) are pushed down.
3. **Mutual nearest neighbours** across archives as the doc-level primitive.

### 2.4 The three serendipity lists (`mole cross`)
All scores exclude sibling scans (`docids`) and near-duplicates (cos ≥ 0.98, listed
apart as "possible duplicates"). The document pair is the unit.

| tab | question | evidence | shown as |
|---|---|---|---|
| **1. Page pairs** (label-free, the headline) | are these two pages one scribe? | mutual cross-archive nearest neighbours by **CSLS** on centered vectors; ranked by CSLS | left: the page; right: its partner first, then the runners-up in the partner's archive |
| **2. Page → foreign hand** | is this page by that hand of another archive? | top-2 mean to the foreign hand (two of its pages agree); the best hand at home shown beside it | page beside the foreign hand's closest pages |
| **3. Hand pairs** | are `comital/KA_8` and `utrecht/KaD` one scribe? | mean of the **two strongest** cross-document cosines (not the average over the hands); reciprocal ranks; mean and cohesion as context | flip through hand a's pages ([ / ]) against hand b's closest pages |

Every row carries the raw cosine (and CSLS / home-best where relevant) and the
**within-archive reference**: the share of same-hand pairs and of different-hand
pairs inside an archive that this score exceeds. That is a yardstick, not a filter.

### 2.5 Review sheet
`mole cross <pool.npy>` (or several `.npy` from one checkpoint + codebook): one
self-contained HTML in the review style, the gap diagnostics in the header, tabs
1–3 (+ duplicates when any), Keep / Reject / Not sure → CSV. `labels.csv` untouched.
The JSON report with every number lands beside it.

## 3. Phase A — rough go with the existing model

### 3.1 Zero-th pass — DONE on the July vectors (CPU, 29 s)
`mole cross` over `outputs/universal_full/{antwerp-bin,utrecht-bin,leroy-bin,flanders-set-bin}.npy`
(pooled July model, one codebook, retired Utrecht/Leroy versions, **no** scale
normalization): 3 092 pages, 1 466 labeled.

* **Gap:** nearest neighbour from the same archive 98 % raw and 98 % centered
  (chance 32 %); centering only levels the medians. ⇒ the archive signature is
  structural (the four sets sit at different script scales: legacy Utrecht ≈45 px,
  Leroy ≈28 px), not an offset. The `purity_other_hand` figure (own scribe masked)
  is the number to watch in Phase A proper.
* **Reference:** within-archive same-hand pairs Q1/med/Q3 = 0.04/0.17/0.30,
  different-hand −0.11/−0.03/0.05.
* **Lists look like handwriting, not scanners:** the Antwerp pages in the top-40
  page pairs are the *earliest* ones (6 of 11 ≤ 1310, against 7 % of the archive),
  paired with 13th-c. Leroy; Utrecht partners cluster in the 1240s. The top hand
  pairs are rank-1 both ways (`flanders/KA_8 ~ utrecht/KaD`, `leroy/20 ~ utrecht/TDTD`,
  `flanders/KA_1 ~ utrecht/KaDui WA`). The Ten Duinen–Ter Doest overlap (§0) surfaced
  here.
* Sheet: `outputs/cross0/july.cross.html` (server, scratch copy `~/mole-cross-tmp`).

### 3.2 Phase A proper — DONE 2026-09-17 (server: codebook ~15 min, embed 21 min, cross ~5 min)

Measured script modules: antwerp-bin 39.3 px (IQR/median 0.47), utrecht-charters 48.4
(0.01), leroy-sauvola 29.1 (0.34), comital 43.0 (0.02) → pinned **41.2 px**; 2 612 of
2 835 pages resampled (70 unmeasurable, left at native scale).

* **Shared-space price = none.** Within-archive macro-mAP (cross-doc) in the one
  index: Antwerp **0.851** (own-codebook ref 0.817), comital **0.620** (0.598 with
  the adapt+intra-norm stack), utrecht-charters 0.667, leroy-sauvola 0.784 (0.816 on
  the retired leroy-bin, auto-labels unfiltered). `outputs/cross/pool.eval.json`.
* **Gap:** own-scribe-masked purity@1 93 % raw → **74 % centered** (July 97 → 83);
  @10 67 % (July 75 %). Scale pinning did real work; the rest is mostly legitimate.
* **Reference:** same-hand Q1/med/Q3 0.07/0.16/0.29; different-hand −0.08/−0.02/0.06.
  The top cross pairs (cos 0.36–0.53) sit above the same-hand Q3.
* **Survived the re-prep (in both runs):** `antwerp-bin/0-0150 (1300)` ↔
  `leroy-sauvola/671o` (#1 page pair twice); `comital/141_2_RAGent K72_29` ↔
  `utrecht-charters/1249.99.99b` (cos 0.525, and that page's best foreign hand is
  `comital/KA_10` 0.49 vs 0.37 at home); `antwerp-bin/0-0317 (1306)` → `leroy/20`.
* **New:** comital↔Utrecht dominates the page pairs (19/50), Utrecht partners in
  1245–1250; hand pairs rank-1 both ways: `comital/KA_10 ~ utrecht/DevB`,
  `KA_10 ~ leroy/102`, `KA_9 ~ utrecht/MiC`, `KA_7 ~ leroy/86`. No duplicates.
* Sheet: `outputs/cross/pool.cross.html` (76 MB; image cache `outputs/cross/imgcache`
  makes rebuilds fast). Under review by Mike.

Runbook as run (`mole codebook` needs the four folders, not the pool root):

Model: `runs/pooled_bin_ft/checkpoint.pth` — the only checkpoint trained on all
collections at once (July: one general model ≈ five specialists). The September
specialists (`comital_ssl_sauvola`, `sluis_leroy_ssl_sauvola`,
`utrecht-charters_ssl_sauvola`) each define a different space and cannot be mixed.

```bash
screen -S cross
cd ~/mole && git pull
# 1. pool the four live Sauvola sets (symlinks, no copies)
POOL=data/cross-pool bash scripts/assemble_pooled.sh \
    data/antwerp-bin data/utrecht-charters data/leroy-sauvola data/comital
# 2. one codebook, one pinned script module, over the whole pool
#    (comital 42.9 px, utrecht-charters 48.4 px, antwerp/leroy un-normalized ->
#     every archive is rescaled to the pooled median at embed time)
mole codebook runs/pooled_bin_ft/checkpoint.pth data/cross-pool/antwerp-bin \
    data/cross-pool/utrecht-charters data/cross-pool/leroy-sauvola data/cross-pool/comital \
    --out outputs/cross/fit.codebook.npy --scale-target auto --device cuda:5
# 3. one embedding file for the whole pool (intra-norm ON for the whole index)
mole embed runs/pooled_bin_ft/checkpoint.pth data/cross-pool outputs/cross/pool.npy \
    --pooling vlad --codebook-from outputs/cross/fit.codebook.npy --vlad-intra-norm \
    --device cuda:5
# 4. price of the shared space, per archive
mole eval outputs/cross/pool.npy data/cross-pool --cross-doc-only --per-hand
# 5. the sheet (CPU, ~1 min; Leroy auto-matches demoted to unlabeled below 0.5 conf)
mole cross outputs/cross/pool.npy --limit 50 --min-confidence 0.5 \
    --out outputs/cross/pool.cross.html
```
Then `scp mike:~/mole/outputs/cross/pool.cross.html ~/Downloads/mole-cross/`.

## 4. Validation without ground truth

1. **Ten Duinen–Ter Doest** (§0): pages of that fonds exist in both the Utrecht set
   (`TDTD*` hands) and the Flanders set. Whether the same scribes (or charters)
   are in both is checkable by a historian from the shelfmarks alone — the
   cheapest route to a first confirmed cross-archive hand.
2. **In-domain positive control — comital across repositories.** The comital labels
   (`KA_n`) span RAGent, ADN Lille, RABrugge, SAGent, … — different institutions,
   different scanners, *same hand labels*. Split `comital` into per-repository
   sub-folders by shelfmark prefix (symlinks + per-folder `labels.csv`) and run
   `mole eval` on that: the **cross-dataset** block is a real "same hand, different
   digitization" mAP, in the target domain, today. This is the number Phase B has to
   move.
3. **Re-prep control** — `flanders-set-bin` (July prep) vs `comital` (Sept prep) share
   charters and labels: a weaker same-scan/different-pipeline check, useful to bound
   what centering alone buys.
4. **Historian review** of tabs 1–3. Accepted pairs become cross-archive
   `hand_aliases.csv` rows (`comital/KA_8,utrecht-charters/KaD`), which `mole eval`
   can then score as true cross-dataset positives — the first real ground truth.

## 5. Phase B — the weekend model

"Larger" should mean **more data in one space**, not a bigger ViT: raven is a
`vit_small` and the only pretrained weights we have; a bigger backbone from scratch on
~3 k pages would lose. What is stale is the *pool*: `pooled_bin_ft` never saw
`utrecht-charters`, `leroy-sauvola` or `comital`.

* Pool: `antwerp-bin` + `utrecht-charters` + `leroy-sauvola` + `comital` ≈ 2 835 pages. **Before pooling, re-prep to one script
  module** (`mole scale-target` over the four → one `--target-module` for all; comital
  and Utrecht were each auto-targeted to their own median) so the model never learns
  scale as an archive cue.
* Init: `checkpoints/raven_checkpoint.pth`, same recipe as the September specialists
  (`configs/pooled_bin.yaml`, overlap 0.5, ~20 epochs). July's 20 epochs on 3 392
  pages took ~45 h wall-clock; ~85 % of the gain was in by epoch 10 — plan for a
  checkpoint at 10 and eval it Saturday evening.
* Optional, if the gap diagnostics (§2.2) show purity ≈ 100 %: archive-balanced
  window sampling so no archive dominates the batch. Not domain-adversarial training —
  too much for a weekend and the centering/CSLS step handles most of it.
* Then repeat Phase A steps 2–5 with the new checkpoint.

**Success criteria** (all measurable without new labels):
* comital cross-repository mAP (§4.2) up vs Phase A;
* kNN archive purity down at equal-or-better within-archive macro-mAP per archive;
* the top candidates' within-archive reference shares hold or rise;
* and at least one hand pair the historian accepts.

## 6. Code (built 2026-09-17, CPU-only)

1. `mole/data/docids.py`: alias `("comital", "flanders")` — sibling scans were
   uncollapsed in that folder.
2. `mole/review/cross.py`: `load_pool` (one pooled `.npy` or several from one model),
   `center_by_archive`, `gap_diagnostics` (purity@k, own-scribe-masked purity,
   medians), `reference_distributions`, `csls_matrix`, the three lists + duplicates,
   `build_cross`, `format_report`.
3. `mole cross` (CLI) → HTML + JSON report.
4. `mole/review/render.py`: `render_cross` on the case-sheet chrome; the left page
   can flip through a hand's pages ([ / ]); generic number bits.
5. `tests/test_review_cross.py`: planted 3-archive pool with per-archive scanner
   offsets; centering, CSLS symmetry, every list finds its plant, the sheet is
   self-contained and only asks questions.

## 7. Open questions
* Leroy labels: `--min-confidence 0.5` demotes the weak auto-matches to unlabeled
  (they still appear in tab 1); is that the floor you want?
* Ten Duinen–Ter Doest: are the Flanders-set pages of that fonds the same charters
  as the Utrecht `TDTD*` ones, or different charters of the same scriptorium?
