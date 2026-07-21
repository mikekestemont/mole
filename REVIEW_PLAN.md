# `mole review` — a label-review sheet you can email

*Proposal, 2026-07-21. Companion to `SUPERVISED_PLAN.md` §3 (`mole suggest`), whose scoring
this shares. Status: the analysis engine is BUILT and tested (`mole/review/suggest.py`);
this document proposes the report on top of it.*

---

## 1. What it is

One self-contained HTML file, produced on the GPU box, downloaded, and **emailed to a
colleague who has no software and no technical background**. They open it, look at ranked
suggestions about their archive, inspect the actual handwriting, tick the ones they accept,
and send back a small CSV. Nothing is installed, nothing is loaded, no `labels.csv` is ever
written by us (D5).

## 2. The six lists (built, `mole/review/suggest.py`)

| list | question it asks | calibrated? |
|---|---|---|
| **attributions** | this unlabeled charter — whose hand? | **yes** |
| **merges** | are these two names one scribe? | no (percentile) |
| **splits** | is this one name really two scribes? | no (percentile) |
| **new hands** | do these unlabeled charters form an unknown hand? | no |
| **doubts** | does this labeled charter sit with the wrong hand? | no |
| **duplicates** | is this the same charter twice? | n/a (threshold) |
| *isolated* | is this even a charter? (blanks, covers) | n/a |

Two invariants, both tested: **sibling scans are never evidence** (scores exclude documents
sharing the query's `doc_id`), and **only attributions carry a probability** — obtained by
hiding each labeled document in turn, which is real ground truth. Everything else gets
comparative wording, never a number that looks like a certainty.

## 3. The images — measured, not guessed

The first instinct (downscaled page thumbnails) fails: at ~5% of native scale you cannot
judge letterforms. The second instinct (JPEG crops of running text) fails differently —
**JPEG is the worst format for binarized pages**, because sharp black/white edges are pure
high-frequency content. Measured on `data/leroy-bin`:

| encoding | full page | 1200px crop |
|---|---|---|
| JPEG q80 | — | 90–220 KB |
| WebP lossy q60 | — | 67–106 KB |
| **WebP lossless** | **38–49 KB** | 16–25 KB |
| PNG 1-bit | 42–56 KB | 19–27 KB |

The pages are truly bilevel (2 unique levels), so lossless wins by 3–5×. **Therefore: embed
the whole page, at native resolution, losslessly.** No crop selection, no fuzziness, no risk
of framing a seal instead of text. The reviewer zooms wherever they like.

Colour archives get binarized on the fly for the report only (`mole prep`'s Sauvola, already
built) — which also matches what the model sees. A `-bin` sibling directory is preferred when
one exists.

## 4. Size budget

Inline data-URIs cost +33% for base64, so ~40 KB on disk ≈ **54 KB in the file**.

| documents with an image | file size |
|---|---|
| 100 | ~5.5 MB |
| 180 | ~10 MB |
| 300 | ~16 MB |

Gmail accepts 25 MB, many corporate servers 10 MB. So the default targets **≤10 MB**:
`--max-mb 10` is honoured by degrading in a fixed order — first cap documents per list, then
drop to page width 1200, then 900 — and the tool **prints what it did and the final size**.
Only documents appearing in a list carry an image; the other ~3,000 dots stay text-only.

`--max-mb 0` disables the cap for local use.

## 5. The report

**Layout.** Scatter left (reusing `viz/scatter.py`'s projection and dots), review panel right,
six collapsible sections with counts.

**Plain language, and wording that matches the evidence.** Attributions state a calibrated
fact — *"This charter looks like hand P. Of 40 suggestions this confident, 36 were correct."*
The uncalibrated lists ask questions — *"Do these two names belong to one scribe? Their
charters sit closer together than most scribes sit to themselves."*, *"This division is
sharper than 95% of random divisions of the same charters."* No cosine appears in the default
view; a "show the numbers" toggle reveals them for whoever wants them.

**Hover a row, and the map answers.** Everything dims except that row's documents; the subject
pulses, its supporting exemplars get rings, a line joins them. A merge row lights both hands
in contrasting colours, so two clouds visibly overlap — or visibly don't. Click pins it.

**Hover a dot, and the charter appears.** The image is already in the file; no cache, no
fetch.

**Comparison is stacked, not side-by-side.** Two charters one above the other at identical
scale: ascenders, `d` forms and abbreviation marks line up vertically, which is how the eye
catches a different hand fastest.

**Decisions leave as a CSV.** Each row has accept / reject / unsure. A "download my decisions"
button writes `decisions.csv` (`kind,document,hand,decision,note`) via a Blob URL — offline,
no server. You merge it manually; `labels.csv` is never touched by the tool.

**An escape hatch to the original.** Each row links to the full-resolution scan: a `file://`
path when reviewing on the machine holding the images, or `--image-url "https://…/{filename}"`
for an archive with an online viewer. Zero bytes.

## 6. CLI

```
mole review EMB.npy [--clusters report.json] [--out review.html]
                    [--limit 25] [--max-mb 10] [--image-cache DIR]
                    [--image-url TPL] [--no-images] [--seed 0]
```

`--image-cache` holds the encoded pages between builds, so re-running after a model change is
fast. **It is a build-time artifact only** — it never travels with the report, which is a
single file with no sidecar and no relative paths. That is the specific failure mode of the
previous thumbnail-cache approach, and it is designed out rather than mitigated.

## 7. Open decisions

- **D1 — default `--limit`.** 25 per list ≈ 180 documents ≈ 10 MB. Higher is more thorough
  but stops being emailable.
- **D2 — do unlabeled dots get images?** Currently only listed documents. Giving all ~3,400
  dots an image would be ~180 MB: local-only territory (`--max-mb 0`).
- **D3 — decision CSV columns.** Proposed `kind,document,hand,decision,note`; is a free-text
  note worth the UI space?
- **D4 — who the file is for.** Written for a colleague who owns the archive. If it also goes
  to people who do not, the exemplar filenames may need masking.
