# Handoff: the "Learnable Handwriter" for scribal letterform visualization

**Purpose.** Explore using the *Learnable Handwriter* to produce per-character **prototype
visualizations** for our charter corpus — the "here is how this scribe writes their *g*"
letterforms — and to chain it with mole's scribal-hand discovery. This doc is self-contained;
a new session can start from here with no prior context.

**Status:** idea only, nothing built. Motivated by a conversation (2026-08-24) where the user
flagged these visualizations as ones they "love" and want for the charter data.

---

## 1. What it is

**The Learnable Handwriter** — Malamatenia Vlachou Efstathiou et al. (IRHT Paris + LIGM / École
des Ponts), *"An Interpretable Deep Learning Approach for Morphological Script Type Analysis"*,
IWCP 2024. An adaptation of **The Learnable Typewriter** (Siglidis, Aubry et al.) to handwriting.

- Paper: https://arxiv.org/pdf/2408.11150 · abs https://arxiv.org/abs/2408.11150
- Project site: https://learnable-handwriter.github.io/
- Code: https://github.com/malamatenia/learnable-handwriter

**Method gist** (from the abstract/site — the full paper/repo has NOT been read yet; do that
first): a transformer-based detection architecture plus a **prototype-based line reconstruction**
module. From **line-level transcriptions** it learns, per character class, an explicit **prototype
image** (a canonical rendered letterform) together with each instance's **deformation** and
**positioning**, and reconstructs each text line as a sequence of deformed, placed prototypes.
Because a prototype is a real, inspectable image, the method turns qualitative palaeographic
observation into **quantitative, comparable morphological measurements**.

**The visualizations (the thing we want):** the learned per-character prototypes (clean letterforms
per script/scribe), their deformations, and cross-script / cross-scribe prototype comparisons.

## 2. Why it fits mole / scripy — and the crucial dependency

It is **complementary** to mole, not a competitor:

| | answers | needs |
|---|---|---|
| **mole** | *who wrote what* — text-independent writer retrieval / scribal-hand clustering | just images |
| **Learnable Handwriter** | *what do this hand's letterforms look like* — prototypes + morphology | **line-level transcriptions** |

**The natural pipeline: mole discovers the scribal groups → the Learnable Handwriter visualizes and
compares the letterforms of a group.** mole never needs text; the LH step is a downstream,
transcription-fed visualization layer.

⚠️ **The load-bearing dependency is transcriptions.** LH needs line-level transcriptions to learn
prototypes. This is exactly the line mole's text-independent design stays on the *other* side of.
So the first feasibility question is: **do we have transcriptions for any charter collection?**
LEAD: the **Antwerp** images were derived from **Transkribus PAGE XML** (see mole memory
`mole-status` / the layout-cropping finding), so transcriptions may already exist there — check.
Utrecht/Flanders/others: unknown. Medieval Latin / Middle-Dutch HTR is otherwise hard.

## 3. Do NOT confuse with VLAC (Raven's interpretability paper)

Same conversation surfaced a *different* explainability method — **VLAC = Vectors of Locally
Aggregated Characters** (Raven, Christlein, Fink, ICDAR 2025,
https://link.springer.com/chapter/10.1007/978-3-032-04627-7_25). VLAC is **discriminative**: it
aggregates writer *features* per character (via a character *annotator* network) and reports
**per-character distances** ("the *g* is what makes these two hands differ"). It produces
*attributions*, NOT letterform *images*. The Learnable Handwriter is **generative/reconstructive**:
it produces the prototype *images* the user wants. Different paradigms — VLAC won't give the
visualizations; the Learnable Handwriter will.

## 4. Exploration plan for the new session

1. **Read the repo README + paper** (https://github.com/malamatenia/learnable-handwriter,
   arXiv:2408.11150). Nail down: exact input format (images + transcription format — ALTO? PAGE?
   plain line strings?), training procedure, compute needs, and how prototypes are exported/rendered.
2. **Transcription audit.** Do we have line transcriptions for any charter collection?
   - Check the Antwerp Transkribus PAGE XML (likely on `~/Desktop/images/antwerp/` per memory, and
     the server `data/antwerp-page`) — does it carry line text, or only layout?
   - If none exist, scope: run HTR (Transkribus / Kraken) on one collection, or start with any
     collection that already has GT.
3. **Reproduce their demo** on the authors' own data first (sanity that the tool runs + we can
   render prototypes), before touching charters.
4. **Chain with mole.** Take ONE mole-discovered scribal group (e.g. a clean Antwerp hand, or a
   Flanders KA_* hand) + its transcriptions → train LH → render that scribe's per-character
   prototypes. Then compare prototypes across two hands mole says are different → does the letterform
   comparison agree?
5. **Deliverable to aim for:** a per-scribe letterform prototype sheet for a charter hand, and a
   side-by-side prototype comparison of two hands — the visualization the user wants.

## 5. Open questions to resolve early
- Transcription availability + format (the gate — resolve this first).
- Does LH need a fixed alphabet / does it handle abbreviations & medieval Latin ligatures?
- Per-document vs per-corpus training (does it learn one prototype set per manuscript, or shared)?
- Compute: it's training-based → GPU. Fits the `mike` server (conda env `mole` or a fresh env;
  avoid GPU 3 — flaky).
- Is this a mole feature, a scripy feature, or a standalone tool wrapped for our data? (Likely
  standalone tool + a thin adapter that feeds it mole clusters + transcriptions.)

## 6. Sources
- Learnable Handwriter: [arXiv:2408.11150](https://arxiv.org/pdf/2408.11150) ·
  [site](https://learnable-handwriter.github.io/) · [code](https://github.com/malamatenia/learnable-handwriter)
- Underlying method — The Learnable Typewriter (Siglidis, Monnier, Aubry): search
  "The Learnable Typewriter generative approach text analysis".
- VLAC (the *other*, discriminative interpretability method, for contrast):
  [Springer](https://link.springer.com/chapter/10.1007/978-3-032-04627-7_25)
