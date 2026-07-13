# MOLE

**Continual self-supervised handwriting embeddings for premodern documents.**

Named after *mole*, the Mexican sauce continually remade from the previous day's
leftovers — the model is continually re-pretrained on a mix of old and new data.

MOLE is a clean, packaged rewrite of Tim Raven's adaptation of **AttMask**
(Kakogeorgiou et al.), itself in the **DINO** / **iBOT** lineage. The end goal is
extracting embeddings for handwriting identification on premodern documents, and
later a similarity-search engine over large image collections.

> ⚠️ **Under construction** — being built in reviewable phases. Full documentation
> (config tables, CLI reference, recipes) lands in the final phase.

## Install

Training + embedding:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
# On the CUDA server, install matching torch wheels, e.g.:
#   pip install torch==2.8.0 torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cu128
```

Preprocessing (`mole prep`, kraken) — **separate environment** (kraken pins its
own torch versions):

```bash
python -m venv .venv-prep && . .venv-prep/bin/activate
pip install -r requirements-prep.txt
pip install -e .
```

## Usage

```bash
mole --help
```

Commands: `prep`, `augview`, `train`, `finetune`, `embed`, `eval`, `models`.

See **[WORKFLOW.md](WORKFLOW.md)** for the recommended end-to-end pipeline and exact
commands, and **[ARCHITECTURE.md](ARCHITECTURE.md)** for design/decisions/build state.

## Acknowledgements

- **AttMask** — Kakogeorgiou et al., *What to Hide from Your Students:
  Attention-Guided Masked Image Modeling* (ECCV 2022).
- **iBOT** — Zhou et al. · **DINO** — Caron et al.
- Tim Raven's writer-identification adaptation, which MOLE is refactored from.
