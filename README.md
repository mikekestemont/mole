# MOLE

**Continual self-supervised handwriting embeddings for premodern documents.**

Named after *mole*, the Mexican sauce continually remade from the previous day's
leftovers — the model is continually re-pretrained on a mix of old and new data.

MOLE is a clean, packaged rewrite of the writer-retrieval method of **Raven,
Matei & Fink** (*Self-Supervised Vision Transformers for Writer Retrieval*, ICDAR 2024,
[arXiv:2409.00751](https://arxiv.org/abs/2409.00751)), which adapts **AttMask**
(Kakogeorgiou et al.), itself in the **DINO** / **iBOT** lineage. The end goal is
extracting embeddings for handwriting identification on premodern documents, and
later a similarity-search engine over large image collections.

> **v0.1.0** is the first tagged release: self-supervised training, VLAD retrieval,
> and vocabulary adaptation. The supported recipe is in the GitHub release notes.
> Later phases (continual replay, a `finetune` CLI, lineage registry) are still
> in progress.

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

Commands: `prep`, `augview`, `train`, `embed`, `codebook`, `eval`, `viz`,
`cluster`, `review`, `cross`.

See **[WORKFLOW.md](WORKFLOW.md)** for the recommended end-to-end pipeline and exact
commands, and **[ARCHITECTURE.md](ARCHITECTURE.md)** for design/decisions/build state.

## Acknowledgements

- **AttMask** — Kakogeorgiou, Gidaris, Psomas, Avrithis, Bursuc, Karantzalos &
  Komodakis, *What to Hide from Your Students: Attention-Guided Masked Image
  Modeling*, ECCV 2022 ([arXiv:2203.12719](https://arxiv.org/abs/2203.12719)).
- **iBOT** — Zhou, Wei, Wang, Shen, Xie, Yuille & Kong, *iBOT: Image BERT
  Pre-Training with Online Tokenizer*, ICLR 2022
  ([arXiv:2111.07832](https://arxiv.org/abs/2111.07832)).
- **DINO** — Caron, Touvron, Misra, Jégou, Mairal, Bojanowski & Joulin, *Emerging
  Properties in Self-Supervised Vision Transformers*, ICCV 2021
  ([arXiv:2104.14294](https://arxiv.org/abs/2104.14294)).
- **VLAD** — Jégou, Douze, Schmid & Pérez, *Aggregating Local Descriptors into a
  Compact Image Representation*, CVPR 2010; intra-normalisation and vocabulary
  adaptation from Arandjelović & Zisserman, *All About VLAD*, CVPR 2013.
- **CSLS** (cross-archive page pairs) — Conneau, Lample, Ranzato, Denoyer & Jégou,
  *Word Translation Without Parallel Data*, ICLR 2018
  ([arXiv:1710.04087](https://arxiv.org/abs/1710.04087)).
- **SGR reranking** — Peer, Kleber & Sablatnig, *Towards Writer Retrieval for
  Historical Datasets*, ICDAR 2023.
- **Raven, Matei & Fink** — *Self-Supervised Vision Transformers for Writer
  Retrieval*, ICDAR 2024 ([doi:10.1007/978-3-031-70536-6_23](https://doi.org/10.1007/978-3-031-70536-6_23),
  [arXiv:2409.00751](https://arxiv.org/abs/2409.00751); TU Dortmund).
  MOLE is refactored from their code and checkpoint; the method is theirs.

See `HWI_REPRODUCTION.md` for the Historical-WI reproduction record.
