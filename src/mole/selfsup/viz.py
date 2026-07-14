"""In-training embedding projector for TensorBoard.

Periodically embeds a fixed sample of documents with the live teacher backbone and
logs them via ``SummaryWriter.add_embedding`` — the interactive PCA / t-SNE / UMAP
"PROJECTOR" tab. Points are coloured by ``hand_id`` (from ``labels.csv``) when
available, else by dataset, so you can watch same-hand documents cluster as training
proceeds. Optional per-point thumbnails show the actual document crop on hover.

Document embedding = L2-normalised mean of the teacher's patch tokens over a few
windows per page (the same deterministic resize as ``mole embed``). Never fatal:
any failure is caught by the caller so it can't interrupt training.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from mole.data.datasets import load_labels
from mole.data.patches import load_rgb
from mole.progress import track

# Kept modest so the extra forward passes and sprite stay cheap.
_WINDOWS_PER_IMAGE = 4
_THUMB_PX = 48


def _resize_to_tensor(model_size: int):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose([
        transforms.Resize((model_size, model_size), interpolation=InterpolationMode.BICUBIC,
                           antialias=True),
        transforms.ToTensor(),
    ])


def log_document_projector(writer, teacher, dataset, device, step: int, *,
                           max_images: int = 300, model_size: int = 224,
                           thumbnails: bool = True, seed: int = 0) -> int:
    """Embed a sample of the dataset's pages and log them to the TB projector.

    Returns the number of documents logged (0 if nothing was logged).
    """
    import torch

    by_image: dict[Path, list] = defaultdict(list)
    for path, win in dataset.index:
        by_image[path].append(win)
    images = sorted(by_image)
    rng = random.Random(seed)
    if len(images) > max_images:
        images = sorted(rng.sample(images, max_images))

    resize = _resize_to_tensor(model_size)
    thumb = _resize_to_tensor(_THUMB_PX) if thumbnails else None
    dir_labels: dict[Path, object] = {}

    def hand_for(path: Path) -> str:
        parent = path.parent
        if parent not in dir_labels:
            dir_labels[parent] = load_labels(parent)
        return dir_labels[parent].hand_by_filename.get(path.name, "unlabeled")

    backbone = teacher.backbone
    nct = backbone.num_class_tokens
    was_training = teacher.training
    teacher.eval()
    vecs, meta_rows, thumbs = [], [], []
    try:
        with torch.no_grad():
            for path in track(images, "Projector embeddings", unit="img", leave=False):
                wins = by_image[path]
                if len(wins) > _WINDOWS_PER_IMAGE:
                    idx = sorted(rng.sample(range(len(wins)), _WINDOWS_PER_IMAGE))
                    wins = [wins[i] for i in idx]
                img = load_rgb(path)
                crops = torch.stack([resize(img.crop((w.x, w.y, w.x + w.size, w.y + w.size)))
                                     for w in wins]).to(device)
                tokens = backbone(crops, return_attention=False, return_all_tokens=True)
                vec = tokens[:, nct:].mean(dim=(0, 1))          # mean over windows + patches
                vec = torch.nn.functional.normalize(vec, dim=0)
                vecs.append(vec.cpu())
                meta_rows.append([hand_for(path), path.parent.name, path.name])
                if thumbnails:
                    w0 = wins[0]
                    thumbs.append(thumb(img.crop((w0.x, w0.y, w0.x + w0.size, w0.y + w0.size))))
    finally:
        teacher.train(was_training)

    if not vecs:
        return 0
    mat = torch.stack(vecs)
    label_img = torch.stack(thumbs) if thumbnails else None
    writer.add_embedding(mat, metadata=meta_rows, metadata_header=["hand", "dataset", "file"],
                         label_img=label_img, global_step=step, tag="documents")
    return len(vecs)
