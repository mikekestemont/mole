"""
Visualize the augmented crops (global + local) as presented to the model
during training with DataAugmentationMorph.

Usage:
    python visualize_augmentations.py --image_path ../assets/Deeds_Batch_1/0-0458_21_10_1339-01.png

Place this script in the same directory as main_attmask.py so imports work.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from torchvision import transforms
from kornia import morphology as morph

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- reproduce augmentation components from main_attmask.py ----------

try:
    import utils  # for GaussianBlur
    GaussianBlur = utils.GaussianBlur
except ImportError:
    # Fallback: simple Gaussian blur matching the DINO/iBOT convention
    import torchvision.transforms.functional as TF
    import random

    class GaussianBlur:
        def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
            self.prob = p
            self.radius_min = radius_min
            self.radius_max = radius_max

        def __call__(self, img):
            if isinstance(img, torch.Tensor):
                if random.random() < self.prob:
                    sigma = random.uniform(self.radius_min, self.radius_max)
                    ks = int(2 * round(3 * sigma) + 1)
                    if ks % 2 == 0:
                        ks += 1
                    return TF.gaussian_blur(img, kernel_size=ks, sigma=sigma)
                return img
            else:
                if random.random() < self.prob:
                    from PIL import ImageFilter
                    sigma = random.uniform(self.radius_min, self.radius_max)
                    return img.filter(ImageFilter.GaussianBlur(radius=sigma))
                return img


def get_random_kernel():
    k = torch.rand(3, 3)
    k -= 0.2
    k = k.round()
    k[1, 1] = 1
    return k


class Erosion:
    def __init__(self):
        self.fn = morph.erosion

    def __call__(self, img):
        kernel = get_random_kernel()
        return self.fn(img.unsqueeze(0), kernel)[0]


class Dilation:
    def __init__(self):
        self.fn = morph.dilation

    def __call__(self, img):
        kernel = get_random_kernel()
        return self.fn(img.unsqueeze(0), kernel)[0]


class DataAugmentationMorph:
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number,
                 patch_size=16, drop_rate=0.0):
        self.patch_size = patch_size
        normalize = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            GaussianBlur(0.25),
            normalize,
            transforms.RandomApply([Erosion()], p=0.5),
            transforms.RandomApply([Dilation()], p=0.5),
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            GaussianBlur(0.1),
            normalize,
            transforms.RandomApply([Dilation()], p=0.5),
            transforms.RandomApply([Erosion()], p=0.5),
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            GaussianBlur(p=0.5),
            normalize,
            transforms.RandomApply([Erosion()], p=0.5),
            transforms.RandomApply([Dilation()], p=0.5),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


# ---------- visualisation helpers ----------

def tensor_to_numpy(t):
    """Convert a [C,H,W] tensor to [H,W,C] numpy clipped to [0,1]."""
    img = t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img, 0, 1)


def draw_patch_grid(ax, img_np, patch_size, title=""):
    """Draw image with a semi-transparent patch grid overlay."""
    ax.imshow(img_np, cmap="gray" if img_np.shape[-1] == 1 else None)
    h, w = img_np.shape[:2]
    for x in range(0, w, patch_size):
        ax.axvline(x, color="red", linewidth=0.4, alpha=0.6)
    for y in range(0, h, patch_size):
        ax.axhline(y, color="red", linewidth=0.4, alpha=0.6)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of independent augmentation samples to show")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="augmentation_grid.pdf")
    # Training params (defaults match your command)
    parser.add_argument("--global_crops_scale", type=float, nargs=2, default=[0.25, 1.0])
    parser.add_argument("--local_crops_scale", type=float, nargs=2, default=[0.05, 0.25])
    parser.add_argument("--local_crops_number", type=int, default=4)
    args = parser.parse_args()

    image = Image.open(args.image_path).convert("RGB")

    augmenter = DataAugmentationMorph(
        global_crops_scale=tuple(args.global_crops_scale),
        local_crops_scale=tuple(args.local_crops_scale),
        local_crops_number=args.local_crops_number,
        patch_size=args.patch_size,
    )

    n_crops = 2 + args.local_crops_number  # 2 global + N local
    n_samples = args.n_samples

    row_labels = (
        ["Global crop 1 (224×224)", "Global crop 2 (224×224)"]
        + [f"Local crop {i+1} (96×96)" for i in range(args.local_crops_number)]
    )

    # --- Generate all samples first ---
    all_samples = []
    for _ in range(n_samples):
        all_samples.append(augmenter(image))

    # --- Layout: rows = crop types, cols = samples ---
    fig, axes = plt.subplots(
        nrows=n_crops, ncols=n_samples,
        figsize=(2 * n_samples, 2 * n_crops),
        dpi=300,
    )
    if n_samples == 1:
        axes = axes[:, np.newaxis]  # ensure 2D

    for s in range(n_samples):
        crops = all_samples[s]
        for r, crop_tensor in enumerate(crops):
            ax = axes[r, s]
            crop_np = tensor_to_numpy(crop_tensor)
            h_px = crop_tensor.shape[-1]  # 224 for global, 96 for local

            # Patch grid spacing: 16 px for 224 crops, proportional for 96 crops
            if h_px == 224:
                grid_ps = args.patch_size
            else:
                grid_ps = int(args.patch_size * h_px / 224)

            draw_patch_grid(ax, crop_np, grid_ps, title="")

            # Column header
            if r == 0:
                ax.set_title(f"Sample {s+1}", fontsize=11, fontweight="bold", pad=8)

            # Row label on left column
            if s == 0:
                ax.set_ylabel(row_labels[r], fontsize=9, rotation=0,
                              labelpad=120, ha="right", va="center")

    fig.suptitle(
        f"DataAugmentationMorph   |   patch_size={args.patch_size}   |   "
        f"global_scale={args.global_crops_scale}   local_scale={args.local_crops_scale}",
        fontsize=12, y=1.005, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", facecolor="white", dpi=300)
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()