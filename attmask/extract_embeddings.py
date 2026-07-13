#!/usr/bin/env python3
"""
Extract VLAD-encoded document embeddings for t-SNE visualization.
Memory-efficient version: saves features to disk incrementally.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
import tempfile
import os


# ============ ViT Model Definition ============

from functools import partial

def trunc_normal_(tensor, std=.02):
    nn.init.trunc_normal_(tensor, std=std)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, :1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / N**0.5, h0 / N**0.5),
            mode='bicubic',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def vit_small(patch_size=16):
    return VisionTransformer(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)


# ============ Helper Functions ============

def load_teacher_weights(model, checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    teacher_state = ckpt['teacher']
    
    new_state = {}
    for k, v in teacher_state.items():
        if k.startswith('backbone.'):
            new_key = k.replace('backbone.', '')
            if 'masked_embed' not in new_key:
                new_state[new_key] = v
    
    model.load_state_dict(new_state, strict=False)
    return model


def extract_patches(image_array, patch_size=256, stride=32, foreground_threshold=0.02):
    H, W = image_array.shape
    patches = []
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image_array[y:y + patch_size, x:x + patch_size]
            foreground_ratio = np.sum(patch < 255) / (patch_size * patch_size)
            if foreground_ratio >= foreground_threshold:
                patches.append(patch)
    
    return patches


def patches_to_tensor(patches):
    patches_array = np.stack(patches)
    patches_array = np.stack([patches_array] * 3, axis=1)
    patches_tensor = torch.from_numpy(patches_array).float() / 255.0
    return patches_tensor


def get_foreground_mask(patches_tensor, patch_size=16, threshold=0.02):
    B = patches_tensor.shape[0]
    pooled = torch.nn.functional.avg_pool2d(patches_tensor[:, 0:1], patch_size)
    pooled = pooled.squeeze(1).view(B, -1)
    foreground_mask = pooled < (1.0 - threshold)
    return foreground_mask


def extract_patch_features(model, patches_tensor, device, batch_size=32):
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i:i + batch_size].to(device)
            tokens = model(batch)
            patch_tokens = tokens[:, 1:, :]
            fg_mask = get_foreground_mask(batch)
            
            # Extract foreground features immediately
            for j in range(len(patch_tokens)):
                mask = fg_mask[j]
                fg_feat = patch_tokens[j][mask].cpu().numpy()
                if len(fg_feat) > 0:
                    all_features.append(fg_feat)
            
            # Clear GPU memory
            del batch, tokens, patch_tokens, fg_mask
            torch.cuda.empty_cache()
    
    if len(all_features) > 0:
        return np.concatenate(all_features, axis=0)
    return np.zeros((0, 384), dtype=np.float32)


def vlad_encode(descriptors, cluster_centers, powernorm=True):
    if len(descriptors) == 0:
        return np.zeros(cluster_centers.shape[0] * cluster_centers.shape[1], dtype=np.float32)
    
    descriptors = np.nan_to_num(descriptors, nan=0)
    kdt = KDTree(cluster_centers, metric='euclidean')
    _, indices = kdt.query(descriptors, k=1)
    indices = indices.flatten()
    
    D = cluster_centers.shape[1]
    K = cluster_centers.shape[0]
    vlad = np.zeros((K, D), dtype=np.float32)
    
    for i, idx in enumerate(indices):
        vlad[idx] += descriptors[i] - cluster_centers[idx]
    
    vlad = vlad.flatten()
    
    if powernorm:
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    
    vlad = normalize(vlad.reshape(1, -1), norm='l2').flatten()
    return vlad


def process_single_image(model, img_path, patch_size, stride, device, batch_size):
    """Process one image and return features."""
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    
    patches = extract_patches(img_array, patch_size, stride)
    
    if len(patches) == 0:
        return np.zeros((0, 384), dtype=np.float32)
    
    patches_tensor = patches_to_tensor(patches)
    features = extract_patch_features(model, patches_tensor, device, batch_size)
    
    del patches, patches_tensor
    return features


# ============ Main ============

def parse_args():
    parser = argparse.ArgumentParser(description="Extract VLAD embeddings for t-SNE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="embeddings.npz")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-features-for-kmeans", type=int, default=500000)
    parser.add_argument("--max-features-per-doc", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Device: {args.device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    
    model = vit_small(patch_size=16)
    model = load_teacher_weights(model, args.checkpoint, device=args.device)
    model = model.to(args.device)
    model.eval()
    
    input_dir = Path(args.input)
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions])
    
    print(f"Found {len(image_files)} images")
    
    # Create temp directory for features
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Temp directory: {temp_dir}")
    
    # Pass 1: Extract features and save to disk
    print("\n=== Pass 1: Extracting features ===")
    features_for_kmeans = []
    
    for img_path in tqdm(image_files, desc="Extracting"):
        features = process_single_image(model, img_path, args.patch_size, args.stride, args.device, args.batch_size)
        
        # Save to temp file
        np.save(temp_dir / f"{img_path.stem}.npy", features)
        
        # Sample for k-means
        if len(features) > 0:
            if len(features) > 500:
                idx = np.random.choice(len(features), 500, replace=False)
                features_for_kmeans.append(features[idx])
            else:
                features_for_kmeans.append(features)
        
        # Check if we have enough for k-means
        total_kmeans = sum(len(f) for f in features_for_kmeans)
        if total_kmeans >= args.max_features_for_kmeans:
            # Trim older samples
            features_for_kmeans = features_for_kmeans[-200:]
    
    features_for_kmeans = np.concatenate(features_for_kmeans, axis=0)
    print(f"Features for k-means: {len(features_for_kmeans)}")
    
    # K-means
    print(f"\n=== K-means clustering (k={args.n_clusters}) ===")
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, n_init=1, batch_size=100000, verbose=1)
    kmeans.fit(features_for_kmeans)
    cluster_centers = kmeans.cluster_centers_
    
    del features_for_kmeans
    
    # Pass 2: VLAD encode
    print("\n=== Pass 2: VLAD encoding ===")
    vlad_dim = args.n_clusters * 384
    embeddings = np.zeros((len(image_files), vlad_dim), dtype=np.float32)
    filenames = []
    
    for i, img_path in enumerate(tqdm(image_files, desc="VLAD encoding")):
        features = np.load(temp_dir / f"{img_path.stem}.npy")
        
        # Limit features per doc
        if len(features) > args.max_features_per_doc:
            idx = np.random.choice(len(features), args.max_features_per_doc, replace=False)
            features = features[idx]
        
        vlad = vlad_encode(features, cluster_centers)
        embeddings[i] = vlad
        filenames.append(img_path.name)
    
    # Cleanup temp
    import shutil
    shutil.rmtree(temp_dir)
    
    # Save
    print(f"\n=== Saving to {args.output} ===")
    np.savez(args.output, embeddings=embeddings, filenames=np.array(filenames), cluster_centers=cluster_centers)
    
    print(f"Saved {len(filenames)} embeddings, shape: {embeddings.shape}")
    print("Done!")


if __name__ == "__main__":
    main()

