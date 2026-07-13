"""Vision Transformer backbone (DINO / iBOT / AttMask lineage).

Faithful port of Tim Raven's ``models/vision_transformer.py`` (itself copy-pasted
from DINO / timm / iBOT), cleaned of the unused ``timm.register_model`` import and
the dead commented masked-image-modeling branch in ``forward``.

Supports masked image modeling (``masked_im_modeling``) and multiple class tokens
(``num_class_tokens``); the last block returns its attention map (used by AttMask).
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn

from mole.selfsup._nn import trunc_normal_


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Stochastic depth per sample (in the main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
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
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class VisionTransformer(nn.Module):
    """Vision Transformer with MIM + multi-CLS support. Last block returns attention."""

    def __init__(self, img_size=(224,), patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 return_all_tokens=False, init_values=0, use_mean_pooling=False,
                 masked_im_modeling=False, num_class_tokens=1):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens
        self.num_class_tokens = num_class_tokens

        self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, num_class_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_class_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_class_tokens
        N = self.pos_embed.shape[1] - self.num_class_tokens
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, :self.num_class_tokens]
        patch_pos_embed = self.pos_embed[:, self.num_class_tokens:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x, mask=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def drop_tokens(self, x, ids_keep):
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x_masked = torch.gather(x[:, 1:], 1, ids_keep_expanded)
        x_masked = torch.cat((x[:, 0].unsqueeze(1), x_masked), dim=1)
        return x_masked

    def forward(self, x_orig, return_all_tokens=None, mask=None, keep_ids=None,
                return_attention=True, **kwargs):
        x = self.prepare_tokens(x_orig)
        if keep_ids is not None:
            x = self.drop_tokens(x, keep_ids)
        if self.masked_im_modeling and mask is not None:
            x = self.mask_sequence(x, mask)

        for i, blk in enumerate(self.blocks):
            x, att = blk(x)  # att from the last block is what AttMask consumes

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

        return_all_tokens = self.return_all_tokens if return_all_tokens is None else return_all_tokens
        if return_attention:
            if return_all_tokens:
                return x, att
            return x[:, :self.num_class_tokens].reshape(-1, self.num_class_tokens * self.embed_dim), att
        if return_all_tokens:
            return x
        return x[:, :self.num_class_tokens].reshape(-1, self.num_class_tokens * self.embed_dim)

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x, _ = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x, _ = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_num_layers(self):
        return len(self.blocks)

    def mask_sequence(self, x, mask):
        x[:, self.num_class_tokens:][mask] = self.masked_embed.to(x.dtype)
        return x

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x


def vit_tiny(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, embed_dim=192, depth=12, num_heads=3,
                             mlp_ratio=4, qkv_bias=True, **kwargs)


def vit_small(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
                             mlp_ratio=4, qkv_bias=True, **kwargs)


def vit_base(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
                             mlp_ratio=4, qkv_bias=True, **kwargs)


def vit_large(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
                             mlp_ratio=4, qkv_bias=True, **kwargs)


VIT_ARCHS = {"vit_tiny": vit_tiny, "vit_small": vit_small, "vit_base": vit_base, "vit_large": vit_large}


def build_vit(arch: str = "vit_small", **kwargs) -> VisionTransformer:
    """Build a ViT by name (``vit_tiny|small|base|large``)."""
    arch = arch.replace("deit", "vit")
    if arch not in VIT_ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {list(VIT_ARCHS)}")
    return VIT_ARCHS[arch](**kwargs)
