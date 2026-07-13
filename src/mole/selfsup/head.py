"""iBOT projection head (DINO/iBOT).

Faithful port of ``models/head.py``, with the multi-GPU sync-BatchNorm variants
(``CSyncBatchNorm`` / ``PSyncBatchNorm``, only used for ``norm='csyncbn'|'psyncbn'``)
removed for the single-GPU-first design. Supported norms: ``bn``, ``ln``, ``None``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mole.selfsup._nn import trunc_normal_


class CustomSequential(nn.Sequential):
    """Sequential that transposes channel dim for BatchNorm on >2D inputs."""

    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1))
                perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]
                inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, act="gelu", last_norm=None, nlayers=3,
                 hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, **kwargs):
        super().__init__()
        norm = self._build_norm(norm, hidden_dim)
        last_norm = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        act = self._build_act(act)
        self.in_dim = in_dim

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim if bottleneck_dim > 0 else out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            layers.append(nn.Linear(hidden_dim, bottleneck_dim if bottleneck_dim > 0 else out_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)

        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None
        self.last_norm = last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == "bn":
            return nn.BatchNorm1d(hidden_dim, **kwargs)
        if norm == "ln":
            return nn.LayerNorm(hidden_dim, **kwargs)
        assert norm is None, f"unknown/removed norm type {norm!r} (single-GPU build: use bn/ln/None)"
        return None

    def _build_act(self, act):
        if act == "relu":
            return nn.ReLU()
        if act == "gelu":
            return nn.GELU()
        raise AssertionError(f"unknown act type {act}")


class iBOTHead(DINOHead):
    def __init__(self, *args, patch_out_dim=8192, norm=None, act="gelu", last_norm=None, nlayers=3,
                 hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, shared_head=False,
                 num_class_tokens=1, **kwargs):
        super().__init__(*args, norm=norm, act=act, last_norm=last_norm, nlayers=nlayers,
                         hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim,
                         norm_last_layer=norm_last_layer, **kwargs)
        self.num_class_tokens = num_class_tokens
        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None
            self.last_norm2 = self._build_norm(last_norm, patch_out_dim, affine=False, **kwargs)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None
            self.last_norm2 = self.last_norm

    def forward(self, x):
        if len(x.shape) == 2:
            return super().forward(x)
        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x[:, :self.num_class_tokens])
            x2 = self.last_layer2(x[:, self.num_class_tokens:])
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, :self.num_class_tokens])
            x2 = self.mlp2(x[:, self.num_class_tokens:])
        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)
        return x1, x2
