"""MultiCropWrapper: run the backbone once per resolution, then the head.

Faithful port of ``utils.MultiCropWrapper``. The training loop calls global and
local crops in separate forward passes, so within any single call all inputs share
a resolution (attention maps concatenate cleanly).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiCropWrapper(nn.Module):
    """Forward each resolution group through the backbone, then the head."""

    def __init__(self, backbone, head=None):
        super().__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = nn.Identity() if head is None else head

    def forward(self, x, mask=None, return_backbone_feat=False, return_attention=False, **kwargs):
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)

        start_idx, output, attention = 0, None, None
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx:end_idx])
            if mask is not None:
                kwargs.update(dict(mask=torch.cat(mask[start_idx:end_idx])))
            _out, _att = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output, attention = _out, _att
            else:
                output = torch.cat((output, _out))
                attention = torch.cat((attention, _att))
            start_idx = end_idx

        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        if return_attention:
            return output_, attention
        return output_


def has_batchnorms(model: nn.Module) -> bool:
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    return any(isinstance(m, bn_types) for m in model.modules())


def get_params_groups(model: nn.Module):
    """Split params into regularized (weights) and not-regularized (biases/norms)."""
    regularized, not_regularized = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]
