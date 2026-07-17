"""PyTorch adaptation of FB-tCNN for SSVEP classification.

Reference architecture:
W. Ding et al., "Filter Bank Convolutional Neural Network for Short
Time-Window Steady-State Visual Evoked Potential Classification," TNSRE, 2021.

Input shape: (batch, n_subbands, n_channels, n_samples)
Output: logits (batch, n_classes)

The implementation preserves the central design of the public Keras code:
- four/sub-band inputs;
- three convolutional layers whose convolution weights are shared across bands;
- branch-specific batch normalization;
- additive fusion of sub-band features;
- final temporal fusion convolution and linear classifier.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv2d(nn.Module):
    """Conv2d with TensorFlow-style SAME padding for arbitrary stride."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1),
                 bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=0, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh = math.ceil(ih / sh)
        ow = math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + kh - ih, 0)
        pad_w = max((ow - 1) * sw + kw - iw, 0)
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        x = F.pad(x, (left, right, top, bottom))
        return self.conv(x)


class FBT_CNN(nn.Module):
    """Filter-bank time-domain CNN (FB-tCNN).

    Parameters mirror the public implementation where practical. The number of
    classes is generalized from the original four-class example to arbitrary
    SSVEP target counts.
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 250,
        n_classes: int = 40,
        n_subbands: int = 4,
        branch_filters: int = 16,
        fusion_filters: int = 32,
        temporal_stride: int = 5,
        local_kernel: int = 5,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if n_channels < 1 or n_samples < local_kernel:
            raise ValueError("Invalid n_channels or n_samples for FB-tCNN")
        if n_subbands < 1:
            raise ValueError("n_subbands must be positive")

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_subbands = n_subbands
        self.dropout_p = dropout

        # Shared convolution kernels across all filter-bank branches.
        self.conv_spatial = nn.Conv2d(
            1, branch_filters, kernel_size=(n_channels, 1), bias=True
        )
        self.conv_global_temporal = SamePadConv2d(
            branch_filters, branch_filters,
            kernel_size=(1, n_samples), stride=(1, temporal_stride), bias=True
        )
        self.conv_local_temporal = nn.Conv2d(
            branch_filters, branch_filters,
            kernel_size=(1, local_kernel), bias=True
        )

        # The public implementation uses independent BN layers per branch while
        # sharing convolution kernels. Preserve that behavior.
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(branch_filters) for _ in range(n_subbands)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(branch_filters) for _ in range(n_subbands)])
        self.bn3 = nn.ModuleList([nn.BatchNorm2d(branch_filters) for _ in range(n_subbands)])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

        # Infer the post-branch temporal width and fuse it exactly as in the
        # original final Conv2D(kernel=(1, remaining_width)).
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            z = self.conv_spatial(dummy)
            z = self.conv_global_temporal(z)
            z = self.conv_local_temporal(z)
            fused_width = int(z.shape[-1])
        if fused_width < 1:
            raise ValueError("Temporal window is too short for FB-tCNN")

        self.conv_fusion = nn.Conv2d(
            branch_filters, fusion_filters,
            kernel_size=(1, fused_width), bias=True
        )
        self.bn_fusion = nn.BatchNorm2d(fusion_filters)
        self.classifier = nn.Linear(fusion_filters, n_classes)
        self.feature_dim = fusion_filters

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _forward_branch(self, x: torch.Tensor, band_idx: int) -> torch.Tensor:
        x = self.activation(self.bn1[band_idx](self.conv_spatial(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2[band_idx](self.conv_global_temporal(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3[band_idx](self.conv_local_temporal(x)))
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input (B, bands, channels, samples), got {tuple(x.shape)}"
            )
        if x.shape[1] != self.n_subbands or x.shape[2] != self.n_channels:
            raise ValueError(
                f"Expected bands/channels=({self.n_subbands},{self.n_channels}), "
                f"got ({x.shape[1]},{x.shape[2]})"
            )

        branch_outputs = []
        for b in range(self.n_subbands):
            xb = x[:, b:b + 1, :, :]
            branch_outputs.append(self._forward_branch(xb, b))

        fused = torch.stack(branch_outputs, dim=0).sum(dim=0)
        fused = self.dropout(fused)
        fused = self.activation(self.bn_fusion(self.conv_fusion(fused)))
        return fused.flatten(1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.classifier(self.dropout(features))
        return (logits, features) if return_features else logits

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
