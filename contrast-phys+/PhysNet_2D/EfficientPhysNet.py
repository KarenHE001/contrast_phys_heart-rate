#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EfficientPhysNet — PhysNet accuracy with EfficientPhys 2D+TSM efficiency.

Key changes vs PhysNet:
  - Conv3d  → Conv2d + TSM  (no 3D ops at all)
  - U-Net decoder removed   (TSM provides temporal context implicitly)
  - AdaptiveAvgPool3d → AvgPool2d (2D only)
  - Output: (B, N, T) ST-rPPG block — fully compatible with existing pipeline
  - ONNX/TFLite friendly: no dynamic Resize, no parity padding
  - Dynamic T: TSM n_segment is inferred at runtime, not fixed at __init__
               so the same model weights work for any clip length (T=300,
               T=120 with --5s, T=150 for windowed inference, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# TSM: Temporal Shift Module (from EfficientPhys)
# Shifts 1/fold_div channels left/right across the T dimension
# to inject temporal context into 2D convolutions — zero cost.
#
# Dynamic version: n_segment is NOT stored at __init__.
# Instead, forward() receives B and T as arguments and computes
# n_segment = T on the fly. This allows the same trained weights
# to run on any clip length without re-instantiating the model.
# ─────────────────────────────────────────────────────────────
class TSM(nn.Module):
    def __init__(self, fold_div=4):
        super().__init__()
        self.fold_div = fold_div
        # n_segment is NOT stored here — inferred dynamically in forward()

    def forward(self, x, T):
        """
        x : (B*T, C, H, W)
        T : number of frames — passed in at runtime so this module is
            compatible with any clip length.
        """
        nt, c, h, w = x.size()
        B = nt // T
        x    = x.view(B, T, c, h, w)
        fold = c // self.fold_div
        out  = torch.zeros_like(x)
        out[:, :-1, :fold]       = x[:, 1:,  :fold]       # shift left  (future)
        out[:, 1:,  fold:2*fold] = x[:, :-1, fold:2*fold] # shift right (past)
        out[:, :,   2*fold:]     = x[:, :,   2*fold:]      # no shift
        return out.view(nt, c, h, w)


# ─────────────────────────────────────────────────────────────
# Conv2d block: TSM → Conv2d → BN → ELU
# T is forwarded through so TSM stays dynamic.
# ─────────────────────────────────────────────────────────────
class TSMConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.tsm  = TSM()                                          # no fixed T
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ELU(inplace=True)

    def forward(self, x, T):
        x = self.tsm(x, T)
        x = self.act(self.bn(self.conv(x)))
        return x


# ─────────────────────────────────────────────────────────────
# EfficientPhysNet
# ─────────────────────────────────────────────────────────────
class EfficientPhysNet(nn.Module):
    """
    Drop-in replacement for PhysNet.

    Input:  (B, C, T, H, W)   — same as PhysNet, T can be any value
    Output: (B, N, T)          — same ST-rPPG block as PhysNet
              N = S*S + 1
              • x[:,  0:S*S, :]  — one signal per spatial cell  (B, S*S, T)
              • x[:, -1,     :]  — spatial average rPPG signal  (B,   1, T)
                                   (same as PhysNet's model_output[:,-1])

    Dynamic T: no fixed clip length at init. Train with T=300,
               infer with T=120 (--5s) or any window size.
    """

    def __init__(self, S=2, in_ch=3, input_size=128):
        super().__init__()
        self.S = S
        self.input_size = input_size

        # ── Input normalization (same as PhysNet) ─────────────
        # done in forward(), not as a layer

        # ── Stage 0: initial projection ───────────────────────
        # Input: (B*T, C, H, W)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )

        # ── Stage 1–4 ─────────────────────────────────────────
        self.pool1   = nn.AvgPool2d(2, 2)
        self.block1a = TSMConvBlock(32, 64)
        self.block1b = TSMConvBlock(64, 64)
        self.pool2   = nn.AvgPool2d(2, 2)
        self.block2a = TSMConvBlock(64, 64)
        self.block2b = TSMConvBlock(64, 64)
        self.pool3   = nn.AvgPool2d(2, 2)
        self.block3a = TSMConvBlock(64, 64)
        self.block3b = TSMConvBlock(64, 64)
        self.pool4   = nn.AvgPool2d(2, 2)
        self.block4a = TSMConvBlock(64, 64)
        self.block4b = TSMConvBlock(64, 64)

        # ── Output head ───────────────────────────────────────
        # input_size // 16 = feature map H/W after 4× pool2d(2,2)
        _feat_hw = input_size // (2 ** 4)
        _k = max(1, _feat_hw // S)
        self.out_pool = nn.AvgPool2d(_k, _k)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, bias=False)

    # ──────────────────────────────────────────────────────────
    def forward(self, x):
        """
        x: (B, C, T, H, W)  — T can be any value, no re-instantiation needed
        """
        B, C, T, H, W = x.shape
        if H != self.input_size or W != self.input_size:
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
            x = x.view(B, T, C, self.input_size, self.input_size).permute(0, 2, 1, 3, 4)
            H, W = self.input_size, self.input_size

        # ── Normalize (same as PhysNet) ────────────────────────
        means = x.mean(dim=(2, 3, 4), keepdim=True)
        stds  = x.std (dim=(2, 3, 4), keepdim=True)
        x = (x - means) / (stds + 1e-7)

        # ── Reshape to (B*T, C, H, W) for 2D processing ───────
        x = x.permute(0, 2, 1, 3, 4)          # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)          # (B*T, C, H, W)

        # ── Stem ──────────────────────────────────────────────
        x = self.stem(x)                        # (B*T, 32, H,    W)

        # ── Stage 1 ───────────────────────────────────────────
        x = self.pool1(x)                       # (B*T, 32, H/2,  W/2)
        x = self.block1a(x, T)                  # (B*T, 64, H/2,  W/2)
        x = self.block1b(x, T)                  # (B*T, 64, H/2,  W/2)

        # ── Stage 2 ───────────────────────────────────────────
        x = self.pool2(x)                       # (B*T, 64, H/4,  W/4)
        x = self.block2a(x, T)                  # (B*T, 64, H/4,  W/4)
        x = self.block2b(x, T)                  # (B*T, 64, H/4,  W/4)

        # ── Stage 3 ───────────────────────────────────────────
        x = self.pool3(x)                       # (B*T, 64, H/8,  W/8)
        x = self.block3a(x, T)                  # (B*T, 64, H/8,  W/8)
        x = self.block3b(x, T)                  # (B*T, 64, H/8,  W/8)

        # ── Stage 4 ───────────────────────────────────────────
        x = self.pool4(x)                       # (B*T, 64, H/16, W/16)
        x = self.block4a(x, T)                  # (B*T, 64, H/16, W/16)
        x = self.block4b(x, T)                  # (B*T, 64, H/16, W/16)

        # ── Output head ───────────────────────────────────────
        x = self.out_pool(x)                    # (B*T, 64, S, S)
        x = self.out_conv(x)                    # (B*T,  1, S, S)

        # ── Reshape back to ST-rPPG block (B, N, T) ───────────
        # Fix: view directly to (B, T, S, S), then index spatial cells.
        # This exactly mirrors PhysNet's end block output layout.
        x = x.view(B, T, self.S, self.S)       # (B, T, S, S)
        x = x.permute(0, 2, 3, 1)              # (B, S, S, T)

        # Collect one (B, 1, T) signal per spatial cell — same as PhysNet
        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:, a, b, :].unsqueeze(1))  # (B, 1, T)

        x_avg = sum(x_list) / (self.S * self.S)            # (B, 1, T)  spatial mean
        X = torch.cat(x_list + [x_avg], dim=1)             # (B, N, T)  N = S*S+1
        return X
        # ── Output layout (identical to PhysNet) ──────────────
        #   X[:, 0:S*S, :]  — per-cell rPPG signals
        #   X[:, -1,    :]  — spatial-average rPPG  ← used in train.py as model_output[:,-1]