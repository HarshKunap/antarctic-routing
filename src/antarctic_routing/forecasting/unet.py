"""Compact U-Net for multi-lead sea-ice concentration forecasting.

Inputs are padded (replicate) to a multiple of 2**depth and the output is
cropped back, so any grid size works.

Two output modes:

* direct (``persistence_channel=None``): C_hat = sigmoid(head);
* residual (``persistence_channel=i``): C_hat = clip(C_t + head, 0, 1), where
  C_t is input channel i (today's map). The head is zero-initialised, so an
  untrained model *is* persistence and training only has to learn the change.
  This is what lets the network match persistence at short leads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class IceUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base: int = 16, depth: int = 3,
                 persistence_channel: int | None = None) -> None:
        super().__init__()
        self.depth = depth
        self.persistence_channel = persistence_channel
        chans = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(in_channels, chans[0])] +
                                  [_block(chans[i], chans[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth))]
        )
        self.dec = nn.ModuleList([_block(chans[i] * 2, chans[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chans[0], out_channels, 1)
        if persistence_channel is not None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        today = x[:, self.persistence_channel: self.persistence_channel + 1] if self.persistence_channel is not None \
            else None
        m = 2**self.depth
        ph, pw = (-h) % m, (-w) % m
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < self.depth:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec, strict=True):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        out = self.head(x)[..., :h, :w]
        if today is None:
            return torch.sigmoid(out)
        return torch.clamp(today + out, 0.0, 1.0)
