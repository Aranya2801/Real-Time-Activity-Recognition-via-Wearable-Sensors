"""Augmentations for IMU windows (applied on the fly, on torch tensors).

The one that matters most is the random rotation. The phone sits in the
trouser pocket at a slightly different angle for every person, so a
model trained on 19 people sees only 19 orientations. Rotating every
xyz-triplet with the same random rotation matrix simulates other
placements. Jitter and scaling help a bit on top of that.
"""

from __future__ import annotations

import math

import torch


def random_rotation_matrices(n: int, max_deg: float, device=None) -> torch.Tensor:
    """Rotation about a random axis by an angle in [-max_deg, max_deg]."""
    axis = torch.randn(n, 3, device=device)
    axis = axis / axis.norm(dim=1, keepdim=True)
    theta = (torch.rand(n, device=device) * 2 - 1) * math.radians(max_deg)
    K = torch.zeros(n, 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    eye = torch.eye(3, device=device).expand(n, 3, 3)
    s, c = torch.sin(theta)[:, None, None], torch.cos(theta)[:, None, None]
    return eye + s * K + (1 - c) * (K @ K)  # Rodrigues


class Augmenter:
    def __init__(self, rotate_deg: float = 30.0, jitter: float = 0.05, scale: float = 0.1,
                 p: float = 0.8, channel_std=None):
        # channel_std: (C,) std of each raw channel, so jitter is relative to the signal
        self.std = None if channel_std is None else torch.as_tensor(channel_std).float().view(1, -1, 1)
        self.rotate_deg = rotate_deg
        self.jitter = jitter
        self.scale = scale
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) in raw sensor units. Rotation needs C to be a multiple of 3."""
        B, C, _ = x.shape
        apply = (torch.rand(B, device=x.device) < self.p).float()[:, None, None]
        out = x

        if self.rotate_deg > 0 and C % 3 == 0:
            R = random_rotation_matrices(B, self.rotate_deg, x.device)
            groups = out.view(B, C // 3, 3, -1)
            rotated = torch.einsum("bij,bgjl->bgil", R, groups).reshape(B, C, -1)
            out = apply * rotated + (1 - apply) * out

        if self.scale > 0:
            s = 1 + (torch.rand(B, C, 1, device=x.device) * 2 - 1) * self.scale
            out = out * (apply * s + (1 - apply))

        if self.jitter > 0:
            noise = torch.randn_like(out) * self.jitter
            if self.std is not None:
                noise = noise * self.std.to(x.device)
            out = out + apply * noise
        return out
