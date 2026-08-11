"""FID (Frechet Inception Distance) helpers for eval_flir.py.

The feature extractor mirrors pytorch-fid: torchvision InceptionV3 truncated at
``Mixed_7c``, input resized to 299x299 and normalized to [-1, 1]. This keeps
values comparable to the standard pytorch-fid numbers without adding a
dependency (torchvision is already required for ``--seg-mode deeplab``).

``calculate_fid`` only needs numpy + scipy and is importable without torch;
``build_fid_inception`` imports torch/torchvision lazily.
"""
from __future__ import annotations

import numpy as np


def calculate_fid(real: np.ndarray, fake: np.ndarray) -> float:
    """Frechet distance between two feature sets of shape (N, D)."""
    from scipy.linalg import sqrtm

    mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mu2, sigma2 = fake.mean(axis=0), np.cov(fake, rowvar=False)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float((mu1 - mu2) @ (mu1 - mu2) + np.trace(sigma1 + sigma2 - 2 * covmean))
    return max(fid, 0.0)


def build_fid_inception(device="cuda"):
    """Return a truncated InceptionV3 producing 2048-dim feature vectors."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models

    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
        aux_logits=False,
    )
    blocks = [
        inception.Conv2d_1a_3x3,
        inception.Conv2d_2a_3x3,
        inception.Conv2d_2b_3x3,
        nn.MaxPool2d(kernel_size=3, stride=2),
        inception.Conv2d_3b_1x1,
        inception.Conv2d_4a_3x3,
        nn.MaxPool2d(kernel_size=3, stride=2),
        inception.Mixed_5b,
        inception.Mixed_5c,
        inception.Mixed_5d,
        inception.Mixed_6a,
        inception.Mixed_6b,
        inception.Mixed_6c,
        inception.Mixed_6d,
        inception.Mixed_6e,
        inception.Mixed_7a,
        inception.Mixed_7b,
        inception.Mixed_7c,
    ]

    class _FIDInception(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.blocks = nn.ModuleList(blocks)
            self.eval()
            self.to(device)

        @torch.no_grad()
        def forward(self, x):
            # x: (B,3,299,299) in [-1,1]
            for block in self.blocks:
                x = block(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            return x.flatten(1)  # (B,2048)

    return _FIDInception(blocks)
