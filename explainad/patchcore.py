"""Compact PatchCore, operating on numpy arrays.

Written rather than taken from anomalib because we must run the *identical* detector
on two different modalities (RGB and depth). anomalib is built around image folders and
its own datamodules; feeding it in-memory depth arrays costs more code than the method
itself, and any difference between the two streams would contaminate the ablation.

Follows Roth et al., PatchCore: WideResNet50-2 features from layers 2 and 3, local
neighbourhood aggregation, a memory bank of normal patches, and a per-patch nearest
neighbour distance as the anomaly score.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SIZE = 256  # SegAD resizes everything to 256x256

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PatchCore:
    def __init__(self, bank_size: int = 40_000, seed: int = 0, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bank_size = bank_size
        self.seed = seed
        self.bank: torch.Tensor | None = None
        self._net = None

    # -- backbone ---------------------------------------------------------
    def _backbone(self):
        if self._net is None:
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            net.eval().to(self.device)
            for p in net.parameters():
                p.requires_grad_(False)
            self._net = net
        return self._net

    @torch.no_grad()
    def _embed(self, images: np.ndarray) -> torch.Tensor:
        """(N, H, W, 3) uint8 or (N, H, W) float -> (N, P, C) patch embeddings."""
        x = torch.as_tensor(images)
        if x.ndim == 3:  # single-channel map (depth): replicate to 3 channels so the
            x = x[..., None].repeat(1, 1, 1, 3)  # ImageNet backbone sees what it expects
        x = x.permute(0, 3, 1, 2).float()
        if x.max() > 1.5:
            x = x / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(self.device)

        net = self._backbone()
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x = net.layer1(x)
        f2 = net.layer2(x)
        f3 = net.layer3(f2)
        # Local neighbourhood aggregation (PatchCore Sec. 3.1): 3x3 average pooling
        # makes each patch descriptor cover a wider receptive field without deeper,
        # more ImageNet-class-specific features.
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        emb = torch.cat([f2, f3], dim=1)  # (N, C, h, w)
        self.grid = emb.shape[-2:]
        return emb.flatten(2).permute(0, 2, 1)  # (N, P, C)

    # -- fit / score ------------------------------------------------------
    def fit(self, images, batch: int = 8) -> "PatchCore":
        rng = np.random.default_rng(self.seed)
        keep = []
        per_image = max(1, self.bank_size // max(1, len(images)) * 4)
        for i in tqdm(range(0, len(images), batch), desc="patchcore fit", unit="batch", mininterval=30):
            e = self._embed(np.stack(images[i:i + batch]))
            e = e.reshape(-1, e.shape[-1])
            idx = rng.choice(e.shape[0], min(per_image * batch, e.shape[0]), replace=False)
            keep.append(e[torch.as_tensor(idx, device=e.device)].half())
        bank = torch.cat(keep)
        if bank.shape[0] > self.bank_size:
            idx = rng.choice(bank.shape[0], self.bank_size, replace=False)
            bank = bank[torch.as_tensor(idx, device=bank.device)]
        # ponytail: random subsampling instead of greedy coreset selection. Coreset
        # buys ~0.5 AUROC in the paper; swap it in if the baseline lands short.
        self.bank = bank.contiguous()
        return self

    @torch.no_grad()
    def score(self, images, batch: int = 8, sigma: float = 4.0) -> np.ndarray:
        """(N, SIZE, SIZE) anomaly maps."""
        assert self.bank is not None, "call fit() first"
        out = []
        for i in range(0, len(images), batch):
            e = self._embed(np.stack(images[i:i + batch])).half()  # (B, P, C)
            n, p, _ = e.shape
            d = torch.cdist(e.reshape(n * p, -1).float(), self.bank.float())
            s = d.min(dim=1).values.reshape(n, 1, *self.grid)
            s = F.interpolate(s, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
            out.append(_gaussian_blur(s, sigma)[:, 0].cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    normal = [rng.integers(0, 60, (SIZE, SIZE, 3), dtype=np.uint8) for _ in range(6)]
    odd = normal[0].copy()
    odd[100:150, 100:150] = 255  # a bright square nothing in training has

    pc = PatchCore(bank_size=4000).fit(normal)
    maps = pc.score([normal[1], odd])
    assert maps.shape == (2, SIZE, SIZE), maps.shape
    assert np.isfinite(maps).all()
    patch = maps[1][100:150, 100:150].mean()
    elsewhere = maps[1][:50, :50].mean()
    assert patch > elsewhere * 1.5, f"defect not localised: {patch} vs {elsewhere}"
    assert maps[1].max() > maps[0].max(), "anomalous image did not score higher"
    print("patchcore ok", round(float(maps[0].max()), 2), round(float(maps[1].max()), 2))
