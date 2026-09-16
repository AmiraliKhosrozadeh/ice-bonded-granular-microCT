"""
Self-contained 3D U-Net for 3-class border-core sand grain segmentation.
========================================================================
Shared by train_bordercore.py (training) and infer_bordercore.py (the real
user scan).  Deliberately small so the whole forward/backward fits an 8 GB
RTX 3070 at a 96^3 patch, and simple enough to read end-to-end.

Classes: 0 = background (air/ice/void), 1 = grain CORE, 2 = grain BORDER.
The border ring keeps touching grains topologically separated so a plain
connected-components + watershed decode (decode_instances.py) recovers the
individual grains.

GroupNorm (not BatchNorm) because patch batches are tiny (1-2); GroupNorm is
batch-size independent and gives stable stats on small 3D batches.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _gn(ch, groups=8):
    g = min(groups, ch)
    while ch % g:
        g -= 1
    return nn.GroupNorm(g, ch)


class ConvBlock(nn.Module):
    """(conv -> GN -> LeakyReLU) x2."""
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.LeakyReLU(0.01, True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), _gn(cout), nn.LeakyReLU(0.01, True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, base=24, depth=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth)]      # 24,48,96,192
        self.enc = nn.ModuleList()
        cin = in_ch
        for c in chs:
            self.enc.append(ConvBlock(cin, c))
            cin = c
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(chs[-1], chs[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cprev = chs[-1] * 2
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose3d(cprev, c, 2, stride=2))
            self.dec.append(ConvBlock(c * 2, c))
            cprev = c
        self.head = nn.Conv3d(chs[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = upconv(x)
            # pad if odd-sized (defensive; patches are even here)
            if x.shape[2:] != skip.shape[2:]:
                d = [skip.shape[2 + i] - x.shape[2 + i] for i in range(3)]
                x = F.pad(x, [0, d[2], 0, d[1], 0, d[0]])
            x = dec(torch.cat([x, skip], 1))
        return self.head(x)


# ---------------------------------------------------------------------------
# Normalization (ParticleSeg3D-style per-volume z-score on foreground-ish)
# ---------------------------------------------------------------------------
def znorm(img):
    """Per-volume z-score; robust to the bimodal CT histogram."""
    img = img.astype(np.float32)
    m, s = img.mean(), img.std()
    return (img - m) / (s + 1e-6)


# ---------------------------------------------------------------------------
# Sliding-window inference (returns class-probability volume, C,Z,Y,X)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sliding_window_logits(model, vol, patch=96, overlap=0.5, device='cuda',
                          out_ch=3, amp=True, acc_device='cpu'):
    """vol: float32 normalized 3D array. Gaussian-weighted patch blending.

    Patch forward runs on `device` (GPU); the full-volume accumulators live on
    `acc_device` ('cpu' by default) so a large real scan does not OOM an 8 GB
    card.  Each patch's softmax is moved to acc_device before blending.
    """
    model.eval()
    Z0, Y0, X0 = vol.shape
    # a bbox dimension smaller than `patch` would make the boundary patch smaller
    # than the Gaussian window -> shape mismatch in `prob * wt`.  Reflect-pad up
    # to `patch` (divisible by the U-Net's 2^depth), run, then crop back.
    pz, py, px = max(patch, Z0), max(patch, Y0), max(patch, X0)
    if (pz, py, px) != (Z0, Y0, X0):
        vol = np.pad(vol, ((0, pz - Z0), (0, py - Y0), (0, px - X0)), mode='reflect')
    Z, Y, X = vol.shape
    step = max(1, int(patch * (1 - overlap)))

    def starts(n):
        if n <= patch:
            return [0]
        s = list(range(0, n - patch + 1, step))
        if s[-1] != n - patch:
            s.append(n - patch)
        return s

    zs, ys, xs = starts(Z), starts(Y), starts(X)
    # gaussian weight window
    ax = np.linspace(-1, 1, patch)
    g1 = np.exp(-(ax ** 2) / (2 * 0.25))
    w = (g1[:, None, None] * g1[None, :, None] * g1[None, None, :]).astype(np.float32)
    wt = torch.from_numpy(w)[None].to(acc_device)

    acc = torch.zeros((out_ch, Z, Y, X), device=acc_device)
    wsum = torch.zeros((1, Z, Y, X), device=acc_device)
    vt = torch.from_numpy(vol)
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                p = vt[z0:z0 + patch, y0:y0 + patch, x0:x0 + patch][None, None].to(device)
                with torch.autocast('cuda', enabled=amp):
                    logit = model(p)[0].float()
                prob = torch.softmax(logit, 0).to(acc_device)
                acc[:, z0:z0 + patch, y0:y0 + patch, x0:x0 + patch] += prob * wt
                wsum[:, z0:z0 + patch, y0:y0 + patch, x0:x0 + patch] += wt
    acc /= wsum.clamp_min(1e-6)
    # crop back to the original (unpadded) volume so callers get the shape they
    # passed in; the reflect-pad above only exists to satisfy the Gaussian window.
    return acc[:, :Z0, :Y0, :X0].cpu().numpy()
