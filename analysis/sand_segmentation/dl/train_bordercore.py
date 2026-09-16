"""
Train the 3D border-core U-Net on the synthetic sand volumes.
=============================================================
Targets are the r=1 border-core encodings in synth/bordercoreTr/ (0=bg,
1=core, 2=border).  Inputs are the CT-like grayscale in synth/imagesTr/.

Loss = weighted CrossEntropy + soft Dice (foreground classes).  The BORDER
class is upweighted because it is the thin ring that actually separates
touching grains -- get the ring wrong and grains merge on decode.

Patch-based (96^3), AMP, AdamW + cosine LR.  Validates by full-volume
sliding-window prediction (the same code path used at inference on the real
scan) and selects the best checkpoint on mean foreground Dice.

Run (WSL, GPU):
    ~/ps3d/bin/python /mnt/e/RPTU-images/CT_images/Sand/pipeline/dl/train_bordercore.py \
        --epochs 200 --patch 96 --batch 2
"""
import os
import csv
import time
import argparse
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from unet3d import UNet3D, znorm, sliding_window_logits

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, 'sand_atlas', 'synth')
IMG = os.path.join(SYN, 'imagesTr')
LAB = os.path.join(SYN, 'bordercoreTr')
CKPT_DIR = os.path.join(HERE, 'model')
os.makedirs(CKPT_DIR, exist_ok=True)
N_CLASSES = 3


def read_split():
    tr, va = [], []
    with open(os.path.join(SYN, 'manifest.csv')) as fh:
        for row in csv.DictReader(fh):
            (tr if row['split'] == 'train' else va).append(row['name'])
    return tr, va


# ---------------------------------------------------------------------------
# Dataset: random augmented 96^3 patches
# ---------------------------------------------------------------------------
class PatchDS(Dataset):
    def __init__(self, names, patch=96, samples_per_vol=8, train=True):
        self.patch = patch
        self.train = train
        self.spv = samples_per_vol
        self.vols, self.labs = [], []
        for n in names:
            img = tifffile.imread(os.path.join(IMG, f'{n}_0000.tif'))
            lab = tifffile.imread(os.path.join(LAB, f'{n}.tif')).astype(np.int64)
            self.vols.append(znorm(img))
            self.labs.append(lab)
        self.names = names

    def __len__(self):
        return len(self.vols) * self.spv

    def _crop(self, vol, lab):
        Z, Y, X = vol.shape
        p = self.patch
        if self.train:
            z0 = np.random.randint(0, Z - p + 1)
            y0 = np.random.randint(0, Y - p + 1)
            x0 = np.random.randint(0, X - p + 1)
        else:
            z0, y0, x0 = (Z - p) // 2, (Y - p) // 2, (X - p) // 2
        v = vol[z0:z0 + p, y0:y0 + p, x0:x0 + p]
        l = lab[z0:z0 + p, y0:y0 + p, x0:x0 + p]
        return v, l

    def _augment(self, v, l):
        # random flips
        for ax in range(3):
            if np.random.rand() < 0.5:
                v = np.flip(v, ax); l = np.flip(l, ax)
        # random 90-deg rotation in a random plane
        if np.random.rand() < 0.75:
            k = np.random.randint(1, 4)
            plane = [(0, 1), (0, 2), (1, 2)][np.random.randint(3)]
            v = np.rot90(v, k, plane); l = np.rot90(l, k, plane)
        v = np.ascontiguousarray(v); l = np.ascontiguousarray(l)
        # intensity jitter (post-znorm)
        if np.random.rand() < 0.5:
            v = v * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)
        if np.random.rand() < 0.3:
            v = v + np.random.normal(0, 0.05, v.shape).astype(np.float32)
        return v, l

    def __getitem__(self, idx):
        vi = idx % len(self.vols)
        v, l = self._crop(self.vols[vi], self.labs[vi])
        if self.train:
            v, l = self._augment(v, l)
        return torch.from_numpy(v.astype(np.float32))[None], torch.from_numpy(l.copy())


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def soft_dice(logits, target, eps=1.0):
    """Mean Dice over foreground classes (core=1, border=2)."""
    prob = torch.softmax(logits, 1)
    oh = F.one_hot(target, N_CLASSES).permute(0, 4, 1, 2, 3).float()
    dice = 0.0
    for c in (1, 2):
        p = prob[:, c]; t = oh[:, c]
        inter = (p * t).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
        dice = dice + (2 * inter + eps) / (denom + eps)
    return 1.0 - (dice / 2).mean()


# ---------------------------------------------------------------------------
# Validation: full-volume sliding window, foreground Dice
# ---------------------------------------------------------------------------
def validate(model, names, patch, device):
    dices = {1: [], 2: []}
    for n in names:
        img = tifffile.imread(os.path.join(IMG, f'{n}_0000.tif'))
        lab = tifffile.imread(os.path.join(LAB, f'{n}.tif')).astype(np.int64)
        prob = sliding_window_logits(model, znorm(img), patch=patch, device=device)
        pred = prob.argmax(0)
        for c in (1, 2):
            p = (pred == c); t = (lab == c)
            inter = (p & t).sum()
            denom = p.sum() + t.sum()
            dices[c].append((2 * inter + 1) / (denom + 1))
    d1 = float(np.mean(dices[1])); d2 = float(np.mean(dices[2]))
    return d1, d2, (d1 + d2) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patch', type=int, default=96)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--base', type=int, default=24)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--spv', type=int, default=8, help='patches sampled per volume per epoch')
    ap.add_argument('--val_every', type=int, default=5)
    ap.add_argument('--border_w', type=float, default=3.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--init', type=str, default='',
                    help='checkpoint to warm-start from (fine-tune)')
    ap.add_argument('--ckpt_tag', type=str, default='',
                    help='suffix for output checkpoints, e.g. _v2 -> bordercore_unet_v2_best.pt')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr_names, va_names = read_split()
    print(f"train {len(tr_names)} vols | val {len(va_names)} vols | device {device}")

    tr_ds = PatchDS(tr_names, patch=a.patch, samples_per_vol=a.spv, train=True)
    tr_ld = DataLoader(tr_ds, batch_size=a.batch, shuffle=True,
                       num_workers=a.workers, pin_memory=True, drop_last=True,
                       persistent_workers=a.workers > 0)

    model = UNet3D(1, N_CLASSES, base=a.base).to(device)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"UNet3D base={a.base}  {nparam:.1f} M params")
    if a.init:
        ck = torch.load(a.init, map_location=device)
        sd = ck.get('model', ck)
        model.load_state_dict(sd)
        print(f"warm-started from {a.init} (epoch {ck.get('epoch','?')}, "
              f"val dice {ck.get('val_dice_mean','?')})")

    ce_w = torch.tensor([1.0, 1.5, a.border_w], device=device)  # bg, core, border
    ce = nn.CrossEntropyLoss(weight=ce_w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=3e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler('cuda')

    tagn = a.ckpt_tag
    best_path = os.path.join(CKPT_DIR, f'bordercore_unet{tagn}_best.pt')
    last_path = os.path.join(CKPT_DIR, f'bordercore_unet{tagn}_last.pt')
    best = -1.0
    log_path = os.path.join(CKPT_DIR, f'train_log{tagn}.csv')
    with open(log_path, 'w', newline='') as fh:
        csv.writer(fh).writerow(['epoch', 'train_loss', 'val_dice_core',
                                 'val_dice_border', 'val_dice_mean', 'lr', 'sec'])

    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in tr_ld:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                logit = model(x)
                loss = ce(logit, y) + soft_dice(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item()
        sched.step()
        tl = running / max(1, len(tr_ld))

        dc = db = dm = float('nan')
        if ep % a.val_every == 0 or ep == a.epochs:
            dc, db, dm = validate(model, va_names, a.patch, device)
            if dm > best:
                best = dm
                torch.save({'model': model.state_dict(), 'base': a.base,
                            'epoch': ep, 'val_dice_mean': dm}, best_path)
                tag = '  <- best'
            else:
                tag = ''
            print(f"ep {ep:3d}  loss {tl:.4f}  dice core {dc:.3f} border {db:.3f} "
                  f"mean {dm:.3f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{time.time()-t0:.0f}s{tag}", flush=True)
        else:
            print(f"ep {ep:3d}  loss {tl:.4f}  {time.time()-t0:.0f}s", flush=True)

        with open(log_path, 'a', newline='') as fh:
            csv.writer(fh).writerow([ep, f"{tl:.4f}", f"{dc:.4f}", f"{db:.4f}",
                                     f"{dm:.4f}", f"{sched.get_last_lr()[0]:.3e}",
                                     f"{time.time()-t0:.0f}"])

    torch.save({'model': model.state_dict(), 'base': a.base,
                'epoch': a.epochs, 'val_dice_mean': dm}, last_path)
    print(f"\nDone. best val mean dice {best:.3f} -> {best_path}")


if __name__ == '__main__':
    main()
