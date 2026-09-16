"""
Run the trained border-core U-Net on a real CT volume, then decode to
per-grain instance labels.
==========================================================================
Input can be a single 3D .tif OR a directory of 2D slice .tifs (e.g.
E:\\RPTU-images\\CT_images\\Sand\\Sand_100_500_T5_01).  The volume is
z-score normalized exactly as in training, predicted by Gaussian-weighted
sliding window, argmaxed to the 3-class border-core map, then handed to
decode_instances.decode() -> integer per-grain labels (same format as the
Stage-3 watershed draft, so it drops into the existing PSD/SPAM stack).

Run (WSL, GPU):
    ~/ps3d/bin/python infer_bordercore.py \
        --in /mnt/e/RPTU-images/CT_images/Sand/Sand_100_500_T5_01 \
        --out /mnt/e/RPTU-images/CT_images/Sand/pipeline/dl/results \
        [--zmin 0 --zmax 620] [--patch 96]
"""
import os
import glob
import argparse
import numpy as np
import tifffile
import torch

from unet3d import UNet3D, znorm, sliding_window_logits
from decode_instances import decode

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_CKPT = os.path.join(HERE, 'model', 'bordercore_unet_best.pt')


def load_volume(path, zmin=None, zmax=None):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.tif')) +
                       glob.glob(os.path.join(path, '*.tiff')))
        if zmin is not None or zmax is not None:
            files = files[(zmin or 0):(zmax if zmax is not None else len(files))]
        print(f"  loading {len(files)} slices from {path}")
        sl0 = tifffile.imread(files[0])
        vol = np.empty((len(files),) + sl0.shape, sl0.dtype)
        vol[0] = sl0
        for i, f in enumerate(files[1:], 1):
            vol[i] = tifffile.imread(f)
        return vol
    vol = tifffile.imread(path)
    if zmin is not None or zmax is not None:
        vol = vol[(zmin or 0):(zmax if zmax is not None else len(vol))]
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--ckpt', default=DEF_CKPT)
    ap.add_argument('--patch', type=int, default=96)
    ap.add_argument('--overlap', type=float, default=0.5)
    ap.add_argument('--min', type=int, default=20, help='min grain voxels for decode')
    ap.add_argument('--zmin', type=int, default=None)
    ap.add_argument('--zmax', type=int, default=None)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ck = torch.load(a.ckpt, map_location=device)
    model = UNet3D(1, 3, base=ck.get('base', 24)).to(device)
    model.load_state_dict(ck['model'])
    print(f"loaded {a.ckpt} (epoch {ck.get('epoch')}, val dice {ck.get('val_dice_mean'):.3f})")

    print("loading volume ...")
    raw = load_volume(a.inp, a.zmin, a.zmax)
    print(f"  volume {raw.shape} dtype {raw.dtype} range {raw.min()}..{raw.max()}")

    print("sliding-window inference ...")
    prob = sliding_window_logits(model, znorm(raw), patch=a.patch,
                                 overlap=a.overlap, device=device)
    pred = prob.argmax(0).astype(np.uint8)
    bc_path = os.path.join(a.out, 'bordercore_pred.tif')
    tifffile.imwrite(bc_path, pred)
    frac = [(pred == c).mean() for c in (0, 1, 2)]
    print(f"  border-core fractions bg/core/border = "
          f"{frac[0]:.3f}/{frac[1]:.3f}/{frac[2]:.3f} -> {bc_path}")

    print("decoding instances ...")
    labels, n = decode(pred, a.min)
    dt = np.uint16 if labels.max() < 65535 else np.uint32
    lab_path = os.path.join(a.out, 'dl_grain_labels.tif')
    tifffile.imwrite(lab_path, labels.astype(dt))
    print(f"  {n} grain instances -> {lab_path}")

    # quick PSD summary
    if n:
        sizes = np.bincount(labels.ravel())[1:]
        sizes = sizes[sizes > 0]
        VOX = 24.7660229
        d = 2.0 * (3.0 * sizes * VOX ** 3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        print(f"  equiv-diam um  d10/d50/d90 = "
              f"{np.percentile(d,10):.0f}/{np.percentile(d,50):.0f}/{np.percentile(d,90):.0f}")


if __name__ == '__main__':
    main()
