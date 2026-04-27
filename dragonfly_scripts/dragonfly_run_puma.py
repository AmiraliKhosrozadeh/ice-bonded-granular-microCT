"""Dragonfly + PuMA driver — auto-pulls config from scanNN_segmentation.py.

Assumes you already have a Dragonfly ORSSession loaded for ANCHOR_SCAN.
Does NOT publish a new CT channel.

All per-scan parameters (thresholds, voxel size, origin, file prefix, etc.)
are read from `<specimen>/scanNN_dragonfly/scanNN_segmentation.py` via the
`scan_config` helper. Edit segmentation.py once → every downstream script
picks up the change automatically.

Single-line invocation in Dragonfly's Python console:
    exec(open(r"E:\\RPTU-images\\CT_images\\Glass\\<SPECIMEN>\\<SPAM>\\dragonfly_run_puma.py", encoding='utf-8').read(), globals())
"""
import os, sys
import numpy as np
import tifffile

ANCHOR_SCAN = 1
PUSH_ORIENTATION_TO_DRAGONFLY = False

# Material thermal conductivities (W/m·K) — used for the conductivity solve
K_TH = {'air': 0.025, 'ice': 2.20, 'glass': 1.05}

# exec(open(...)) doesn't set __file__; fall back to the absolute spam dir
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = r'E:\RPTU-images\CT_images\Glass\<SPECIMEN>\<SPAM>'
sys.path.insert(0, HERE)
from scan_config import load_scan_config

cfg = load_scan_config(ANCHOR_SCAN)
print(f'PuMA driver — scan {ANCHOR_SCAN}, parameters from {cfg["_seg_path"]}')
print('=' * 70)
for k in ('VOXEL_SIZE_UM', 'CT_ORIGIN_UM', 'T_AIR_ICE', 'T_ICE_GLASS',
          'T_GLASS_AL', 'FILE_PREFIX'):
    if k in cfg:
        print(f'  {k:<18}= {cfg[k]}')

VOXEL_M = float(cfg['VOXEL_SIZE_UM']) * 1e-6
DATA_DIR = os.path.join(HERE, 'data')

# ---- find pumapy ---------------------------------------------------------
import importlib
puma = None
for name in ('pumapy', 'puma', 'PuMA'):
    try:
        m = importlib.import_module(name)
        if any(a.startswith('compute_') for a in dir(m)):
            puma = m
            print(f'IMPORTED  {name!r}')
            break
    except ModuleNotFoundError:
        pass
if puma is None:
    print('PuMA not importable; aborting.')
    sys.exit(1)

# ---- load the non-aligned CT (already on disk, matches loaded session) ---
ct_path   = os.path.join(DATA_DIR, f'ct_scan{ANCHOR_SCAN:02d}.tif')
mask_path = os.path.join(DATA_DIR, f'specimen_mask_scan{ANCHOR_SCAN:02d}.tif')
print(f'\nLoading {ct_path}')
ct_arr   = tifffile.imread(ct_path)
mask_arr = tifffile.imread(mask_path) if os.path.exists(mask_path) else None
print(f'  CT shape = {ct_arr.shape}')

T_AIR   = float(cfg['T_AIR_ICE'])
T_ICE   = float(cfg['T_ICE_GLASS'])
T_GLASS = float(cfg['T_GLASS_AL'])
print(f'  thresholds: air<{T_AIR}  ice<{T_ICE}  glass<{T_GLASS}')

# Air-phase mask, restricted to the specimen
if mask_arr is not None:
    air_phase = (ct_arr < T_AIR) & (mask_arr > 0)
else:
    air_phase = ct_arr < T_AIR
print(f'  air-phase voxels: {int(air_phase.sum()):,}')

ws_air = puma.Workspace.from_array(air_phase.astype(np.int16))
print(f'  Workspace (air) built. shape={ws_air.matrix.shape}')

# 1) Continuum tortuosity along z
if hasattr(puma, 'compute_continuum_tortuosity'):
    print('\n--- Continuum tortuosity (air, z direction) ---')
    try:
        res = puma.compute_continuum_tortuosity(
            ws_air, cutoff=(1, 1), direction='z',
            tolerance=1e-4, maxiter=10000, solver_type='cg',
            display_iter=False)
        print(f'  result: {res}')
    except Exception as e:
        print(f'  call failed: {e}')

# 2) Effective thermal conductivity along z
if hasattr(puma, 'compute_thermal_conductivity'):
    print('\n--- Effective thermal conductivity (z direction) ---')
    try:
        cond = np.zeros(ct_arr.shape, dtype=np.float32)
        cond[ct_arr < T_AIR]                                = K_TH['air']
        cond[(ct_arr >= T_AIR) & (ct_arr < T_ICE)]          = K_TH['ice']
        cond[(ct_arr >= T_ICE) & (ct_arr < T_GLASS)]        = K_TH['glass']
        ws_cond = puma.Workspace.from_array(cond)
        res_k = puma.compute_thermal_conductivity(
            ws_cond, cond_map=None, direction='z',
            side_bc='symmetric', tolerance=1e-4, maxiter=5000,
            solver_type='cg', display_iter=False)
        print(f'  result: {res_k}')
    except Exception as e:
        print(f'  call failed: {e}')

# 3) Orientation distribution
if hasattr(puma, 'compute_orientation_st'):
    print('\n--- Orientation distribution (structure tensor) ---')
    try:
        ori = puma.compute_orientation_st(
            ws_air, cutoff=(1, 1), sigma=0.7, rho=1.4, edt=False)
        ori_arr = ori if hasattr(ori, 'shape') else getattr(ori, 'matrix', None)
        if ori_arr is not None:
            print(f'  orientation field shape: {ori_arr.shape}')
        if PUSH_ORIENTATION_TO_DRAGONFLY and ori_arr is not None:
            # Find the loaded scan-N channel and copy its origin/spacing
            from ORSModel import createChannelFromNumpyArray, Channel
            ref_ch = None
            for ch_ in Channel.getAllObjectsOfClass(Channel.getClassNameStatic()):
                if cfg['CT_CHANNEL_NAME'] in (ch_.getTitle() or ''):
                    ref_ch = ch_; break
            if ref_ch is None:
                print(f'  WARNING: scan {ANCHOR_SCAN} not loaded — skipping channel publish')
            else:
                scalar = ori_arr[..., 0] if ori_arr.ndim == 4 else ori_arr
                ch = createChannelFromNumpyArray(scalar.astype(np.float32))
                ch.setXSpacing(ref_ch.getXSpacing())
                ch.setYSpacing(ref_ch.getYSpacing())
                ch.setZSpacing(ref_ch.getZSpacing())
                ch.setOrigin(ref_ch.getOrigin())
                ch.setTitle(f'PuMA_orientation_z_scan{ANCHOR_SCAN}')
                ch.publish()
                print('  published orientation channel (aligned)')
    except Exception as e:
        print(f'  call failed: {e}')

# 4) Surface area
if hasattr(puma, 'compute_surface_area'):
    print('\n--- Surface area (air) ---')
    try:
        sa = puma.compute_surface_area(ws_air, cutoff=(1, 1))
        print(f'  result: {sa}')
    except Exception as e:
        print(f'  call failed: {e}')

print('\nDone.  Numbers are printed in this console.')
