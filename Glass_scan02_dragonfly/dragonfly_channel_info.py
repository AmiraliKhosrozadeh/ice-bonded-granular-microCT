"""
Dragonfly diagnostic — prints every Channel's title, shape, origin, and
voxel spacing. Run once per session (once with scan 1 loaded, once with
scan 2 loaded) and paste the printed block back.

Usage (Dragonfly Python console):
  exec(open(r'E:\\RPTU-images\\CT_images\\Glass\\Glass_scan02_dragonfly\\dragonfly_channel_info.py', encoding='utf-8').read(), globals())
"""

print("\n" + "="*72)
print("DRAGONFLY CHANNEL INFO")
print("="*72)

cls_name = Channel.getClassNameStatic()
print(f"Class name: {cls_name!r}")

all_ch = Channel.getAllObjectsOfClass(cls_name)
ch_list = list(all_ch) if all_ch is not None else []
print(f"Found {len(ch_list)} channel(s):\n")

for i, ch in enumerate(ch_list):
    # Title
    title = ''
    for tmethod in ['getTitle', 'getPrivateTitle', 'getName', 'getLabel']:
        if hasattr(ch, tmethod):
            try:
                title = getattr(ch, tmethod)()
                if title and 'orsObj' not in str(title):
                    break
            except Exception:
                pass
    if not title or 'orsObj' in str(title):
        title = str(ch)

    # Origin: meters -> micrometers
    ox_um = oy_um = oz_um = None
    try:
        o = ch.getOrigin()
        ox_um, oy_um, oz_um = o.getX()*1e6, o.getY()*1e6, o.getZ()*1e6
    except Exception as e:
        print(f"    origin read failed: {e}")

    # Spacing
    sx = sy = sz = None
    try:
        sx = ch.getXSpacing() * 1e6
        sy = ch.getYSpacing() * 1e6
        sz = ch.getZSpacing() * 1e6
    except Exception:
        pass

    # Shape
    nx = ny = nz = None
    try:
        nx = ch.getXSize()
        ny = ch.getYSize()
        nz = ch.getZSize()
    except Exception:
        pass

    print(f"[{i}] title     : {title!r}")
    print(f"    shape XYZ : ({nx}, {ny}, {nz})")
    if ox_um is not None:
        print(f"    origin um : ({ox_um:.3f}, {oy_um:.3f}, {oz_um:.3f})")
    if sx is not None:
        print(f"    spacing   : ({sx:.4f}, {sy:.4f}, {sz:.4f}) um")
    print()

print("="*72)
print("Paste the block above back to Claude.")
print("="*72 + "\n")
