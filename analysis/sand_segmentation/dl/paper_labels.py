"""Central paper-facing label map. EDIT HERE to relabel ALL figures consistently.
Mapping (2026-06-08): 100_500->S1, 75_200->S2, 25mm->S3; T5_0x->Stage x.
Format: 'Sand S2 - Stage 2' (en-dash)."""
import re
MATERIAL = "Sand"
SPEC_NUM = {"100_500": 1, "75_200": 2, "25mm": 3}
DASH = "–"

def series_of(name):
    for k in SPEC_NUM:
        if name.startswith(k):
            return k
    return None

def stage_num(name):
    m = re.search(r"T5_0?(\d)", name)
    return int(m.group(1)) if m else None

def specimen(series):            # '100_500' -> 'Sand S1'
    return f"{MATERIAL} S{SPEC_NUM[series]}"

def stage(name):                 # '..._T5_02' -> 'Stage 2'
    n = stage_num(name)
    return f"Stage {n}" if n else ""

def full(name):                  # '75_200_T5_02' -> 'Sand S2 - Stage 2'
    s = series_of(name)
    return f"{specimen(s)} {DASH} {stage(name)}" if s else name
