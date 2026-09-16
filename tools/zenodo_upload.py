"""Upload the full-resolution volumes to a Zenodo deposition.

The reconstructions, phase maps and grain labels of the thirty scans are about
60 GB, beyond what a git repository holds; Zenodo takes 50 GB per record, so
the set is split into two records (glass and alumina; sand).  Each stack is
zlib-compressed before upload.

Usage, with a personal access token created in your Zenodo account settings
(scope deposit:write) placed in the environment, never in this file:

    set ZENODO_TOKEN=...              (Windows)   /   export ZENODO_TOKEN=...
    python tools/zenodo_upload.py --list files_beads.txt --title "..." [--sandbox]

files.txt holds one path per line.  The script creates the deposition, uploads
every file, prints the reserved DOI, and leaves the record unpublished so the
metadata can be checked in the browser before Publish is pressed.
"""
import argparse
import os
import sys

import requests

BASE = {"real": "https://zenodo.org/api", "sandbox": "https://sandbox.zenodo.org/api"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="text file, one path per line")
    ap.add_argument("--title", required=True)
    ap.add_argument("--sandbox", action="store_true")
    a = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("set ZENODO_TOKEN in the environment first")
    base = BASE["sandbox" if a.sandbox else "real"]
    hdr = {"Authorization": f"Bearer {token}"}
    r = requests.post(f"{base}/deposit/depositions", json={}, headers=hdr)
    r.raise_for_status()
    dep = r.json()
    bucket = dep["links"]["bucket"]
    print("deposition", dep["id"], "reserved DOI", dep["metadata"].get("prereserve_doi", {}).get("doi"))
    meta = {"metadata": {
        "title": a.title,
        "upload_type": "dataset",
        "description": ("Micro-CT reconstructions (24.77 um voxel), three-phase maps and grain "
                        "labels of ice-bonded granular columns under in-situ uniaxial compression "
                        "at -5 C; companion to the paper 'Grain-scale kinematics, ice-bond failure "
                        "and damage evolution in ice-bonded granular media under in-situ uniaxial "
                        "compression' and to https://github.com/AmiraliKhosrozadeh/ice-bonded-granular-microCT"),
        "creators": [{"name": "Khosrozadeh, Amirali"}, {"name": "Sinnwell, Yannik"},
                     {"name": "Pietsch-Braune, Swantje"}, {"name": "Nikolaus, Kai"},
                     {"name": "Antonyuk, Sergiy"}, {"name": "Heinrich, Stefan"}],
        "access_right": "open", "license": "cc-by-4.0",
    }}
    requests.put(f"{base}/deposit/depositions/{dep['id']}", json=meta, headers=hdr).raise_for_status()
    for line in open(a.list):
        p = line.strip()
        if not p:
            continue
        name = os.path.basename(p)
        print("uploading", name, f"{os.path.getsize(p)/1e9:.2f} GB", flush=True)
        with open(p, "rb") as fh:
            rr = requests.put(f"{bucket}/{name}", data=fh, headers=hdr)
        rr.raise_for_status()
    print("done; check the record in the browser, then Publish")


if __name__ == "__main__":
    main()
