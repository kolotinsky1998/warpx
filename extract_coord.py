#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def step_from_filename(path: str) -> int:
    m = re.search(r"openpmd_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def find_first_key(obj: Any, target_key: str) -> Optional[Any]:
    """
    Recursively find the first occurrence of a dict key `target_key`
    anywhere in a nested JSON structure. Returns the VALUE of that key.
    """
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for v in obj.values():
            got = find_first_key(v, target_key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = find_first_key(v, target_key)
            if got is not None:
                return got
    return None


def get_species_block(root: Dict[str, Any], species: str) -> Dict[str, Any]:
    """
    Find the 'particles' block anywhere, then get species within it.
    """
    particles = root.get("particles", None)
    if particles is None:
        particles = find_first_key(root, "particles")
    if particles is None or not isinstance(particles, dict):
        raise KeyError("Could not find a 'particles' dict anywhere in this JSON file.")
    if species not in particles:
        raise KeyError(f"Found 'particles' but species '{species}' not present. Available: {list(particles.keys())}")
    return particles[species]


def extract_xyz_ids(
    species_block: Dict[str, Any],
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """
    From your snippet (openPMD/Warpx-like):
      position/x/data, position/y/data OR position/y/attributes/value/value, position/z/data
      id/data
    """
    pos = species_block["position"]

    x = list(map(float, pos["x"]["data"]))
    z = list(map(float, pos["z"]["data"]))

    # y can be stored as data array OR scalar attribute "value"
    y_node = pos.get("y", {})
    if isinstance(y_node, dict) and "data" in y_node:
        y = list(map(float, y_node["data"]))
    else:
        y0 = float(y_node.get("attributes", {}).get("value", {}).get("value", 0.0))
        y = [y0] * len(x)

    ids = list(map(int, species_block["id"]["data"]))

    if not (len(x) == len(y) == len(z) == len(ids)):
        raise ValueError(f"Length mismatch: len(x)={len(x)}, len(y)={len(y)}, len(z)={len(z)}, len(id)={len(ids)}")

    return x, y, z, ids


def extract_uxuyuz(species_block: Dict[str, Any], n: int) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract momentum components from:
      momentum/x/data, momentum/y/data OR constant value, momentum/z/data
    Returns ux, uy, uz arrays of length n.
    """
    mom = species_block["momentum"]

    ux = list(map(float, mom["x"]["data"]))
    uz = list(map(float, mom["z"]["data"]))

    uy_node = mom.get("y", {})
    if isinstance(uy_node, dict) and "data" in uy_node:
        uy = list(map(float, uy_node["data"]))
    else:
        uy0 = float(uy_node.get("attributes", {}).get("value", {}).get("value", 0.0))
        uy = [uy0] * len(ux)

    if not (len(ux) == len(uy) == len(uz) == n):
        raise ValueError(
            f"Length mismatch in momentum: len(ux)={len(ux)}, len(uy)={len(uy)}, len(uz)={len(uz)}, expected={n}"
        )

    return ux, uy, uz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Folder with openpmd_*.json")
    ap.add_argument("--species", default="electrons", help="Species name inside particles (default electrons)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int, help="Track particle by ID (from .../id/data)")
    g.add_argument("--index", type=int, help="Track particle by index (0-based)")
    ap.add_argument("--dt", type=float, default=None, help="Time step. If set: time = step*dt else time=step.")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--debug", action="store_true", help="Print where particles/species were found")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "openpmd_*.json")), key=step_from_filename)
    if not files:
        raise SystemExit("No openpmd_*.json files found")

    rows = []
    tracked_id: Optional[int] = None

    for path in files:
        step = step_from_filename(path)

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        try:
            sp = get_species_block(d, args.species)
        except KeyError as e:
            if args.debug:
                print(f"[skip] {os.path.basename(path)}: {e}")
            continue

        try:
            x, y, z, ids = extract_xyz_ids(sp)
        except Exception as e:
            raise SystemExit(f"Failed to extract xyz/ids from {path}: {e}")
        try:
            ux, uy, uz = extract_uxuyuz(sp, len(ids))
        except Exception as e:
            raise SystemExit(f"Failed to extract momentum from {path}: {e}")

        if args.id is not None:
            try:
                idx = ids.index(args.id)
            except ValueError:
                continue
            tracked_id = args.id
        else:
            idx = args.index
            if idx < 0 or idx >= len(ids):
                continue
            tracked_id = ids[idx]

        t = float(step) if args.dt is None else float(step) * float(args.dt)

        rows.append({
            "time": t,
            "x": x[idx],
            "y": y[idx],
            "z": z[idx],
            "ux": ux[idx],
            "uy": uy[idx],
            "uz": uz[idx],
            "step": step,
            "id": tracked_id,
            "file": os.path.basename(path),
        })

    if not rows:
        raise SystemExit(
            "No rows extracted.\n"
            "Possible causes:\n"
            " - species name differs (use --debug)\n"
            " - particle with given --id not present\n"
            " - JSON structure differs (send first ~60 lines of a file)\n"
        )

    if args.out is None:
        tag = f"id{args.id}" if args.id is not None else f"idx{args.index}_id{tracked_id}"
        args.out = os.path.join(args.dir, f"traj_{args.species}_{tag}.csv")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time", "x", "y", "z", "ux", "uy", "uz", "step", "id", "file"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {args.out} (rows={len(rows)})")


if __name__ == "__main__":
    main()
