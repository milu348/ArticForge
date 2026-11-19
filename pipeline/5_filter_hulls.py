#!/usr/bin/env python3
"""Step 5: filter body collision hulls that block the moving parts' sweep
volumes (equivalent to cutting the drawer slot and door opening into the
collision model). This is the generic fix for the rest-pose-micro-open trap
(demo.md pitfall #2).

Everything is auto-derived by urdfkit from the adapted URDF + hull meshes:
per joint, the child part's swept AABB over its full travel. Body hulls
overlapping any swept volume are dropped; the full hull set stays on disk.

Reads assets/<asset>/collision/{collision-report.json, *_hull*.stl},
writes assets/<asset>/collision/collision-filtered.json.

Usage:
  .venv/bin/python pipeline/5_filter_hulls.py --asset nightstand-02
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import urdfkit  # noqa: E402


def overlaps(lo1, hi1, lo2, hi2, tol):
    return np.all(lo1 <= hi2 + tol) and np.all(lo2 <= hi1 + tol)


def main():
    ap = common.add_asset_arg(argparse.ArgumentParser(
        description="Step 5: sweep-volume filtering of body hulls"))
    args = ap.parse_args()

    cfg = common.config(args.asset)
    asset = common.asset_dir(args.asset)
    coll = asset / "collision"
    rep = json.loads((coll / "collision-report.json").read_text())

    model = urdfkit.parse_urdf(asset / f"{args.asset}.urdf")
    root = urdfkit.root_link(model)
    tol = float(cfg.get("sweep_tol", 0.004))
    sweeps = urdfkit.swept_aabbs(asset, model, tol=tol)

    body_pos, _ = urdfkit.body_placement(asset, model)
    kept, dropped = [], []
    for h in rep[root]["hulls"]:
        m = trimesh.load(coll / Path(h["file"]).name)
        lo, hi = m.bounds  # hull coords are root-link-local (Z-up)
        lo_w, hi_w = lo + body_pos, hi + body_pos
        blocked = any(overlaps(lo_w, hi_w, lo_s, hi_s, tol)
                      for lo_s, hi_s in sweeps.values())
        if blocked:
            dropped.append(h)
        else:
            kept.append(h)
    out = {root: {"hulls": kept}, "dropped": dropped,
           "sweptVolumes": {k: [lo.tolist(), hi.tolist()]
                            for k, (lo, hi) in sweeps.items()}}
    with open(coll / "collision-filtered.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"kept {len(kept)} body hulls, dropped {len(dropped)}:")
    for d in dropped:
        print("  -", Path(d["file"]).name)


if __name__ == "__main__":
    main()
