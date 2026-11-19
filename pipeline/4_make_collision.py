#!/usr/bin/env python3
"""Step 4: collision geometry via convex decomposition (CoACD).

The part list is auto-derived from the adapted URDF (step 3): the root link
(the one with child joints) gets CoACD convex decomposition; leaf parts get
a single convex hull. Override per link with config "decompose": {"name": true}.

Reads assets/<asset>/meshes/*.stl, writes convex hulls + collision-report.json
into assets/<asset>/collision/.

Usage:
  .venv/bin/python pipeline/4_make_collision.py --asset nightstand-02
"""
import argparse
import json
import sys
import time
from pathlib import Path

import coacd
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import urdfkit  # noqa: E402


def prep(mesh, target_faces=30000):
    m = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    m.merge_vertices()
    if len(m.faces) > target_faces:
        m = m.simplify_quadric_decimation(face_count=target_faces)
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
    return m


def main():
    ap = common.add_asset_arg(argparse.ArgumentParser(
        description="Step 4: CoACD convex-decomposition collision geometry"))
    args = ap.parse_args()

    cfg = common.config(args.asset)
    asset = common.asset_dir(args.asset)
    meshes = asset / "meshes"
    coll = asset / "collision"
    coll.mkdir(parents=True, exist_ok=True)

    model = urdfkit.parse_urdf(asset / f"{args.asset}.urdf")
    root = urdfkit.root_link(model)
    decompose_cfg = cfg.get("decompose", {})
    links = [l["name"] for l in
             json.loads((asset / "adapt-report.json").read_text())["links"]]

    report = {}
    for name in links:
        decompose = decompose_cfg.get(name, name == root)
        t0 = time.time()
        m = prep(trimesh.load(meshes / f"{name}.stl"))
        if decompose:
            coacd.set_log_level("error")
            parts = coacd.run_coacd(
                coacd.Mesh(m.vertices, m.faces),
                threshold=0.08, max_convex_hull=12, resolution=1000,
            )
            hulls = []
            for i, (v, f) in enumerate(parts):
                h = trimesh.Trimesh(np.asarray(v), np.asarray(f), process=True)
                path = coll / f"{name}_hull{i:02d}.stl"
                h.export(path)
                hulls.append({"file": f"collision/{name}_hull{i:02d}.stl",
                              "tris": len(h.faces)})
        else:
            h = m.convex_hull
            path = coll / f"{name}_hull00.stl"
            h.export(path)
            hulls = [{"file": f"collision/{name}_hull00.stl", "tris": len(h.faces)}]
        report[name] = {
            "sourceTris": len(trimesh.load(meshes / f"{name}.stl").faces),
            "hulls": hulls,
            "seconds": round(time.time() - t0, 1),
        }
        print(name, "->", len(hulls), "hulls,", report[name]["seconds"], "s")
    with open(coll / "collision-report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
