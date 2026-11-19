#!/usr/bin/env python3
"""Step 3: adapt the Lux3D articulation URDF package for MuJoCo / ROS.

- Convert per-link GLB meshes to binary STL (decimated to <=180k faces,
  MuJoCo's 200k-per-mesh STL limit)
- Rotate the model from glTF Y-up to URDF/MuJoCo Z-up (bake RotX(+90) into
  vertices, joint origins and axes)
- Add collision geometries and physically plausible inertials
- Auto-correct the joint-origin frame: JOINT_Y_SHIFT = adapted root-link STL
  min_z (body-centered exports -> -half height; floor-origin exports -> 0).
  Override with config "joint_y_shift" if a delivery ever breaks the rule.

All per-asset values are config-overridable (configs/<asset>.json): "mass"
for absolute link masses; otherwise mass = volume x density (root link x3).

Reads the extracted articulation package (robot.urdf + per-link GLBs) from
output/downloads/<asset>/urdf/, writes into assets/<asset>/:
  <asset>.urdf, meshes/*.stl, adapt-report.json

Usage:
  .venv/bin/python pipeline/3_adapt_urdf.py --asset nightstand-02 \
      [--src output/downloads/nightstand-02/urdf] [--out assets/nightstand-02]
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MAX_FACES = 180_000  # MuJoCo STL limit is 200k — leave headroom

# RotX(+90): (x, y, z) -> (x, -z, y)  maps glTF +Y-up to +Z-up
R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def rot_vec(v):
    return R @ np.asarray(v, dtype=np.float64)


def load_link_mesh(path, max_faces=MAX_FACES):
    sc = trimesh.load(path, force="scene")
    parts = [g for g in sc.geometry.values() if isinstance(g, trimesh.Trimesh)]
    m = trimesh.util.concatenate(parts)
    # Merge duplicated vertices FIRST: glTF splits vertices by normal/UV, so
    # quadric decimation on unmerged copies tears the mesh apart (boundary
    # edges 0.3% -> 23.8% = visible gaps). Merge order matters.
    m.merge_vertices(merge_tex=True, merge_norm=True)
    if len(m.faces) > max_faces:
        m = m.simplify_quadric_decimation(face_count=max_faces)
    v = m.vertices @ R.T
    return trimesh.Trimesh(vertices=v, faces=m.faces, process=True)


def main():
    ap = common.add_asset_arg(argparse.ArgumentParser(
        description="Step 3: adapt LUX3D URDF package for MuJoCo/ROS"))
    ap.add_argument("--src", default=None,
                    help="dir with robot.urdf + per-link GLBs (default: output/downloads/<asset>/urdf)")
    ap.add_argument("--out", default=None,
                    help="asset output dir (default: assets/<asset>)")
    args = ap.parse_args()

    cfg = common.config(args.asset)
    src = Path(args.src) if args.src else common.downloads_dir(args.asset) / "urdf"
    out = Path(args.out) if args.out else common.asset_dir(args.asset)
    os.makedirs(out / "meshes", exist_ok=True)

    tree = ET.parse(src / "robot.urdf")
    uroot = tree.getroot()

    # root link = the link that is nobody's child
    children = {j.find("child").get("link") for j in uroot.findall("joint")}
    root_name = next((l.get("name") for l in uroot.findall("link")
                      if l.get("name") not in children),
                     uroot.find("link").get("name"))

    density = float(cfg.get("density", 700.0))
    report = {"links": [], "joints": [], "world": {}}
    link_mass_props, mesh_minz = {}, {}

    for link in uroot.findall("link"):
        name = link.get("name")
        vis = link.find("visual/geometry/mesh").get("filename")
        m = load_link_mesh(src / vis)
        stl_rel = f"meshes/{name}.stl"
        m.export(out / stl_rel)
        mesh_minz[name] = float(m.bounds[0][2])

        p = m.mass_properties
        vol = abs(p["volume"]) or 1e-6
        if name in cfg.get("mass", {}):
            mass = float(cfg["mass"][name])
        else:
            mass = float(np.clip(vol * density * (3.0 if name == root_name else 1.0),
                                 0.5, 120.0))
        rho = mass / vol
        inertia = np.abs(np.diag(p["inertia"])) * rho
        inertia = np.maximum(inertia, 1e-5)
        com = p["center_mass"]
        link_mass_props[name] = dict(mass=mass, com=com, inertia=inertia)
        report["links"].append(dict(
            name=name, tris=len(m.faces), mass=round(mass, 3),
            com=[round(x, 4) for x in com],
            inertia=[round(x, 6) for x in inertia], stl=stl_rel))

    # --- joint-origin frame correction (pitfall #1), auto-derived -----------
    shift = (float(cfg["joint_y_shift"]) if "joint_y_shift" in cfg
             else mesh_minz[root_name])
    report["world"] = {
        "rootLink": root_name,
        "jointYShift": round(shift, 6),
        "jointYShiftAuto": "joint_y_shift" not in cfg,
        "bodyPosZ": round(-mesh_minz[root_name], 6),
    }

    # rewrite URDF in Z-up with STL meshes, collisions, inertials
    new = ET.Element("robot", {"name": args.asset})
    for link in uroot.findall("link"):
        name = link.get("name")
        mp = link_mass_props[name]
        le = ET.SubElement(new, "link", {"name": name})
        i, c = mp["inertia"], mp["com"]
        ine = ET.SubElement(le, "inertial")
        ET.SubElement(ine, "origin", {"xyz": f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"})
        ET.SubElement(ine, "mass", {"value": f"{mp['mass']:.4f}"})
        ET.SubElement(ine, "inertia",
                      {"ixx": f"{i[0]:.6f}", "iyy": f"{i[1]:.6f}", "izz": f"{i[2]:.6f}",
                       "ixy": "0", "ixz": "0", "iyz": "0"})
        for sec in ("visual", "collision"):
            g = ET.SubElement(ET.SubElement(le, sec), "geometry")
            ET.SubElement(g, "mesh", {"filename": f"meshes/{name}.stl"})
        # NOTE: no material-only <visual> block — SAPIEN's urchin parser
        # crashes on a <visual> without <geometry> (required subelement).
    for joint in uroot.findall("joint"):
        name, jtype = joint.get("name"), joint.get("type")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        org = [float(x) for x in joint.find("origin").get("xyz").split()]
        org[1] += shift  # floor-frame -> body-center frame (Y-up, pre-rotation)
        ax = [float(x) for x in joint.find("axis").get("xyz").split()]
        lim = joint.find("limit")
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        effort = max(50.0, float(lim.get("effort", 10)) * 5)
        org_z, ax_z = rot_vec(org), rot_vec(ax)
        ax_z /= (np.linalg.norm(ax_z) or 1.0)
        je = ET.SubElement(new, "joint", {"name": name, "type": jtype})
        ET.SubElement(je, "parent", {"link": parent})
        ET.SubElement(je, "child", {"link": child})
        ET.SubElement(je, "origin",
                      {"xyz": f"{org_z[0]:.6f} {org_z[1]:.6f} {org_z[2]:.6f}", "rpy": "0 0 0"})
        ET.SubElement(je, "axis",
                      {"xyz": f"{ax_z[0]:.6f} {ax_z[1]:.6f} {ax_z[2]:.6f}"})
        ET.SubElement(je, "limit",
                      {"lower": f"{lo:.6f}", "upper": f"{hi:.6f}",
                       "effort": f"{effort:.1f}", "velocity": "1.0"})
        report["joints"].append(dict(name=name, type=jtype, parent=parent, child=child,
                                     origin=[round(x, 5) for x in org_z],
                                     axis=[round(x, 5) for x in ax_z],
                                     limits=[lo, hi]))

    ET.indent(new, space="  ")
    ET.ElementTree(new).write(out / f"{args.asset}.urdf", encoding="unicode")
    with open(out / "adapt-report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["world"], ensure_ascii=False, indent=2))
    print(f"links: {len(report['links'])}, joints: {len(report['joints'])} -> {out}/{args.asset}.urdf")


if __name__ == "__main__":
    main()
