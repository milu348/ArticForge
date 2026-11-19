#!/usr/bin/env python3
"""Step 3b: rewrite <collision> entries in the adapted URDF to CoACD convex
hulls (multi-<collision> per link).

Why: the adapted URDF points collision at the *visual* STL. MuJoCo never
cares (urdfkit.build_mjcf swaps in the hulls when building scenes), but any
URDF-native consumer (SAPIEN/ManiSkill, ROS Gazebo) cooks the raw triangle
mesh into a convex approximation, fusing moving parts to the body. Point
collision at the hull STLs instead.

Sources:
  assets/<asset>/collision/collision-report.json      (all hulls per link)
  assets/<asset>/collision/collision-filtered.json    (root-link keep list)

Usage: .venv/bin/python pipeline/3b_urdf_collision_hulls.py --asset <name>
Idempotent: re-running replaces existing mesh collisions from scratch.
"""
import argparse
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    ap = common.add_asset_arg(argparse.ArgumentParser(
        description="Point URDF collisions at CoACD convex hulls"))
    args = ap.parse_args()
    ad = common.asset_dir(args.asset)
    urdf_path = ad / f"{args.asset}.urdf"

    report = json.loads((ad / "collision" / "collision-report.json").read_text())
    filtered = json.loads((ad / "collision" / "collision-filtered.json").read_text())
    # root-link hulls: filtered (post sweep-filter); child links: all hulls
    hulls = {}
    root_key = next(k for k in filtered
                    if isinstance(filtered[k], dict) and "hulls" in filtered[k])
    hulls[root_key] = [h["file"] for h in filtered[root_key]["hulls"]]
    for link, info in report.items():
        if link != root_key:
            hulls[link] = [h["file"] for h in info["hulls"]]

    tree = ET.parse(urdf_path)
    uroot = tree.getroot()
    replaced = {}
    for link in uroot.findall("link"):
        lname = link.get("name")
        files = hulls.get(lname)
        if not files:
            continue
        for col in list(link.findall("collision")):
            link.remove(col)
        for f in files:
            col = ET.SubElement(link, "collision")
            g = ET.SubElement(col, "geometry")
            ET.SubElement(g, "mesh", {"filename": f})
        replaced[lname] = len(files)
    tree.write(urdf_path, encoding="utf-8", xml_declaration=False)
    print(f"{args.asset}: collision -> convex hulls {replaced}")


if __name__ == "__main__":
    main()
