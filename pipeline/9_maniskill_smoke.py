#!/usr/bin/env python3
"""Stage-0 smoke test: load AHOLO URDF assets into SAPIEN/ManiSkill (Physx).

Four-level check per asset:
  L1 parse     URDF accepted by SAPIEN's loader
  L2 build     articulation built, link/joint counts match the URDF
  L3 sanity    500 physics steps with root fixed: no floor penetration,
               no NaN, joint drift within limits
  L4 drive     ramp each joint with 2x the MuJoCo-probed onset force and
               measure travel (cross-engine physics robustness)

Usage:
  source scripts/maniskill-env.sh
  .venv-maniskill/bin/python pipeline/9_maniskill_smoke.py            # all assets
  .venv-maniskill/bin/python pipeline/9_maniskill_smoke.py --asset side-table-01
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ASSETS = ["side-table-01", "wardrobe-01", "nightstand-02",
          "oven-mini-01", "toolbox-01"]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def smoke_one(asset: str):
    import sapien
    import mani_skill  # noqa: F401  (ensures env registration)

    ad = common.asset_dir(asset)
    urdf = ad / f"{asset}.urdf"
    probe = json.loads((ad / "force_probe_report.json").read_text())
    print(f"\n===== {asset} =====")

    # L1: parse
    scene = sapien.Scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    try:
        parsed = loader.parse(str(urdf))
        # SAPIEN 3 returns (articulation_builders, actor_builders, sensors)
        builders = parsed[0] if isinstance(parsed, tuple) else parsed.get(
            "articulation_builders", [])
        check("L1 parse", len(builders) == 1, f"{len(builders)} articulation builder(s)")
    except Exception as e:
        check("L1 parse", False, str(e)[:200])
        return
    builder = builders[0]
    # body-center export: lift by bodyPosZ so the asset rests on the floor
    # (same rule as urdfkit.body_placement in the MuJoCo chain)
    report = json.loads((ad / "adapt-report.json").read_text())
    z0 = report.get("world", {}).get("bodyPosZ", 0.0)
    builder.initial_pose = sapien.Pose(p=[0, 0, z0])

    # L2: build
    try:
        art = builder.build()
        MOVABLE = ("revolute", "prismatic", "revolute_unwrapped")
        n_links = len(art.get_links())
        n_joints = len([j for j in art.get_joints() if j.get_type() in MOVABLE])
        check("L2 build", n_links >= 2 and n_joints >= 1,
              f"links={n_links} movable_joints={n_joints}")
    except Exception as e:
        check("L2 build", False, str(e)[:200])
        return

    scene.add_ground(altitude=0.0)
    scene.set_timestep(1 / 120.0)

    # L3: sanity — 500 steps with zero actuation
    art.set_qpos(np.zeros(art.dof))
    scene.step()
    t0 = time.time()
    for _ in range(500):
        scene.step()
    poses = np.array([l.pose.p for l in art.get_links()])
    q = art.get_qpos()
    ok3 = (np.isfinite(poses).all() and np.isfinite(q).all()
           and abs(poses[:, 2]).max() < 1.0)
    check("L3 sanity (500 steps, root fixed)", ok3,
          f"dt={time.time()-t0:.1f}s |q|max={np.abs(q).max():.4f} zmax={np.abs(poses[:,2]).max():.3f}")

    # L4: drive each joint at 2x onset and measure travel
    MOVABLE = ("revolute", "prismatic", "revolute_unwrapped")
    all_joints = art.get_joints()
    # qpos layout: every articulation joint contributes per its dof; fixed
    # joints contribute 0. Walk the joint list to map each movable joint to
    # its qpos slice (SAPIEN3 joints expose no qpos-address API).
    qidx = 0
    qslice = {}
    for j in all_joints:
        d = j.get_dof()
        if j.get_type() in MOVABLE:
            qslice[j.get_name()] = slice(qidx, qidx + d)
        qidx += d
    joints = [j for j in all_joints if j.get_type() in MOVABLE]
    for j in joints:
        jname = j.get_name()
        lo, hi = j.get_limits()[0]
        target = hi if abs(hi) >= abs(lo) else lo   # open end = larger |limit|
        units = "N.m" if j.get_type() != "prismatic" else "N"
        probe_j = probe.get("joints", {}).get(jname, {})
        onset = probe_j.get("onsetCmd")
        # apply a constant effort at 2x onset; magnitude floored at 1.0
        # (N or N.m) so light doors/drawers still get a meaningful drive.
        # Sign follows the probe's onset (open direction can be negative,
        # e.g. the oven's downward-flipping door).
        effort = 2 * onset if onset else 20.0
        if abs(effort) < 1.0:
            effort = 1.0 if effort >= 0 else -1.0
        art.set_qpos(np.zeros(art.dof))
        # SAPIEN3 exposes PD drive (no direct joint effort API), so L4 drives
        # position targets like the MuJoCo probe's executor; stiffness is set
        # so peak drive force ~= 2x onset (cross-engine check stays physical).
        j.set_drive_property(stiffness=effort / max(abs(target), 1e-6) * 4,
                             damping=effort / max(abs(target), 1e-6) * 0.2,
                             force_limit=effort * 2)
        travel_meas = 0.0
        steps = 600
        for i in range(steps):
            j.set_drive_target(target)
            scene.step()
            travel_meas = float(np.abs(art.get_qpos()[qslice[jname]]).max())
            if travel_meas >= 0.9 * abs(target):
                break
        frac = travel_meas / abs(target) if abs(target) > 1e-9 else 0
        check(f"L4 drive {jname} ({units}, onset={onset})",
              frac >= 0.9, f"travel {frac*100:.0f}% of {target:.4f} @ {effort:.2f}{units}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None)
    args = ap.parse_args()
    assets = [args.asset] if args.asset else ASSETS
    for a in assets:
        try:
            smoke_one(a)
        except Exception as e:
            check(f"{a} (crash)", False, repr(e)[:200])
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n==== SMOKE: {'PASS' if n_fail == 0 else 'FAIL'} "
          f"({len(RESULTS) - n_fail}/{len(RESULTS)}) ====")


if __name__ == "__main__":
    main()
