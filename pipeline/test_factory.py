#!/usr/bin/env python3
"""Offline end-to-end validation of the local factory chain (steps 3-8) with
ZERO credits: builds the worst-case fixture (all four demo.md pitfalls), runs
the pipeline on it, and asserts every auto-derivation landed correctly.

Run:  .venv/bin/python pipeline/test_factory.py
"""
import json
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ASSET = "fixture-cabinet"
PY = sys.executable
PIPE = common.root() / "pipeline"
ENV = {**os.environ, "MUJOCO_GL": "egl", "AHOLO_ROOT": str(common.root())}

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def run(step, script, extra=None):
    t0 = time.time()
    cmd = [PY, str(PIPE / script), "--asset", ASSET] + (extra or [])
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=1200)
    dt = time.time() - t0
    print(f"step {step} ({script}) {dt:.1f}s rc={r.returncode}")
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-3000:])
    return r, dt


def main():
    asset = common.asset_dir(ASSET)

    print("== fixture ==")
    r, _ = run("fx", "fixtures/make_fixture.py")
    assert r.returncode == 0, "fixture build failed"

    print("== step 3: adapt ==")
    r, _ = run("3", "3_adapt_urdf.py")
    assert r.returncode == 0
    rep = json.loads((asset / "adapt-report.json").read_text())
    w = rep["world"]
    check("3: auto JOINT_Y_SHIFT = -0.275 (pitfall #1)",
          abs(w["jointYShift"] + 0.275) < 1e-3 and w["jointYShiftAuto"],
          f"shift={w['jointYShift']}")
    check("3: bodyPosZ = 0.275 (sits on floor)", abs(w["bodyPosZ"] - 0.275) < 1e-3)
    body_tris = next(l["tris"] for l in rep["links"] if l["name"] == "body")
    check("3: body decimated below 180k (pitfall #3)", body_tris <= 180_000,
          f"{body_tris} tris")
    u = ET.parse(asset / f"{ASSET}.urdf").getroot()
    org = {j.get("name"): [float(x) for x in j.find("origin").get("xyz").split()]
           for j in u.findall("joint")}
    ax = {j.get("name"): [float(x) for x in j.find("axis").get("xyz").split()]
          for j in u.findall("joint")}
    d_o = org.get("drawer_slide", [9, 9, 9])
    check("3: drawer origin corrected (0, -0.05, 0.075)",
          max(abs(a - b) for a, b in zip(d_o, [0, -0.05, 0.075])) < 2e-3, str(d_o))
    check("3: drawer axis -> (0,-1,0) Z-up",
          max(abs(a - b) for a, b in zip(ax.get("drawer_slide", [9, 9, 9]), [0, -1, 0])) < 1e-3)
    d_h = org.get("door_hinge", [9, 9, 9])
    check("3: door hinge origin (-0.25, -0.2, -0.135)",
          max(abs(a - b) for a, b in zip(d_h, [-0.25, -0.2, -0.135])) < 2e-3, str(d_h))
    inr = u.find("link/inertial")
    check("3: inertial single element (origin/mass/inertia children)",
          inr is not None and inr.find("origin") is not None
          and inr.find("mass") is not None and inr.find("inertia") is not None)

    print("== step 4: collision ==")
    r, _ = run("4", "4_make_collision.py")
    assert r.returncode == 0
    coll = json.loads((asset / "collision" / "collision-report.json").read_text())
    check("4: parts auto-derived from URDF (3 links)",
          set(coll) == {"body", "drawer_top", "door_bottom"}, str(sorted(coll)))
    check("4: body decomposed, leaves single-hull",
          len(coll["body"]["hulls"]) >= 2 and len(coll["drawer_top"]["hulls"]) == 1
          and len(coll["door_bottom"]["hulls"]) == 1,
          f"body={len(coll['body']['hulls'])} hulls")

    print("== step 5: sweep filter ==")
    r, _ = run("5", "5_filter_hulls.py")
    assert r.returncode == 0
    filt = json.loads((asset / "collision" / "collision-filtered.json").read_text())
    n_kept, n_drop = len(filt["body"]["hulls"]), len(filt["dropped"])
    check("5: filter cut blocking hulls (pitfall #2)", n_drop >= 1,
          f"kept={n_kept} dropped={n_drop}")
    check("5: some body hulls kept", n_kept >= 1, f"kept={n_kept}")
    check("5: swept volumes recorded for every joint",
          set(filt["sweptVolumes"]) == {"drawer_slide", "door_hinge"})
    # the closed front plate region must be inside a swept volume
    front_center = [0.0, -0.195, 0.275]
    in_sweep = any(all(lo[i] - 1e-6 <= front_center[i] <= hi[i] + 1e-6 for i in range(3))
                   for lo, hi in [tuple(map(list, v)) for v in filt["sweptVolumes"].values()])
    check("5: front plate region covered by sweep math", in_sweep)

    print("== step 6: force probe ==")
    r, _ = run("6", "6_force_probe.py")
    assert r.returncode == 0
    probe = json.loads((asset / "force_probe_report.json").read_text())
    check("6: QA pass — every joint reaches >=90% travel (pitfall #4 auto-sign)",
          probe["pass"] and all(j["travelFrac"] >= 0.9 for j in probe["joints"].values()),
          str({k: (v["onsetCmd"], v["travelFrac"]) for k, v in probe["joints"].items()}))

    print("== step 7: physics video ==")
    r, _ = run("7", "7_physics_video.py")
    assert r.returncode == 0
    pv = asset / f"{ASSET}_physics.mp4"
    check("7: physics video rendered", pv.exists() and pv.stat().st_size > 100_000,
          f"{pv.stat().st_size//1024 if pv.exists() else 0} KB")

    print("== step 8: demo video ==")
    r, _ = run("8", "8_render_video.py")
    assert r.returncode == 0
    dv = asset / f"{ASSET}_demo.mp4"
    snaps = sorted(p.name for p in asset.glob("snapshot_*.png"))
    check("8: demo video + 3 snapshots", dv.exists() and dv.stat().st_size > 100_000
          and len(snaps) == 3, f"{len(snaps)} snapshots")

    total = sum(1 for _, ok, _ in CHECKS)
    bad = [(n, d) for n, ok, d in CHECKS if not ok]
    print(f"\n==== FACTORY VALIDATION: {'PASS' if not bad else 'FAIL'} "
          f"({total - len(bad)}/{total}) ====")
    for n, d in bad:
        print(f"  FAILED: {n} {d}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
