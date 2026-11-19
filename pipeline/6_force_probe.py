#!/usr/bin/env python3
"""Step 6: contact-grade scene + force probing for EVERY joint.

Builds assets/<asset>/scene_physics.mjcf via urdfkit (collision hulls +
visual meshes, motor actuators, one joint per URDF joint), then for each
joint: settles the model, ramps force/torque toward the "open" limit
endpoint (the endpoint with the larger |value| — resolves axis-sign
dependence, pitfall #4), and measures motion onset (static friction) and
late-phase plateau. Saves per-joint CSV + force_curves.png + a
force_probe_report.json whose "pass" flags double as the automatic QC gate
(asset loads, every joint reaches >=90% of its travel).

Usage:
  .venv/bin/python pipeline/6_force_probe.py --asset nightstand-02
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import urdfkit  # noqa: E402

RAMP_T, HOLD_T = 4.0, 2.0


def ramp_probe(mujoco, model, data, act_idx, qpos_idx, sign, out_csv):
    """Ramp cmd from 0 toward sign*ctrlrange_max, record (t, cmd, qpos, qvel)."""
    rows = []
    n_ramp = int(RAMP_T / model.opt.timestep)
    n_hold = int(HOLD_T / model.opt.timestep)
    max_cmd = model.actuator_ctrlrange[act_idx][1]
    for step in range(n_ramp + n_hold):
        frac = min(1.0, step / n_ramp)
        cmd = sign * frac * max_cmd
        data.ctrl[act_idx] = cmd
        mujoco.mj_step(model, data)
        if step % 10 == 0:
            rows.append((step * model.opt.timestep, cmd,
                         data.qpos[qpos_idx], data.qvel[qpos_idx]))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "cmd", "qpos", "qvel"])
        w.writerows(rows)
    return rows


def analyze(rows, target):
    """Motion-onset cmd (static friction) and late-phase plateau estimate."""
    onset = next((r[1] for r in rows if abs(r[3]) > 0.002), None)
    tail = [r for r in rows if r[0] > rows[-1][0] * 0.9]
    plateau = (float(np.mean([r[1] for r in tail]))
               if tail and np.mean([abs(r[3]) for r in tail]) > 0.005 else None)
    travel = rows[-1][2]
    frac = abs(travel / target) if abs(target) > 1e-9 else 0.0
    return onset, plateau, travel, frac


def main():
    ap = common.add_asset_arg(argparse.ArgumentParser(
        description="Step 6: build physics scene + per-joint force probing"))
    args = ap.parse_args()

    import mujoco
    cfg = common.config(args.asset)
    asset = common.asset_dir(args.asset)

    model_urdf = urdfkit.parse_urdf(asset / f"{args.asset}.urdf")
    joints = urdfkit.ordered_joints(model_urdf)

    scene = asset / "scene_physics.mjcf"
    scene.write_text(urdfkit.build_mjcf(asset, model_urdf, cfg, physics=True))
    model = mujoco.MjModel.from_xml_path(str(scene))

    results, all_pass = {}, True
    for i, j in enumerate(joints):
        if j["type"] not in ("prismatic", "revolute"):
            continue
        unit = "N" if j["type"] == "prismatic" else "N.m"
        target, sign = urdfkit.open_target(j)
        data = mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model, data)  # settle contacts
        rows = ramp_probe(mujoco, model, data, i, i, sign,
                          asset / f"{j['name']}_force_curve.csv")
        onset, plateau, travel, frac = analyze(rows, target)
        ok = bool(frac >= 0.90)
        all_pass = bool(all_pass and ok)
        results[j["name"]] = {
            "type": j["type"], "unit": unit,
            "onsetCmd": None if onset is None else round(float(onset), 4),
            "plateauCmd": None if plateau is None else round(float(plateau), 4),
            "target": round(float(target), 5), "travel": round(float(travel), 5),
            "travelFrac": round(float(frac), 3), "sign": sign, "pass": ok,
        }
        print(f"{j['name']}: onset {onset} {unit}, travel {travel:.4f} / {target:.4f} "
              f"({frac*100:.0f}%) {'PASS' if ok else 'FAIL'}")

    # curves: one subplot per joint
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(5 * max(n, 1), 4), squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        rows = np.loadtxt(asset / f"{name}_force_curve.csv", delimiter=",",
                          skiprows=1, ndmin=2)
        ax.plot(rows[:, 0], rows[:, 1], label="cmd")
        ax.plot(rows[:, 0], rows[:, 2], label="qpos")
        ax.set_xlabel("t (s)")
        ax.set_title(f"{name} ({r['unit']}) {'PASS' if r['pass'] else 'FAIL'}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(asset / "force_curves.png", dpi=110)

    report = {"pass": all_pass, "joints": results}
    (asset / "force_probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print("QA:", "PASS" if all_pass else "FAIL", "-> force_probe_report.json")


if __name__ == "__main__":
    main()
