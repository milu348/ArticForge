#!/usr/bin/env python3
"""Step 7: physics-scene choreography video: force-driven open/hold/close on
the contact-enabled model built by step 6 (no servo cheating — friction &
limits are real).

The per-joint force profile is auto-derived from force_probe_report.json:
peak = force_scale x measured onset (config "force_scale", default 1.5),
applied along each joint's "open" direction (urdfkit.open_target). If a
joint's onset was not measurable, the peak falls back to half its ctrlrange.

Usage:
  .venv/bin/python pipeline/7_physics_video.py --asset nightstand-02
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import urdfkit  # noqa: E402

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

# choreography phases (s): settle | open ramp | hold | close ramp | settle
T_SETTLE, T_OPEN, T_HOLD, T_CLOSE, T_END = 0.5, 1.3, 2.2, 3.2, 4.0


def smoothstep(x):
    return x * x * (3 - 2 * x)


def make_profile(joints, probe, cfg):
    """Per joint: (sign, peak). peak = force_scale x onset, fallback ctrlrange/2."""
    scale = float(cfg.get("force_scale", 1.5))
    prof = []
    for j in joints:
        _, sign = urdfkit.open_target(j)
        r = probe["joints"].get(j["name"], {})
        onset = r.get("onsetCmd")
        peak = (abs(onset) * scale if onset else 0.0)
        prof.append((sign, peak))
    return prof


def force_targets(t, prof):
    out = []
    for sign, peak in prof:
        if t < T_SETTLE:
            out.append(0.0)
        elif t < T_OPEN:
            out.append(sign * peak * smoothstep((t - T_SETTLE) / (T_OPEN - T_SETTLE)))
        elif t < T_HOLD:
            out.append(sign * peak * 0.3)
        elif t < T_CLOSE:
            out.append(-sign * peak * 1.1 * smoothstep((t - T_HOLD) / (T_CLOSE - T_HOLD)))
        else:
            out.append(-sign * peak * 0.5)
    return out


def main():
    ap = common.add_asset_arg(argparse.ArgumentParser(
        description="Step 7: force-driven physics choreography video"))
    args = ap.parse_args()

    cfg = common.config(args.asset)
    asset = common.asset_dir(args.asset)
    model = mujoco.MjModel.from_xml_path(str(asset / "scene_physics.mjcf"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)

    model_urdf = urdfkit.parse_urdf(asset / f"{args.asset}.urdf")
    joints = urdfkit.ordered_joints(model_urdf)
    probe = json.loads((asset / "force_probe_report.json").read_text())
    prof = make_profile(joints, probe, cfg)

    cam_d = urdfkit.auto_camera(asset, model_urdf, cfg)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = cam_d["lookat"]
    cam.distance = cam_d["distance"]
    cam.azimuth = cam_d["azimuth"]
    cam.elevation = cam_d["elevation"]

    frames = []
    total = int(T_END / model.opt.timestep)
    for step in range(total):
        t = step * model.opt.timestep
        for a, v in enumerate(force_targets(t, prof)):
            data.ctrl[a] = v
        mujoco.mj_step(model, data)
        if step % 10 == 0:  # 25 fps
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    out = asset / f"{args.asset}_physics.mp4"
    imageio.mimwrite(out, frames, fps=25, quality=8)
    print("video:", out, f"({len(frames)} frames)")
    for i, j in enumerate(joints):
        r = probe["joints"].get(j["name"], {})
        print(f"final {j['name']}: qpos={data.qpos[i]:+.4f} "
              f"(target {r.get('target', 0):+.4f})")
    renderer.close()


if __name__ == "__main__":
    main()
