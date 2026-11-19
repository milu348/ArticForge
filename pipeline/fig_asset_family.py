#!/usr/bin/env python3
"""Render the 5-asset family figure for the competition report.

For each asset this script:
  1. builds the kinematic demo scene (urdfkit.build_mjcf, physics=False — the
     same scene step 8 uses, so the figure always matches the shipped look),
  2. sets every joint to a SHOWCASE pose (a fraction of its open target, so
     drawers/doors/trays read clearly as articulation, not as closed boxes),
  3. frames the camera automatically on the showcase-pose AABB (per-point
     cone fit -> identical margins across assets, no crop / no tiny-asset
     cell), with per-asset azimuth/elevation overrides,
  4. composites a 3+2 grid with labels -> docs/report-assets/fig-asset-family.png.

Usage:
  .venv/bin/python pipeline/fig_asset_family.py                # full figure
  .venv/bin/python pipeline/fig_asset_family.py --only side-table-01 --debug
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import urdfkit  # noqa: E402

# ---------------------------------------------------------------- parameters

OUT_REL = Path("docs/report-assets/fig-asset-family.png")
DEBUG_DIR = "output/figs-debug"
BACKUP_DIR = "output/figs-backup"

CELL_W, CELL_H = 760, 560   # final cell size (rendered at SS, downscaled)
SS = 2                      # supersampling factor
GAP, MARGIN = 10, 10        # grid gutters / outer margin (background color)
BG = (13, 13, 13)

# per-asset showcase pose + camera. pose = joint-name -> fraction of the open
# target (slide: how far out; hinge: how far swung). camera azimuth/elevation
# in MuJoCo free-camera degrees; all 5 assets face -Y.
ASSETS = [
    dict(name="side-table-01", label="side-table-01", sub="drawer · prismatic",
         pose={"drawer_joint": 0.70}, azimuth=135, elevation=-20),
    dict(name="wardrobe-01", label="wardrobe-01", sub="double doors · revolute",
         pose={"door_left_joint": 0.60, "door_right_joint": 0.60},
         azimuth=135, elevation=-18),
    dict(name="nightstand-02", label="nightstand-02", sub="drawer + door",
         pose={"drawer_top_joint": 0.70, "door_bottom_joint": 0.60},
         azimuth=135, elevation=-20),
    dict(name="oven-mini-01", label="oven-mini-01", sub="drop-front door · revolute",
         pose={"door_joint": 0.65}, azimuth=135, elevation=-24),
    dict(name="toolbox-01", label="toolbox-01", sub="tray · prismatic",
         pose={"tray_joint": 0.75}, azimuth=135, elevation=-20),
]

MARGIN_FIT = 1.22   # cone-fit margin (1.0 = AABB corners touch the frame)
FOVY = 45.0         # MuJoCo free-camera default vertical fov


# ---------------------------------------------------------------- geometry

def _hull_mesh(asset_dir: Path, link: str):
    coll = asset_dir / "collision" / f"{link}_hull00.stl"
    vis = asset_dir / "meshes" / f"{link}.stl"
    import trimesh
    return trimesh.load(coll if coll.exists() else vis)


def showcase_aabb(asset_dir: Path, model: dict, pose_frac: dict):
    """World AABB of the whole asset AT the showcase pose (root placed on the
    floor + every child link transformed to its showcase joint value)."""
    body_pos, _ = urdfkit.body_placement(asset_dir, model)

    lo, hi = _hull_mesh(asset_dir, urdfkit.root_link(model)).bounds
    pts = [np.vstack([lo, hi]) + body_pos]

    for j in model["joints"]:
        frac = pose_frac.get(j["name"], 0.0)
        tgt, sign = urdfkit.open_target(j)
        t = tgt * frac
        V = np.asarray(_hull_mesh(asset_dir, j["child"]).vertices)
        if j["type"] == "prismatic":
            P = V + j["pos"] + j["axis"] * t
        else:
            R = trimesh_transform(j["axis"], t)
            P = (R @ V.T).T + j["pos"]
        pts.append(P + body_pos)

    P = np.vstack(pts)
    return P.min(0), P.max(0)


def trimesh_transform(axis, angle):
    import trimesh
    return trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]


def fit_camera(aabb_lo, aabb_hi, azimuth, elevation, aspect):
    """Distance + lookat so every AABB corner sits inside the frustum with an
    even margin (per-point cone fit; more even than a bounding sphere)."""
    c = (aabb_lo + aabb_hi) / 2.0
    az, el = np.radians(azimuth), np.radians(elevation)
    # MuJoCo free-camera convention (verified by render): camera sits at
    # lookat + d*(cos el*cos az, -cos el*sin az, -sin el) -> the view forward
    # is the negated horizontal part, tilted down by |el| (negative el = above).
    fwd = np.array([-np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    fwd /= np.linalg.norm(fwd)

    tan_v = np.tan(np.radians(FOVY) / 2.0)
    tan_h = tan_v * aspect
    corners = np.array([[x, y, z] for x in (aabb_lo[0], aabb_hi[0])
                        for y in (aabb_lo[1], aabb_hi[1])
                        for z in (aabb_lo[2], aabb_hi[2])])
    dist = 0.0
    for p in corners:
        v = p - c
        along = v @ fwd
        perp = np.linalg.norm(v - along * fwd)
        # vertical fov is the binding constraint; horizontal is wider (aspect>1)
        t = along + perp / tan_v
        dist = max(dist, t)
    return c, dist * MARGIN_FIT


# ---------------------------------------------------------------- render

def render_cell(entry: dict, debug=False):
    import imageio.v2 as imageio
    import mujoco

    name = entry["name"]
    cfg = common.config(name)
    asset = common.asset_dir(name)
    model_urdf = urdfkit.parse_urdf(asset / f"{name}.urdf")

    # same builder as step 8, but a scratch scene with a bigger offscreen
    # buffer (SS-scaled cell exceeds the shipped scene's 1280x720 default)
    scene = asset / "_fig_scene.mjcf"
    xml = urdfkit.build_mjcf(asset, model_urdf, cfg, physics=False)
    scene.write_text(xml.replace('offwidth="1280" offheight="720"',
                                 f'offwidth="{CELL_W * SS}" offheight="{CELL_H * SS}"'))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    joints = urdfkit.ordered_joints(model_urdf)
    pose_frac = entry["pose"]
    for i, j in enumerate(joints):
        tgt, _ = urdfkit.open_target(j)
        data.qpos[i] = tgt * pose_frac.get(j["name"], 0.0)
    mujoco.mj_forward(model, data)

    lo, hi = showcase_aabb(asset, model_urdf, pose_frac)
    lookat, dist = fit_camera(lo, hi, entry["azimuth"], entry["elevation"],
                              aspect=CELL_W / CELL_H)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = entry["azimuth"]
    cam.elevation = entry["elevation"]

    renderer = mujoco.Renderer(model, height=CELL_H * SS, width=CELL_W * SS)
    renderer.update_scene(data, camera=cam)
    img = renderer.render().copy()
    renderer.close()

    import PIL.Image as Image
    cell = Image.fromarray(img).resize((CELL_W, CELL_H), Image.LANCZOS)
    if debug:
        out = common.root() / DEBUG_DIR / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cell.save(out)
        print(f"[debug] {name}: qpos=" +
              ", ".join(f"{data.qpos[i]:+.3f}" for i in range(len(joints))) +
              f" dist={dist:.2f} -> {out}")
    return cell


def compose(cells, out_path: Path):
    from PIL import Image, ImageDraw, ImageFont

    font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    font_r = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)

    labeled = []
    for cell, meta in zip(cells, ASSETS):
        img = cell.copy()
        d = ImageDraw.Draw(img)
        x, y = 18, CELL_H - 66
        d.text((x, y), meta["label"], font=font_b, fill=(255, 255, 255),
               stroke_width=3, stroke_fill=(0, 0, 0))
        d.text((x, y + 36), meta["sub"], font=font_r, fill=(210, 210, 210),
               stroke_width=2, stroke_fill=(0, 0, 0))
        labeled.append(img)

    W = 3 * CELL_W + 2 * GAP + 2 * MARGIN
    H = 2 * CELL_H + GAP + 2 * MARGIN
    board = Image.new("RGB", (W, H), BG)
    pos3 = [(MARGIN + i * (CELL_W + GAP), MARGIN) for i in range(3)]
    x0 = MARGIN + (CELL_W // 2) + GAP // 2            # bottom row of 2, centered
    pos2 = [(x0, MARGIN + CELL_H + GAP),
            (x0 + CELL_W + GAP, MARGIN + CELL_H + GAP)]
    for img, p in zip(labeled, pos3 + pos2):
        board.paste(img, p)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(out_path)
    print("figure:", out_path, f"({W}x{H})")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Render the 5-asset family figure")
    ap.add_argument("--only", help="render a single asset cell (debug)")
    ap.add_argument("--debug", action="store_true", help="write debug cell pngs")
    ap.add_argument("--out", default=str(OUT_REL), help="output path (relative to root)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")  # headless offscreen

    if args.only:
        entry = next(e for e in ASSETS if e["name"] == args.only)
        render_cell(entry, debug=True)
        return

    out = common.root() / args.out
    if out.exists():                       # keep the previous version, no-VCS insurance
        bdir = common.root() / BACKUP_DIR
        bdir.mkdir(parents=True, exist_ok=True)
        b = bdir / out.name
        if not b.exists():
            b.write_bytes(out.read_bytes())
            print("backed up old figure ->", b)

    cells = [render_cell(e, debug=args.debug) for e in ASSETS]
    compose(cells, out)


if __name__ == "__main__":
    main()
