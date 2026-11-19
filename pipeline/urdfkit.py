#!/usr/bin/env python3
"""urdfkit — the auto-derivation core of the AHOLO factory.

Everything the physics steps (5-8) used to hard-code per asset is derived
here from two sources of truth written by earlier steps:
  - assets/<asset>/<asset>.urdf   (adapted URDF: Z-up, joint origins/axes/limits)
  - assets/<asset>/adapt-report.json  (link masses, STL paths, world placement)

Auto-derivation rules (demo.md pitfall notes #1/#2/#4 resolved generically):
  JOINT_Y_SHIFT : adapted root-link STL min_z
                  (body-centered export -> min_z = -half height -> shift down
                   by half height; floor-origin export -> min_z = 0 -> no shift)
  body placement: world pos = (0, 0, -min_z)  -> the asset sits on the floor
  sweep volumes : per joint, the child link's hull (link-local frame, origin
                  at the joint) transformed over the joint's full travel:
                  slide -> translate along the axis by lo and hi;
                  hinge -> sample rotations about the axis through the joint
  open direction: the limit endpoint with the larger |value| is "open" (this
                  resolves axis-sign dependence, pitfall #4)
  camera        : from the root link STL bounds
"""
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

FMT = lambda v: " ".join(f"{x:.6f}" for x in np.asarray(v).ravel())


# ---------------------------------------------------------------- URDF parse

def parse_urdf(path) -> dict:
    root = ET.parse(str(path)).getroot()
    links = [l.get("name") for l in root.findall("link")]
    joints = []
    for j in root.findall("joint"):
        org = j.find("origin")
        ax = j.find("axis")
        lim = j.find("limit")
        pos = np.array([float(x) for x in (org.get("xyz") if org is not None else "0 0 0").split()])
        axis = np.array([float(x) for x in (ax.get("xyz") if ax is not None else "1 0 0").split()])
        n = np.linalg.norm(axis)
        if n > 0:
            axis = axis / n
        joints.append(dict(
            name=j.get("name"), type=j.get("type"),
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            pos=pos, axis=axis,
            lo=float(lim.get("lower")) if lim is not None else 0.0,
            hi=float(lim.get("upper")) if lim is not None else 0.0,
            effort=float(lim.get("effort")) if lim is not None else 10.0))
    return {"name": root.get("name"), "links": links, "joints": joints}


def root_link(model) -> str:
    children = {j["child"] for j in model["joints"]}
    for l in model["links"]:
        if l not in children:
            return l
    return model["links"][0]


def ordered_joints(model) -> list:
    """Joints in kinematic-tree (DFS) order — matches MJCF/qpos/actuator order."""
    by_parent = defaultdict(list)
    for j in model["joints"]:
        by_parent[j["parent"]].append(j)
    out = []

    def walk(link):
        for j in by_parent.get(link, []):
            out.append(j)
            walk(j["child"])
    walk(root_link(model))
    return out


def open_target(joint):
    """(target, sign): the limit endpoint with the larger |value| is 'open'."""
    return (joint["hi"], 1.0) if abs(joint["hi"]) >= abs(joint["lo"]) else (joint["lo"], -1.0)


# ------------------------------------------------------- placement & sweeps

def _link_stl(asset_dir: Path, link: str) -> trimesh.Trimesh:
    return trimesh.load(asset_dir / "meshes" / f"{link}.stl")


def body_placement(asset_dir: Path, model: dict):
    """(world_pos, root_min_z): world pos puts the root link's lowest point on the floor."""
    z0 = _link_stl(asset_dir, root_link(model)).bounds[0][2]
    return np.array([0.0, 0.0, -z0]), float(z0)


def joint_y_shift(asset_dir: Path, model: dict) -> float:
    """Auto joint-origin correction: the root STL's min_z (see module docstring)."""
    return body_placement(asset_dir, model)[1]


def swept_aabbs(asset_dir: Path, model: dict, tol=0.004) -> dict:
    """Per joint: conservative world-frame AABB of the child part over its travel."""
    coll = asset_dir / "collision"
    fallback = asset_dir / "meshes"
    body_pos, _ = body_placement(asset_dir, model)
    out = {}
    for j in model["joints"]:
        hull = coll / f"{j['child']}_hull00.stl"
        m = trimesh.load(hull if hull.exists() else fallback / f"{j['child']}.stl")
        V = np.asarray(m.vertices)
        pts = []
        if j["type"] == "slide":
            for t in (j["lo"], j["hi"]):
                pts.append(V + j["pos"] + j["axis"] * t)
        else:  # hinge / continuous: sample the swing
            for a in np.linspace(j["lo"], j["hi"], 25):
                R = trimesh.transformations.rotation_matrix(a, j["axis"])[:3, :3]
                pts.append((R @ V.T).T + j["pos"])
        P = np.vstack(pts) + body_pos
        out[j["name"]] = (P.min(0) - tol, P.max(0) + tol)
    return out


# ---------------------------------------------------------------- MJCF build

def load_adapt_report(asset_dir: Path) -> dict:
    return json.loads((asset_dir / "adapt-report.json").read_text())


def load_collision(asset_dir: Path, filtered=True) -> dict:
    """link -> {'hulls': [{'file':...}, ...]}; prefers collision-filtered.json."""
    coll = asset_dir / "collision"
    full = json.loads((coll / "collision-report.json").read_text())
    if filtered:
        fp = coll / "collision-filtered.json"
        if fp.exists():
            data = json.loads(fp.read_text())
            # the filter writes the kept hulls under the root link's name
            # (older files used the literal key "body")
            root_key = next((k for k in data if isinstance(data[k], dict)
                             and "hulls" in data[k]), None)
            if root_key:
                full[root_key] = data[root_key]
    return full


def _mass_map(report) -> dict:
    return {l["name"]: l["mass"] for l in report["links"]}


def _hull_assets(rep, physics: bool) -> list:
    out = []
    for link, info in rep.items():
        if physics:
            for i, h in enumerate(info["hulls"]):
                out.append(f'    <mesh name="{link}_h{i}" file="{h["file"]}"/>')
        else:
            out.append(f'    <mesh name="{link}" file="{link}.stl"/>')
    return out


def _geom_lines(link, rep, masses, physics: bool, ind: str) -> list:
    out = []
    if physics:
        for i, h in enumerate(rep[link]["hulls"]):
            mass = masses.get(link, 1.0) if i == 0 else 0.0
            out.append(f'{ind}<geom class="col" type="mesh" mesh="{link}_h{i}" mass="{mass:.4f}"/>')
        out.append(f'{ind}<geom class="vis" type="mesh" mesh="{link}_vismesh" mass="0"/>')
    else:
        out.append(f'{ind}<geom class="vis" type="mesh" mesh="{link}" mass="{masses.get(link, 1.0):.4f}"/>')
    return out


def build_mjcf(asset_dir: Path, model: dict, cfg: dict, physics=True) -> str:
    """Contact-enabled physics scene (motor actuators, hull collision) or the
    kinematic demo scene (position actuators, visual meshes only)."""
    report = load_adapt_report(asset_dir)
    masses = _mass_map(report)
    rep = load_collision(asset_dir, filtered=physics)
    body_pos, _ = body_placement(asset_dir, model)
    names = model["name"]

    if physics:
        for link in report["links"]:  # extra visual mesh assets
            rep[link["name"]] = rep.get(link["name"], {"hulls": []})
        vis_assets = [f'    <mesh name="{l["name"]}_vismesh" file="{l["stl"]}"/>'
                      for l in report["links"]]
        geom_cls_def = ('    <default class="col">\n'
                        '      <geom group="3" rgba="0.6 0.85 0.6 0.35"/>\n'
                        '    </default>\n')
    else:
        vis_assets = []
        geom_cls_def = ""

    by_parent = defaultdict(list)
    for j in ordered_joints(model):
        by_parent[j["parent"]].append(j)

    def joint_attrs(j):
        jt = "slide" if j["type"] == "prismatic" else "hinge"
        dmp = _cfg_type(cfg, "damping", jt, 2.0 if jt == "slide" else 0.5)
        fl = _cfg_type(cfg, "frictionloss", jt, 1.0 if jt == "slide" else 0.05)
        return (jt, dmp, fl,
                f'<joint name="{j["name"]}" type="{jt}" axis="{FMT(j["axis"])}" '
                f'range="{j["lo"]:.6f} {j["hi"]:.6f}" damping="{dmp}" '
                f'frictionloss="{fl}" armature="0.01"/>')

    # MJCF semantics: a <joint> moves the body it is *inside of*, relative to
    # its parent.  So each joint must be the first child of the CHILD body,
    # whose pos is the URDF joint origin (parent-link frame).
    def emit(link, pos_str, ind, out, jxml=None):
        out.append(f'{ind}<body name="{link}" pos="{pos_str}">')
        if jxml:
            out.append(f'{ind}  {jxml}')
        out += _geom_lines(link, rep, masses, physics, ind + "  ")
        for j in by_parent.get(link, []):
            _, _, _, child_jxml = joint_attrs(j)
            emit(j["child"], FMT(j["pos"]), ind + "  ", out, jxml=child_jxml)
        out.append(f'{ind}</body>')

    bodies = []
    emit(root_link(model), FMT(body_pos), "    ", bodies)

    actuators = []
    for j in ordered_joints(model):
        jt = "slide" if j["type"] == "prismatic" else "hinge"
        if physics:
            c = _cfg_type(cfg, "ctrlrange", jt, 60.0 if jt == "slide" else 12.0)
            actuators.append(f'    <motor name="{j["name"]}_motor" joint="{j["name"]}" '
                             f'gear="1" ctrlrange="-{c} {c}"/>')
        else:
            kp = _cfg_type(cfg, "kp", jt, 600.0 if jt == "slide" else 40.0)
            actuators.append(f'    <position name="{j["name"]}_act" joint="{j["name"]}" '
                             f'kp="{kp}"/>')

    meshdir = "." if physics else "meshes"
    return f"""
<mujoco model="{names}_{'physics' if physics else 'demo'}">
  <compiler angle="radian" meshdir="{meshdir}"/>
  <option timestep="{0.001 if physics else 0.002}" gravity="0 0 -9.81" cone="elliptic"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <default>
    <geom solref="0.005 1" solimp="0.95 0.99 0.001"/>
    <default class="vis"><geom contype="0" conaffinity="0" group="2" rgba="0.78 0.63 0.44 1"/></default>
{geom_cls_def}  </default>
  <asset>
{_hull_assets(rep, physics)}
{chr(10).join(vis_assets)}
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.92 0.9" rgb2="0.8 0.8 0.78" width="300" height="300"/>
    <material name="floor_mat" texture="grid" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="1.2 -1.5 1.8" dir="-0.5 0.6 -0.6" diffuse="0.9 0.9 0.85" castshadow="false"/>
    <light pos="-1.0 -1.2 1.4" dir="0.4 0.5 -0.6" diffuse="0.5 0.5 0.55" castshadow="false"/>
    <geom name="floor" type="plane" size="2.5 2.5 0.05" material="floor_mat"/>
{chr(10).join(bodies)}  </worldbody>
  <actuator>
{chr(10).join(actuators)}
  </actuator>
</mujoco>
"""


def _cfg_type(cfg, key, jtype, default):
    v = cfg.get(key, default)
    if isinstance(v, dict):
        return float(v.get(jtype, default))
    return float(v)


def auto_camera(asset_dir: Path, model: dict, cfg: dict) -> dict:
    m = _link_stl(asset_dir, root_link(model))
    lo, hi = m.bounds
    _, z0 = body_placement(asset_dir, model)   # root STL min_z (world offset)
    # lookat must be in WORLD frame: link meshes are link-local, the root body
    # is lifted by -min_z so it rests on the floor (pitfall #1 placement rule).
    c = (lo + hi) / 2 + np.array([0.0, 0.0, -z0])
    cam = cfg.get("camera", {})
    dist = cam.get("distance") or float(np.max(hi - lo)) * 1.9
    return dict(lookat=np.array([c[0], c[1], c[2]]), distance=float(dist),
                azimuth=float(cam.get("azimuth", -150)),
                elevation=float(cam.get("elevation", -22)))
