#!/usr/bin/env python3
"""Build a synthetic LUX3D-articulation-like delivery that deliberately
bakes in ALL FOUR documented pitfalls (demo.md), so the factory's
auto-derivations can be validated offline with zero credits:

  #1 joint origins in a floor-based frame while link meshes are exported
     body-centered  -> auto JOINT_Y_SHIFT must recover -0.275
  #2 rest-pose micro-open: the drawer protrudes 1.5 cm past the front face
     and the door rests 1.5 deg open -> sweep filter must cut the front
  #3 body mesh way above MuJoCo's 200k-face STL limit (700k) -> decimation
  #4 open direction depends on axis sign (door opens toward NEGATIVE angles)

Geometry (glTF floor frame, +y up, +z front):
  cabinet 0.5 x 0.55 x 0.4, six panels including a CLOSED front plate
  (worst case: no openings in the mesh at all)
  drawer: 0.4 x 0.12 x 0.3, rest center (0, 0.35, 0.05), slides +z (open),
          protrudes +z 1.5 cm at rest
  door:   0.48 x 0.26 panel, hinge at left edge (-0.25, 0.14, 0.2), rotates
          about +y, opens toward -90 deg (outward), rests 1.5 deg open

Writes output/downloads/fixture-cabinet/urdf/{robot.urdf, body.glb,
drawer_top.glb, door_bottom.glb} — the exact shape step 3 expects.
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

HALF_H = 0.275


def box(x0, x1, y0, y1, z0, z1, n=4):
    return trimesh.creation.box(bounds=[[x0, y0, z0], [x1, y1, z1]])


def body_mesh():
    """Six-panel cabinet incl. closed front, in glTF floor coords, then
    re-centered on the body center (y -= HALF_H) and face-inflated to ~700k."""
    panels = [
        box(-0.25, 0.25, 0.00, 0.03, -0.20, 0.20),   # bottom
        box(-0.25, 0.25, 0.52, 0.55, -0.20, 0.20),   # top
        box(-0.25, -0.22, 0.03, 0.52, -0.20, 0.20),  # left
        box(0.22, 0.25, 0.03, 0.52, -0.20, 0.20),    # right
        box(-0.22, 0.22, 0.03, 0.52, 0.17, 0.20),    # back
        box(-0.25, 0.25, 0.00, 0.55, 0.19, 0.20),    # front (CLOSED — trap #2)
    ]
    m = trimesh.util.concatenate(panels)
    while len(m.faces) < 600_000:                    # trap #3: ~786k faces
        m = m.subdivide()
    m.vertices[:, 1] -= HALF_H                       # body-centered export
    return m


def drawer_mesh():
    """Link-local (origin at the joint): 0.4x0.12x0.3 box, protruding 1.5 cm."""
    m = box(-0.20, 0.20, -0.06, 0.06, -0.15, 0.15)
    m.vertices[:, 2] += 0.015                        # trap #2 (front = +z)
    return m


def door_mesh():
    """Link-local (origin at the hinge): panel along +x, resting 1.5 deg open."""
    m = box(0.0, 0.48, -0.13, 0.13, -0.001, 0.001)
    R = trimesh.transformations.rotation_matrix(np.radians(-1.5), [0, 1, 0])[:3, :3]
    m.vertices = (R @ m.vertices.T).T                # trap #2 micro-open
    return m


URDF = """<?xml version="1.0"?>
<robot name="fixture-cabinet">
  <!-- LUX3D-style delivery: joint origins in FLOOR frame (trap #1) -->
  <link name="body">
    <visual><geometry><mesh filename="body.glb"/></geometry></visual>
  </link>
  <link name="drawer_top">
    <visual><geometry><mesh filename="drawer_top.glb"/></geometry></visual>
  </link>
  <link name="door_bottom">
    <visual><geometry><mesh filename="door_bottom.glb"/></geometry></visual>
  </link>
  <joint name="drawer_slide" type="prismatic">
    <parent link="body"/><child link="drawer_top"/>
    <origin xyz="0 0.35 0.05"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0.19" effort="10" velocity="1"/>
  </joint>
  <joint name="door_hinge" type="revolute">
    <parent link="body"/><child link="door_bottom"/>
    <origin xyz="-0.25 0.14 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="0" effort="5" velocity="1"/>
  </joint>
</robot>
"""


def main():
    out = common.downloads_dir("fixture-cabinet") / "urdf"
    out.mkdir(parents=True, exist_ok=True)
    trimesh.Scene(body_mesh()).export(out / "body.glb")
    trimesh.Scene(drawer_mesh()).export(out / "drawer_top.glb")
    trimesh.Scene(door_mesh()).export(out / "door_bottom.glb")
    (out / "robot.urdf").write_text(URDF)
    for f in sorted(out.iterdir()):
        print(f"{f.name}: {f.stat().st_size/1024:.0f} KB")
    print("fixture ready:", out)


if __name__ == "__main__":
    main()
