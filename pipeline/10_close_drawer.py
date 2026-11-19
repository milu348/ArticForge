#!/usr/bin/env python3
"""Step 10 — Push an AHOLO drawer closed with the Panda arm (ManiSkill3).

This is the *runnable* phase-1 counterpart of the OpenAholoDrawer-v0 task:
a scripted, contact-based controller drives a real Panda robot (mplib-free;
pd_ee_delta_pose Cartesian control) to push a partially-open drawer back to
its closed limit. No joint teleporting, no position-server cheating — the
drawer moves only because the fingers physically push it.

The ManiSkill task environment ``CloseAholoDrawer-v0`` and the driver live
together in this file so step 10 is self-contained (only the shared asset
loading / open-task base classes are imported from ``aholo_tasks``).

Usage (CPU, no display needed):
  source scripts/maniskill-env.sh
  .venv-maniskill/bin/python pipeline/10_close_drawer.py --asset side-table-01
  .venv-maniskill/bin/python pipeline/10_close_drawer.py --asset side-table-01 --view
  # offscreen render to a video
  .venv-maniskill/bin/python pipeline/10_close_drawer.py \
      --asset side-table-01 --record-every 1 \
      --record docs/report-assets/panda-close-drawer-offscreen.mp4
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# CPU Vulkan ICD on this machine (no NVIDIA ICD installed).
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
os.environ.setdefault("SAPIEN_VULKAN_LIBRARY_PATH", "")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.registration import register_env  # noqa: E402

from aholo_tasks import OpenAholoDrawerEnv  # noqa: E402  (shared asset loading)


# --------------------------------------------------------------------------- #
# ManiSkill task: CloseAholoDrawer-v0
# --------------------------------------------------------------------------- #
@register_env("CloseAholoDrawer-v0", max_episode_steps=200)
class CloseAholoDrawerEnv(OpenAholoDrawerEnv):
    """Panda pushes a partially-open AHOLO drawer back to its closed limit.

    The drawer starts 30% open (PhysX-stable; handle inside the arm
    workspace — see the 70% reach note on the open-task base class). Success
    means q reached the closed limit within ``closed_tol_frac`` of travel
    AND the link is static (no shoving-through by inertia).

    Reward mirrors ``OpenAholoJointEnv`` but drives ``q -> closed`` instead
    of ``q -> open`` (official OpenCabinetDrawer-style reach + progress).
    """

    # 30% keeps the drawer at a PhysX-stable open pose (~0.1 m travel); higher
    # start values settle back during reset because of the rest-pose contact.
    start_open_frac = 0.3
    closed_tol_frac = 0.05

    def __init__(self, *args, start_open_frac=None, closed_tol_frac=None, **kwargs):
        # set before super(): scene setup calls _after_reconfigure / reset
        if start_open_frac is not None:
            self.start_open_frac = start_open_frac
        if closed_tol_frac is not None:
            self.closed_tol_frac = closed_tol_frac
        super().__init__(*args, **kwargs)

    def _after_reconfigure(self, options):
        super()._after_reconfigure(options)
        lo, hi = self._target_limits
        # closed end = the limit endpoint with smaller |value| (factory rule)
        self.closed_q = lo if abs(lo) <= abs(hi) else hi
        signed_travel = self.open_target - self.closed_q
        # goal: within closed_tol_frac of travel from the closed limit
        self.close_goal = self.closed_q + signed_travel * self.closed_tol_frac

    def evaluate(self):
        qpos = self._asset.get_qpos()
        idx = self.target_qpos_index()
        q = qpos[..., idx] if qpos.dim() == 1 else \
            qpos[..., idx:idx + 1].squeeze(-1)
        # drawer closed: q has returned to within tol of the closed limit
        if self.open_target >= self.closed_q:
            done = q <= self.close_goal
        else:
            done = q >= self.close_goal
        link_static = torch.linalg.norm(
            self.target_link.angular_velocity, axis=-1) <= 1.0
        link_static = link_static & (
            torch.linalg.norm(self.target_link.linear_velocity, axis=-1) <= 0.1)
        return {"success": done & link_static,
                "closed_enough": done,
                "grasp_anchor_pos": self.grasp_anchor_position()}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_anchor = torch.linalg.norm(
            self.agent.tcp.pose.p - info["grasp_anchor_pos"], axis=-1)
        reaching = 1 - torch.tanh(5 * tcp_to_anchor)
        qpos = self._asset.get_qpos()
        idx = self.target_qpos_index()
        q = qpos[..., idx] if qpos.dim() == 1 else \
            qpos[..., idx:idx + 1].squeeze(-1)
        remaining = torch.abs(q - self.close_goal) / max(self._travel, 1e-6)
        closing = 2 * (1 - remaining)
        # single-env tensors are 0-dim: use torch.where instead of mask set
        reaching = torch.where(remaining < 0.999,
                               torch.tensor(2.0, device=reaching.device), reaching)
        closing = torch.where(info["closed_enough"],
                              torch.tensor(3.0, device=closing.device), closing)
        reward = reaching + closing
        reward = torch.where(info["success"],
                             torch.tensor(5.0, device=reward.device), reward)
        return reward


# --------------------------------------------------------------------------- #
# Scripted Panda controller
# --------------------------------------------------------------------------- #
# Handle / front-top push point in the movable link's LOCAL frame.
# side-table-01 was measured from the mesh (7.3 cm rod at x≈0, y≈-0.21, z≈0.069).
# Other assets fall back to the front-top edge of the link mesh bbox.
HANDLE_LOCAL = {
    "side-table-01": [0.0, -0.21, 0.069],
}


def push_point_local(asset: str, env) -> np.ndarray:
    if asset in HANDLE_LOCAL:
        return np.asarray(HANDLE_LOCAL[asset], dtype=np.float64)
    # Generic fallback: front-top edge of the target link's visual mesh.
    e = env.unwrapped
    report = json.loads((ROOT / "assets" / asset / "adapt-report.json").read_text())
    link_name = e.target_link.name
    meta = next(l for l in report["links"] if l["name"] == link_name)
    import trimesh
    mesh = trimesh.load(ROOT / "assets" / asset / meta["stl"])
    v = np.asarray(mesh.vertices)
    axis = np.asarray(next(j["axis"] for j in report["joints"]
                           if j["child"] == link_name), dtype=np.float64)
    # front = extreme vertex along the opening direction (joint axis)
    proj = v @ axis
    front_idx = int(np.argmax(proj) if axis.sum() >= 0 else np.argmin(proj))
    point = v[front_idx].copy()
    point[2] = v[:, 2].max() - 0.005   # near the top edge
    point[0] = float(np.mean(v[:, 0]))  # centred across the facade
    print(f"[info] auto push point for {asset}/{link_name}: {np.round(point, 3)}")
    return point


def run(asset: str, seed: int, view: bool, start_open_frac: float,
        max_steps: int, verbose: bool, record: str = None,
        record_every: int = 2):
    env_kwargs = dict(
        sim_backend="cpu",
        render_backend="cpu",
        control_mode="pd_ee_delta_pose",
        asset_name=asset,
        start_open_frac=start_open_frac,
        max_episode_steps=max_steps,
    )
    frames = []
    if record:
        # Offscreen camera: look from the robot's side so the drawer front
        # (which faces the robot) and the pushing gripper are both visible.
        from mani_skill.utils import sapien_utils
        pose = sapien_utils.look_at(eye=[-0.55, -0.95, 0.95],
                                    target=[0.28, 0.0, 0.55])
        p = np.asarray(pose.p).ravel()
        q = np.asarray(pose.q).ravel()
        env_kwargs.update(
            render_mode="rgb_array",
            human_render_camera_configs={
                "render_camera": {
                    "pose": [*p, *q],
                    "width": 960, "height": 720, "fov": 1.0,
                    "shader_pack": "default",
                }
            },
        )
    env = gym.make("CloseAholoDrawer-v0", **env_kwargs)
    env.reset(seed=seed)
    e = env.unwrapped
    handle_local = push_point_local(asset, env)

    def world_handle() -> np.ndarray:
        T = e.target_link.pose.sp.to_transformation_matrix()
        return (T @ np.r_[handle_local, 1.0])[:3]

    def action(delta: np.ndarray, gripper: float = -1.0) -> np.ndarray:
        # pd_ee_delta_pose: pos delta in [-0.1, 0.1] per axis -> action [-1, 1]
        pos = np.clip(np.asarray(delta, dtype=np.float64) / 0.1, -1.0, 1.0)
        return np.r_[pos, [0.0, 0.0, 0.0], [gripper]].astype(np.float32)

    def tcp() -> np.ndarray:
        return e.agent.tcp.pose.p[0].cpu().numpy()

    def qpos() -> float:
        return float(e._asset.get_qpos()[0, e.target_qpos_index()].item())

    step_count = 0

    def do_step(act: np.ndarray):
        nonlocal step_count
        out = env.step(act)
        step_count += 1
        if record and step_count % record_every == 0:
            img = np.asarray(env.render())
            while img.ndim > 3:
                img = img[0]          # drop the num_envs batch dim
            frames.append(img)
        return out[-1]

    def move_to(target: np.ndarray, k: float = 0.5, tol: float = 0.004,
                limit: int = 120, tag: str = "move"):
        for i in range(limit):
            d = np.asarray(target) - tcp()
            if np.linalg.norm(d) < tol:
                return True
            info = do_step(action(d * k))
            if verbose and i % 20 == 0:
                print(f"  [{tag}] step {i:3d} tcp={np.round(tcp(),3)} q={qpos():.4f}")
        return False

    lo, hi = e._target_limits
    print(f"asset={asset}  limits=({lo:.4f}, {hi:.4f})  "
          f"start q={qpos():.4f} ({start_open_frac*100:.0f}%)  "
          f"goal q<={e.close_goal:.4f}  handle={np.round(world_handle(),3)}")

    # 1. Small lift to clear the tabletop on the way in (collision-free approach).
    move_to(tcp() + np.array([0.0, 0.0, 0.05]), k=0.5, limit=40, tag="lift")

    # 2. Travel horizontally to a pre-push pose 5 cm in front of the handle.
    hp = world_handle()
    move_to(hp + np.array([-0.05, 0.0, 0.12]), tag="align")

    # 3. Lower onto the handle at push height.
    hp = world_handle()
    move_to(hp + np.array([-0.05, 0.0, 0.0]), tag="lower")

    # 4. Push +x (toward the cabinet) in small increments. The target is kept
    #    slightly beyond the live handle position so contact is maintained.
    for i in range(max_steps):
        hp = world_handle()
        d = np.clip(hp + np.array([0.03, 0.0, 0.0]) - tcp(), -0.008, 0.008)
        info = do_step(action(d))
        if verbose and i % 10 == 0:
            print(f"  [push] step {i:3d} q={qpos():.4f} tcp={np.round(tcp(),3)} "
                  f"handle={np.round(hp,3)}")
        if bool(info["closed_enough"].item()):
            print(f"  [push] closed at step {i}, q={qpos():.4f}")
            break
        if view:
            e.render_human()

    # 5. Settle (zero motion) so the link is static -> success condition.
    for _ in range(15):
        info = do_step(action(np.zeros(3)))
        if view:
            e.render_human()

    success = bool(info["success"].item())
    print(f"RESULT: {'SUCCESS' if success else 'FAIL'}  "
          f"final q={qpos():.4f} (goal<={e.close_goal:.4f})  "
          f"closed_enough={bool(info['closed_enough'].item())}")
    env.close()

    if record:
        # Hold the final frame briefly so the closed state is unmistakable.
        for _ in range(12):
            frames.append(frames[-1])
        import imageio.v2 as iio
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        iio.mimwrite(record, frames, fps=24, codec="libx264", quality=7)
        print(f"video: {record}  ({len(frames)} frames @ 24fps "
              f"= {len(frames)/24:.1f}s)")

    return 0 if success else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="side-table-01")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="render the live viewer")
    ap.add_argument("--start-open-frac", type=float, default=0.3)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--record", default=None, metavar="PATH",
                    help="offscreen-render an MP4 to this path")
    ap.add_argument("--record-every", type=int, default=2,
                    help="capture one frame every N env steps")
    args = ap.parse_args()
    sys.exit(run(args.asset, args.seed, args.view, args.start_open_frac,
                 args.max_steps, args.verbose, args.record, args.record_every))


if __name__ == "__main__":
    main()
