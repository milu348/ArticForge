"""AHOLO articulated-asset manipulation tasks for ManiSkill3.

OpenAholoJointEnv: a Panda arm must open a movable joint (drawer slide or
door hinge) of an AHOLO-generated asset to >=90% travel — the same success
line as the factory's MuJoCo force probe (force_probe_report.json QA).

Assets are the factory deliverables themselves:
  assets/<asset>/<asset>.urdf  (collision = CoACD convex hulls, see 3b)
  assets/<asset>/adapt-report.json  (world placement: bodyPosZ)
  assets/<asset>/force_probe_report.json  (per-joint onset, open target)

Task family (drawer manipulation validated; door push-close framework
removed per wrap-up decision, doors remain covered by the Phase-0 smoke):
  OpenAholoJoint-v0    drawer assets, random each reconfigure
  OpenAholoDrawer-v0   prismatic: start 15% open (official rest-pose
                       micro-open, pitfall #2) -> PULL open to 90%+

The push-to-close task (CloseAholoDrawer-v0) and its Panda driver live in
``pipeline/10_close_drawer.py`` (step 10) — it reuses OpenAholoDrawerEnv
defined here. Importing this module alone no longer registers that task.

Reward follows the official OpenCabinetDrawer shape: reach the handle
(link COM) then open the joint, dense + success at 90% travel with the
link static.

Usage (CPU backend, WSL2):
  source scripts/maniskill-env.sh
  .venv-maniskill/bin/python -m mani_skill.examples.demo_random_action \\
      -e OpenAholoDrawer-v0 -b cpu -rb cpu
"""
from typing import Any, Optional, Union

import numpy as np
import sapien
import torch
from pathlib import Path

from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import SimConfig

AHOLO_ROOT = Path(__file__).resolve().parent.parent / "assets"
MOVABLE = ("revolute", "prismatic", "revolute_unwrapped")

# assets with at least one prismatic joint (drawers/trays)
DRAWER_ASSETS = ["side-table-01", "nightstand-02", "toolbox-01"]
# assets with at least one revolute joint (doors)
DOOR_ASSETS = ["wardrobe-01", "nightstand-02", "oven-mini-01"]


class OpenAholoJointEnv(BaseEnv):
    """Base: Panda opens a target joint of an AHOLO asset to 90% travel."""

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]
    asset_names: list[str] = []          # override per task
    # Pull to 70% travel instead of 90%: at 90% the handle world-x reaches
    # ~-0.01, i.e. BEHIND the robot base (x=0), so the arm cannot follow it
    # with a real grasp — the "success" came from the forearm shoving the
    # drawer.  At 70% the handle stays at x≈0.05, inside the arm workspace.
    min_open_frac = 0.7
    # drawer starts this far along its travel (fraction). The open task uses
    # the official rest-pose micro-open (15%); the close task overrides this
    # to leave enough travel to push back (50%).
    start_open_frac = 0.15
    # Per-asset yaw (deg about z): the robot base sits at the WORLD ORIGIN
    # facing +x — mplib 0.1.1 ignores set_base_pose (measured: targets are
    # solved/executed in the base frame = world frame), so the base MUST be
    # at the origin and the ASSET is placed 0.5 m in front instead. Asset
    # fronts (handle side, link-local -y) must face the robot (-x):
    #   front=-y  -> yaw -90 (side-table, nightstand, wardrobe)
    #   open side=+y (toolbox tray / oven flip semantics) -> yaw +90
    ASSET_YAW_DEG = {
        "side-table-01": -90.0,
        "nightstand-02": -90.0,
        "wardrobe-01": -90.0,
        "toolbox-01": 90.0,
        "oven-mini-01": 90.0,
    }
    asset_z = 0.15                       # asset on a nominal 15 cm stand:
    # drawer handles land at z≈0.62; combined with the base at y=-0.42 the
    # grasp point sits at ~0.63 m from the shoulder — Panda's comfort band
    # (0.72 m at z=0.72 was at the reach limit: screw-MP silently stopped
    # mid-path, 6 cm short of the target).

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02,
                 reconfiguration_freq=None, num_envs=1, asset_name=None, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # asset_name pins one asset (used by the single-asset demo scripts);
        # None keeps the default random-per-reconfigure behaviour.
        self.all_model_ids = np.array([asset_name] if asset_name else self.asset_names)
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 10
        super().__init__(*args, robot_uids=robot_uids,
                         reconfiguration_freq=reconfiguration_freq,
                         num_envs=num_envs, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(spacing=8)

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.55, 1.45, 0.75], target=[0.5, 0.35, 0.5])
        return CameraConfig("render_camera", pose=pose, width=640, height=640,
                            fov=1.1, near=0.01, far=100)

    def _load_agent(self, options: dict):
        # mplib 0.1.1 ignores set_base_pose (measured) — the base MUST sit
        # at the world origin; default Panda orientation faces +x, toward
        # the asset placed 0.5 m ahead (see _load_aholo_asset).
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        sapien.set_log_level("warn")
        model_id = str(np.asarray(self._batched_episode_rng.choice(self.all_model_ids)).ravel()[0])
        self._asset = self._load_aholo_asset(str(model_id))
        self._current_model_id = model_id

    def _load_aholo_asset(self, model_id: str) -> Articulation:
        ad = AHOLO_ROOT / model_id
        urdf = ad / f"{model_id}.urdf"
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        parsed = loader.parse(str(urdf))
        builders = parsed[0] if isinstance(parsed, tuple) else parsed.get(
            "articulation_builders", [])
        builder = builders[0]
        import json
        body_pos_z = json.loads(
            (ad / "adapt-report.json").read_text())["world"]["bodyPosZ"]
        # asset rests on an invisible stand: root link lifted to asset_z
        yaw = np.deg2rad(self.ASSET_YAW_DEG.get(model_id, 0.0))
        q_yaw = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]  # wxyz, about +z
        builder.initial_pose = sapien.Pose(
            p=[0.5, 0, self.asset_z + body_pos_z], q=q_yaw)
        # single-env reconfigure rebuilds the scene every reset; the old
        # articulation object is gone but its registry entry persists, so
        # build under a unique name each reconfigure instead (state obs
        # reads live objects, not the registry)
        art = builder.build(name=model_id)
        return art

    def _after_reconfigure(self, options):
        # target joint: first movable joint (our assets have 1-2; prismatic
        # preferred for the drawer task family)
        movable = [j for j in self._asset.get_joints()
                   if j.get_type()[0] in MOVABLE]   # batched: type is a list
        pri = [j for j in movable if j.get_type()[0] == "prismatic"]
        self.target_joint = (pri or movable)[0]
        self.target_link = self.target_joint.get_child_link()
        lim = np.asarray(self.target_joint.get_limits()[0]).ravel()
        lo, hi = float(lim[0]), float(lim[1])
        self._target_limits = (lo, hi)   # cached floats for episode init
        # open end = the limit endpoint with larger |value| (factory rule)
        self.open_target = hi if abs(hi) >= abs(lo) else lo

        # target link COM (link-local) — push-point anchor for doors
        import json
        rep = json.loads(
            (AHOLO_ROOT / self._asset.name / "adapt-report.json").read_text())
        link_meta = next(l for l in rep["links"] if l["name"] == self.target_link.name)
        self._link_com_local = np.array(link_meta["com"], dtype=np.float64)
        # goal joint value: drawer OPEN to 90%+ of travel (pull task)
        self.goal_q = self.open_target * self.min_open_frac
        self._travel = abs(self.open_target - lo)

    def grasp_anchor_position(self):
        """Grasp anchor (world): the target link's origin — the strategy
        anchors on the drawer front edge computed in the MP evaluator; the
        reward only needs a point that tracks the moving link."""
        return self.target_link.pose.p.clone().squeeze(0) \
            if self.target_link.pose.p.dim() > 1 else self.target_link.pose.p.clone()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # drawer task: start 15% open (matches the official rest-pose
            # micro-open, pitfall #2) -> PULL open to 90%+ of travel
            lim = self._target_limits
            closed = lim[0] if abs(lim[0]) <= abs(lim[1]) else lim[1]
            start_q = closed + (self.open_target - closed) * self.start_open_frac
            q = torch.zeros((b, self._asset.dof))
            k = 0
            for j in self._asset.get_joints():
                if j.get_type()[0] in MOVABLE:
                    q[:, k] = start_q
                    k += 1
            self._asset.set_qpos(q)
            # Panda rest pose. Fingers at the fixed FORK aperture (qpos 0.033,
            # pad faces 3.5cm off centre => 7cm gap, straddling the two ends of
            # side-table-01's 7.3cm handle rod). NEVER actuated afterwards —
            # zero finger motion for the whole episode.
            qpos = torch.tensor([0, 0, 0, -np.pi / 4, 0, np.pi / 4, 0,
                                 0.033, 0.033]).repeat(b).reshape(b, -1)
            qpos[:, :7] += torch.randn(b, 7) * self.robot_init_qpos_noise
            self.agent.robot.set_qpos(qpos)

    def target_qpos_index(self) -> int:
        idx = 0
        for j in self._asset.get_joints():
            if j.get_name() == self.target_joint.get_name():
                return idx
            idx += j.get_dof()
        raise RuntimeError("target joint not found")

    def evaluate(self):
        qpos = self._asset.get_qpos()
        idx = self.target_qpos_index()
        q = qpos[..., idx] if qpos.dim() == 1 else \
            qpos[..., idx:idx + 1].squeeze(-1)
        # drawer open: q has crossed 90% of travel toward open (overshoot ok)
        if self.open_target >= 0:
            goal_done = q >= self.goal_q
        else:
            goal_done = q <= self.goal_q
        link_static = torch.linalg.norm(
            self.target_link.angular_velocity, axis=-1) <= 1.0
        link_static = link_static & (
            torch.linalg.norm(self.target_link.linear_velocity, axis=-1) <= 0.1)
        anchor_pos = self.grasp_anchor_position()
        return {"success": goal_done & link_static,
                "open_enough": goal_done,
                "grasp_anchor_pos": anchor_pos}

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_anchor_pos=info["grasp_anchor_pos"] - self.agent.tcp.pose.p,
                target_joint_qpos=self._asset.get_qpos(),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_anchor = torch.linalg.norm(
            self.agent.tcp.pose.p - info["grasp_anchor_pos"], axis=-1)
        reaching = 1 - torch.tanh(5 * tcp_to_anchor)
        # fraction of travel remaining toward the open target
        qpos = self._asset.get_qpos()
        idx = self.target_qpos_index()
        q = qpos[..., idx] if qpos.dim() == 1 else \
            qpos[..., idx:idx + 1].squeeze(-1)
        remaining = torch.abs(q - self.goal_q) / max(self._travel, 1e-6)
        opening = 2 * (1 - remaining)
        # single-env tensors are 0-dim: use torch.where instead of mask set
        reaching = torch.where(remaining < 0.999,
                               torch.tensor(2.0, device=reaching.device), reaching)
        opening = torch.where(info["open_enough"],
                              torch.tensor(3.0, device=opening.device), opening)
        reward = reaching + opening
        reward = torch.where(info["success"],
                             torch.tensor(5.0, device=reward.device), reward)
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0


@register_env("OpenAholoJoint-v0", max_episode_steps=200)
class OpenAholoJointAllEnv(OpenAholoJointEnv):
    """Drawer assets (validated family); doors are covered by the Phase-0
    operability smoke (MP door manipulation is a documented boundary)."""
    asset_names = ["side-table-01", "nightstand-02"]


@register_env("OpenAholoDrawer-v0", max_episode_steps=200)
class OpenAholoDrawerEnv(OpenAholoJointEnv):
    """Prismatic (slide) joints only."""
    asset_names = DRAWER_ASSETS
