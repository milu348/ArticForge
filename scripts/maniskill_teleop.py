#!/usr/bin/env python3
"""Launch ManiSkill's built-in Panda teleop on an AHOLO task env (CPU render).

Why a launcher:
  * ManiSkill's interactive_panda.py hardcodes GPU rendering (render_mode=
    rgb_array -> cuda:0) and does not expose render_backend; on this machine
    we inject sim_backend/render_backend="cpu" via a gym.make wrapper.
  * The custom env (OpenAholoDrawer-v0 / OpenAholoJoint-v0) is only registered
    when aholo_tasks is imported, so we import it before launching.

Usage:
  source scripts/maniskill-env.sh
  .venv-maniskill/bin/python scripts/maniskill_teleop.py \
      -e OpenAholoDrawer-v0 --save-video \
      --viewer-shader default --video-saving-shader minimal \
      --record-dir output/demos

  # GUI keys (print 'h' in the viewer for the full list):
  #   arrow keys / u,j : move the ghost Panda hand (then press n to execute)
  #   g : open/close gripper   c : finish episode, save trajectory
  #   q : quit and write trajectories / videos
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import aholo_tasks  # noqa: F401  (registers custom envs)
import gymnasium as gym
from mani_skill.examples.teleoperation.interactive_panda import main, parse_args
from mani_skill.utils.wrappers.record import RecordEpisode

# ManiSkill version mismatch in interactive_panda.py: it calls
# env.render_human() and env.agent on the RecordEpisode wrapper, but that
# wrapper does not forward either attribute.  Patch forwarding properties so
# the stock teleop driver works without editing site-packages.
for _name in ("render_human", "agent"):
    if not hasattr(RecordEpisode, _name):
        setattr(RecordEpisode, _name,
                property(lambda self, _n=_name: getattr(self.unwrapped, _n)))

_orig_make = gym.make


def _cpu_make(id, **kwargs):
    kwargs.setdefault("sim_backend", "cpu")
    kwargs.setdefault("render_backend", "cpu")
    return _orig_make(id, **kwargs)


gym.make = _cpu_make

if __name__ == "__main__":
    main(parse_args())
