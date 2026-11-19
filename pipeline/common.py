#!/usr/bin/env python3
"""AHOLO pipeline shared helpers: path resolution + optional per-asset config.

Root resolution: the parent of this file's directory (pipeline/..), overridable
with the AHOLO_ROOT environment variable. All pipeline scripts derive their
default paths from this root, so they run from any working directory.

Asset-centric layout (asset name is the single parameter axis, --asset):
  configs/<asset>.json        OPTIONAL per-asset overrides (all keys optional)
  assets/<asset>/             deliverables (URDF, meshes, collision, scenes, videos)
  output/downloads/<asset>/   raw cloud deliveries (regenerable, safe to delete)
  output/records/<asset>/     submission records (quote/approval evidence)

Everything the physics steps need is auto-derived from the adapted URDF and
adapt-report.json (see urdfkit.py). configs/<asset>.json only OVERRIDES the
defaults. Schema (all keys optional):
  {
    "density": 700,                       # kg/m^3 for the mass heuristic
    "mass": {"body": 25.0},               # absolute per-link mass overrides
    "joint_y_shift": -0.275,              # override the auto-derived shift (m, Y-up, pre-rotation)
    "frictionloss": {"slide": 1.0, "hinge": 0.05},
    "damping": {"slide": 2.0, "hinge": 0.5},
    "ctrlrange": {"slide": 60.0, "hinge": 12.0},
    "kp": {"slide": 600.0, "hinge": 40.0},
    "force_scale": 1.5,                   # step 7 peak = scale x measured onset
    "camera": {"distance": null, "azimuth": -128, "elevation": -14},
    "decompose": {"body": true}           # per-link CoACD on/off (default: root only)
  }
"""
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def root() -> Path:
    return Path(os.environ.get("AHOLO_ROOT", _ROOT))


def skill_runtime() -> Path:
    """Official aholo-lux3d Skill runtime (lux3d_client, asset_uploader, ...)."""
    return root() / "skills" / "aholo-lux3d" / "core" / "runtime"


def records_dir(asset: str) -> Path:
    return root() / "output" / "records" / asset


def downloads_dir(asset: str) -> Path:
    return root() / "output" / "downloads" / asset


def asset_dir(asset: str) -> Path:
    return root() / "assets" / asset


def config_path(asset: str) -> Path:
    return root() / "configs" / f"{asset}.json"


def config(asset: str) -> dict:
    """Load the optional per-asset override config; {} when absent."""
    p = config_path(asset)
    return json.loads(p.read_text()) if p.exists() else {}


def add_asset_arg(parser):
    """Register the common --asset flag that drives all default paths."""
    parser.add_argument("--asset", required=True,
                        help="asset name; drives records/downloads/assets paths")
    return parser


def cfg_get(cfg: dict, jtype: str, key: str, default: float) -> float:
    """Read a per-joint-type numeric setting: cfg[key][jtype] or cfg[key] or default."""
    block = cfg.get(key, {})
    if isinstance(block, dict):
        return float(block.get(jtype, block.get("default", default)))
    return float(block)
