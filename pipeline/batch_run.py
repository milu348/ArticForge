#!/usr/bin/env python3
"""Batch factory driver: run pipeline steps across many assets from a batch
manifest, with per-asset state tracking (resumable) and a summary report.

Manifest (configs/<batch>.json):
  {
    "assets": [
      {"name": "nightstand-02",
       "text_prompt": "一句话描述物体外观",
       "motion_prompt": "运动描述（固定件/滑/转）",
       "quote_id": null,          // 1_text_to_3d 的报价 id（审批后填入）
       "version": "G1-Turbo", "region": "cn"}
    ]
  }

Steps: "1" text-to-3d (cloud, ~13 credits), "2" articulation (cloud, ~28
credits; needs step 1 output), "3"-"8" local chain. Default --steps 3-8
(zero credits). State lives in output/records/<asset>/state.json — completed
steps are skipped on re-run, so a failed batch resumes where it stopped.

Usage:
  .venv/bin/python pipeline/batch_run.py --batch batch-001 --steps 3,4,5,6,7,8
  .venv/bin/python pipeline/batch_run.py --batch batch-001 --cloud   # include 1,2
"""
import argparse
import fnmatch
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PY = sys.executable
PIPE = Path(__file__).resolve().parent
LOCAL_STEPS = {"3": "3_adapt_urdf.py", "4": "4_make_collision.py",
               "5": "5_filter_hulls.py", "6": "6_force_probe.py",
               "7": "7_physics_video.py", "8": "8_render_video.py"}
CLOUD_STEPS = {"1", "2"}


def state_path(asset):
    return common.records_dir(asset) / "state.json"


def load_state(asset):
    p = state_path(asset)
    return json.loads(p.read_text()) if p.exists() else {"steps": {}}


def save_state(asset, st):
    p = state_path(asset)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=2))


def sh(cmd, timeout=3600):
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-2500:])
    return r


def step_1(asset_spec, env):
    quote = asset_spec.get("quote_id")
    if not quote:
        print("  SKIP step 1: quote_id missing (approve the quote in the LUX3D "
              "console, then fill configs/<batch>.json)")
        return None
    return sh([PY, PIPE / "1_text_to_3d.py", quote,
               "--asset", asset_spec["name"],
               "--prompt", asset_spec["text_prompt"],
               "--version", asset_spec.get("version", "G1-Turbo"),
               "--region", asset_spec.get("region", "cn")], timeout=3600)


def step_2(asset_spec, env):
    name = asset_spec["name"]
    dl = common.downloads_dir(name)
    # find the static GLB from step 1 (any glb outside an urdf/ dir)
    glbs = [p for p in dl.rglob("*.glb") if "urdf" not in p.parts]
    if not glbs:
        print(f"  FAIL step 2: no GLB found under {dl}")
        return None
    r = sh([PY, common.skill_runtime() / "asset_uploader.py", "upload",
            glbs[0], "--region", asset_spec.get("region", "cn")])
    if r.returncode != 0:
        return r
    import re
    m = re.search(r"https?://\S+", r.stdout)
    if not m:
        print("  FAIL step 2: no glbUrl in uploader output")
        return None
    r = sh([PY, PIPE / "2_articulation_submit.py", "--asset", name,
            "--glb-url", m.group(0).rstrip(".,);\"'"),
            "--prompt", asset_spec["motion_prompt"]], timeout=1800)
    if r.returncode != 0:
        return r
    for attempt in range(60):  # poll up to ~30 min
        r = sh([PY, PIPE / "2_articulation_resume.py", "--asset", name],
               timeout=900)
        if r.returncode != 0:
            return r
        blob = (r.stdout or "") + (r.stderr or "")
        if any(k in blob.upper() for k in ("SUCCEED", "SUCCESS", "DONE", "DELIVERED")):
            break
        if "FAIL" in blob.upper():
            return None
        print(f"  ...polling articulation ({attempt + 1})")
        time.sleep(30)
    # locate + unpack the delivered URDF package
    urdf_dir = dl / "urdf"
    if not (urdf_dir / "robot.urdf").exists():
        for z in sorted(dl.rglob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                if any(n.endswith("robot.urdf") for n in names):
                    urdf_dir.mkdir(parents=True, exist_ok=True)
                    zf.extractall(urdf_dir)
                    print("  unpacked", z.name, "->", urdf_dir)
                    break
    return r if (urdf_dir / "robot.urdf").exists() else None


def main():
    ap = argparse.ArgumentParser(description="AHOLO batch factory driver")
    ap.add_argument("--batch", required=True,
                    help="batch name: configs/<batch>.json")
    ap.add_argument("--steps", default="3,4,5,6,7,8",
                    help="comma list subset of 1,2,3,4,5,6,7,8 (default: local chain)")
    ap.add_argument("--cloud", action="store_true",
                    help="shorthand for --steps 1,2,3,4,5,6,7,8")
    args = ap.parse_args()

    manifest_path = common.root() / "configs" / f"{args.batch}.json"
    manifest = json.loads(manifest_path.read_text())
    steps = list(dict.fromkeys(("1,2," + args.steps if args.cloud else args.steps).split(",")))

    summary = []
    for spec in manifest["assets"]:
        name = spec["name"]
        st = load_state(name)
        print(f"\n===== {name} =====")
        for step in steps:
            key = f"step{step}"
            done = st["steps"].get(key, {}).get("status") == "done"
            if done:
                print(f"[{key}] done (cached) — {st['steps'][key]['seconds']}s")
                summary.append((name, step, "cached", st["steps"][key]["seconds"]))
                continue
            t0 = time.time()
            if step in CLOUD_STEPS:
                r = (step_1 if step == "1" else step_2)(spec, None)
            elif step in LOCAL_STEPS:
                r = sh([PY, PIPE / LOCAL_STEPS[step], "--asset", name])
            else:
                print(f"[{key}] unknown step, skipped")
                summary.append((name, step, "unknown", 0))
                continue
            dt = round(time.time() - t0, 1)
            ok = bool(r is not None and getattr(r, "returncode", 1) == 0)
            st["steps"][key] = {"status": "done" if ok else "failed", "seconds": dt}
            save_state(name, st)
            if r is None:
                summary.append((name, step, "skipped", dt))
            else:
                summary.append((name, step, "ok" if ok else "FAILED", dt))
                if not ok:
                    print(f"[{key}] FAILED — fix and re-run (state saved, will resume)")
                    break

    print("\n==== BATCH SUMMARY ====")
    print(f"{'asset':<24}{'step':<6}{'status':<10}{'sec':>8}")
    fails = 0
    for name, step, status, dt in summary:
        print(f"{name:<24}{step:<6}{status:<10}{dt:>8}")
        fails += status in ("FAILED", "skipped")
    print(f"\n{fails} non-ok entr"
          f"{'y' if fails == 1 else 'ies'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
