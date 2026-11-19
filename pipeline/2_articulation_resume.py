#!/usr/bin/env python3
"""Step 3b: resume (poll + download) an existing articulation task.

Never re-creates the task — resumes by providerTaskId from the submission
record, so an interrupted poll never double-charges.

Usage:
  .venv/bin/python pipeline/2_articulation_resume.py --asset nightstand-02
"""
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

sys.path.insert(0, str(common.skill_runtime()))
import lux3d_client as c  # noqa: E402

CST = timezone(timedelta(hours=8))


def main():
    ap = common.add_asset_arg(argparse.ArgumentParser(
        description="Step 3b: resume articulation task (never re-create)"))
    ap.add_argument("--region", default="cn")
    ap.add_argument("--record", default=None,
                    help="submission record JSON (default: output/records/<asset>/attempt-02-articulation.json)")
    ap.add_argument("--out-base", default=None,
                    help="download base dir (default: output/downloads/<asset>/articulated)")
    args = ap.parse_args()

    record = (Path(args.record) if args.record
              else common.records_dir(args.asset) / "attempt-02-articulation.json")
    out_base = (Path(args.out_base) if args.out_base
                else common.downloads_dir(args.asset) / "articulated")

    with open(record, encoding="utf-8") as f:
        rec = json.load(f)
    assert rec["providerTaskId"], "no task id; cannot resume"
    assert rec["submitState"] in ("submitted", "completed"), rec["submitState"]

    result = c.resume_articulation_animation(
        rec["providerTaskId"], str(out_base), region=args.region, max_attempts=80, interval=15
    )
    rec["providerStatus"] = 3
    rec["timestamps"]["completedAt"] = datetime.now(CST).isoformat()
    rec["downloadedArtifacts"] = result.get("downloads", [])
    rec["artifactUrl"] = result.get("artifactUrl", "")
    rec["submitState"] = "completed"
    with open(record, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    # Unpack the delivery ZIP in place (urdf/robot.urdf + meshes/*.glb) so the
    # local chain (step 3+) works whether driven via batch_run or by hand.
    import zipfile
    for item in result.get("downloads", []):
        p = item["path"] if isinstance(item, dict) else item
        if str(p).endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                z.extractall(out_base)
            print("unpacked:", p, "->", out_base)
    print("completed:", json.dumps(result.get("downloads", []), ensure_ascii=False))


if __name__ == "__main__":
    main()
