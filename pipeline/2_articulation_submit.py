#!/usr/bin/env python3
"""Step 3a: articulation submission + poll + download.

Preconditions (quote/approval step, outside this script):
  - submission record exists at output/records/<asset>/attempt-02-articulation.json
    with submitState == "not_started", retryCount == 0 and
    parameters.glbUrl / parameters.prompt matching the CLI arguments

Delivers a URDF package (robot.urdf + per-link GLB) into
output/downloads/<asset>/articulated/ (extract/rearrange so that
output/downloads/<asset>/urdf/ holds robot.urdf + GLBs for step 4).

Usage:
  .venv/bin/python pipeline/2_articulation_submit.py --asset nightstand-02 \
      --glb-url <uploaded glb url> --prompt "<运动描述（固定件/滑/转）>"
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


def now():
    return datetime.now(CST).isoformat()


def main():
    ap = common.add_asset_arg(argparse.ArgumentParser(
        description="Step 3a: articulation submit + poll + download"))
    ap.add_argument("--glb-url", required=True, help="uploaded static GLB url")
    ap.add_argument("--prompt", required=True, help="motion description prompt")
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

    def save():
        with open(record, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    assert rec["submitState"] == "not_started", f"attempt in state {rec['submitState']}"
    assert rec["retryCount"] == 0
    assert (rec["parameters"]["glbUrl"] == args.glb_url
            and rec["parameters"]["prompt"] == args.prompt), "record/args mismatch"

    rec["submitState"] = "started"
    rec["submissionStartedAt"] = now()
    save()

    task_id = c.create_articulation_animation_task(args.glb_url, args.prompt, region=args.region)

    rec["providerTaskId"] = str(task_id)
    rec["submitState"] = "submitted"
    rec["timestamps"]["submittedAt"] = now()
    save()
    print("providerTaskId:", task_id, flush=True)

    result = c.resume_articulation_animation(
        str(task_id), str(out_base), region=args.region, max_attempts=80, interval=15
    )
    rec["providerStatus"] = 3
    rec["timestamps"]["completedAt"] = now()
    rec["downloadedArtifacts"] = result.get("downloads", [])
    rec["artifactUrl"] = result.get("artifactUrl", "")
    rec["submitState"] = "completed"
    save()
    print("completed:", json.dumps(result.get("downloads", []), ensure_ascii=False))


if __name__ == "__main__":
    main()
