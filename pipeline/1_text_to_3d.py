#!/usr/bin/env python3
"""Step 1: text-to-3D submission + poll + download (G1-Turbo).

Preconditions (quote/approval step, outside this script):
  - submission record exists at output/records/<asset>/attempt-01-text-to-3d.json
    with submitState == "not_started", retryCount == 0 and the matched quoteId

Behavior: single create call (content_id carries the quoteId), never
re-create; polls to terminal state and downloads the result zip into
output/downloads/<asset>/.

Usage:
  .venv/bin/python pipeline/1_text_to_3d.py <quote-id> \
      --asset nightstand-02 --prompt "<物体描述>"
"""
import argparse
import json
import os
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
        description="Step 1: text-to-3D submit + poll + download"))
    ap.add_argument("quote_id", help="quote id matched to the submission record")
    ap.add_argument("--prompt", required=True, help="text-to-3D prompt")
    ap.add_argument("--version", default="G1-Turbo", help="generation model version")
    ap.add_argument("--region", default="cn")
    ap.add_argument("--record", default=None,
                    help="submission record JSON (default: output/records/<asset>/attempt-01-text-to-3d.json)")
    ap.add_argument("--out-dir", default=None,
                    help="download directory (default: output/downloads/<asset>)")
    args = ap.parse_args()

    record = (Path(args.record) if args.record
              else common.records_dir(args.asset) / "attempt-01-text-to-3d.json")
    out_dir = Path(args.out_dir) if args.out_dir else common.downloads_dir(args.asset)

    with open(record, encoding="utf-8") as f:
        rec = json.load(f)

    def save():
        with open(record, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    # Pre-dispatch checks
    assert rec["submitState"] == "not_started", f"attempt already in state {rec['submitState']}"
    assert rec["retryCount"] == 0, "retry accounting mismatch"
    assert rec["quoteId"] == args.quote_id, "quote mismatch"

    rec["submitState"] = "started"
    rec["submissionStartedAt"] = now()
    save()

    # Single create call; content_id carries the matched quoteId
    task_id = c.create_text_to_3d_task(
        args.prompt,
        version=args.version,
        region=args.region,
        content_id=args.quote_id,
    )

    # Save returned ID before any polling
    rec["providerTaskId"] = str(task_id)
    rec["submitState"] = "submitted"
    rec["timestamps"]["submittedAt"] = now()
    save()
    print("providerTaskId:", task_id, flush=True)

    # Poll to terminal state and download expected outputs. NOTE: for a single
    # expected format the downloader appends the extension to output_path
    # (side-table-01 -> side-table-01.zip), so record the actual file.
    expected = c.generation_result_formats(args.version, None)
    result = c.complete_task(
        str(task_id),
        output_path=str(out_dir),
        expected_formats=expected,
        region=args.region,
        max_attempts=60,
        interval=15,
    )
    # complete_task returns parse_task_outputs(task) (dict) when output_path is
    # None, else a list of (path, byte_count) download pairs — handle both.
    if isinstance(result, list):
        downloads = result
        rec["providerStatus"] = c.get_task(str(task_id), region=args.region).get("status")
    else:
        downloads = []
        rec["providerStatus"] = result.get("status")
    rec["timestamps"]["completedAt"] = now()
    rec["downloadedArtifacts"] = [
        {"path": str(p), "bytes": int(n), "format": "zip"} for p, n in downloads
    ]
    save()
    print("final status:", rec["providerStatus"])
    for p, n in downloads:
        print(f"  downloaded: {p} ({n} bytes)")


if __name__ == "__main__":
    main()
