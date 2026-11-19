#!/usr/bin/env python3
"""Poll already-submitted text-to-3d tasks to completion and download results.

Zero-credit helper: tasks were created by 1_text_to_3d.py (or its inline
create path) and only need polling + download + record backfill. Never
re-creates a task. Handles the single-format download shape of
complete_task ((path, bytes) list) and unpacks the delivery ZIP in place
(the same unpack batch_run.step_2 performs).
"""
import argparse
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CST = timezone(timedelta(hours=8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True,
                    help="JSON mapping asset name -> provider task id")
    ap.add_argument("--region", default="cn")
    ap.add_argument("--max-attempts", type=int, default=80)
    ap.add_argument("--interval", type=int, default=15)
    args = ap.parse_args()

    sys.path.insert(0, str(common.root() / "skills" / "aholo-lux3d" / "core" / "runtime"))
    import lux3d_client as c

    tasks = json.loads(args.tasks)
    pending = dict(tasks)
    done = {}
    for attempt in range(args.max_attempts):
        for name, tid in list(pending.items()):
            try:
                t = c.get_task(tid, region=args.region)
            except Exception as e:  # transient API error: keep polling
                print(f"[{name}] get_task error: {e}", flush=True)
                continue
            status = t.get("status")
            if status == 3:
                out = common.downloads_dir(name)
                result = c.complete_task(str(tid), output_path=str(out),
                                         expected_formats=c.generation_result_formats(
                                             "G1-Turbo", None),
                                         region=args.region, max_attempts=2,
                                         interval=5)
                downloads = result if isinstance(result, list) else []
                rec_p = common.records_dir(name) / "attempt-01-text-to-3d.json"
                rec = json.loads(rec_p.read_text(encoding="utf-8"))
                rec["providerStatus"] = 3
                rec["submitState"] = "completed"
                rec.setdefault("timestamps", {})["completedAt"] = \
                    datetime.now(CST).isoformat()
                rec["downloadedArtifacts"] = [
                    {"path": str(p), "bytes": int(n), "format": "zip"}
                    for p, n in downloads]
                rec_p.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
                # unpack in place (mesh/tex_mesh.glb for step 2 upload)
                for p, _ in downloads:
                    if str(p).endswith(".zip"):
                        with zipfile.ZipFile(p) as z:
                            z.extractall(out)
                files = sorted(str(f.relative_to(out))
                               for f in out.rglob("*") if f.is_file())
                print(f"[{name}] task {tid} DONE -> {files}", flush=True)
                done[name] = tid
                del pending[name]
            elif isinstance(status, int) and status not in (0, 1, 2):
                print(f"[{name}] task {tid} terminal status {status} (FAILED?)",
                      flush=True)
                del pending[name]
        if not pending:
            break
        time.sleep(args.interval)
    for name, tid in pending.items():
        print(f"[{name}] task {tid} still pending after polling window",
              flush=True)
    print(json.dumps({"completed": done, "pending": pending}, ensure_ascii=False))


if __name__ == "__main__":
    main()
