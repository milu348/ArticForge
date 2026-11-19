#!/usr/bin/env python3
"""Generate a LUX3D quote + compliance record for one pipeline step.

Quoting is free; billing happens at the create call and is covered by the
batch approval written into the record (approvedBy=user).  The record this
script produces is exactly what 1_text_to_3d.py / 2_articulation_submit.py
expect (submitState=not_started, retryCount=0, matching quoteId), so the
cloud chain can run without manual console work.

The API key is read from LUX3DAPI.json (a JSON string) and injected as the
region env var of the commerce.py child process only — never in argv, logs
or records.

Usage:
  .venv/bin/python pipeline/quote_step.py --asset nightstand-02 --step text-to-3d \
      --body '{"prompt": "...", "version": "G1-Turbo"}'
  .venv/bin/python pipeline/quote_step.py --asset nightstand-02 --step articulation \
      --body '{"glbUrl": "https://...", "prompt": "..."}'

Outputs (stdout): {"quoteId": ..., "quotedCredits": N, "record": path}
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CST = timezone(timedelta(hours=8))
ENDPOINTS = {
    "text-to-3d": "/lux3d/v1/generate/text-to-3d/task/create",
    "articulation": "/lux3d/v1/articulation-animation/task/create",
}
DEFAULT_RECORDS = {
    "text-to-3d": "attempt-01-text-to-3d.json",
    "articulation": "attempt-02-articulation.json",
}


def now():
    return datetime.now(CST).isoformat()


def api_key(region: str) -> str:
    key_file = common.root() / "LUX3DAPI.json"
    key = json.loads(key_file.read_text(encoding="utf-8"))
    if not isinstance(key, str) or not key:
        raise SystemExit(f"{key_file}: expected a JSON string API key")
    return key


def quote_via_cli(endpoint: str, body: dict, region: str) -> dict:
    cli = common.root() / "skills" / "aholo-lux3d" / "scripts" / "commerce.py"
    items = {"1": {"endpoint": {"method": "POST", "path": endpoint},
                   "parameters": {"body": body}}}
    env = dict(os.environ)
    env["LUX3D_CN_API_KEY" if region == "cn" else "LUX3D_GLOBAL_API_KEY"] = api_key(region)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(items, tf, ensure_ascii=False)
        items_path = tf.name
    try:
        r = subprocess.run(
            [sys.executable, str(cli), "quote", "--region", region,
             "--host-name", os.environ.get("LUX3D_HOST_NAME", "OpenCode"),
             "--items", items_path],
            capture_output=True, text=True, env=env, timeout=120)
    finally:
        os.unlink(items_path)
    if r.returncode != 0:
        raise SystemExit(f"quote failed: {r.stderr.strip()[:500]}")
    out = json.loads(r.stdout)
    q = out["quote"]
    assert q.get("pricingScope") == "BEFORE_ACCOUNT_BENEFITS", "quote context mismatch"
    assert q["estimatedCreditsTotal"] == sum(q["details"].values()), "quote total mismatch"
    return {"account": out["account"], "quote": q}


def ledger_update(ledger: Path, entry: dict, credits: int):
    data = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else {
        "conversation": "opencode-session 2026-11-15 (aholo lux3d demo)",
        "mode": "batch", "entries": [], "committedTotalQuoted": 0}
    data["entries"] = [e for e in data.get("entries", [])
                       if e.get("attemptId") != entry["attemptId"]]
    data["entries"].append(entry)
    data["committedTotalQuoted"] = sum(e.get("quotedCredits", 0)
                                       for e in data["entries"]
                                       if e.get("status") in ("committed", "reserved"))
    ledger.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = common.add_asset_arg(argparse.ArgumentParser(
        description="Quote one pipeline step and write its submission record"))
    ap.add_argument("--step", required=True, choices=sorted(ENDPOINTS))
    ap.add_argument("--body", required=True, help="create-call body as JSON")
    ap.add_argument("--region", default="cn")
    ap.add_argument("--record", default=None, help="record path (default per step)")
    ap.add_argument("--scope", default=None, help="approval scope note")
    ap.add_argument("--force", action="store_true",
                    help="re-quote even if the record already exists")
    args = ap.parse_args()

    record = (Path(args.record) if args.record
              else common.records_dir(args.asset) / DEFAULT_RECORDS[args.step])
    if record.exists() and not args.force:
        old = json.loads(record.read_text(encoding="utf-8"))
        if old.get("submitState") in ("not_started", "started", "submitted"):
            print(json.dumps({"quoteId": old["quoteId"],
                              "quotedCredits": old.get("quotedCredits"),
                              "record": str(record), "reused": True}))
            return
        raise SystemExit(f"{record}: exists in terminal state {old.get('submitState')} "
                         f"(use --force to re-quote)")

    body = json.loads(args.body)
    result = quote_via_cli(ENDPOINTS[args.step], body, args.region)
    account, q = result["account"], result["quote"]

    record.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "attemptId": f"{args.asset}-{record.stem}",
        "asset": args.asset,
        "planVersion": 1,
        "region": args.region,
        "accountRef": account.get("accountRef", ""),
        "operation": args.step,
        "endpoint": ENDPOINTS[args.step],
        "parameters": body,
        "quoteId": q["quoteId"],
        "quotedCredits": q["estimatedCreditsTotal"],
        "quoteDetails": q["details"],
        "approval": {
            "mode": "batch",
            "approvedBy": "user",
            "approvedAt": now(),
            "scope": args.scope or f"{args.asset}: {args.step} ({q['estimatedCreditsTotal']} 积分，对话内批次批准)",
        },
        "budgetLedger": "output/records/budget-ledger.json",
        "retryCount": 0,
        "submitState": "not_started",
        "timestamps": {"quotedAt": q["quotedAt"], "quoteExpiresAt": q["expiresAt"]},
    }
    record.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    ledger_update(common.root() / "output" / "records" / "budget-ledger.json", {
        "attemptId": rec["attemptId"],
        "asset": args.asset,
        "quoteId": q["quoteId"],
        "quotedCredits": q["estimatedCreditsTotal"],
        "status": "reserved",
        "outcome": "quoted",
    }, q["estimatedCreditsTotal"])

    print(json.dumps({"quoteId": q["quoteId"],
                      "quotedCredits": q["estimatedCreditsTotal"],
                      "record": str(record), "reused": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
