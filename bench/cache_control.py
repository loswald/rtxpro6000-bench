#!/usr/bin/env python3
"""Reset an idle benchmark server's prefix cache, requiring a success receipt.

No running requests are aborted. Old vLLM versions returning empty HTTP 200
cannot establish success and fail closed. Use dedicated benchmark endpoints.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import urllib.error
import urllib.request


def acknowledged(engine, body):
    if engine == "vllm":
        try:
            receipt = json.loads(body)
        except (ValueError, TypeError):
            return False
        return isinstance(receipt, dict) and receipt.get("success") is True
    return body.strip().startswith("Cache flushed.")


def reset(engine, port):
    path = "/reset_prefix_cache" if engine == "vllm" else "/flush_cache"
    url = f"http://127.0.0.1:{port}{path}"
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            ok = response.status == 200 and acknowledged(engine, body)
        return {"port": port, "url": url, "verified": ok, "response": body[:1000]}
    except (OSError, ValueError) as error:
        return {"port": port, "url": url, "verified": False, "error": str(error)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--ports", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if any(not 1 <= p <= 65535 for p in args.ports):
        ap.error("ports must be between 1 and 65535")
    receipts = [reset(args.engine, p) for p in args.ports]
    result = {"engine": args.engine, "created": dt.datetime.now(dt.timezone.utc).isoformat(),
              "verified": all(r["verified"] for r in receipts), "receipts": receipts}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["verified"]:
        print("Cache reset was not confirmed; refusing to label this run as reset.")
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
