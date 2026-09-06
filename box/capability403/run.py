"""Matched, audited adapter for the original six-family 403-case suite."""
from __future__ import annotations

import argparse
import contextvars
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent / "evalsuite"
FAMILIES = {"tools": 70, "code": 75, "math": 80, "longctx": 48, "knowledge": 70, "ifeval": 60}
CAPS = {"math": 32768, "code": 20480, "knowledge": 20480, "ifeval": 16384, "tools": 8192, "longctx": 6144}
CONFIG = {"families": FAMILIES, "completion_caps": CAPS, "temperature": 1.0, "top_p": 0.95,
          "seed": 20260903, "concurrency": 64, "time_budget_s": 7260, "grace_s": 120,
          "request_timeout_s": 3600, "retries": 3, "retry_backoff_s": [2, 8, 20],
          "context_window": 40960, "reasoning": True, "tools_mode": "prompt", "save_responses": "full"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_suite():
    lock = json.loads((HERE / "suite.lock.json").read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        path = SUITE / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Pinned suite file differs or is missing: " + relative)
    cases = {}
    for family, expected_count in FAMILIES.items():
        rows = [r for r in read_jsonl(SUITE / "data" / "items" / (family + ".jsonl")) if "id" in r]
        if len(rows) != expected_count:
            raise ValueError(f"{family}: expected {expected_count}, found {len(rows)}")
        for row in rows:
            if row["id"] in cases:
                raise ValueError("Duplicate case id: " + row["id"])
            cases[row["id"]] = family
    if len(cases) != 403:
        raise ValueError("Expected exactly 403 cases")
    manifest = json.loads((SUITE / "data" / "manifest.json").read_text(encoding="utf-8"))
    return {"lock_sha256": digest(HERE / "suite.lock.json"), "manifest_file_sha256": digest(SUITE / "data" / "manifest.json"),
            "historical_manifest_digest": manifest["manifest_sha256"],
            "file_count": len(lock["files"]), "case_count": len(cases), "case_families": cases}


def validate_urls(urls):
    for url in urls.split(","):
        p = urlparse(url)
        if p.scheme != "http" or p.hostname not in ("127.0.0.1", "::1") or p.username or p.password or p.query or p.fragment or p.path not in ("", "/"):
            raise ValueError("Use numeric loopback HTTP endpoints without credentials, e.g. http://127.0.0.1:8000")
        if p.port is None:
            raise ValueError("An explicit endpoint port is required")


def body_from_call(client, original, messages, kwargs):
    b = inspect.signature(original).bind(client, messages, **kwargs)
    b.apply_defaults()
    k = b.arguments
    body = {"model": k["model"] or client.model, "messages": messages, "max_tokens": int(k["max_tokens"]),
            "temperature": float(k["temperature"]), "top_p": float(k["top_p"]), "stream": False}
    if k["seed"] is not None:
        body["seed"] = int(k["seed"])
    if k["tools"]:
        body.update({"tools": k["tools"], "tool_choice": k["tool_choice"] if k["tool_choice"] is not None else "auto"})
    if k["extra_body"]:
        body.update(k["extra_body"])
    return body


class RequestLedger:
    def __init__(self, path, cases, replay=None):
        self.path, self.cases = Path(path), cases
        self.context = contextvars.ContextVar("cap403_item", default=None)
        self.entries = {}
        self.replay = {}
        if replay:
            for row in read_jsonl(replay):
                key = (row["id"], row["turn"])
                if key in self.replay or canonical_hash(row["body"]) != row["sha256"]:
                    raise ValueError("Replay ledger has duplicate keys or corrupted body hash")
                self.replay[key] = row
            if set(self.replay) != {(item_id, 0) for item_id in cases}:
                raise ValueError("Replay ledger must contain exactly one logical request for each of the 403 cases")
        self.path.touch(exist_ok=False)

    def install(self, common, runner):
        original_one = runner.run_one
        original_chat = common.ChatClient.chat
        async def run_one(w, *args, **kwargs):
            token = self.context.set([w.id, w.family, 0])
            try:
                return await original_one(w, *args, **kwargs)
            finally:
                self.context.reset(token)
        async def chat(client, messages, **kwargs):
            current = self.context.get()
            if current is None:
                raise ValueError("Unattributed chat request is not permitted")
            item_id, family, turn = current
            current[2] += 1
            key = (item_id, turn)
            proposed = body_from_call(client, original_chat, messages, kwargs)
            body = proposed
            if self.replay:
                if key not in self.replay:
                    raise ValueError("Request is absent from replay: " + str(key))
                body = self.replay[key]["body"]
                # Long-context calibration can change rendering; replay freezes its prompt.
                fields = set(proposed) | set(body)
                if any(proposed.get(k) != body.get(k) for k in fields if k != "messages" or family != "longctx"):
                    raise ValueError("Non-prompt request settings differ from baseline: " + str(key))
            row = {"id": item_id, "family": family, "turn": turn, "sha256": canonical_hash(body), "body": body}
            if key in self.entries:
                raise ValueError("Repeated logical request: " + str(key))
            self.entries[key] = row
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
            return await original_chat(client, body["messages"], route_key=kwargs.get("route_key", 0), extra_body=body)
        runner.run_one = run_one
        common.ChatClient.chat = chat

    def complete(self):
        return set(self.entries) == {(item_id, 0) for item_id in self.cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify frozen suite and configuration without an endpoint or code execution")
    parser.add_argument("--preflight", action="store_true", help="Run only the fixed sandbox tests; never contact a model")
    parser.add_argument("--tag")
    parser.add_argument("--base-urls", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="m")
    parser.add_argument("--model-revision", help="Exact model revision; must match between comparison arms")
    parser.add_argument("--engine-label", help="Descriptive engine/patch label")
    parser.add_argument("--provenance-file", action="append", default=[], help="Launch/config/patch file to hash (repeatable)")
    parser.add_argument("--out", default="results/capability403")
    parser.add_argument("--replay-requests", help="Baseline .requests.jsonl to freeze the optimized requests")
    args = parser.parse_args(argv)
    verified = verify_suite()
    if args.check:
        print(json.dumps({"suite": {k: v for k, v in verified.items() if k != "case_families"}, "configuration": CONFIG}, indent=2))
        return 0
    from sandbox import Sandbox
    sandbox = Sandbox()
    saved_key = os.environ.pop("EVAL_API_KEY", None)
    try:
        sandbox_report = sandbox.prepare()
        if args.preflight:
            print(json.dumps(sandbox_report, indent=2))
            return 0
        if not args.tag or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag) or not args.model_revision or not args.engine_label:
            parser.error("--tag (safe filename), --model-revision, and --engine-label are required")
        validate_urls(args.base_urls)
        out = Path(args.out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        prefix = out / args.tag
        if any(out.glob(args.tag + ".*")):
            raise ValueError("Run tag already exists; use a fresh tag to avoid mixing results")
        sys.path.insert(0, str(SUITE))
        import common
        import run_eval
        from families import code
        sandbox.patch_code_module(code)
        ledger = RequestLedger(str(prefix) + ".requests.jsonl", verified["case_families"], args.replay_requests)
        ledger.install(common, run_eval)
        original_tokenize = common.ChatClient.tokenize
        async def tokenize(client, *a, **kw):
            result = await original_tokenize(client, *a, **kw)
            if client.max_model_len is not None and client.max_model_len != CONFIG["context_window"]:
                raise ValueError(f"Endpoint context window {client.max_model_len} differs from required 40960")
            return result
        common.ChatClient.tokenize = tokenize
        command = ["--tag", args.tag, "--base-urls", args.base_urls, "--model", args.model, "--out", str(out),
                   "--families", ",".join(FAMILIES), "--reasoning", "--temperature", "1", "--top-p", "0.95",
                   "--seed", "20260903", "--concurrency", "64", "--max-tokens", "32768",
                   "--max-tokens-family", ",".join(f"{k}={v}" for k, v in CAPS.items()), "--time-budget", "7260",
                   "--grace", "120", "--request-timeout", "3600", "--retries", "3", "--retry-backoff", "2,8,20",
                   "--save-responses", "full", "--data-dir", str(SUITE / "data")]
        metadata = {"schema": 1, "suite": verified, "configuration": CONFIG, "model": args.model,
                    "model_revision": args.model_revision, "engine_label": args.engine_label,
                    "provenance_files": {str(Path(p).resolve()): digest(p) for p in args.provenance_file},
                    "adapter_files": {p.name: digest(p) for p in HERE.glob("*.py")},
                    "sandbox": sandbox_report, "replay_source_sha256": digest(args.replay_requests) if args.replay_requests else None,
                    "suite_argv": command, "complete": False}
        metadata_path = str(prefix) + ".audit.json"
        write_json(metadata_path, metadata)
        try:
            exit_code = run_eval.main(command)
        finally:
            items_path = str(prefix) + ".items.jsonl"
            rows = read_jsonl(items_path) if Path(items_path).exists() else []
            ids = [r["id"] for r in rows]
            bad = [r["id"] for r in rows if r["status"] in ("error", "cancelled", "skipped")]
            metadata.update({"complete": len(ids) == 403 and set(ids) == set(verified["case_families"]) and not bad and ledger.complete(),
                             "record_count": len(rows), "execution_error_ids": bad,
                             "logical_request_count": len(ledger.entries), "requests_sha256": digest(ledger.path),
                             "items_sha256": digest(items_path) if Path(items_path).exists() else None})
            write_json(metadata_path, metadata)
        return 0 if exit_code == 0 and metadata["complete"] else 1
    finally:
        sandbox.close()
        if saved_key is not None:
            os.environ["EVAL_API_KEY"] = saved_key


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print("capability403: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
