#!/usr/bin/env python3
"""Paired, provenance-checked capability evaluation via installed lm-eval.

The runner needs lm_eval[api] in its interpreter; init/compare and tests use only
the standard library. No packages are installed, model servers launched, or
generated code executed. See QUALITY.md for the evidence boundary and commands.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import http.server
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


SCHEMA = 1
DEFAULT_TASKS = [
    {"name": "gsm8k_cot", "capability": "math", "metric": "exact_match,strict-match", "num_fewshot": 8},
    {"name": "bbh_cot_fewshot_boolean_expressions", "capability": "logic", "metric": "exact_match,get-answer", "num_fewshot": 3},
    {"name": "ifeval", "capability": "instruction_following", "metric": "prompt_level_strict_acc,none", "num_fewshot": 0},
]


def json_default(value):
    if callable(value):
        return {"callable": f"{value.__module__}.{value.__qualname__}"}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot preserve {type(value).__name__} as JSON")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=json_default)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=json_default) + "\n", encoding="utf-8")


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_suite(suite):
    if suite.get("schema_version") != SCHEMA or suite.get("purpose") not in {"accuracy", "smoke"}:
        raise ValueError("A version-1 suite must explicitly declare purpose accuracy or smoke")
    identity = suite.get("identity", {})
    for key in ("model_id", "model_revision", "tokenizer_revision", "prompt_format_sha256"):
        value = identity.get(key)
        if not isinstance(value, str) or not value.strip() or value.lower() in {"main", "latest", "unknown"} or "<" in value:
            raise ValueError(f"Pin identity.{key}; mutable revisions and placeholders are not provenance")
    if not re.fullmatch(r"[0-9a-f]{64}", identity["prompt_format_sha256"]):
        raise ValueError("identity.prompt_format_sha256 must fingerprint the actual prompt-format artifact")
    for key in ("max_gen_toks", "concurrency"):
        if not positive_int(suite.get(key)):
            raise ValueError(f"{key} must be a positive integer")
    if suite.get("limit") is not None and not positive_int(suite["limit"]):
        raise ValueError("limit must be null (full task) or a positive integer")
    if not isinstance(suite.get("seed"), int) or isinstance(suite["seed"], bool):
        raise ValueError("An explicit integer seed is required")
    gen = suite.get("generation")
    if not isinstance(gen, dict) or not {"temperature", "top_p", "chat_template_kwargs", "until"} <= gen.keys():
        raise ValueError("generation must explicitly set temperature, top_p, chat_template_kwargs, and until")
    if not isinstance(gen["chat_template_kwargs"], dict):
        raise ValueError("chat_template_kwargs must be an object ({} preserves server defaults)")
    if not isinstance(gen["until"], list) or len(gen["until"]) > 4 or not all(isinstance(s, str) and s for s in gen["until"]):
        raise ValueError("generation.until must be [] or at most four nonempty stop strings")
    reserved = {"model", "messages", "prompt", "seed", "max_tokens", "max_gen_toks", "max_completion_tokens", "stream", "n", "stop", "truncate_prompt_tokens"}
    if reserved & gen.keys():
        raise ValueError(f"Reserved generation keys: {sorted(reserved & gen.keys())}")
    if not isinstance(suite.get("fewshot_as_multiturn"), bool) or "think_end_token" not in suite or "system_instruction" not in suite:
        raise ValueError("Declare fewshot_as_multiturn, think_end_token, and system_instruction explicitly")
    tasks = suite.get("tasks", [])
    if not tasks or len({t["name"] for t in tasks}) != len(tasks):
        raise ValueError("Task names must be nonempty and unique")
    for task in tasks:
        if not all(isinstance(task.get(key), str) and task[key] for key in ("name", "capability", "metric")):
            raise ValueError("Every task needs name, capability, and metric")
        if "," not in task["metric"] or not isinstance(task.get("num_fewshot"), int) or task["num_fewshot"] < 0:
            raise ValueError("Task metrics must include their filter (e.g. exact_match,strict-match), with explicit num_fewshot")
    canonical(suite)  # Reject NaN/infinities instead of silently accepting invalid provenance.


def normalized_request(payload):
    # Served aliases may differ between launches of the same pinned checkpoint.
    return {k: v for k, v in payload.items() if k != "model"}


class CaptureProxy:
    """Loopback-only recording proxy; authorization headers are never persisted."""

    def __init__(self, upstream, capture_path, timeout=1800):
        parsed = urllib.parse.urlsplit(upstream)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("--url must be an HTTP(S) server URL without credentials/query/fragment")
        self.upstream = upstream.rstrip("/")
        self.capture_path = Path(capture_path)
        self.timeout = timeout
        self.records = []
        self.lock = threading.Lock()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                record = {"path": self.path, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
                started = time.monotonic()
                status, raw = 502, b""
                try:
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    record["request"] = payload
                    if self.path != "/v1/chat/completions" or payload.get("stream"):
                        raise ValueError("Only non-streaming chat completions are supported")
                    headers = {"Content-Type": "application/json"}
                    if os.environ.get("QUALITY_API_KEY"):
                        headers["Authorization"] = "Bearer " + os.environ["QUALITY_API_KEY"]
                    req = urllib.request.Request(owner.upstream + self.path, data=canonical(payload).encode("utf-8"), headers=headers)
                    try:
                        with urllib.request.urlopen(req, timeout=owner.timeout) as response:
                            status, raw = response.status, response.read()
                    except urllib.error.HTTPError as exc:
                        status, raw = exc.code, exc.read()
                    record["response"] = json.loads(raw)
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    if not raw:
                        raw = canonical({"error": "Quality capture proxy request failed; see capture log"}).encode("utf-8")
                record.update(status=status, duration_s=time.monotonic() - started,
                              raw_response=raw.decode("utf-8", errors="replace"))
                with owner.lock:
                    owner.records.append(record)
                    with owner.capture_path.open("a", encoding="utf-8") as stream:
                        stream.write(canonical(record) + "\n")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"

    def __enter__(self):
        self.capture_path.touch(exist_ok=False)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def capture_issues(records, suite, expected_count):
    issues = []
    if len(records) != expected_count:
        issues.append(f"Expected {expected_count} requests, captured {len(records)} (missing requests/retries/repeats)")
    for index, record in enumerate(records):
        prefix = f"request {index}"
        request = record.get("request", {})
        if record.get("error") or record.get("status") != 200:
            issues.append(f"{prefix}: HTTP or capture error")
            continue
        if not request.get("messages") or request.get("seed") != suite["seed"]:
            issues.append(f"{prefix}: missing chat messages or mismatched seed")
        budgets = [request[k] for k in ("max_tokens", "max_completion_tokens") if k in request]
        if budgets != [suite["max_gen_toks"]]:
            issues.append(f"{prefix}: missing/changed/ambiguous output budget")
        for key, value in suite["generation"].items():
            request_key = "stop" if key == "until" else key
            if request.get(request_key) != value:
                issues.append(f"{prefix}: generation setting {key} was not preserved")
        choices = record.get("response", {}).get("choices", [])
        if len(choices) != 1:
            issues.append(f"{prefix}: expected exactly one response choice")
            continue
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            issues.append(f"{prefix}: incomplete/unknown finish_reason={choice.get('finish_reason')!r}")
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            issues.append(f"{prefix}: missing final answer (including reasoning-only output)")
    return issues


def selected_samples(raw, task):
    metric, filter_name = task["metric"].split(",", 1)
    samples = []
    for sample in raw.get("samples", {}).get(task["name"], []):
        if sample.get("filter") != filter_name:
            continue
        score = sample.get(metric)
        if not isinstance(score, (bool, int, float)) or score not in (0, 1):
            raise ValueError(f"{task['name']}: only binary per-example metrics are supported; got {score!r}")
        for key in ("doc_id", "doc_hash", "prompt_hash", "target_hash", "target", "arguments", "resps", "filtered_resps"):
            if key not in sample or sample[key] is None:
                raise ValueError(f"{task['name']}: missing per-example provenance/output {key}")
        samples.append({**sample, "score": int(score)})
    if not samples:
        raise ValueError(f"{task['name']}: no scored samples for {task['metric']}")
    if len({str(s["doc_id"]) for s in samples}) != len(samples):
        raise ValueError(f"{task['name']}: duplicate document IDs")
    counts = raw.get("n-samples", {}).get(task["name"], {})
    if counts.get("effective") != len(samples) or not positive_int(counts.get("original")):
        raise ValueError(f"{task['name']}: missing/mismatched effective and original sample counts")
    reported = raw.get("results", {}).get(task["name"], {}).get(task["metric"])
    observed = sum(s["score"] for s in samples) / len(samples)
    if not isinstance(reported, (int, float)) or not math.isclose(reported, observed, abs_tol=1e-9):
        raise ValueError(f"{task['name']}: aggregate metric does not match retained per-example scores")
    config = raw.get("configs", {}).get(task["name"], {})
    if config.get("output_type") != "generate_until":
        raise ValueError(f"{task['name']}: only generate_until tasks have captured final answers")
    return samples


def package_fingerprint(package):
    root = Path(package.__file__).resolve().parent
    files = sorted(p for p in root.rglob("*") if p.suffix in {".py", ".yaml", ".yml", ".jinja", ".json"})
    sha = hashlib.sha256()
    for path in files:
        sha.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        sha.update(path.read_bytes())
    return sha.hexdigest()


def run_suite(args):
    suite = load(args.suite)
    validate_suite(suite)
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)  # Never mix or overwrite older evidence.
    launch_bytes = Path(args.launch).read_bytes()
    write(out / "suite.json", suite)
    (out / "launch.json").write_bytes(launch_bytes)
    import lm_eval  # Requires the existing isolated evaluation environment, never the serving interpreter.

    manifest = {"schema_version": SCHEMA, "suite": suite, "suite_sha256": digest(suite),
                "harness_version": importlib.metadata.version("lm_eval"),
                "harness_sha256": package_fingerprint(lm_eval),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "label": args.label, "served_model": args.served_model,
                "launch_sha256": hashlib.sha256(launch_bytes).hexdigest(), "launch": load(args.launch),
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "tasks": {}, "issues": []}
    for task in suite["tasks"]:
        name = task["name"]
        print(f"quality: {args.label}: {name}, limit={suite['limit']}, max_gen_toks={suite['max_gen_toks']}", flush=True)
        try:
            with CaptureProxy(args.url, out / f"{name}.http.jsonl", args.timeout) as proxy:
                model_args = {"model": args.served_model, "base_url": proxy.url,
                              "tokenizer_backend": None, "tokenized_requests": False,
                              "num_concurrent": suite["concurrency"], "max_retries": 1,
                              "max_gen_toks": suite["max_gen_toks"], "seed": suite["seed"],
                              "think_end_token": suite["think_end_token"], "timeout": args.timeout}
                raw = lm_eval.simple_evaluate(
                    model="local-chat-completions", model_args=model_args, tasks=[name],
                    num_fewshot=task["num_fewshot"], batch_size=1, limit=suite["limit"],
                    apply_chat_template=True, fewshot_as_multiturn=suite["fewshot_as_multiturn"],
                    system_instruction=suite["system_instruction"],
                    gen_kwargs={**suite["generation"], "max_gen_toks": suite["max_gen_toks"]},
                    log_samples=True, bootstrap_iters=0, use_cache=None, cache_requests=False,
                    random_seed=suite["seed"], numpy_random_seed=suite["seed"],
                    torch_random_seed=suite["seed"], fewshot_random_seed=suite["seed"],
                    confirm_run_unsafe_code=False,
                )
            write(out / f"{name}.lm_eval.json", raw)
            samples = selected_samples(raw, task)
            issues = capture_issues(proxy.records, suite, len(samples))
            original = raw["n-samples"][name]["original"]
            if len(samples) != (min(suite["limit"], original) if suite["limit"] else original):
                issues.append("Captured sample count does not satisfy the frozen suite limit/full split")
            metadata = {key: raw.get(key, {}).get(name) for key in ("configs", "versions", "n-samples")}
            manifest["tasks"][name] = {
                "task": task, "metadata": metadata, "metadata_sha256": digest(metadata),
                "samples": samples, "requests": proxy.records,
                "accuracy": sum(s["score"] for s in samples) / len(samples), "issues": issues,
            }
            manifest["issues"].extend(f"{name}: {issue}" for issue in issues)
        except Exception as exc:
            manifest["issues"].append(f"{name}: {type(exc).__name__}: {exc}")
        write(out / "quality.json", manifest)  # Preserve completed work if a later task fails.
    manifest["status"] = "invalid" if manifest["issues"] else "complete"
    write(out / "quality.json", manifest)
    print(f"quality: {manifest['status']}: {out / 'quality.json'}", flush=True)
    return 2 if manifest["issues"] else 0


def binomial_cdf(k, n, p):
    if p <= 0:
        return 1.0
    if p >= 1:
        return float(k >= n)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(t - peak) for t in terms))


def binomial_upper(k, n, alpha):
    """Exact one-sided Clopper-Pearson upper bound; no normal/zero-variance shortcut."""
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(alpha) / n)
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        if binomial_cdf(k, n, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def paired_statistics(base, candidate, alpha):
    n = len(base)
    losses = sum(a == 1 and b == 0 for a, b in zip(base, candidate))
    gains = sum(a == 0 and b == 1 for a, b in zip(base, candidate))
    # Two binomial proportions from the paired outcomes. Bonferroni needs no
    # independence between gains/losses. Caller also splits alpha across tasks.
    gain_lower = 1 - binomial_upper(n - gains, n, alpha / 2)
    loss_upper = binomial_upper(losses, n, alpha / 2)
    return {"n": n, "baseline_accuracy": sum(base) / n, "candidate_accuracy": sum(candidate) / n,
            "gains": gains, "losses": losses, "delta": (gains - losses) / n,
            "delta_lower_bound": gain_lower - loss_upper,
            "bound_method": "exact binomial gain/loss bounds with Bonferroni correction"}


def document_key(sample):
    return str(sample["doc_id"])


def compare_runs(base, candidate, margin=0.0, confidence=0.95):
    if not 0 <= margin < 1 or not 0 < confidence < 1:
        raise ValueError("margin must be in [0,1), confidence in (0,1)")
    issues, reports = [], {}
    result = {"schema_version": SCHEMA, "status": "invalid", "pass": False,
              "margin": margin, "confidence": confidence, "tasks": reports, "issues": issues,
              "scope": "Only the configured tasks, fixed sample set and pinned serving/request settings; not index equivalence or universal quality preservation."}
    try:
        for label, run in (("baseline", base), ("candidate", candidate)):
            validate_suite(run.get("suite", {}))
            if run.get("schema_version") != SCHEMA or run.get("status") != "complete" or run.get("issues"):
                issues.append(f"{label}: incomplete/invalid evaluation")
            if run.get("suite_sha256") != digest(run["suite"]):
                issues.append(f"{label}: suite fingerprint missing or inconsistent")
            for key in ("harness_version", "harness_sha256", "runner_sha256", "launch_sha256"):
                if not run.get(key):
                    issues.append(f"{label}: missing {key}")
        for key in ("suite_sha256", "harness_version", "harness_sha256", "runner_sha256"):
            if base.get(key) != candidate.get(key):
                issues.append(f"Mismatched {key}; model/task/seed/sampling/budgets/provenance must match")
        if base["suite"]["purpose"] != "accuracy" or candidate["suite"]["purpose"] != "accuracy":
            issues.append("Smoke results cannot establish capability accuracy")
        task_names = {t["name"] for t in base["suite"]["tasks"]}
        if set(base.get("tasks", {})) != task_names or set(candidate.get("tasks", {})) != task_names:
            issues.append("Missing/unexpected tasks")
        if issues:
            return result
        alpha = (1 - confidence) / len(task_names)
        for name in sorted(task_names):
            a, b = base["tasks"][name], candidate["tasks"][name]
            for label, entry, run in (("baseline", a, base), ("candidate", b, candidate)):
                if entry.get("issues"):
                    issues.append(f"{name}/{label}: recorded integrity issues")
                if entry.get("metadata_sha256") != digest(entry.get("metadata")):
                    issues.append(f"{name}/{label}: task metadata fingerprint mismatch")
                expected_task = next(t for t in run["suite"]["tasks"] if t["name"] == name)
                if entry.get("task") != expected_task:
                    issues.append(f"{name}/{label}: task/metric differs from frozen suite")
                metric_name = expected_task["metric"].split(",", 1)[0]
                metadata = entry.get("metadata", {})
                if metadata.get("versions") is None:
                    issues.append(f"{name}/{label}: missing task version")
                reconstructed = {key: {name: metadata.get(key)} for key in ("configs", "versions", "n-samples")}
                reconstructed.update(samples={name: entry.get("samples", [])},
                                     results={name: {expected_task["metric"]: entry.get("accuracy")}})
                checked_samples = selected_samples(reconstructed, expected_task)
                if any(s["score"] != s[metric_name] for s in entry["samples"]):
                    issues.append(f"{name}/{label}: retained score differs from raw scored sample")
                original = metadata["n-samples"]["original"]
                expected_n = min(run["suite"]["limit"], original) if run["suite"]["limit"] else original
                if len(checked_samples) != expected_n:
                    issues.append(f"{name}/{label}: incomplete sample coverage")
                issues.extend(f"{name}/{label}: {issue}" for issue in capture_issues(entry.get("requests", []), run["suite"], len(entry.get("samples", []))))
            if a.get("metadata_sha256") != b.get("metadata_sha256"):
                issues.append(f"{name}: task/dataset/scoring configuration mismatch")
            request_sets = [Counter(digest(normalized_request(r.get("request", {}))) for r in x["requests"]) for x in (a, b)]
            if request_sets[0] != request_sets[1]:
                issues.append(f"{name}: actual prompts/generation payloads differ")
            mapped = [{document_key(s): s for s in x["samples"]} for x in (a, b)]
            if not mapped[0] or any(len(m) != len(x["samples"]) for m, x in zip(mapped, (a, b))) or mapped[0].keys() != mapped[1].keys():
                issues.append(f"{name}: missing, duplicate or unpaired samples")
                continue
            scores_a, scores_b = [], []
            for key in sorted(mapped[0]):
                sa, sb = mapped[0][key], mapped[1][key]
                for field in ("doc_hash", "prompt_hash", "target_hash", "arguments", "target", "filter"):
                    if field not in sa or field not in sb or sa[field] != sb[field]:
                        issues.append(f"{name}/{key}: missing/mismatched {field}")
                for sample in (sa, sb):
                    if sample.get("score") not in (0, 1) or "resps" not in sample or "filtered_resps" not in sample:
                        raise ValueError(f"{name}/{key}: missing binary score/raw answers")
                scores_a.append(sa["score"])
                scores_b.append(sb["score"])
            stats = paired_statistics(scores_a, scores_b, alpha)
            stats["observed_regression"] = stats["delta"] < -margin
            stats["noninferiority_supported"] = stats["delta_lower_bound"] >= -margin
            stats["capability"] = a["task"]["capability"]
            stats["changed_documents"] = [{"doc_id": key, "baseline": mapped[0][key]["score"], "candidate": mapped[1][key]["score"]}
                                          for key in sorted(mapped[0]) if mapped[0][key]["score"] != mapped[1][key]["score"]]
            reports[name] = stats
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        issues.append(f"Malformed/incomplete evidence: {exc}")
    if issues:
        return result
    if any(t["observed_regression"] for t in reports.values()):
        result["status"] = "observed_regression"
    elif reports and all(t["noninferiority_supported"] for t in reports.values()):
        result["status"], result["pass"] = "noninferiority_supported", True
    else:
        result["status"] = "inconclusive"
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Write a frozen suite; review model-specific reasoning settings before using it")
    for key in ("model-id", "model-revision", "tokenizer-revision", "prompt-format", "out"):
        init.add_argument("--" + key, required=True)
    init.add_argument("--generation-json", help="JSON object with the full generation settings; supports nested reasoning/template options")
    init.add_argument("--max-gen-toks", type=int, default=32768)
    init.add_argument("--limit", type=int, default=200, help="0 uses complete task splits")
    init.add_argument("--concurrency", type=int, default=8)
    init.add_argument("--seed", type=int, default=1234)
    run = commands.add_parser("run", help="Evaluate a running server using this interpreter's installed lm-eval")
    for key in ("suite", "url", "served-model", "label", "launch", "out"):
        run.add_argument("--" + key, required=True)
    run.add_argument("--timeout", type=float, default=1800)
    compare = commands.add_parser("compare", help="Offline paired comparison (standard library only)")
    for key in ("baseline", "candidate", "out"):
        compare.add_argument("--" + key, required=True)
    compare.add_argument("--margin", type=float, default=0.0, help="Predeclared allowed accuracy drop, e.g. 0.01 = 1 percentage point")
    compare.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args(argv)
    try:
        if args.command == "init":
            generation = load(args.generation_json) if args.generation_json else {"temperature": 0, "top_p": 1, "chat_template_kwargs": {}, "until": []}
            suite = {"schema_version": SCHEMA, "purpose": "accuracy",
                     "identity": {"model_id": args.model_id, "model_revision": args.model_revision,
                                  "tokenizer_revision": args.tokenizer_revision,
                                  "prompt_format_sha256": hashlib.sha256(Path(args.prompt_format).read_bytes()).hexdigest()},
                     "seed": args.seed, "limit": args.limit or None, "concurrency": args.concurrency,
                     "max_gen_toks": args.max_gen_toks, "generation": generation,
                     "think_end_token": "</think>", "fewshot_as_multiturn": True,
                     "system_instruction": None, "tasks": DEFAULT_TASKS}
            validate_suite(suite)
            if Path(args.out).exists():
                raise ValueError("Refusing to overwrite an existing frozen suite")
            write(args.out, suite)
            print(args.out)
            return 0
        if args.command == "run":
            return run_suite(args)
        report = compare_runs(load(args.baseline), load(args.candidate), args.margin, args.confidence)
        report.update(baseline=str(Path(args.baseline).resolve()), candidate=str(Path(args.candidate).resolve()))
        write(args.out, report)
        print(json.dumps({"status": report["status"], "pass": report["pass"], "issues": report["issues"]}, indent=2))
        return 0 if report["pass"] else (2 if report["status"] == "invalid" else 3)
    except (OSError, ValueError, ImportError) as exc:
        print(f"quality: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
