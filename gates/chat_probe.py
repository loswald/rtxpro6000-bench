#!/usr/bin/env python3
"""Run the existing 20 chat tripwires with explicit sampling and durable raw evidence.

This is a corruption/profile probe, not a capability benchmark. The output
directory must not exist. The source prompt list is read as a Python literal;
box/quality20.py is never imported or executed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_prompts(path):
    source = path.read_bytes()
    tree = ast.parse(source.decode("utf-8-sig"), filename=str(path))
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PROMPTS" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Expected one literal PROMPTS assignment")
    prompts = ast.literal_eval(assignments[0].value)
    if not isinstance(prompts, list) or len(prompts) != 20:
        raise ValueError("Expected the complete 20-prompt tripwire")
    for item in prompts:
        if (not isinstance(item, (tuple, list)) or len(item) != 2
                or not isinstance(item[0], str) or not item[0].strip()
                or (item[1] is not None and not isinstance(item[1], str))):
            raise ValueError("Malformed prompt/expected-substring pair")
    if len({item[0] for item in prompts}) != 20:
        raise ValueError("Duplicate prompts cannot supply 20-item coverage")
    return prompts, sha256(source)


def validate_request_config(config):
    if not isinstance(config, dict):
        raise ValueError("Request config must be a JSON object")
    reserved = set(config) & {"model", "messages", "prompt", "seed", "stream", "n"}
    if reserved:
        raise ValueError("Runner-owned request keys: " + ", ".join(sorted(reserved)))
    if not isinstance(config.get("max_tokens"), int) or isinstance(config["max_tokens"], bool) or config["max_tokens"] <= 0:
        raise ValueError("Request config needs a positive integer max_tokens")
    for key, low, high in (("temperature", 0, None), ("top_p", 0, 1), ("min_p", 0, 1)):
        if key not in config:
            continue
        value = config[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < low or (high is not None and value > high)):
            raise ValueError("Invalid request config value for " + key)
    if "temperature" not in config or "top_p" not in config:
        raise ValueError("Request config must explicitly specify temperature and top_p")
    canonical(config)  # reject NaN and Infinity anywhere in the supplied config


def item_seed(base_seed, index, prompt):
    digest = hashlib.sha256(f"{base_seed}:{index}:{prompt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def make_request(model, prompt, config, seed):
    return {**config, "model": model, "messages": [{"role": "user", "content": prompt}],
            "seed": seed, "stream": False, "n": 1}


def repetition(text):
    words = re.findall(r"\S+", text)
    if not words:
        return {"word_count": 0, "top6gram": 0, "distinct_ratio": None, "flag": False}
    ratio = len(set(words)) / len(words)
    top = max(Counter(tuple(words[i:i + 6]) for i in range(len(words) - 5)).values(), default=0)
    return {"word_count": len(words), "top6gram": top, "distinct_ratio": ratio,
            "flag": len(words) >= 12 and (top >= 4 or ratio < 0.35)}


def assess_response(response, expected, max_tokens):
    if not isinstance(response, dict) or response.get("error"):
        raise ValueError("Response is not a successful chat-completion object")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Response choice has no chat message")
    final = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(final, str) or not isinstance(reasoning, str):
        raise ValueError("Final and reasoning content must be strings or null")
    finish = choice.get("finish_reason")
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    at_budget = (isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool)
                 and completion_tokens >= max_tokens)
    capped = finish == "length" or at_budget
    final_rep, reasoning_rep = repetition(final), repetition(reasoning)
    # Check the answer channel only: a correct phrase in reasoning cannot rescue
    # an incorrect final answer. This retains the original lightweight tripwire.
    expected_found = expected is None or expected.casefold() in final.casefold()
    empty_final = not final.strip()
    leaked_thinking = "<think>" in final.casefold() or "</think>" in final.casefold()
    reasons = []
    if capped:
        reasons.append("completion reached the token budget")
    if empty_final:
        reasons.append("final answer is empty")
    if final_rep["flag"]:
        reasons.append("final answer repetition tripwire fired")
    if reasoning_rep["flag"]:
        reasons.append("reasoning repetition tripwire fired")
    if not expected_found:
        reasons.append("expected substring absent from final answer")
    if leaked_thinking:
        reasons.append("thinking markers leaked into the final answer channel")
    if finish not in ("stop", "length"):
        reasons.append("unexpected finish reason")
    if capped:
        verdict = "degenerate" if final_rep["flag"] or reasoning_rep["flag"] else "truncated"
    elif empty_final:
        verdict = "empty"
    elif finish != "stop" or leaked_thinking:
        verdict = "format"
    elif final_rep["flag"]:
        verdict = "degenerate"
    elif not expected_found:
        verdict = "wrong"
    else:
        verdict = "ok"
    return {
        "verdict": verdict, "reasons": reasons,
        "text": final, "reasoning": reasoning, "reasoning_chars": len(reasoning),
        "finish": finish, "empty_final": empty_final, "capped": capped,
        "expected_found_in_final": expected_found,
        "expected_found_in_reasoning": expected is not None and expected.casefold() in reasoning.casefold(),
        "final_repetition": final_rep, "reasoning_repetition": reasoning_rep,
        "usage": usage,
    }


def run_item(index, prompt, expected, request, url, directory, timeout):
    item_dir = directory / f"{index:02d}"
    item_dir.mkdir()
    body = canonical(request)
    (item_dir / "request.json").write_bytes(body)
    started = time.monotonic()
    record = {"index": index, "prompt": prompt, "expected": expected, "seed": request["seed"],
              "request": request, "request_sha256": sha256(body), "url": url, "started_at": now(),
              "http_status": None, "response": None, "response_sha256": None}
    raw = b""
    try:
        http_request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(http_request, timeout=timeout) as response:
                raw = response.read()
                record["http_status"] = response.status
                record["response_content_type"] = response.headers.get("Content-Type")
        except urllib.error.HTTPError as error:
            raw = error.read()
            record["http_status"] = error.code
            raise RuntimeError(f"HTTP {error.code}") from error
        parsed = json.loads(raw)
        record["response"] = parsed
        record.update(assess_response(parsed, expected, request["max_tokens"]))
    except Exception as error:
        if raw and record["response"] is None:
            try:
                record["response"] = json.loads(raw)
            except (ValueError, UnicodeError):
                pass
        record.update(verdict="error", text="", reasoning="", finish="error",
                      error={"type": type(error).__name__, "message": str(error)})
    finally:
        (item_dir / "response.raw").write_bytes(raw)
        record["response_sha256"] = sha256(raw)
        record["raw_response_file"] = str(Path("items") / f"{index:02d}" / "response.raw")
        record["elapsed_s"] = round(time.monotonic() - started, 6)
        record["ended_at"] = now()
        write_json(item_dir / "result.json", record)
    return record


def run_probe(*, model, base_url, out_dir, request_config, prompts_source, base_seed=1234,
              concurrency=10, timeout=600):
    validate_request_config(request_config)
    if not 1 <= concurrency <= 20 or timeout <= 0:
        raise ValueError("Concurrency must be 1..20 and timeout must be positive")
    prompts, source_hash = load_prompts(prompts_source)
    out_dir.mkdir(parents=True, exist_ok=False)  # never replace a prior greedy/profile result
    items_dir = out_dir / "items"
    items_dir.mkdir()
    url = base_url.rstrip("/") + "/v1/chat/completions"
    manifest = {
        "schema_version": 1, "purpose": "corruption_profile_tripwire_not_capability_benchmark",
        "started_at": now(), "model": model, "url": url, "request_config": request_config,
        "request_config_sha256": sha256(canonical(request_config)),
        "prompt_source": str(prompts_source.resolve()), "prompt_source_sha256": source_hash,
        "prompt_set_sha256": sha256(canonical(prompts)), "base_seed": base_seed,
        "seed_method": "sha256(base_seed:index:prompt), first 32 bits masked to 31 bits",
        "concurrency": concurrency, "timeout_s": timeout,
        "n_planned": len(prompts), "n_completed": 0, "partial": True, "pass": False,
    }
    write_json(out_dir / "run.json", manifest)
    records = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for index, (prompt, expected) in enumerate(prompts):
            request = make_request(model, prompt, request_config, item_seed(base_seed, index, prompt))
            futures.append(pool.submit(run_item, index, prompt, expected, request, url, items_dir, timeout))
        for future in as_completed(futures):
            records.append(future.result())
            manifest["n_completed"] = len(records)
            manifest["counts"] = dict(Counter(record["verdict"] for record in records))
            write_json(out_dir / "run.json", manifest)
    records.sort(key=lambda record: record["index"])
    write_json(out_dir / "results.json", records)
    counts = dict(Counter(record["verdict"] for record in records))
    passed = len(records) == len(prompts) and counts.get("ok", 0) == len(prompts)
    manifest.update(ended_at=now(), partial=False, counts=counts, **{"pass": passed})
    manifest["final_repetition_flags"] = sum(record.get("final_repetition", {}).get("flag", False) for record in records)
    manifest["reasoning_repetition_flags"] = sum(record.get("reasoning_repetition", {}).get("flag", False) for record in records)
    manifest["empty_final_count"] = sum(record.get("empty_final", False) for record in records)
    manifest["capped_count"] = sum(record.get("capped", False) for record in records)
    manifest["quality_preservation_established"] = False
    write_json(out_dir / "run.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--request-config", required=True, type=Path)
    parser.add_argument("--prompts-source", type=Path, default=Path(__file__).resolve().parents[1] / "box/quality20.py")
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.request_config.read_text(encoding="utf-8-sig"))
        result = run_probe(model=args.model, base_url=args.base_url, out_dir=args.out_dir,
                           request_config=config, prompts_source=args.prompts_source,
                           base_seed=args.base_seed, concurrency=args.concurrency, timeout=args.timeout)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
