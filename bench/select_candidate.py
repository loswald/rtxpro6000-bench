#!/usr/bin/env python3
"""Choose the next configuration to evaluate; never deploy or certify a winner.

Raw benchmark JSON must have all requests, exact fixed outputs, and no errors.
Every item in its corruption/format smoke check must explicitly have verdict='ok'.
The default also requires run_integrity provenance. --legacy-screen permits only
unknown provenance to nominate archived runs for a NEW measurement and full
paired quality evaluation. A smoke pass cannot establish capability preservation.

Example: python bench/select_candidate.py --probe-dir results/probe --shape router
  --concurrency 1024 --output-tokens 128 --legacy-screen --out candidate.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

try:
    from .run_integrity import load_result, validate_run
except ImportError:
    from run_integrity import load_result, validate_run


def smoke_check(items, expected_items=20):
    reasons = []
    if not isinstance(items, list) or len(items) != expected_items:
        reasons.append("expected exactly %d smoke items" % expected_items)
        return {"pass": False, "reasons": reasons}
    prompts = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            reasons.append("smoke item %d is malformed" % index)
            continue
        if item.get("verdict") != "ok":
            reasons.append("smoke item %d verdict is %r" % (index, item.get("verdict")))
        if not isinstance(item.get("prompt"), str) or not item["prompt"]:
            reasons.append("smoke item %d has no prompt" % index)
        else:
            prompts.append(item["prompt"])
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            reasons.append("smoke item %d has no response" % index)
        if item.get("finish") not in ("stop", "length"):
            reasons.append("smoke item %d has a failed or unknown completion" % index)
    if len(set(prompts)) != len(prompts):
        reasons.append("duplicate smoke prompts")
    return {"pass": not reasons, "reasons": reasons}


def screen_candidate(tag, shards, meta, smoke, output_tokens, expected_shards=1, legacy_screen=False):
    """Strict corruption prefilter for expensive paired evaluation, not promotion."""
    integrity = validate_run(shards, meta)
    checked_smoke = smoke_check(smoke)
    reasons = list(checked_smoke["reasons"])
    if len(shards) != expected_shards:
        reasons.append("expected %d benchmark shards, found %d" % (expected_shards, len(shards)))
    if integrity["errors"]:
        reasons.extend(integrity["errors"])
    if not integrity["valid"] and not legacy_screen:
        reasons.extend(integrity["unknowns"])
    for index, item in enumerate(shards):
        if not isinstance(item, dict):
            continue
        requested, completed = item.get("num_prompts"), item.get("completed")
        if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0 or completed != requested:
            reasons.append("shard %d does not have all requested completions" % index)
        elif item.get("total_output_tokens") != requested * output_tokens:
            reasons.append("shard %d has incomplete or unexpected fixed-length output" % index)
        if item.get("failed") != 0:
            reasons.append("shard %d failure count is nonzero or missing" % index)
    rates = [item.get("output_throughput", 0) for item in shards if isinstance(item, dict)]
    try:
        rate = sum(float(value) for value in rates)
    except (ValueError, TypeError):
        rate = None
    # Invalid rates are already excluded by run_integrity; keep JSON finite.
    if rate is not None and (rate != rate or rate in (float("inf"), float("-inf"))):
        rate = None
    return {"tag": tag, "screening_eligible": not reasons, "output_tok_s": rate,
            "reasons": list(dict.fromkeys(reasons)), "integrity": integrity,
            "smoke": checked_smoke, "requires_new_integrity_run": not integrity["valid"],
            "quality_preservation_established": False, "promotion_eligible": False}


def select(candidates):
    eligible = [item for item in candidates if item["screening_eligible"]]
    best = max(eligible, key=lambda item: (item["output_tok_s"], item["tag"])) if eligible else None
    return {"status": "candidate_for_evaluation" if best else "no_eligible_candidate",
            "selected": best["tag"] if best else None,
            "selected_run": best.get("run_id") if best else None,
            "purpose": "Nominate a configuration for fresh throughput and paired capability evaluation only",
            "promotion_eligible": False, "candidates": candidates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--quality-dir", help="default: --probe-dir; files named TAG_quality20.json")
    parser.add_argument("--shape", required=True)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--output-tokens", required=True, type=int)
    parser.add_argument("--expected-shards", type=int, default=1)
    parser.add_argument("--legacy-screen", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if min(args.concurrency, args.output_tokens, args.expected_shards) <= 0:
        parser.error("concurrency, output-tokens and expected-shards must be positive")
    root = Path(args.probe_dir)
    quality = Path(args.quality_dir) if args.quality_dir else root
    candidates = []
    pattern = re.compile(r"__" + re.escape(args.shape) + r"__c" + str(args.concurrency) + r"(?:__|$)")
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        grouped = {}
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith((".meta.json", ".integrity.json", ".skipped.json")) or not pattern.search(path.stem):
                continue
            shard = load_result(str(path))
            run_id = shard.get("run_id") or re.sub(r"__p\d+$", "", path.stem)
            grouped.setdefault(run_id, []).append(shard)
        for run_id, shards in grouped.items():
            meta_path = directory / (run_id + ".meta.json")
            meta = load_result(str(meta_path)) if meta_path.exists() else {}
            smoke_path = quality / (directory.name + "_quality20.json")
            try:
                smoke = json.loads(smoke_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                smoke = None
            result = screen_candidate(directory.name, shards, meta, smoke, args.output_tokens,
                                      args.expected_shards, args.legacy_screen)
            result.update(run_id=run_id, benchmark_files=[str(directory / row["_file"]) for row in shards],
                          smoke_file=str(smoke_path),
                          smoke_sha256=hashlib.sha256(smoke_path.read_bytes()).hexdigest() if smoke_path.exists() else None)
            candidates.append(result)
    output = select(candidates)
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if output["selected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
