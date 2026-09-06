"""Reject mismatched/incomplete runs and report paired 403-case quality changes."""
import argparse
import json
import math
from pathlib import Path
from run import canonical_hash, digest, read_jsonl, write_json


def load(prefix):
    prefix = str(prefix)
    meta = json.loads(Path(prefix + ".audit.json").read_text(encoding="utf-8"))
    if not meta["complete"] or not meta["sandbox"]["passed"]:
        raise ValueError("Run is incomplete or failed sandbox preflight: " + prefix)
    for kind, suffix in (("items", ".items.jsonl"), ("requests", ".requests.jsonl")):
        if digest(prefix + suffix) != meta[kind + "_sha256"]:
            raise ValueError("Artifact hash mismatch: " + prefix + suffix)
    rows = read_jsonl(prefix + ".items.jsonl")
    items = {r["id"]: r for r in rows}
    cases = meta["suite"]["case_families"]
    if len(rows) != 403 or set(items) != set(cases) or any(r["status"] in ("error", "cancelled", "skipped") for r in rows):
        raise ValueError("Run must have all 403 unique cases without infrastructure failures")
    requests = {}
    for row in read_jsonl(prefix + ".requests.jsonl"):
        key = (row["id"], row["turn"])
        if key in requests or row["sha256"] != canonical_hash(row["body"]):
            raise ValueError("Request ledger is corrupt")
        requests[key] = row["sha256"]
    if set(requests) != {(i, 0) for i in cases}:
        raise ValueError("Expected exactly 403 attributed requests")
    return meta, items, requests


def compare(baseline, optimized):
    a, ar, aq = load(baseline)
    b, br, bq = load(optimized)
    for key in ("configuration", "model", "model_revision", "suite", "adapter_files"):
        if a[key] != b[key]:
            raise ValueError("Runs differ in " + key)
    if aq != bq:
        raise ValueError("Request bodies differ; optimized run must replay baseline requests")
    families = {}
    for family in a["configuration"]["families"]:
        ids = sorted(i for i, f in a["suite"]["case_families"].items() if f == family)
        regressions = [i for i in ids if ar[i]["correct"] and not br[i]["correct"]]
        improvements = [i for i in ids if not ar[i]["correct"] and br[i]["correct"]]
        n = len(regressions) + len(improvements)
        p = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(len(regressions), len(improvements)) + 1)) / 2**n) if n else 1.0
        def count(rows, field):
            return sum(field(rows[i]) for i in ids)
        families[family] = {"n": len(ids), "baseline_correct": count(ar, lambda r: bool(r["correct"])),
                            "optimized_correct": count(br, lambda r: bool(r["correct"])),
                            "correct_delta": len(improvements) - len(regressions), "regressions": regressions,
                            "improvements": improvements, "mcnemar_exact_two_sided_p": p,
                            "baseline_length_finishes": count(ar, lambda r: r.get("finish_reason") == "length"),
                            "optimized_length_finishes": count(br, lambda r: r.get("finish_reason") == "length"),
                            "baseline_reasoning_rescues": count(ar, lambda r: "answer_from_reasoning" in r.get("flags", [])),
                            "optimized_reasoning_rescues": count(br, lambda r: "answer_from_reasoning" in r.get("flags", []))}
    return {"matched_complete_comparison": True, "case_count": 403, "request_bodies_identical": True,
            "baseline": str(baseline), "optimized": str(optimized), "families": families,
            "baseline_correct": sum(r["correct"] for r in ar.values()), "optimized_correct": sum(r["correct"] for r in br.values()),
            "quality_parity_certified": False,
            "interpretation": "Original scorer retained. Aggregate equality or a nonsignificant paired test does not establish capability equivalence. Review family regressions and the separate strict quality gate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Prefix without .audit.json")
    parser.add_argument("optimized", help="Prefix without .audit.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.optimized)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
