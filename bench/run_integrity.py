#!/usr/bin/env python3
"""Fail closed before ranking serving runs or deriving costs (Python stdlib only).

validate_run(bench, meta) accepts parsed shard dictionaries. The optional ``_file``
key identifies a shard by its ``__pPORT.json`` suffix. Missing evidence is
``unknown``, not a fabricated pass or a proven failure. This checks measurement
integrity only; a valid run still requires a separate task-quality evaluation.

CLI: python bench/run_integrity.py --meta RUN.meta.json [--out CHECK.json] SHARD...
Exit 0: verified; 1: invalid or unknown; 2: command-line usage error.
"""
import argparse
import json
import math
import os
import re


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _integer(value):
    result = _number(value)
    return int(result) if result is not None and result.is_integer() else None


def _flag(value):
    if value is True or str(value).lower() in ("1", "true", "yes", "on"):
        return True
    if value is False or str(value).lower() in ("0", "false", "no", "off"):
        return False
    return None


def _ports(value):
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value]
    return [p for p in re.split(r"[,\s]+", str(value).strip()) if p]


def validate_run(bench, meta):
    """Return validity, explicit reasons, and independent cache-evidence status."""
    errors, unknowns = [], []
    if not isinstance(meta, dict):
        errors.append("metadata is not a JSON object")
        meta = {}
    if meta.get("_load_error"):
        errors.append("metadata could not be read: " + str(meta["_load_error"]))
    if not meta:
        unknowns.append("run metadata missing")
    if not isinstance(bench, list):
        errors.append("benchmark shards are not a list")
        bench = []
    if not bench:
        errors.append("no benchmark result shards")

    expected_ports = _ports(meta.get("ports"))
    if not expected_ports:
        unknowns.append("expected shard ports missing")
    elif len(set(expected_ports)) != len(expected_ports):
        errors.append("duplicate expected ports in metadata")
    if expected_ports and len(bench) != len(expected_ports):
        errors.append("shard count mismatch: expected %d, found %d" % (len(expected_ports), len(bench)))

    rc = _integer(meta.get("bench_exit_code"))
    if rc is None:
        unknowns.append("benchmark client exit code missing or invalid")
    elif rc != 0:
        errors.append("benchmark client exited %d" % rc)
    exit_codes = meta.get("bench_exit_codes", []) or []
    if not isinstance(exit_codes, (list, tuple)):
        errors.append("shard client exit codes are not a list")
        exit_codes = []
    for value in exit_codes:
        shard_rc = _integer(value)
        if shard_rc is None or shard_rc != 0:
            errors.append("a shard client exit code is nonzero or invalid")

    cache_policy = meta.get("cache_policy")
    cache_verified = _flag(meta.get("cache_reset_verified"))
    if cache_policy == "reset" and cache_verified is True:
        cache_status = "verified"
    elif cache_policy == "disabled" and _flag(meta.get("prefix_cache_enabled")) is False:
        cache_status = "verified"
    elif cache_policy == "reset" and cache_verified is False:
        cache_status = "invalid"
        errors.append("requested prefix-cache reset was not verified")
    else:
        cache_status = "unknown"
        unknowns.append("cross-run prefix-cache state is unverified")

    fixed_out = _integer(meta.get("out_len"))
    expected_total_output = _integer(meta.get("expected_total_output_tokens"))
    ignore_eos = _flag(meta.get("ignore_eos"))
    if ignore_eos is not True:
        unknowns.append("ignore-eos fixed-output behavior is unverified")
    if (fixed_out is None or fixed_out <= 0) and (expected_total_output is None or expected_total_output <= 0):
        unknowns.append("expected generated-token count missing (variable outputs need expected_total_output_tokens)")

    seen_ports, seen_files = [], []
    total_completed = total_requested = total_output = 0
    for index, shard in enumerate(bench):
        label = "shard %d" % index
        if not isinstance(shard, dict):
            errors.append(label + " is not a JSON object")
            continue
        filename = shard.get("_file")
        if filename:
            label = os.path.basename(str(filename))
            seen_files.append(str(filename))
        if shard.get("_load_error"):
            errors.append(label + " could not be read: " + str(shard["_load_error"]))
            continue
        port = shard.get("replica_port", shard.get("port"))
        match = re.search(r"__p(\d+)\.json$", str(filename or ""))
        if match:
            if port is not None and str(port) != match.group(1):
                errors.append(label + " filename and port metadata disagree")
            port = match.group(1)
        if port is not None:
            seen_ports.append(str(port))
        elif len(expected_ports) > 1:
            unknowns.append(label + " replica port is unidentified")

        completed = _integer(shard.get("completed"))
        requested = _integer(shard.get("num_prompts"))
        failed = _integer(shard.get("failed"))
        if completed is None or completed <= 0:
            errors.append(label + " has no positive completed-request count")
        if requested is None or requested <= 0:
            unknowns.append(label + " requested count missing or invalid")
        if failed is None:
            unknowns.append(label + " failure count missing or invalid")
        elif failed != 0:
            errors.append(label + " reports %d failed requests" % failed)
        if completed is not None and requested is not None and completed != requested:
            errors.append(label + " completed %d/%d requests" % (completed, requested))
        if shard.get("errors") or shard.get("error"):
            errors.append(label + " contains benchmark errors")
        shard_rc = shard.get("bench_exit_code", shard.get("exit_code"))
        if shard_rc is not None and _integer(shard_rc) != 0:
            errors.append(label + " has a nonzero or invalid client exit code")

        generated = _integer(shard.get("total_output_tokens"))
        if generated is None or generated <= 0:
            errors.append(label + " has no positive generated-token count")
        elif completed is not None and fixed_out is not None and fixed_out > 0 and generated != completed * fixed_out:
            errors.append(label + " generated %d tokens; expected exactly %d" % (generated, completed * fixed_out))
        duration = _number(shard.get("duration"))
        if duration is None or duration <= 0:
            errors.append(label + " duration is missing, nonfinite or nonpositive")
        for field in ("request_throughput", "output_throughput", "total_token_throughput"):
            value = _number(shard.get(field))
            if value is None or value <= 0:
                errors.append(label + " " + field + " is missing, nonfinite or nonpositive")
        total_completed += completed or 0
        total_requested += requested or 0
        total_output += generated or 0

    if len(seen_files) != len(set(seen_files)):
        errors.append("duplicate benchmark files")
    if len(seen_ports) != len(set(seen_ports)):
        errors.append("duplicate benchmark ports")
    if expected_ports and seen_ports:
        unexpected = sorted(set(seen_ports) - set(expected_ports))
        missing = sorted(set(expected_ports) - set(seen_ports))
        if unexpected:
            errors.append("unexpected benchmark ports: " + ",".join(unexpected))
        if missing and len(expected_ports) > 1:
            errors.append("missing benchmark ports: " + ",".join(missing))
    expected_requested = _integer(meta.get("num_prompts"))
    if expected_requested is None or expected_requested <= 0:
        unknowns.append("total requested count missing from metadata")
    elif total_requested != expected_requested or total_completed != expected_requested:
        errors.append("run totals differ from metadata: completed=%d, requested=%d, expected=%d" %
                      (total_completed, total_requested, expected_requested))
    if expected_total_output is not None and total_output != expected_total_output:
        errors.append("run generated %d tokens; metadata expects %d" % (total_output, expected_total_output))
    errors = list(dict.fromkeys(errors))
    unknowns = list(dict.fromkeys(unknowns))
    status = "invalid" if errors else "unknown" if unknowns else "valid"
    valid = status == "valid"
    return {"valid": valid, "status": status, "reasons": errors + unknowns,
            "errors": errors, "unknowns": unknowns, "cache_status": cache_status,
            "headline_eligible": valid, "cost_eligible": valid,
            "completed": total_completed, "requested": total_requested,
            "total_output_tokens": total_output, "shards": len(bench)}


def load_result(path):
    """Retain parse failures as diagnostic shards instead of silently dropping them."""
    try:
        with open(path, encoding="utf-8-sig") as handle:
            result = json.load(handle)
        if not isinstance(result, dict):
            raise ValueError("expected a JSON object")
        result["_file"] = os.path.basename(path)
        return result
    except (OSError, ValueError) as exc:
        return {"_file": os.path.basename(path), "_load_error": str(exc)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out")
    parser.add_argument("results", nargs="*")
    args = parser.parse_args(argv)
    meta = load_result(args.meta)
    result = validate_run([load_result(path) for path in args.results], meta)
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(output)
    print(output, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
