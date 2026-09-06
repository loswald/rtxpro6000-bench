#!/usr/bin/env python3
"""pick_best.py <probe-dir-glob> <label> <concurrency> [exclude-substring]

Print the fastest tag whose matching 20-item corruption tripwire passed completely.
This nominates a layout for a full quality run; it does not certify a deployment or
an economic result. Use bench/select_candidate.py for raw-result integrity checks.
"""
import csv
import glob
import json
import math
import os
import sys


def quality_passes(directory):
    """Fail closed on missing, partial, malformed, or failed matching smoke data."""
    try:
        with open(directory.rstrip(os.sep) + "_quality20.json", encoding="utf-8-sig") as handle:
            items = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(items, list) or len(items) != 20:
        return False
    prompts = []
    for item in items:
        if not isinstance(item, dict) or item.get("verdict") != "ok":
            return False
        if item.get("finish") not in ("stop", "length"):
            return False
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            return False
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            return False
        prompts.append(item["prompt"])
    return len(set(prompts)) == 20


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    pattern, label, conc = args[:3]
    exclude = args[3] if len(args) > 3 else None
    best, best_v = None, -1.0
    for directory in sorted(glob.glob(pattern)):
        if not os.path.isdir(directory):
            continue
        tag = os.path.basename(directory)
        if (exclude and exclude in tag) or not quality_passes(directory):
            continue
        for filename in sorted(glob.glob(os.path.join(directory, "summary*.tsv"))):
            with open(filename, encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if row.get("label") != label or str(row.get("C")) != str(conc):
                        continue
                    try:
                        value = float(row["out_tps"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    if not math.isfinite(value) or value <= 0:
                        continue
                    if value > best_v:
                        best, best_v = tag, value
    if best:
        print(best)
        print("%.0f" % best_v, file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
