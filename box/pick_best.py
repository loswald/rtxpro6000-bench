#!/usr/bin/env python3
"""pick_best.py <probe-dir-glob> <label> <concurrency> [exclude-substring]

Prints the tag with the highest output tokens/s for the given shape among the probe directories matching the glob
(each holds the summary*.tsv that agg.py writes). Used by the chains to hand the fastest layout of a sweep to a
quality run, so throughput and quality are measured on the same configuration.
"""
import csv, glob, os, sys

pattern, label, conc = sys.argv[1], sys.argv[2], sys.argv[3]
exclude = sys.argv[4] if len(sys.argv) > 4 else None
best, best_v = None, -1.0
for d in glob.glob(pattern):
    if not os.path.isdir(d):
        continue
    tag = os.path.basename(d)
    if exclude and exclude in tag:
        continue
    for f in glob.glob(os.path.join(d, "summary*.tsv")):
        for r in csv.DictReader(open(f, encoding="utf-8"), delimiter="\t"):
            if r.get("label") != label or str(r.get("C")) != str(conc):
                continue
            try:
                v = float(r["out_tps"])
            except (KeyError, ValueError):
                continue
            if v > best_v:
                best, best_v = tag, v
if best:
    print(best)
    print("%.0f" % best_v, file=sys.stderr)
else:
    sys.exit(1)
