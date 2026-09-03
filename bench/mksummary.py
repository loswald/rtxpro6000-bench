#!/usr/bin/env python3
"""Build results/summary_all.tsv from every per-tag summary the node's agg.py wrote."""
import glob, os, csv
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
files = sorted(glob.glob(f"{root}/probe/*/summary*.tsv")) + sorted(glob.glob(f"{root}/5090/probe/*/summary*.tsv"))
rows, header = [], None
for f in files:
    tag = os.path.basename(os.path.dirname(f))
    host = "5090x8" if "/5090/" in f.replace(os.sep, "/") else ("pro6000-ws400w" if tag.startswith(("c6_", "f2_", "r3_")) else "pro6000-s600w")
    with open(f, newline="") as fh:
        rd = list(csv.reader(fh, delimiter="\t"))
    if not rd:
        continue
    h, body = rd[0], rd[1:]
    tagged = h[0].lower() in ("tag", "model", "run")
    if header is None:
        header = (h if tagged else ["tag"] + h) + ["host"]
    for r in body:
        if not r or not r[0].strip():
            continue
        rows.append((r if tagged else [tag] + r) + [host])
out = f"{root}/summary_all.tsv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
print(f"{len(rows)} rows from {len(files)} files -> summary_all.tsv")
print("header:", header[:10])
tags = sorted({r[0] for r in rows})
print(f"{len(tags)} tags:", ", ".join(tags[:48]))
