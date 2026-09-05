#!/usr/bin/env python3
"""Paired comparison of two quality runs on the items both scored: paired.py <A.items.jsonl> <B.items.jsonl> [labelA labelB]
Prints accuracy on the common items, the only-A-right / only-B-right split, and the per-family split, which is the
comparison the noise-floor table is read against (identical runs split up to 11-to-20)."""
import json, sys, collections
def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        it = json.loads(line)
        k = it.get("id") or it.get("item_id") or it.get("qid")
        if k is None: continue
        d[k] = (bool(it.get("correct")), it.get("family", "?"), it.get("status") or ("truncated" if it.get("finish_reason") == "length" else ""))
    return d
A, B = load(sys.argv[1]), load(sys.argv[2])
la, lb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("A", "B")
common = sorted(set(A) & set(B))
onlyA = [k for k in common if A[k][0] and not B[k][0]]
onlyB = [k for k in common if B[k][0] and not A[k][0]]
accA = sum(A[k][0] for k in common) / len(common); accB = sum(B[k][0] for k in common) / len(common)
print(f"{len(common)} common items: {la} {accA:.3f}, {lb} {accB:.3f}, gap {accB-accA:+.3f}; only {la} right {len(onlyA)}, only {lb} right {len(onlyB)}")
fam = collections.defaultdict(lambda: [0, 0, 0])
for k in common:
    f = A[k][1]; fam[f][2] += 1
    if k in onlyA: fam[f][0] += 1
    if k in onlyB: fam[f][1] += 1
print("  per family (only-%s / only-%s / n): " % (la, lb) + ", ".join(f"{f} {v[0]}/{v[1]}/{v[2]}" for f, v in sorted(fam.items())))
tr = collections.Counter(B[k][2] for k in onlyA); print(f"  status of {lb} on the items only {la} got right: {dict(tr)}")
