#!/usr/bin/env python3
"""Build results/summary_all.tsv from every per-tag summary in the results tree.

Host attribution is load-bearing here, because only the 600 W Server Edition numbers transfer to a
purchase: the replacement box caps the same silicon at 400 W and loses 23-34%, and the 5090 box is a
different product entirely. Earlier versions guessed the host from a tag prefix and silently skipped the
recovered 600 W tree altogether, which is how both compressed-tensors quantiser builds went missing from
the comparison they were run for.

Three trees, three hosts:
  results/600w/probe   the 4x RTX PRO 6000 Server Edition, 600 W (the original box)
  results/600w2/probe  the second Server Edition box, 600 W (rented 5 Sept when the 400 W box could not restart)
  results/5090/probe   the 8x RTX 5090 box                                 (destroyed; results kept)
  results/probe        pulled from whichever PRO 6000 box was current, so it holds BOTH the original 600 W
                       runs (before the outage) and the 400 W replacement's runs (after). Split by tag.

The per-tag summary files do not all share one header: agg.py grew columns over the campaign (summary.tsv
with 10, summary_full.tsv with 18). The first version of this script took the FIRST file's header as the
header for all of them, so every row from a wider file was written mis-aligned - its host label landed in a
latency column and a 400 W quantisation-ladder row read as a 600 W one. Each file is now read with its own
header and written into the union of columns, blanks where a file lacks one.

Runs present in more than one tree are counted once, preferring the copy under its own host's tree.
"""
import csv
import glob
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Tags measured on the original 600 W Server Edition box before it was stopped on 3 Sept. Everything else
# under results/probe was produced by the 400 W Workstation replacement.
SERVER_600W_PREFIXES = (
    "n2_", "n_", "q27_base", "qwen27b_x4", "gptoss_x4", "full_gptoss", "sw_gptoss", "ds4flash",
    "ds_marlin_ep", "gen5_", "moe_", "pr_", "tune_", "vglm_", "sglglm_", "vllm_dp4", "sat_marlin",
)


def host_for(path: str, tag: str) -> str:
    p = path.replace(os.sep, "/")
    if "/5090/" in p:
        return "5090x8"
    if "/600w/" in p or "/600w2/" in p:
        return "pro6000-s600w"
    return "pro6000-s600w" if tag.startswith(SERVER_600W_PREFIXES) else "pro6000-ws400w"


def main() -> None:
    files = []
    for sub in ("600w/probe", "600w2/probe", "5090/probe", "probe"):
        files += sorted(glob.glob(os.path.join(ROOT, sub, "*", "summary*.tsv")))

    columns: list[str] = ["tag"]
    records: list[dict] = []
    for f in files:
        tag = os.path.basename(os.path.dirname(f))
        host = host_for(f, tag)
        with open(f, newline="", encoding="utf-8") as fh:
            rd = [r for r in csv.reader(fh, delimiter="\t") if r]
        if not rd:
            continue
        head, body = rd[0], rd[1:]
        tagged = head[0].strip().lower() in ("tag", "model", "run")
        cols = head if tagged else ["tag"] + head
        for c in cols:
            if c not in columns:
                columns.append(c)
        for r in body:
            if not r or not r[0].strip():
                continue
            vals = r if tagged else [tag] + r
            rec = dict(zip(cols, vals))
            rec["host"] = host
            records.append(rec)
    if "host" not in columns:
        columns.append("host")

    # De-duplicate on the WHOLE row, not on (tag, shape, concurrency). A run reachable through two trees
    # produces byte-identical rows and must be counted once; a genuine repeat of the same configuration
    # produces different numbers and must be kept, because the spread between repeats is the only run-to-run
    # variance estimate this campaign has. FP8 at concurrency 1,024 was measured three times: 3,037 / 3,146 /
    # 3,148 out tok/s, a 3.6% spread worth not hiding.
    rows, seen = [], set()
    for rec in records:
        row = [rec.get(c, "") for c in columns]
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    out = os.path.join(ROOT, "summary_all.tsv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(columns)
        w.writerows(rows)

    hi = columns.index("host")
    by_host: dict[str, int] = {}
    for r in rows:
        by_host[r[hi]] = by_host.get(r[hi], 0) + 1
    print(f"{len(rows)} rows from {len(files)} files -> summary_all.tsv ({len(columns)} columns)")
    print("  by host:", ", ".join(f"{k} {v}" for k, v in sorted(by_host.items())))
    print(f"  {len({r[0] for r in rows})} distinct tags")


if __name__ == "__main__":
    main()
