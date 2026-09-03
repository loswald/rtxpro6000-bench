#!/usr/bin/env python3
"""
prepare_data.py - download the public sources and build the concentrated item sets once.

  python3 evalsuite/prepare_data.py [--data-dir evalsuite/data] [--seed 20260903] [--profile default|full]
      [--only math,code,tools,longctx,knowledge,ifeval] [--refresh] [--allow-short] [--pin]
      [--opt [FAMILY.]KEY=VALUE ...] [--offline-fixtures DIR] [--list]

For every requested family this calls `families/<name>.py::prepare(data_dir, seed, profile, refresh, log,
allow_short, **opts)`, which fetches its raw sources into data/raw/<source>/ (cached, sha256-recorded) and
writes data/items/<name>.jsonl (sorted keys, seeded order -> byte-identical rebuilds).  The item files are
then hashed into data/manifest.json, which run_eval.py verifies before every run and records in its output
(so two runs can be paired only when they used the same items).

Deterministic: all randomness goes through common.seeded_rng(seed, family, subfamily).  Idempotent: cached
sources are reused unless --refresh.  Exits non-zero when any family failed or a pool was smaller than
requested (unless --allow-short).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import traceback
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from common import DEFAULT_DATA_DIR, DEFAULT_SEED, ShortPool  # noqa: E402
import families as famreg  # noqa: E402

SOURCES_JSON = os.path.join(common.EVALSUITE_DIR, "sources.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="build the evalsuite item sets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--profile", choices=["default", "full"], default="default")
    p.add_argument("--only", default=None, help="comma-separated families (default: all non-hidden families)")
    p.add_argument("--refresh", action="store_true", help="re-download cached sources")
    p.add_argument("--allow-short", action="store_true", help="accept pools smaller than requested")
    p.add_argument("--pin", action="store_true", help="write the resolved source revisions/hashes into evalsuite/sources.json")
    p.add_argument("--opt", action="append", default=[], metavar="[FAMILY.]KEY=VALUE",
                   help="family option, e.g. code.lcb_after=2025-02-01, knowledge.categories=law,engineering,history")
    p.add_argument("--offline-fixtures", default=None, help="directory of raw-source fixtures for offline builds (passed to families)")
    p.add_argument("--list", action="store_true", help="list families and exit")
    return p


def parse_opts(specs: list[str]) -> dict:
    out: dict = collections.defaultdict(dict)
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--opt expects [family.]key=value, got {spec!r}")
        k, v = spec.split("=", 1)
        fam, key = (k.split(".", 1) if "." in k else ("*", k))
        out[fam.strip()][key.strip()] = common.json_or_str(v.strip())
    return dict(out)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for name in famreg.discover():
            mod = famreg.load(name)
            print(f"{name:12s} hidden={mod.HIDDEN} subs={mod.SUBFAMILIES} - {mod.DESCRIPTION}")
        return 0
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(os.path.join(data_dir, "items"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
    names = [n.strip() for n in args.only.split(",") if n.strip()] if args.only else famreg.default_families()
    opts_all = parse_opts(args.opt)

    manifest = common.load_manifest(data_dir) or {"schema_version": common.SCHEMA_VERSION, "sources": {}, "items": {}, "counts": {}}
    manifest.setdefault("sources", {}); manifest.setdefault("items", {}); manifest.setdefault("counts", {})
    manifest.update({"built_at": common.now_iso(), "seed": args.seed, "profile": args.profile,
                     "evalsuite_git_sha": common.evalsuite_git_sha()})

    failures: list[tuple[str, str]] = []
    for name in names:
        try:
            mod = famreg.load(name, require=("prepare",))
        except Exception as e:
            log(f"== {name}: cannot load: {e}")
            failures.append((name, f"load: {e}"))
            continue
        opts = dict(opts_all.get("*", {}))
        opts.update(opts_all.get(name, {}))
        if args.offline_fixtures:
            opts["offline_fixtures"] = os.path.abspath(args.offline_fixtures)
        log(f"== {name}: prepare(profile={args.profile}, seed={args.seed}, refresh={args.refresh}, opts={opts})")
        t0 = time.time()
        try:
            res = mod.prepare(data_dir, seed=args.seed, profile=args.profile, refresh=args.refresh, log=log,
                              allow_short=args.allow_short, **opts) or {}
        except ShortPool as e:
            log(f"   {name}: pool smaller than requested: {e} (use --allow-short to accept)")
            failures.append((name, f"short pool: {e}"))
            continue
        except Exception as e:
            log(f"   {name}: FAILED: {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}")
            failures.append((name, f"{type(e).__name__}: {e}"))
            continue
        rel = res.get("file") or f"items/{name}.jsonl"
        path = os.path.join(data_dir, rel)
        if not os.path.exists(path):
            failures.append((name, f"prepare() returned but {path} does not exist"))
            log(f"   {name}: FAILED: {path} missing")
            continue
        rows = common.read_jsonl(path)
        items = [r for r in rows if "id" in r]
        ids = [r["id"] for r in items]
        if len(set(ids)) != len(ids):
            failures.append((name, "duplicate item ids"))
            log(f"   {name}: FAILED: duplicate ids")
            continue
        counts = res.get("counts") or dict(sorted(collections.Counter(r.get("subfamily", name) for r in items).items()))
        manifest["items"][name] = {"file": rel.replace("\\", "/"), "n": len(ids), "sha256": common.sha256_file(path), "ids": ids}
        manifest["counts"][name] = counts
        manifest["sources"].update(res.get("sources") or {})
        manifest.update(res.get("manifest_extra") or {})
        pools = res.get("pools") or {}
        for sub, n in counts.items():
            pool = pools.get(sub)
            log(f"   {name}/{sub}: n={n}" + (f" (pool {pool})" if pool is not None else ""))
        for note in res.get("notes") or []:
            log(f"   note: {note}")
        log(f"   {name}: {len(ids)} items -> {path} ({time.time() - t0:.1f}s)")

    manifest["manifest_sha256"] = common.manifest_digest(manifest)
    common.atomic_write_text(os.path.join(data_dir, "manifest.json"), json.dumps(manifest, indent=1, sort_keys=True))
    log(f"manifest: {os.path.join(data_dir, 'manifest.json')} sha256={manifest['manifest_sha256'][:16]}.. "
        f"families={sorted(manifest['items'])}")

    if args.pin:
        pinned = {}
        if os.path.exists(SOURCES_JSON):
            with open(SOURCES_JSON, "r", encoding="utf-8") as f:
                pinned = json.load(f)
        pinned.update(manifest["sources"])
        common.atomic_write_text(SOURCES_JSON, json.dumps(pinned, indent=1, sort_keys=True))
        log(f"pinned {len(manifest['sources'])} sources into {SOURCES_JSON}")

    if failures:
        log("FAILED: " + "; ".join(f"{n}: {why}" for n, why in failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
