#!/usr/bin/env python3
"""
gates/hwdecisions.py -- the shared hardware / decision contract, read-only helper.

vast/hardware_truth.sh WRITES  $HW_DIR/decisions.env  (HW_DIR defaults to $RESULTS_ROOT/hw),
bench/env.sh SOURCES it, and every gate / co-tenancy / sleep-wake JSON carries the two
flags derived from it so summaries can dagger TP2/TP4 rows (and never TP1 replica rows).

decisions.env is a plain KEY=VALUE file:
  P2P_OK=1             peer access supported on all pairs (bandwidth/latency are recorded, not
                       used to disable P2P; NCCL_P2P_DISABLE is NEVER auto-set on this box)
  CUSTOM_ALLREDUCE=0   vLLM custom all-reduce off by default (CUSTOM_ALLREDUCE=1 for an A/B)
  NCCL_P2P_DISABLE=0   only ever 1 by explicit human decision (peer access unsupported)
  ACS_SUSPECTED=1      PCIe ACS on the host: switch-local P2P is redirected through the root
                       complex (pair 0-1 ~21 GB/s vs cross-switch 0-2 ~38 GB/s all_reduce busbw)
  PESSIMISTIC_TP=1     TP2/TP4 (incl. DP-over-TP) throughput is a pessimistic lower bound
  HOST_RAM_GB=1500
  NOTES="free text"

Precedence: environment (bench/env.sh already sourced the file, or a human override) wins,
the file fills the gaps.  Missing key -> None (unknown), never a silent default.

Library:
    sys.path.insert(0, "<repo>/gates"); from hwdecisions import hw_decisions, pessimistic_flags
    flags = pessimistic_flags(hw_decisions(), tp=2)   # -> {"acs_suspected": True, "pessimistic_tp": True, ...}
CLI:
    python3 gates/hwdecisions.py [--tp N] [--results-root DIR] [--json | --shell]
    eval "$(python3 gates/hwdecisions.py --shell --tp "$TP")"   # HW_ACS_SUSPECTED=1 HW_PESSIMISTIC_THIS=1 ...
Python 3.8+ stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

KEYS = ("P2P_OK", "CUSTOM_ALLREDUCE", "NCCL_P2P_DISABLE", "ACS_SUSPECTED", "PESSIMISTIC_TP", "HOST_RAM_GB", "NOTES")
FLAG_KEYS = ("P2P_OK", "CUSTOM_ALLREDUCE", "NCCL_P2P_DISABLE", "ACS_SUSPECTED", "PESSIMISTIC_TP")


def default_results_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("RESULTS_ROOT") or os.path.join(os.path.dirname(here), "results")


def decisions_path(results_root: str | None = None, hw_dir: str | None = None) -> str:
    hw_dir = hw_dir or os.environ.get("HW_DIR") or os.path.join(results_root or default_results_root(), "hw")
    return os.path.join(hw_dir, "decisions.env")


def read_decisions_env(path: str) -> dict:
    """Parse KEY=VALUE lines (optional `export `, quotes, # comments).  Missing file -> {}."""
    out: dict = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                if k.startswith("export "):
                    k = k[len("export "):].strip()
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                if k:
                    out[k] = v
    except OSError:
        pass
    return out


def to_flag(v):
    """'1'/'true'/'yes' -> True, '0'/'false'/'no'/'' -> False, None -> None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return None


def hw_decisions(results_root: str | None = None, hw_dir: str | None = None) -> dict:
    """Merged view: env > decisions.env.  Adds `_source` (file path or None) and `_present` (bool)."""
    path = decisions_path(results_root, hw_dir)
    filed = read_decisions_env(path)
    # bench/env.sh exports normalised defaults (ACS_SUSPECTED=0, PESSIMISTIC_TP=0, ...) even when the
    # file is missing and says so with HW_DECISIONS_SOURCE=missing; without a file the flags are
    # UNKNOWN here, not 0, so summaries show '?' instead of silently dropping the dagger.
    env_is_default_only = os.environ.get("HW_DECISIONS_SOURCE") == "missing" and not filed
    merged: dict = {}
    for k in KEYS:
        env_v = None if env_is_default_only else os.environ.get(k)
        if k == "NOTES" and env_v in (None, ""):
            env_v = os.environ.get("HW_NOTES")          # bench/env.sh re-exports the file's NOTES as HW_NOTES
        if env_v not in (None, ""):
            merged[k] = env_v
        elif k in filed:
            merged[k] = filed[k]
    # bench/env.sh may have read a differently located file (HW_DECISIONS_FILE / HW_DECISIONS_SOURCE)
    src = path if filed else (os.environ.get("HW_DECISIONS_SOURCE") or None)
    if src and not os.path.isfile(src):
        src = None
    merged["_source"] = src
    merged["_present"] = bool(filed) or bool(src)
    return merged


def pessimistic_flags(dec: dict | None = None, tp=None, dp=None, replicas=None) -> dict:
    """The fields every result JSON should carry.  pessimistic_tp is True only for TP>1 cells
    (TP2/TP4, DP-over-TP) when PESSIMISTIC_TP=1; TP1 replica cells are False; unknown -> None."""
    dec = dec if dec is not None else hw_decisions()
    pess = to_flag(dec.get("PESSIMISTIC_TP"))
    try:
        tp_i = int(tp) if tp not in (None, "") else None
    except (TypeError, ValueError):
        tp_i = None
    if tp_i is not None and tp_i <= 1:
        pessimistic_tp = False
    elif pess is True:
        pessimistic_tp = True if tp_i is not None else None   # TP unknown: cannot say
    elif pess is False:
        pessimistic_tp = False
    else:
        pessimistic_tp = None
    ram = dec.get("HOST_RAM_GB")
    try:
        ram = float(ram) if ram not in (None, "") else None
        if ram is not None and ram.is_integer():
            ram = int(ram)
    except (TypeError, ValueError):
        pass
    return {
        "acs_suspected": to_flag(dec.get("ACS_SUSPECTED")),
        "pessimistic_tp": pessimistic_tp,
        "p2p_ok": to_flag(dec.get("P2P_OK")),
        "custom_allreduce": to_flag(dec.get("CUSTOM_ALLREDUCE")),
        "nccl_p2p_disable": to_flag(dec.get("NCCL_P2P_DISABLE")),
        "host_ram_gb": ram,
        "tp": tp_i,
        "dp": _int_or_none(dp),
        "replicas": _int_or_none(replicas),
        "hw_decisions_file": dec.get("_source"),
        "hw_notes": dec.get("NOTES"),
    }


def _int_or_none(v):
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _shell_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tp", default=None, help="tensor-parallel size of the cell (for pessimistic_tp)")
    ap.add_argument("--dp", default=None)
    ap.add_argument("--replicas", default=None)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--hw-dir", default=None)
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="JSON (default)")
    fmt.add_argument("--shell", action="store_true", help="HW_*=value lines for eval")
    a = ap.parse_args(argv)
    dec = hw_decisions(a.results_root, a.hw_dir)
    flags = pessimistic_flags(dec, a.tp, a.dp, a.replicas)
    if a.shell:
        lines = [
            f"HW_DECISIONS_FILE={shlex.quote(_shell_value(dec.get('_source')))}",
            f"HW_DECISIONS_PRESENT={'1' if dec.get('_present') else '0'}",
        ]
        for k in FLAG_KEYS:
            lines.append(f"HW_{k}={shlex.quote(_shell_value(to_flag(dec.get(k))))}")
        lines.append(f"HW_HOST_RAM_GB={shlex.quote(_shell_value(flags['host_ram_gb']))}")
        lines.append(f"HW_NOTES={shlex.quote(_shell_value(dec.get('NOTES')))}")
        lines.append(f"HW_PESSIMISTIC_THIS={shlex.quote(_shell_value(flags['pessimistic_tp']))}")
        print("\n".join(lines))
    else:
        print(json.dumps({"decisions": {k: v for k, v in dec.items() if not k.startswith("_")},
                          "source": dec.get("_source"), **flags}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
