#!/usr/bin/env bash
# search_offers.sh -- find VERIFIED 4x RTX PRO 6000 Blackwell machines on Vast.ai and print the columns we care about.
#
#   ./vast/search_offers.sh                 # default filters, sorted by $/h
#   ./vast/search_offers.sh --json          # raw JSON (for jq)
#   ./vast/search_offers.sh 'extra=filter'  # append/override query terms, e.g. 'geolocation in [GB,DE,NL,FR,SE,FI] inet_down_cost<0.01'
#
# Env overrides: GPU_NAMES (default RTX_PRO_6000_WS,RTX_PRO_6000_S -- Max-Q (300 W) deliberately excluded),
#                MIN_RAM_GB=256 MIN_DISK_GB=1200 MIN_INET_MBPS=1000 MIN_PORTS=4 MIN_DRIVER=595.0.0 MIN_CUDA=13.0 ORDER=dph_total
# Requires: vastai CLI with the api key set via 'vastai set api-key <KEY>' (the key never lives in this repo).
set -euo pipefail
have() { command -v "$1" >/dev/null 2>&1; }
have vastai || { echo "vastai CLI not found: pip install vastai && vastai set api-key <KEY>" >&2; exit 1; }

GPU_NAMES="${GPU_NAMES:-RTX_PRO_6000_WS,RTX_PRO_6000_S}"
Q="num_gpus=4 gpu_name in [${GPU_NAMES}] verified=true rentable=true rented=any gpu_frac=1"
Q="$Q cuda_vers>=${MIN_CUDA:-13.0} driver_version>=${MIN_DRIVER:-595.0.0} cpu_ram>=${MIN_RAM_GB:-256} disk_space>=${MIN_DISK_GB:-1200}"
Q="$Q inet_down>=${MIN_INET_MBPS:-1000} direct_port_count>=${MIN_PORTS:-4} duration>=${MIN_DAYS:-3}"
ORDER="${ORDER:-dph_total}"        # ascending $/h; use 'pcie_bw-,dph_total' to put the fastest PCIe first
RAW=0; EXTRA=""
for a in "$@"; do case "$a" in --json) RAW=1 ;; *) EXTRA="$EXTRA $a" ;; esac; done
Q="$Q$EXTRA"
echo "query: $Q" >&2; echo "order: $ORDER" >&2

if [ "$RAW" = 1 ]; then exec vastai search offers "$Q" -o "$ORDER" --raw; fi

# a *working* python (the Windows Store 'python3' alias exists but fails); uv can supply one
find_python() {
  local c; for c in python3 python "py -3"; do $c -c 'import sys' >/dev/null 2>&1 && { echo "$c"; return 0; }; done
  have uv && { echo "uv run --no-project -q python"; return 0; }; return 1
}
PY="$(find_python || true)"
if [ -z "$PY" ]; then
  echo "(no python/uv found for the pretty table; here is the CLI's own table)" >&2
  exec vastai search offers "$Q" -o "$ORDER"
fi
# The payload travels through a temp file: a here-doc replaces the script's stdin (shellcheck SC2259), so
# piping the JSON into `$PY -` hands python an empty stdin; an env var would be capped at ~128 KB.
JSON_FILE="$(mktemp "${TMPDIR:-/tmp}/vast_offers.XXXXXX")"
trap 'rm -f "$JSON_FILE"' EXIT
vastai search offers "$Q" -o "$ORDER" --raw > "$JSON_FILE"
$PY - "$JSON_FILE" <<'EOF'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    offers = json.load(f)
if not offers:
    print("no offers match -- relax filters (try MIN_INET_MBPS=500, MIN_PORTS=2, or drop duration>=3)"); sys.exit(0)
cols = [("id","id",8), ("$/h","dph_total",7), ("gpu","gpu_name",18), ("W","gpu_max_power",5), ("cpu","cpu_name",28), ("vcpu","cpu_cores_effective",5),
        ("ram_GB","cpu_ram",7), ("disk_GB","disk_space",8), ("inet_dn","inet_down",8), ("inet_up","inet_up",8), ("pcie_GBs","pcie_bw",8),
        ("gen","pci_gen",4), ("rel","reliability2",6), ("ports","direct_port_count",6), ("cuda","cuda_max_good",5), ("driver","driver_version",11),
        ("stor$/GBmo","storage_cost",10), ("dl$/GB","inet_down_cost",7), ("days","duration",6), ("geo","geolocation",4), ("verif","verification",9), ("mach","machine_id",7)]
def fmt(o, key):
    v = o.get(key)
    if v is None and key == "reliability2": v = o.get("reliability")
    if v is None: return "-"
    if key == "cpu_ram": return f"{v/1024:.0f}"
    if key in ("dph_total","storage_cost","inet_down_cost"): return f"{v:.3f}"
    if key in ("reliability2",): return f"{v:.3f}"
    if key in ("duration",): return f"{v/86400:.0f}"
    if key in ("disk_space","inet_down","inet_up","pcie_bw","gpu_max_power","cpu_cores_effective"): return f"{v:.0f}"
    return str(v)
print(" ".join(f"{n:<{w}}" for n, _, w in cols))
for o in offers:
    print(" ".join(f"{fmt(o,k)[:w]:<{w}}" for _, k, w in cols))
print(f"\n{len(offers)} offers. Rent: ./vast/create_instance.sh <id>   (4x total $/h shown; storage billed extra at stor$/GBmo x disk)")
EOF
