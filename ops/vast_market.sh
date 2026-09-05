#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
# what renters pay today for RTX PRO 6000 capacity, per GPU-hour, across every rentable listing
vastai search offers 'gpu_name in ["RTX PRO 6000 S","RTX PRO 6000 WS","RTX PRO 6000","RTX PRO 6000 Blackwell"] rentable=true' -o dph --raw --limit 500 2>/dev/null | python3 -W ignore -c '
import json,sys,statistics
o=json.load(sys.stdin)
per=[ (x["dph_total"]/x["num_gpus"]) for x in o if x.get("num_gpus") and x.get("dph_total")]
per.sort()
n=len(per)
if n:
    q=lambda f: per[min(n-1,int(f*n))]
    print("  listings=%d  per-GPU $/h: p10=%.2f p25=%.2f median=%.2f p75=%.2f mean=%.2f"%(n,q(.1),q(.25),statistics.median(per),q(.75),statistics.mean(per)))
    rented=[x for x in o if x.get("rented")]
    print("  by gpu name:", {k: round(statistics.median([x["dph_total"]/x["num_gpus"] for x in o if x.get("gpu_name")==k]),2) for k in sorted(set(x.get("gpu_name") for x in o))})
'
