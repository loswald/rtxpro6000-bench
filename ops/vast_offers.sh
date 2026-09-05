#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
# four RTX PRO 6000 cards, CUDA 13-capable driver, room for the roster; both editions, so the power limit is visible
vastai search offers 'gpu_name in ["RTX PRO 6000 S","RTX PRO 6000 WS","RTX PRO 6000","RTX PRO 6000 Blackwell"] num_gpus=4 cuda_max_good>=13.0 disk_space>=1200 rentable=true' -o dph --raw 2>/dev/null | python3 -W ignore -c '
import json,sys
offers=json.load(sys.stdin)
print("  %d offers"%len(offers))
print("  %-10s %-18s %5s %6s %7s %6s %8s %7s %6s %s"%("id","gpu","$/h","disk","inet_dn","rel","cuda","pcie","cpus","geo"))
for o in offers[:14]:
    print("  %-10s %-18s %5.2f %6.0f %7.0f %6.3f %8s %7s %6s %s"%(o.get("id"),o.get("gpu_name"),o.get("dph_total") or 0,o.get("disk_space") or 0,o.get("inet_down") or 0,o.get("reliability2") or 0,o.get("cuda_max_good"),str(o.get("pcie_bw"))[:6],o.get("cpu_cores_effective"),o.get("geolocation")))
'
