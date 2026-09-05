#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
V=$(command -v vastai || ls "$HOME"/.local/bin/vastai 2>/dev/null | head -1)
[ -n "$V" ] || { echo "  vastai CLI not found (PATH=$PATH)"; ls "$HOME/.local/bin" 2>/dev/null | head; exit 1; }
"$V" show instances --raw 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin)
for i in d:
    print("  id=%s state=%s actual=%s intended=%s gpu=%s x%s ssh=%s:%s label=%s"%(i.get("id"),i.get("cur_state"),i.get("actual_status"),i.get("intended_status"),i.get("gpu_name"),i.get("num_gpus"),i.get("ssh_host"),i.get("ssh_port"),i.get("label")))
    print("     status_msg=%s | gpu_util=%s | disk_util=%s"%(str(i.get("status_msg"))[:160].replace("\n"," "),i.get("gpu_util"),i.get("disk_util")))
'
