#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
OFFER="${1:-48388066}"
vastai create instance "$OFFER" --image vllm/vllm-openai:latest --disk 2500 --ssh --direct \
  --env '-p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8003:8003' --label sqwish-rtxpro6000-s2 2>&1 | sed -E "s/'instance_api_key': '[0-9a-f]+'/'instance_api_key': '<redacted>'/" | head -5
sleep 15
vastai show instances --raw 2>/dev/null | python3 -W ignore -c '
import json,sys
for i in json.load(sys.stdin):
    print("  id=%s state=%s actual=%s gpu=%s x%s label=%s ssh=%s:%s msg=%s"%(i.get("id"),i.get("cur_state"),i.get("actual_status"),i.get("gpu_name"),i.get("num_gpus"),i.get("label"),i.get("ssh_host"),i.get("ssh_port"),str(i.get("status_msg"))[:80]))'
