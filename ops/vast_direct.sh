#!/usr/bin/env bash
# print the direct (non-proxy) ssh endpoint of an instance, and try it
export PATH="$HOME/.local/bin:$PATH"
ID="${1:-49694407}"
vastai show instance "$ID" --raw 2>/dev/null | python3 -c '
import json,sys
i=json.load(sys.stdin)
ports=i.get("ports") or {}
p22=(ports.get("22/tcp") or [{}])[0].get("HostPort")
print("  public_ip=%s direct_port_22=%s proxy=%s:%s direct_port_start=%s direct_port_end=%s"%(i.get("public_ipaddr"),p22,i.get("ssh_host"),i.get("ssh_port"),i.get("direct_port_start"),i.get("direct_port_end")))
print("  ports map: %s"%{k:[x.get("HostPort") for x in v] for k,v in ports.items()})
open("/tmp/direct.txt","w").write("%s %s\n"%(i.get("public_ipaddr"),p22))
'
read -r H P < /tmp/direct.txt
if [ -n "$P" ] && [ "$P" != "None" ]; then
  echo "  trying direct ssh root@$H -p $P"
  timeout 30 ssh -i "$HOME/.ssh/id_ed25519" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR "root@$H" 'echo "  direct ok: $(hostname) $(date +%H:%M:%S)"; tmux ls | cut -d: -f1 | paste -sd" " | sed "s/^/  sessions: /"; tail -2 /workspace/results/chain600w.log | cut -c1-150' 2>&1 | head -6
  echo "direct $H $P" > "$(dirname "$0")/inst6000a_direct.ssh"
fi
