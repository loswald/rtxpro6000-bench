#!/usr/bin/env bash
# Recover the results stranded on the original 600 W Server Edition box when it was stopped out from under
# us. These are the numbers that actually transfer to a Server Edition purchase: everything measured since
# was on a 400 W-capped Workstation box.
set -u
KEY="$HOME/.ssh/id_ed25519"
SP=/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad
read -r _ H P _ < "$SP/inst6000c.ssh"
SSH="ssh -i $KEY -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30 -o LogLevel=ERROR root@$H"
SCP="scp -q -i $KEY -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
L=/mnt/c/Users/ushni/Downloads/AIRR
$SSH 'set +e
cd /workspace/results
for f in smoke/*.log; do [ -f "$f" ] || continue
  k=$(grep -m4 -ohE "Using [A-Za-z0-9_]+ (attention backend|for NVFP4 GEMM)|Using .[A-Z0-9_]+. [A-Za-z0-9]+ MoE backend|Selected [A-Za-z0-9]+Kernel|GPU KV cache size: [0-9,]+ tokens" "$f" | sort -u | paste -sd"|")
  printf "%s\t%s\n" "$(basename "$f")" "$k"
done | sort -u > kernels_by_server_host1.tsv
nvidia-smi --query-gpu=name,power.limit,power.max_limit,driver_version --format=csv,noheader | head -1 > host3_gpu.txt
python3 -c "import vllm,flashinfer; print(\"vllm\", vllm.__version__, \"flashinfer\", flashinfer.__version__)" >> host3_gpu.txt 2>/dev/null
tar czf /workspace/results_host3.tgz --exclude=smoke -C /workspace results 2>/dev/null
du -sh /workspace/results_host3.tgz | sed "s/^/  tarball: /"'
timeout 900 $SCP "root@$H:/workspace/results_host3.tgz" "$L/" && echo "  pulled" || { echo "  pull FAILED"; exit 1; }
cd "$L" || exit 1
mkdir -p results/600w2 && tar xzf results_host3.tgz -C results/600w2 --strip-components=1 && python3 -c "import os; os.unlink('results_host3.tgz')" && echo "  merged into results/600w2/"
ls results/600w2/probe 2>/dev/null | grep -v quality20 | wc -l | sed 's/^/  probe dirs recovered: /'
cat results/600w2/host3_gpu.txt 2>/dev/null | sed 's/^/  /'
python3 box/mksummary.py 2>/dev/null | head -2
