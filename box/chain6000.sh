#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 240); do
  grep -q ENGINE-SETUP2-DONE $R/engine_setup2.log 2>/dev/null && python3 -c "import vllm,b12x,flashinfer" 2>/dev/null && grep -q DL6000A-DONE $R/dl6000.log 2>/dev/null && break
  sleep 30
done
python3 -c "import vllm,b12x,flashinfer,torch; assert torch.cuda.is_available(); print(\"engine ok\", vllm.__version__, flashinfer.__version__)" || { echo "engine NOT ok"; exit 1; }
echo "[$(date +%H:%M:%S)] chain: control6000"; bash /workspace/bench/control6000.sh > $R/control6000.log 2>&1
echo "[$(date +%H:%M:%S)] chain: fleet2";      GLM_ARMS=base,mtp bash /workspace/bench/fleet2.sh > $R/fleet2.log 2>&1
echo "[$(date +%H:%M:%S)] chain: roster3";     bash /workspace/bench/roster3.sh > $R/roster3.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN6000 DONE"
