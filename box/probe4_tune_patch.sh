#!/usr/bin/env bash
# add a 'tune' set to probe4.sh: 3 high-information points only
python3 - <<'PY'
p="/workspace/bench/probe4.sh"; s=open(p).read()
if "tune)" not in s:
    s = s.replace('  quick)', '  tune)\n    run router 1024 128 0 256 256; run promptopt 512 256 3072 256 256; run judge 4096 512 0 64 128 ;;\n  quick)', 1)
    open(p,"w").write(s); print("tune set added")
else:
    print("tune set already present")
PY
bash -n /workspace/bench/probe4.sh && echo "probe4 ok"
