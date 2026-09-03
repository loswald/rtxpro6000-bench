#!/usr/bin/env bash
# P2P + NCCL truth. Run ON the box.
set +e
mkdir -p /workspace/bench /workspace/results/hwtruth
cat > /workspace/bench/p2p_torch.py <<'PY'
import torch, time, itertools, json
n = torch.cuda.device_count(); out = {"pairs": []}
for a, b in itertools.permutations(range(n), 2):
    can = torch.cuda.can_device_access_peer(a, b)
    x = torch.empty(256*1024*1024, dtype=torch.float32, device=f"cuda:{a}")
    y = torch.empty_like(x, device=f"cuda:{b}")
    torch.cuda.synchronize(a); torch.cuda.synchronize(b)
    for _ in range(2): y.copy_(x, non_blocking=True)
    torch.cuda.synchronize(b); t0 = time.perf_counter()
    for _ in range(5): y.copy_(x, non_blocking=True)
    torch.cuda.synchronize(b)
    bw = 5 * x.numel() * 4 / (time.perf_counter() - t0) / 1e9
    s = torch.empty(1, dtype=torch.float32, device=f"cuda:{a}"); d = torch.empty_like(s, device=f"cuda:{b}")
    for _ in range(100): d.copy_(s)
    torch.cuda.synchronize(b); t0 = time.perf_counter()
    for _ in range(1000): d.copy_(s)
    torch.cuda.synchronize(b)
    out["pairs"].append({"src": a, "dst": b, "peer": bool(can), "GBps": round(bw,1), "lat_us": round((time.perf_counter()-t0)/1000*1e6,2)})
    del x, y
print(json.dumps(out))
PY
cat > /workspace/bench/nccl_allreduce.py <<'PY'
import time, json, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); w = dist.get_world_size(); torch.cuda.set_device(r)
res = []
for nbytes in [1024*1024, 16*1024*1024, 128*1024*1024, 512*1024*1024]:
    t = torch.ones(nbytes // 4, dtype=torch.float32, device="cuda")
    for _ in range(5): dist.all_reduce(t)
    torch.cuda.synchronize(); dist.barrier()
    iters = 50 if nbytes <= 16*1024*1024 else 10
    t0 = time.perf_counter()
    for _ in range(iters): dist.all_reduce(t)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    res.append({"MB": nbytes//1048576, "us": round(dt*1e6,1), "busbw_GBps": round(nbytes/dt/1e9*2*(w-1)/w, 2)})
if r == 0: print(json.dumps({"world": w, "results": res}))
dist.destroy_process_group()
PY
cd /workspace/bench
echo "== P2P =="; python3 p2p_torch.py 2>/dev/null | tee /workspace/results/hwtruth/p2p.json | python3 -c "
import sys,json; d=json.load(sys.stdin)
for p in d['pairs']: print(f\"{p['src']}->{p['dst']} peer={p['peer']} {p['GBps']} GB/s lat {p['lat_us']} us\")"
echo "== NCCL all_reduce x4 =="
NCCL_IB_DISABLE=1 timeout 300 torchrun --nproc_per_node 4 nccl_allreduce.py 2>/dev/null | tail -1 | tee /workspace/results/hwtruth/nccl4.json
echo "== NCCL pair 0,1 =="
CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 timeout 200 torchrun --nproc_per_node 2 nccl_allreduce.py 2>/dev/null | tail -1 | tee /workspace/results/hwtruth/nccl01.json
echo "== NCCL pair 0,2 =="
CUDA_VISIBLE_DEVICES=0,2 NCCL_IB_DISABLE=1 timeout 200 torchrun --nproc_per_node 2 nccl_allreduce.py 2>/dev/null | tail -1 | tee /workspace/results/hwtruth/nccl02.json
