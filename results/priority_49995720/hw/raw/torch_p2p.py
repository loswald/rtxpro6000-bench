#!/usr/bin/env python3
# torch_p2p.py OUT_JSON -- peer-access matrix (torch.cuda.can_device_access_peer), unidirectional device->device copy
# bandwidth (256 MiB, 10 reps) and small-copy latency (8 B, 200 reps, includes launch overhead) for every GPU pair.
# torch enables peer access lazily on the first cross-device copy, so the copies below ride P2P when it is supported.
import json, sys, time
import torch
n = torch.cuda.device_count()
res = {"tool": "torch", "torch": torch.__version__, "cuda": torch.version.cuda, "device_count": n,
       "names": [torch.cuda.get_device_name(i) for i in range(n)],
       "capability": [list(torch.cuda.get_device_capability(i)) for i in range(n)],
       "can_access_peer": [[True if i == j else bool(torch.cuda.can_device_access_peer(i, j)) for j in range(n)] for i in range(n)],
       "unidir_copy_gbps": [[None] * n for _ in range(n)], "copy_latency_us": [[None] * n for _ in range(n)], "errors": []}
NB = 256 << 20
for i in range(n):
    for j in range(n):
        if i == j or not res["can_access_peer"][i][j]:
            continue
        try:
            src = torch.empty(NB, dtype=torch.uint8, device=f"cuda:{i}")
            dst = torch.empty(NB, dtype=torch.uint8, device=f"cuda:{j}")
            for _ in range(3):
                dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize(i); torch.cuda.synchronize(j)
            t0 = time.perf_counter()
            for _ in range(10):
                dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize(i); torch.cuda.synchronize(j)
            res["unidir_copy_gbps"][i][j] = round(NB * 10 / (time.perf_counter() - t0) / 1e9, 2)
            s8 = torch.zeros(2, dtype=torch.float32, device=f"cuda:{i}")
            d8 = torch.zeros(2, dtype=torch.float32, device=f"cuda:{j}")
            for _ in range(20):
                d8.copy_(s8); torch.cuda.synchronize(j)
            t0 = time.perf_counter()
            for _ in range(200):
                d8.copy_(s8); torch.cuda.synchronize(j)
            res["copy_latency_us"][i][j] = round((time.perf_counter() - t0) / 200 * 1e6, 2)
            del src, dst, s8, d8
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"{i}->{j}: {e!r}")
res["all_pairs_peer_access"] = all(res["can_access_peer"][i][j] for i in range(n) for j in range(n)) if n else None
off = [res["unidir_copy_gbps"][i][j] for i in range(n) for j in range(n) if i != j and res["unidir_copy_gbps"][i][j]]
res["unidir_copy_gbps_min"] = min(off) if off else None
res["unidir_copy_gbps_max"] = max(off) if off else None
lat = [res["copy_latency_us"][i][j] for i in range(n) for j in range(n) if i != j and res["copy_latency_us"][i][j]]
res["copy_latency_us_max"] = max(lat) if lat else None
with open(sys.argv[1], "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps({k: res[k] for k in ("device_count", "all_pairs_peer_access", "unidir_copy_gbps_min", "unidir_copy_gbps_max", "copy_latency_us_max")}))
