import torch, time, itertools, json
n = torch.cuda.device_count()
out = {"pairs": []}
for a, b in itertools.permutations(range(n), 2):
    can = torch.cuda.can_device_access_peer(a, b)
    x = torch.empty(256*1024*1024, dtype=torch.float32, device=f"cuda:{a}")  # 1 GiB
    y = torch.empty_like(x, device=f"cuda:{b}")
    torch.cuda.synchronize(a); torch.cuda.synchronize(b)
    for _ in range(2): y.copy_(x, non_blocking=True)
    torch.cuda.synchronize(b)
    t0 = time.perf_counter()
    for _ in range(5): y.copy_(x, non_blocking=True)
    torch.cuda.synchronize(b)
    bw = 5 * x.numel() * 4 / (time.perf_counter() - t0) / 1e9
    s = torch.empty(1, dtype=torch.float32, device=f"cuda:{a}"); d = torch.empty_like(s, device=f"cuda:{b}")
    for _ in range(100): d.copy_(s)
    torch.cuda.synchronize(b)
    t0 = time.perf_counter()
    for _ in range(1000): d.copy_(s)
    torch.cuda.synchronize(b)
    lat_us = (time.perf_counter() - t0) / 1000 * 1e6
    out["pairs"].append({"src": a, "dst": b, "peer_access": bool(can), "GBps_uni": round(bw, 1), "lat_us": round(lat_us, 2)})
    del x, y
print(json.dumps(out, indent=1))
