import os, time, json, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); w = dist.get_world_size(); torch.cuda.set_device(r)
res = []
for nbytes in [32*1024, 1024*1024, 16*1024*1024, 128*1024*1024, 512*1024*1024]:
    t = torch.ones(nbytes // 4, dtype=torch.float32, device="cuda")
    for _ in range(5): dist.all_reduce(t)
    torch.cuda.synchronize(); dist.barrier()
    iters = 50 if nbytes <= 16*1024*1024 else 10
    t0 = time.perf_counter()
    for _ in range(iters): dist.all_reduce(t)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    algbw = nbytes / dt / 1e9; busbw = algbw * 2 * (w - 1) / w
    res.append({"bytes": nbytes, "us": round(dt*1e6, 1), "algbw_GBps": round(algbw, 2), "busbw_GBps": round(busbw, 2)})
if r == 0:
    print(json.dumps({"world": w, "nccl": torch.cuda.nccl.version(), "results": res}, indent=1))
dist.destroy_process_group()
