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
