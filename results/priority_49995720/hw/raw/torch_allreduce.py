#!/usr/bin/env python3
# torch_allreduce.py WORLD OUT_JSON [MASTER_PORT] -- NCCL all_reduce busbw via torch.distributed; fallback for
# nccl-tests' all_reduce_perf. Uses the GPUs in CUDA_VISIBLE_DEVICES in order (rank r -> cuda:r).
# busbw = algbw * 2*(n-1)/n (nccl-tests convention), so numbers are comparable with all_reduce_perf.
import json, os, sys, time
import torch, torch.distributed as dist, torch.multiprocessing as mp

SIZES = [8, 1 << 20, 8 << 20, 128 << 20, 1 << 30]

def worker(rank, world, port, out):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    except TypeError:
        dist.init_process_group("nccl", rank=rank, world_size=world)
    rows = []
    for nbytes in SIZES:
        n = max(nbytes // 4, 2)
        chk = torch.ones(n, device=dev); dist.all_reduce(chk); torch.cuda.synchronize()
        wrong = int((chk != float(world)).sum().item())
        x = torch.ones(n, device=dev)
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        iters = 20 if nbytes >= (128 << 20) else 100
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        algbw = nbytes / dt / 1e9
        rows.append({"size_bytes": nbytes, "time_us": round(dt * 1e6, 2), "algbw_gbps": round(algbw, 3),
                     "busbw_gbps": round(algbw * 2 * (world - 1) / world, 3), "wrong": wrong})
        del x, chk
    dist.barrier()
    if rank == 0:
        with open(out, "w") as f:
            json.dump({"tool": "torch.distributed all_reduce", "world": world,
                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                       "nccl": ".".join(map(str, torch.cuda.nccl.version())), "rows": rows}, f, indent=2)
    dist.destroy_process_group()

if __name__ == "__main__":
    world = int(sys.argv[1]); out = sys.argv[2]; port = int(sys.argv[3]) if len(sys.argv) > 3 else 29517
    mp.spawn(worker, args=(world, port, out), nprocs=world, join=True)
