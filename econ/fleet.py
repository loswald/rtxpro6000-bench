# What a 2 x B7 fleet (14 x 96GB) could actually host, using our measured numbers.
GPUS, VRAM = 14, 96
print(f"FLEET: {GPUS} x RTX PRO 6000 = {GPUS*VRAM:,} GB VRAM, 2 x 4TB NVMe, 2 x 512GB host RAM\n")

# (name, weights GB, measured or estimated out tok/s per single card at saturation)
models = [
    ("gpt-oss-120b (MXFP4)",        61, 3687, "measured 14,747 across 4 cards"),
    ("Qwen3.8-Flash-Next NVFP4",    78, None, "downloaded, not yet benchmarked"),
    ("Qwen3.8-27B-FP8",             29,  660, "measured 2,642 across 4 cards"),
    ("gpt-oss-20b (MXFP4)",         13, None, "replica-density candidate"),
    ("gemma-4-26B-A4B",             26, None, "MoE, 3.8B active"),
    ("Qwen3.6-35B-A3B",             35, None, "MoE, 3B active"),
    ("Nemotron-3.5-Lightning-30B",  30, None, "NVFP4"),
    ("Mistral-Small-3.2-24B",       24, None, "dense"),
    ("Muse-Glimmer-30B",            30, None, "dense + DFlash draft"),
]
print(f"{'model':<30} {'GB':>5} {'fits 1 card':>12} {'KV headroom':>12}")
for n, gb, tps, note in models:
    kv = VRAM*0.96 - gb
    print(f"{n:<30} {gb:>5} {'yes' if gb < VRAM*0.9 else 'TP2':>12} {kv:>10.0f} GB")

single = [m for m in models if m[1] < VRAM*0.9]
print(f"\n  {len(single)} of {len(models)} run on ONE card -> up to {GPUS} distinct dedicated endpoints at once.")

# disk
disk_tb = 8
total_gb = sum(m[1] for m in models)
print(f"  model library on disk: {total_gb} GB for these {len(models)} = "
      f"{disk_tb*1000/ (total_gb/len(models)):.0f} models fit in {disk_tb} TB")

# throughput if the whole fleet ran the champion
per_card = 14747/4
print(f"\n  if all {GPUS} cards ran gpt-oss-120b: {per_card*GPUS:,.0f} out tok/s "
      f"= {per_card*GPUS*60/1e6:.1f}M out tok/min")
print(f"  at the shared-prefix shape:      {14799/4*GPUS:,.0f} out tok/s alongside "
      f"{207183/4*GPUS/1e6:.1f}M in tok/s")

# on-demand economics
node_net_mo = 3158.0                      # 2 nodes, net of Vast + ERIS
print(f"\n  net fleet cost GBP {node_net_mo:,.0f}/mo = GBP {node_net_mo/GPUS:,.0f} per GPU per month")
print(f"  = GBP {node_net_mo/GPUS/730:.3f} per GPU-hour, against a Vast market price of ~GBP 1.12")

# what we measured about swap latency
print("\nWHAT WE MEASURED ABOUT SPIN-UP (this is the number that decides 'on demand')")
print("  gpt-oss-120b, 61 GB, cold from page cache -> healthy:      70-110 s")
print("  DeepSeek-V4-Flash, 156 GB, TP4, cold -> healthy:           240 s")
print("  weights load itself was ~8 s; the rest is CUDA graph capture")
print("  -> vLLM sleep mode (level 1, weights to host RAM) is the lever worth testing:")
print("     512 GB host RAM per box could park ~8 medium models resident")