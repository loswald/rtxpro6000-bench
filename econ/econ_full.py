GBPUSD = 1.35
# --- the cost stack, per GPU per month, B7-64TR (7 GPUs at GBP 4,375/mo ex VAT) ---
list_gpu   = 4375.00/7
disc       = 0.25                      # committed-use discount, 2-year term
eris       = 0.2697                    # confirmed from the AP1 AIF: 11,463.98 / 42,506.41
vast_gpu   = 1635.0/14                 # incidental income from idle capacity, per GPU
after_disc = list_gpu*(1-disc)
after_eris = after_disc*(1-eris)
after_vast = after_eris - vast_gpu
gpu_hr     = after_vast/730

print("COST STACK, per GPU per month")
print(f"  Scan list, ex VAT                    GBP {list_gpu:>7.2f}")
print(f"  after {disc:.0%} committed-use discount   GBP {after_disc:>7.2f}")
print(f"  after ERIS relief at {eris:.2%}         GBP {after_eris:>7.2f}")
print(f"  after Vast incidental income         GBP {after_vast:>7.2f}")
print(f"  => GBP {gpu_hr:.4f} per GPU-hour  (${gpu_hr*GBPUSD:.3f})")
print(f"  a 4-GPU equivalent costs GBP {gpu_hr*4:.3f}/hr\n")

node4 = gpu_hr*4

# model, out tok/s and in tok/s measured on 4 GPUs, API p10 in $/1M (in, out)
rows = [
  ("gpt-oss-120b x4, FI-CUTLASS",   10913,  87301, 0.037, 0.170),
  ("Qwen3.8-27B-FP8 x4",             2642,  21135, 0.320, 2.500),
  ("DeepSeek-V4-Flash DP4+EP",       1326,  10610, 0.089, 0.177),
  ("DeepSeek, shared-prefix shape",  3002,  42032, 0.089, 0.177),
  ("gpt-oss, shared-prefix shape",  14799, 207183, 0.037, 0.170),
]
print(f"{'configuration':<32} {'API GBP/h':>10} {'node GBP/h':>11} {'x cheaper':>10} {'GBP/1M out':>11}")
res = []
for name, o, i, pin, pout in rows:
    oh, ih = o*3600/1e6, i*3600/1e6                 # millions of tokens per hour
    api = oh*(pout/GBPUSD) + ih*(pin/GBPUSD)
    per1m = node4/oh
    res.append((name, api, node4, api/node4, per1m))
    print(f"{name:<32} {api:>10.2f} {node4:>11.3f} {api/node4:>9.1f}x {per1m:>11.4f}")

print("\nSAME TABLE AT REALISTIC DUTY CYCLES (node cost per useful token rises; API does not)")
print(f"{'configuration':<32} {'100%':>8} {'70%':>8} {'50%':>8} {'30%':>8}")
for (name, api, n4, mult, _), (_, o, i, _, _) in zip(res, rows):
    line = f"{name:<32}"
    for u in (1.0, 0.7, 0.5, 0.3):
        line += f" {api/(n4/u):>7.1f}x"
    print(line)

print("\nWHAT CHANGED versus what I told you earlier")
print("  1. I used GROSS list price. The real base is 64% lower after discount, ERIS and Vast.")
print("  2. I compared output tokens against output-token prices only. Providers bill INPUT too,")
print("     and our input rates are 4-8x our output rates, so that was the larger error.")
print(f"  3. Result: DeepSeek goes from 'loses at 0.2x' to {res[2][3]:.1f}x CHEAPER than the API.")
print("     It is still our weakest model per GPU, but it is no longer uneconomic to self-host.")