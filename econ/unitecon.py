# Can a Scan B7 fleet undercut the commercial inference providers?
GBPUSD = 1.35
NODE_MO_EXVAT = 4375.00          # 1 x B7-64TR, 7 GPUs, ex VAT
DISCOUNT      = 0.25
GPUS          = 7
HOURS         = 730

gpu_mo  = NODE_MO_EXVAT*(1-DISCOUNT)/GPUS
gpu_hr  = gpu_mo/HOURS
print("COST BASE")
print(f"  B7 at {DISCOUNT:.0%} off, ex VAT       GBP {NODE_MO_EXVAT*(1-DISCOUNT):>9,.0f}/mo for {GPUS} GPUs")
print(f"  per GPU                        GBP {gpu_mo:>9,.2f}/mo  = GBP {gpu_hr:.4f}/GPU-hr = ${gpu_hr*GBPUSD:.3f}/GPU-hr")
print(f"  (for comparison, renting an H200 on-demand runs ~$2-3/GPU-hr, a B200 ~$4-6)\n")

# measured output tok/s PER CARD at saturation, from our runs
models = [
    ("gpt-oss-120b, FI-CUTLASS kernel", 14747/4, 0.170, 0.500),
    ("gpt-oss-120b, Marlin",            12924/4, 0.170, 0.500),
    ("Qwen3.8-27B-FP8",                  2642/4, 2.500, 2.775),
    ("DeepSeek-V4-Flash TP4 (so far)",   1091/4, 0.184, 0.280),
]
print(f"{'model':<34} {'out tok/s/card':>14} {'our cost/1M':>12} {'mkt p10':>9} {'mkt med':>9} {'margin p10':>11}")
for name, tps, p10, med in models:
    m_per_hr = tps*3600/1e6
    cost_1m  = gpu_hr/m_per_hr                    # GBP per 1M output tokens, 100% utilisation
    p10g, medg = p10/GBPUSD, med/GBPUSD
    print(f"{name:<34} {tps:>14,.0f} {cost_1m:>11.4f}  {p10g:>8.3f} {medg:>8.3f} {p10g/cost_1m:>10.1f}x")

print("\nSENSITIVITY TO UTILISATION (gpt-oss on the winning kernel, vs p10 market price)")
tps = 14747/4; m_per_hr = tps*3600/1e6; p10g = 0.170/GBPUSD
print(f"{'utilisation':>12} {'cost/1M out':>12} {'price p10':>10} {'gross margin':>13}")
for u in (1.0, 0.7, 0.5, 0.3, 0.15):
    c = gpu_hr/(m_per_hr*u)
    print(f"{u:>11.0%} {c:>12.4f} {p10g:>10.3f} {1-c/p10g:>12.1%}")

print("\nWHAT ONE 7-GPU NODE COULD BILL, if sold at the p10 market price")
for u in (0.3, 0.5, 0.7):
    tok = tps*GPUS*3600*HOURS*u/1e6                # M output tokens/month
    rev = tok*p10g
    cost = NODE_MO_EXVAT*(1-DISCOUNT)
    print(f"  at {u:.0%} utilisation: {tok:>9,.0f}M out tok/mo -> GBP {rev:>9,.0f} revenue "
          f"vs GBP {cost:,.0f} cost  = GBP {rev-cost:>+9,.0f}/mo")

print("\nTHE CATCH")
print("  These are OUTPUT tokens only, priced against providers who also charge for input.")
print("  Our measured input rates are 4-8x the output rate, so real revenue would be higher.")
print("  But: this only holds for models that fit ONE card. DeepSeek needs all four and")
print("  its economics stay marginal until the saturation ladder says otherwise.")