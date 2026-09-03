GBPUSD = 1.35
GPUS = 14                                  # 2 x B7
node_mo   = 4375.00*2*0.75                 # ex VAT, 25% off
eris      = node_mo*0.2697                 # 100% R&D use -> full relief
node_net  = node_mo - eris
per_card  = 14747/4                        # measured gpt-oss out tok/s per card
fleet_tps = per_card*GPUS

# API price for the SAME work, p10 across providers serving gpt-oss-120b
out_1m, in_1m = 0.170/GBPUSD, 0.037/GBPUSD
shapes = {"router 1k/128 (8:1)": 8, "judge 4k/512 (8:1)": 8,
          "rollout 8k/2k (4:1)": 4, "promptopt (14:1 shared prefix)": 14}

print("MAKE vs BUY for Sqwish's own R&D consumption")
print(f"  2 x B7, 14 GPUs, ex VAT, 25% off   GBP {node_mo:>8,.0f}/mo")
print(f"  less ERIS at 26.97% (100% R&D)     GBP {-eris:>8,.0f}/mo")
print(f"  NET CASH COST                      GBP {node_net:>8,.0f}/mo\n")
print(f"  fleet output at saturation         {fleet_tps:>8,.0f} tok/s = {fleet_tps*3600*730/1e9:.0f}B out tok/mo at 100% duty\n")

print("BREAK-EVEN: how many output tokens a month make the node cheaper than the API")
print(f"{'workload shape':<32} {'API GBP/1M out-equiv':>21} {'break-even/mo':>16} {'fleet duty':>11}")
for lab, ratio in shapes.items():
    api_per_1m = out_1m + ratio*in_1m
    be = node_net/api_per_1m                       # M output tokens
    duty = be*1e6/(fleet_tps*3600*730)
    print(f"{lab:<32} {api_per_1m:>21.3f} {be:>13,.0f}M {duty:>10.1%}")

print("\nWHAT THE BREAK-EVEN LOOKS LIKE IN RESEARCH TERMS (router shape)")
api_per_1m = out_1m + 8*in_1m
be = node_net/api_per_1m
print(f"  break-even = {be:,.0f}M output tokens/month = {be/30:,.0f}M/day")
for lab, out_len in [("agentic rollouts (2,048 out)", 2048), ("judge calls (512 out)", 512),
                     ("counterfactual rewrites (256 out)", 256)]:
    print(f"    {lab:<34} {be*1e6/out_len/30:>12,.0f} per day to break even")

print("\nSAVINGS IF YOU EXCEED IT")
print(f"{'monthly output tokens':>22} {'API cost':>12} {'node net':>10} {'saving':>12}")
for b in (5, 14, 30, 60, 135):
    api = b*1000*api_per_1m
    print(f"{b:>19}B {api:>12,.0f} {node_net:>10,.0f} {api-node_net:>+12,.0f}")

print("\nTHE COMPLICATION: your credits only cover part of this")
creds = (150000+100000)/GBPUSD + 10000
print(f"  Azure + AWS + Google credits        GBP {creds:>9,.0f}")
print(f"  but they buy CLOSED-model access (Azure AI Foundry, Vertex, Bedrock).")
print(f"  The cheap open-weight endpoints you would replace (gpt-oss, Qwen, DeepSeek at")
print(f"  p10 on OpenRouter) are NOT credit-funded, so that spend is real cash today.")
print(f"  If open-weight inference is GBP X/mo of real cash, the node pays back at X > GBP {node_net:,.0f}.")