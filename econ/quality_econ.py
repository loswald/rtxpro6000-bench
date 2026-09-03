# Artificial Analysis Intelligence Index v4.1.1 (Sept 2026, open weights) vs OUR measured throughput.
# Until now I ranked models purely on tokens/s. That is only half the decision.
GBPUSD = 1.35
gpu_hr = (4375.00/7)*0.75*(1-0.2697)/730 - (1635.0/14)/730   # discount + ERIS + Vast, per GPU-hour
node4  = gpu_hr*4

# name, AA index, GB on disk, cards needed, measured out tok/s (None = not yet run)
M = [
 ("GLM-5.3-Flash",            57, 197.8, 4, None),
 ("Qwen3.8-27B (xhigh)",      52,  29.0, 1, 2642),   # measured across 4 replicas
 ("MiniMax-M3",               45, 259.0, 4, None),
 ("Nemotron 3 Ultra",         38, None,  None, None),
 ("Muse Glimmer (high)",      35,  59.6, 1, None),
 ("gpt-oss-120b (high)",      24,  61.0, 1, 14747),  # our throughput champion
 ("Nemotron 3.5 Lightning",   24,  21.6, 1, None),
]
print("QUALITY vs THROUGHPUT -- the tradeoff I had not been pricing")
print(f"{'model':<26} {'AA idx':>7} {'GB':>7} {'cards':>6} {'our out tok/s':>14} {'GBP/1M out':>11} {'idx per GBP/1M':>15}")
for n, idx, gb, cards, tps in M:
    if tps:
        per1m = node4/(tps*3600/1e6)
        val = idx/per1m
        print(f"{n:<26} {idx:>7} {gb if gb else '-':>7} {cards if cards else '-':>6} {tps:>14,} {per1m:>11.4f} {val:>15,.0f}")
    else:
        print(f"{n:<26} {idx:>7} {str(gb) if gb else '-':>7} {str(cards) if cards else '-':>6} {'not yet run':>14} {'-':>11} {'-':>15}")

print("\nTHE UNCOMFORTABLE FINDING")
print("  gpt-oss-120b, our throughput champion, scores 24 -- joint LOWEST on the chart.")
print("  Qwen3.8-27B scores 52, more than double, and fits on ONE card at 29 GB.")
print("  We have been optimising the fastest model, not the most useful one.")
print()
q, g = 52, 24
tq, tg = 2642, 14747
pq, pg = node4/(tq*3600/1e6), node4/(tg*3600/1e6)
print(f"  Qwen3.8-27B : index {q}, GBP {pq:.4f}/1M out  -> {q/pq:,.0f} index-points per GBP/1M")
print(f"  gpt-oss-120b: index {g}, GBP {pg:.4f}/1M out  -> {g/pg:,.0f} index-points per GBP/1M")
print(f"  gpt-oss is {tg/tq:.1f}x faster but {q/g:.1f}x weaker. On index-per-pound it still wins {(g/pg)/(q/pq):.1f}x,")
print(f"  BUT only if the task tolerates a 24-index model. For agentic rollouts and judging, it may not.")
print()
print("REPRIORITISED TEST ORDER (quality first, then throughput)")
for n, idx, gb, cards, _ in sorted([m for m in M if m[2]], key=lambda x: -x[1]):
    fit = "1 card" if cards == 1 else f"{cards} cards"
    print(f"  {idx:>3}  {n:<26} {gb:>6.1f} GB  {fit}")
print("\n  DROP from the campaign: gemma-4-26B, gemma-4-31B, Qwen3.6-35B -- none appear on the")
print("  index at all, so they are below 23 and not worth GPU time.")