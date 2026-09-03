# Cost note -- 4x RTX PRO 6000 Blackwell on Vast.ai, 48 h campaign

Snapshot date: **2026-09-02**. Vast prices are set by hosts and move hourly; treat everything below as a
planning range and read the live `dph_total` / `storage_cost` / `inet_down_cost` columns from
`search_offers.sh` before renting.

**Actual (2026-09-02):** the rented verified 4x RTX PRO 6000 Blackwell **Server Edition** box bills **~$4/h** all-in
(below the $5.50 planning figure), with only ~390 GB of container disk. Two consequences for the budget: the
campaign is disk-serialised (one big model on disk at a time, download -> sweep -> delete), so download time sits
inside GPU-billed hours instead of overlapping with the first sweeps, and the 48 h estimate scales to
**~$190 for 48 h of compute** (~$4/h x 48) plus ingress. Downloads that were already on disk when this note was
written: DeepSeek-V4-Flash-0731 (167 GB), Qwen3.8-27B-FP8 (31 GB), gpt-oss-120b (65 GB).

## Per-GPU hourly prices observed (USD / GPU / h)

| Source (date) | RTX PRO 6000 on Vast.ai | Notes |
|---|---|---|
| Vast price page, Workstation edition (page title) | from **$0.67** | floor across all hosts incl. unverified / interruptible |
| Vast price page, Server edition (page title) | from **$0.87** | floor |
| getdeploying.com provider table (2026-09-02) | on-demand **$1.20**, spot **$0.84** | their tracked Vast listing |
| thundercompute.com pricing blog (2026-09-01) | median **$1.76** | "median of 12 verified US/CA hosts" |
| Vast pricing page (no numbers rendered server-side) | -- | per-second billing, no minimum, three tiers: On-Demand, Interruptible (50%+ cheaper), Reserved |

Vast's own RTX PRO 6000 pages render prices client-side, so the fetched HTML only carried the "from $X" title
figure. The verified-host filters this campaign needs (`verified=true cpu_ram>=256 disk_space>=1200 inet_down>=1000
direct_port_count>=4`) push you toward the upper half of the range.

Sources:
- https://vast.ai/pricing/gpu/RTX-PRO-6000-WS  (title: "for $0.67/hr")
- https://vast.ai/pricing/gpu/RTX-PRO-6000-S   (title: "for $0.87/hr")
- https://vast.ai/pricing  (billing model; live grid not fetchable server-side)
- https://getdeploying.com/gpus/nvidia-rtx-pro-6000  (Vast on-demand $1.20, spot $0.84; updated 2026-09-02)
- https://www.thundercompute.com/blog/nvidia-rtx-pro-6000-pricing  (Vast median $1.76; reviewed 2026-09-01)

## Planning numbers for the 4-GPU machine

| Line item | Low | Expected | High | Basis |
|---|---|---|---|---|
| GPU compute, 4x, $/h | $3.40 | **$5.50** | $7.20 | $0.85 / $1.35 / $1.80 per GPU-h (verified, 256 GB+ RAM, Gen5 board) |
| Container disk, 1200 GB, $/h | $0.10 | **$0.25** | $0.50 | `storage_cost` typically $0.06-0.30 /GB/month -> x1200 / 730 h |
| Ingress, ~1.0-1.1 TB of model weights | $0 | **$10** | $55 | `inet_down_cost` is $0 on most hosts, up to ~$0.05/GB on some -- filter `inet_down_cost<0.01` |
| Egress (results JSON, logs) | ~$0 | ~$0 | ~$1 | negligible |
| **48 h campaign total, primary instance** | **~$175** | **~$285** | **~$425** | 48 x (compute + disk) + ingress |
| Optional SGLang A/B instance, ~8 h | $30 | **$50** | $65 | same machine class, short-lived, downloads only 2-3 models |
| **Campaign total incl. A/B** | **~$205** | **~$335** | **~$490** | |

Rules of thumb:
- Billing is per second while the instance exists. A **stopped** instance still bills storage (`storage_cost x disk`),
  so `vastai destroy instance ID` when done -- do not just stop it.
- The ~1 TB of weights (see README "Disk budget") takes roughly 2.5 h at a real 1 Gbit/s and is the single biggest
  idle-cost item: the GPUs bill while weights land. On the 390 GB box the downloads cannot all run up front; the
  cheapest pattern is to start the *next* model's `bench/prefetch.sh` while the current model's sweep runs (as long as
  both fit: e.g. 127 GB + 167 GB), then delete the finished one. `hardware_truth.sh` (~5 min) runs before any engine.
- At ~$4/h, each agent-shape point at C=64 (~4 min) costs ~$0.25 and a full three-shape sweep of one cell with the
  per-shape concurrency lists in the top-level README costs roughly $6-10; the 30 min agent C=256 points that were
  dropped would have cost ~$2 each for little decision value.
- Interruptible (bid) pricing is ~50% cheaper but a preemption mid-sweep costs you the model reload (2-10 min per cell)
  and the partial sweep. Use on-demand for the 48 h run; interruptible is fine for the SGLang A/B if you are watching it.
- Reserved pricing (up to 50% off) only makes sense once the Scan-hosted node decision is made, not for this trial.

## What the numbers mean for the Scan decision

At the expected ~$5.50/h, a 4x RTX PRO 6000 box on Vast costs ~$4,000/month if run 24/7 -- compare against the Scan
node's monthly lease plus colo/power when you write the decision memo. Fill in the measured tok/s from `bench/summarise.py`
to get $/M tokens per cell: `usd_per_M_tok = dph_total / (total_tok_s * 3600 / 1e6)`.
