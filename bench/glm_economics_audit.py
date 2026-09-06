#!/usr/bin/env python3
"""Reproduce the bounded GLM cost audit from complete AIRR raw measurements.

All price observations are dated 2026-09-05. The GBP/USD conversion, full VAT
recovery, 70% demand utilization and 100% qualified-output retention are stated
scenario assumptions, not forecasts or claims of passed quality/SLO gates.
"""
import hashlib
import json
from pathlib import Path
from economics import compare, read_benchmark

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "priority"
SCAN = "https://www.scan.co.uk/products/3xs-sc-pb4-32t-1-month-4x-96gb-nvidia-rtx-pro-6000-512gb-ddr5-ecc-amd-epyc-9354p"
ZAI = "https://docs.z.ai/guides/overview/pricing"
OPENROUTER = "https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints"
TAG = "glm53f_dp4ep4_s192"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prices = {"promotion": {"input_rate": .075, "cached_input_rate": .015, "output_rate": .25},
              "list": {"input_rate": .15, "cached_input_rate": .03, "output_rate": .50}}
    cost = 1999.98 / 1.2 / 720 * 1.35
    observations = {}
    cases = []
    for shape in ("router", "promptopt"):
        path = ROOT / "results" / "600w" / "probe" / TAG / f"{TAG}__{shape}__c1024__p8000.json"
        workload = read_benchmark(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        observations[shape] = dict(workload, source_sha256=sha(path),
                                  median_ttft_seconds=raw["p50_ttft_ms"] / 1000,
                                  p99_ttft_seconds=raw["p99_ttft_ms"] / 1000,
                                  request_goodput=raw.get("request_goodput"))
        for cache in ([0] if shape == "router" else [3072 / 3584, 0]):
            for band, rates in prices.items():
                scenario = {"node_hourly_cost": cost, "utilization": .7, "goodput_fraction": 1,
                            "output_tps": workload["output_tps"], "input_tokens": workload["input_tokens"],
                            "output_tokens": workload["output_tokens"], "cache_hit_fraction": cache, **rates}
                cases.append({"shape": shape, "price_band": band, "cached_prefix": bool(cache),
                              "inputs": scenario, "comparison": compare(**scenario),
                              "quality_gate": "unverified", "slo_gate": "unverified", "eligible_for_selection": False})
    mixed = {}
    for band in prices:
        pair = [r for r in cases if r["price_band"] == band and (r["shape"] == "router" or r["cached_prefix"])]
        local = sum(r["comparison"]["local_cost_per_request"] for r in pair) / 2
        api = sum(r["comparison"]["provider_cost_per_request"] for r in pair) / 2
        mixed[band] = {"assumption": "Equal numbers of router and promptopt requests, all shared prefix tokens billed as cache hits",
                       "local_cost_per_request": local, "api_cost_per_request": api,
                       "local_divided_by_api": local / api,
                       "uniform_throughput_multiplier_for_break_even": local / api}
    result = {"date": "2026-09-05", "scope": "GLM-5.3-Flash; historical NVFP4 throughput as a provisional engineering reference",
              "sources": {"scan": SCAN, "zai_prices": ZAI, "provider_endpoints": OPENROUTER},
              "node_cost": {"verified_inc_vat_gbp": 1999.98, "verified_subscription_hours": 720,
                            "assumed_vat_rate": .2, "assumed_vat_recovery_fraction": 1,
                            "ex_vat_gbp": 1666.65, "assumed_usd_per_gbp": 1.35,
                            "usd_per_calendar_hour": cost, "usd_per_active_hour_at_70_percent_demand": cost / .7,
                            "commitment_discount": 0, "tax_relief": 0, "resale_credit": 0},
              "prices_usd_per_million": prices, "promotion_end_utc": "2026-09-09T16:00:00Z",
              "observations": observations, "scenarios": cases, "equal_request_mix": mixed,
              "excluded": {"glm53f_dp2tp2ep2_s384": "Corruption tripwire failed: 8 degenerate and 1 wrong of 20",
                           "glm53f_tp4ep4_s512": "Each router/promptopt raw result has 976 completed and 3120 failed of 4096 requested"},
              "quality_precision_caveat": "OpenRouter reports FP8 for several cheapest providers and FP4 for DeepInfra. Exact local/provider checkpoint bytes and task-quality equivalence are unverified. No eligible serving recommendation follows from this arithmetic."}
    (OUT / "glm_economics_20260905.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in cases:
        c = row["comparison"]
        print(f"{row['shape']:10} cached={row['cached_prefix']!s:5} {row['price_band']:9} "
              f"required={c['break_even_output_tps_at_given_utilization']:.1f} out/s "
              f"multiplier={c['required_throughput_multiplier']:.3f} "
              f"min_demand={c['break_even_utilization_at_measured_output_tps']:.3%}")
    print(json.dumps(mixed, indent=2))


if __name__ == "__main__":
    main()
