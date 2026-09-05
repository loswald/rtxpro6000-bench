#!/usr/bin/env python3
"""Matched-workload inference break-even calculator (Python 3.8+, stdlib only).

All monetary inputs must already be in the same currency, after recoverable VAT,
with node-hourly-cost covering the whole node over paid calendar hours. No Scan
quote or provider price is assumed. Utilization is realized billable demand as a
fraction of benchmark capacity, NOT nvidia-smi utilization.

Example (illustrative inputs, not a provider or Scan quote):
  python bench/economics.py --currency USD --node-hourly-cost 4 --utilization .5 \
    --output-tps 1000 --input-tokens 4096 --output-tokens 512 \
    --input-rate .2 --output-rate .6 --cache-hit-fraction 0

Use --benchmark-json FILE instead of the three throughput/token arguments to
read one raw vLLM bench result. The file must represent the whole costed node;
one replica's result must not be compared with a four-replica node price.
"""
import argparse
import json
import math
from pathlib import Path
import sys


def number(value, name, minimum=0.0, maximum=None, strict_min=False):
    """Reject missing, non-finite, negative and nonsensical fractional inputs."""
    if isinstance(value, bool):
        raise ValueError("{} must be numeric, not a boolean".format(name))
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a finite number".format(name))
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    if value < minimum or (strict_min and value == minimum):
        raise ValueError("{} must be {} {}".format(name, ">" if strict_min else ">=", minimum))
    if maximum is not None and value > maximum:
        raise ValueError("{} must be <= {}".format(name, maximum))
    return value


def effective_vat_cost(ex_vat_cost, vat_rate, recoverable_fraction):
    """Arithmetic helper only; caller establishes tax/recovery eligibility."""
    base = number(ex_vat_cost, "ex_vat_cost")
    rate = number(vat_rate, "vat_rate")
    recoverable = number(recoverable_fraction, "recoverable_fraction", maximum=1)
    return base * (1 + rate * (1 - recoverable))


def read_benchmark(path):
    """Use observed token counts, never configured max_tokens or total tok/s."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("benchmark must be a single raw result object")
    completed = number(data.get("completed"), "benchmark completed", strict_min=True)
    failed = number(data.get("failed"), "benchmark failed")
    if failed:
        raise ValueError("benchmark has failed requests; use a qualified complete run")
    requested = number(data.get("num_prompts"), "benchmark num_prompts", strict_min=True)
    if completed != requested:
        raise ValueError("benchmark is incomplete: completed != num_prompts")
    duration = number(data.get("duration"), "benchmark duration", strict_min=True)
    inputs = number(data.get("total_input_tokens"), "benchmark total_input_tokens")
    outputs = number(data.get("total_output_tokens"), "benchmark total_output_tokens", strict_min=True)
    throughput = outputs / duration
    if data.get("output_throughput") is not None:
        reported = number(data["output_throughput"], "benchmark output_throughput", strict_min=True)
        if not math.isclose(reported, throughput, rel_tol=0.01, abs_tol=0.01):
            raise ValueError("benchmark output_throughput conflicts with output tokens / duration")
    return {
        "output_tps": throughput,
        "input_tokens": inputs / completed,
        "output_tokens": outputs / completed,
        "benchmark_path": str(Path(path).resolve()),
        "model": data.get("model_id", data.get("model")),
        "completed": completed,
        "duration_seconds": duration,
    }


def compare(*, node_hourly_cost, utilization, output_tps, input_tokens,
            output_tokens, input_rate, output_rate, cache_hit_fraction,
            cached_input_rate=None, goodput_fraction=1.0):
    """Price identical useful outputs and their associated input/cache mix.

    goodput_fraction is the fraction of generated output tokens retained after
    quality and latency qualification. It must not double-count losses already
    excluded from output_tps. Provider rates assume the same accepted workload;
    include provider retry/billing overhead in those rates if applicable.
    """
    cost = number(node_hourly_cost, "node_hourly_cost")
    util = number(utilization, "utilization", maximum=1, strict_min=True)
    tps = number(output_tps, "output_tps", strict_min=True)
    inputs = number(input_tokens, "input_tokens")
    outputs = number(output_tokens, "output_tokens", strict_min=True)
    in_rate = number(input_rate, "input_rate")
    out_rate = number(output_rate, "output_rate")
    hit = number(cache_hit_fraction, "cache_hit_fraction", maximum=1)
    goodput = number(goodput_fraction, "goodput_fraction", maximum=1, strict_min=True)
    if hit and cached_input_rate is None:
        raise ValueError("cached_input_rate is required when cache_hit_fraction > 0")
    cached = None if cached_input_rate is None else number(cached_input_rate, "cached_input_rate")
    ratio = inputs / outputs
    blended_input_rate = (1 - hit) * in_rate + hit * (cached or 0)
    provider_per_m_output = out_rate + ratio * blended_input_rate
    useful_tps = tps * util * goodput
    local_per_m_output = cost * 1e6 / (3600 * useful_tps)
    # JSON null denotes no finite positive-price break-even, never Infinity/NaN.
    threshold_tps = (cost * 1e6 / (3600 * util * goodput * provider_per_m_output)
                     if provider_per_m_output else (0.0 if cost == 0 else None))
    threshold_util = (cost * 1e6 / (3600 * tps * goodput * provider_per_m_output)
                      if provider_per_m_output else (0.0 if cost == 0 else None))
    saving = provider_per_m_output - local_per_m_output
    tied = math.isclose(local_per_m_output, provider_per_m_output, rel_tol=1e-12, abs_tol=1e-12)
    return {
        "input_output_ratio": ratio,
        "provider_blended_input_rate_per_million": blended_input_rate,
        "effective_useful_output_tps": useful_tps,
        "local_cost_per_million_useful_output_tokens": local_per_m_output,
        "provider_cost_per_million_useful_output_tokens_including_input": provider_per_m_output,
        "local_cost_per_request": local_per_m_output * outputs / 1e6,
        "provider_cost_per_request": provider_per_m_output * outputs / 1e6,
        "savings_per_million_useful_output_tokens": saving,
        "savings_fraction": saving / provider_per_m_output if provider_per_m_output else None,
        "break_even_output_tps_at_given_utilization": threshold_tps,
        "required_throughput_multiplier": threshold_tps / tps if threshold_tps is not None else None,
        "break_even_utilization_at_measured_output_tps": threshold_util,
        "break_even_possible_at_measured_output_tps": threshold_util is not None and threshold_util <= 1,
        "maximum_competitive_node_hourly_cost": provider_per_m_output * useful_tps * 3600 / 1e6,
        "arithmetic_result": "equal" if tied else ("local_cheaper" if saving > 0 else "provider_cheaper"),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--currency", required=True, help="common currency of ALL cost/rate inputs; no FX conversion")
    parser.add_argument("--node-hourly-cost", required=True, type=float,
                        help="whole-node cost per paid calendar hour, after recoverable VAT")
    parser.add_argument("--utilization", required=True, type=float,
                        help="realized demand / measured capacity, (0,1]; not GPU SM utilization")
    parser.add_argument("--input-rate", required=True, type=float, help="provider currency / 1M uncached input tokens")
    parser.add_argument("--output-rate", required=True, type=float, help="provider currency / 1M output tokens, including reasoning")
    parser.add_argument("--cached-input-rate", type=float, help="provider currency / 1M cached input tokens")
    parser.add_argument("--cache-hit-fraction", required=True, type=float,
                        help="fraction of provider INPUT TOKENS billed at cache-hit rate, [0,1]")
    parser.add_argument("--output-tps", type=float, help="whole-node generated output tokens/s on matching workload")
    parser.add_argument("--input-tokens", type=float, help="mean input tokens per accepted request")
    parser.add_argument("--output-tokens", type=float, help="mean actual output tokens per accepted request, including reasoning")
    parser.add_argument("--benchmark-json", type=Path, help="one raw vLLM result covering the whole costed node")
    parser.add_argument("--goodput-fraction", type=float, default=1,
                        help="fraction of output tokens meeting quality/SLO, (0,1]; default 1 is an assumption")
    parser.add_argument("--quality-gate", choices=("pass", "fail", "unverified"), default="unverified")
    parser.add_argument("--slo-gate", choices=("pass", "fail", "unverified"), default="unverified")
    parser.add_argument("--model", help="exact local checkpoint/revision and reasoning mode label")
    parser.add_argument("--provider", help="matched provider endpoint/model/region/time-band label")
    parser.add_argument("--rate-source", help="URL or invoice reference and rate date")
    parser.add_argument("--node-cost-source", help="quote/invoice reference and date")
    parser.add_argument("--out", type=Path, help="also write the JSON result to this file")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        workload = {"output_tps": args.output_tps, "input_tokens": args.input_tokens, "output_tokens": args.output_tokens}
        if args.benchmark_json:
            if any(value is not None for value in workload.values()):
                raise ValueError("--benchmark-json cannot be combined with manual throughput/token inputs")
            workload = read_benchmark(args.benchmark_json)
        elif any(value is None for value in workload.values()):
            raise ValueError("supply --benchmark-json or all of --output-tps, --input-tokens, --output-tokens")
        inputs = {
            "node_hourly_cost": args.node_hourly_cost, "utilization": args.utilization,
            "output_tps": workload["output_tps"], "input_tokens": workload["input_tokens"],
            "output_tokens": workload["output_tokens"], "input_rate": args.input_rate,
            "output_rate": args.output_rate, "cached_input_rate": args.cached_input_rate,
            "cache_hit_fraction": args.cache_hit_fraction, "goodput_fraction": args.goodput_fraction,
        }
        calculations = compare(**inputs)
        qualified = args.quality_gate == "pass" and args.slo_gate == "pass"
        result = {
            "schema_version": 1, "currency": args.currency, "inputs": inputs,
            "workload_source": workload,
            "provenance": {"model": args.model, "provider": args.provider,
                           "rate_source": args.rate_source, "node_cost_source": args.node_cost_source},
            "qualification": {"quality_gate": args.quality_gate, "slo_gate": args.slo_gate,
                              "eligible_for_selection": qualified,
                              "status": "qualified" if qualified else "provisional_not_a_serving_recommendation"},
            "assumptions": [
                "Node cost and provider prices share one currency and consistent VAT treatment.",
                "Node cost includes all paid calendar hours, power, hosting and allocated support/capital costs.",
                "Throughput covers the whole costed node at this same input/output/context/cache workload.",
                "Provider and local workload use matched model quality, reasoning budgets and latency requirements.",
                "Provider cache hit fraction is token weighted and does not imply a measured local prefix-cache speedup.",
                "Output tokens include reasoning; accepted workload excludes unusable or truncated answers.",
                "Quality/SLO labels are caller attestations; this calculator does not run evaluations.",
            ],
            "comparison": calculations,
        }
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(encoded, encoding="utf-8")
        sys.stdout.write(encoded)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
