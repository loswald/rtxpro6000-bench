"""Economic decision tests: mixed billing, utilization, cache, and invalid evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from bench.economics import compare, effective_vat_cost, main, read_benchmark


class EconomicsTests(unittest.TestCase):
    def scenario(self, **changes):
        args = dict(node_hourly_cost=4, utilization=.5, output_tps=1000,
                    input_tokens=4000, output_tokens=1000, input_rate=.2,
                    output_rate=.6, cache_hit_fraction=0)
        args.update(changes)
        return compare(**args)

    def test_input_billing_changes_break_even(self):
        result = self.scenario()
        self.assertAlmostEqual(result["provider_cost_per_request"], .0014)
        self.assertAlmostEqual(result["local_cost_per_request"], 4 / 1800)
        self.assertAlmostEqual(result["break_even_output_tps_at_given_utilization"], 1587.3015873015872)
        self.assertAlmostEqual(result["break_even_utilization_at_measured_output_tps"], .7936507936507936)
        self.assertTrue(result["break_even_possible_at_measured_output_tps"])

    def test_half_utilization_doubles_local_unit_cost(self):
        half, full = self.scenario(), self.scenario(utilization=1)
        self.assertAlmostEqual(half["local_cost_per_request"], 2 * full["local_cost_per_request"])
        self.assertEqual(full["arithmetic_result"], "local_cheaper")
        self.assertEqual(half["arithmetic_result"], "provider_cheaper")

    def test_provider_cache_discount_can_reverse_result(self):
        no_cache = self.scenario(utilization=1)
        cached = self.scenario(utilization=1, cache_hit_fraction=.9, cached_input_rate=.01)
        self.assertEqual(no_cache["arithmetic_result"], "local_cheaper")
        self.assertEqual(cached["arithmetic_result"], "provider_cheaper")
        self.assertAlmostEqual(cached["provider_cost_per_request"], .000716)
        self.assertEqual(no_cache["effective_useful_output_tps"], cached["effective_useful_output_tps"])

    def test_quality_slo_losses_reduce_useful_capacity(self):
        base, lossy = self.scenario(), self.scenario(goodput_fraction=.8)
        self.assertAlmostEqual(lossy["local_cost_per_request"], base["local_cost_per_request"] / .8)
        self.assertAlmostEqual(lossy["effective_useful_output_tps"], 400)

    def test_vat_removed_once_not_as_twenty_percent_discount(self):
        self.assertEqual(effective_vat_cost(100, .2, 1), 100)
        self.assertEqual(effective_vat_cost(100, .2, 0), 120)
        self.assertAlmostEqual(effective_vat_cost(100, .2, .5), 110)

    def test_zero_provider_price_has_no_finite_paid_node_break_even(self):
        result = self.scenario(input_rate=0, output_rate=0)
        self.assertIsNone(result["break_even_output_tps_at_given_utilization"])
        self.assertFalse(result["break_even_possible_at_measured_output_tps"])
        json.dumps(result, allow_nan=False)
        free = self.scenario(input_rate=0, output_rate=0, node_hourly_cost=0)
        self.assertEqual(free["arithmetic_result"], "equal")

    def test_invalid_financial_and_workload_values_fail(self):
        for change in (dict(utilization=0), dict(utilization=1.1), dict(output_tps=-1),
                       dict(output_tokens=0), dict(input_tokens=-1), dict(input_rate=float("nan")),
                       dict(node_hourly_cost=float("inf")), dict(cache_hit_fraction=.2),
                       dict(cache_hit_fraction=2), dict(goodput_fraction=0), dict(output_rate=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.scenario(**change)

    def test_raw_benchmark_uses_observed_tokens_and_wall_time(self):
        data = dict(completed=2, failed=0, num_prompts=2, duration=4,
                    total_input_tokens=8000, total_output_tokens=2000, output_throughput=500)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bench.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            result = read_benchmark(path)
            self.assertEqual(result["output_tps"], 500)
            self.assertEqual(result["input_tokens"], 4000)
            self.assertEqual(result["output_tokens"], 1000)
            for change in (dict(failed=1), dict(completed=1), dict(output_throughput=2000),
                           dict(total_output_tokens=0), dict(duration=0)):
                path.write_text(json.dumps(dict(data, **change)), encoding="utf-8")
                with self.subTest(change=change), self.assertRaises(ValueError):
                    read_benchmark(path)

    def test_cli_defaults_to_provisional_even_if_arithmetic_favours_local(self):
        args = ["--currency", "USD", "--node-hourly-cost", "4", "--utilization", "1",
                "--output-tps", "1000", "--input-tokens", "4000", "--output-tokens", "1000",
                "--input-rate", ".2", "--output-rate", ".6", "--cache-hit-fraction", "0"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(args), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["comparison"]["arithmetic_result"], "local_cheaper")
        self.assertFalse(result["qualification"]["eligible_for_selection"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(args + ["--quality-gate", "pass", "--slo-gate", "pass"])
        self.assertTrue(json.loads(output.getvalue())["qualification"]["eligible_for_selection"])


if __name__ == "__main__":
    unittest.main()
