import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.select_candidate import screen_candidate, select, smoke_check


def smoke():
    return [{"prompt": "prompt %d" % i, "text": "answer", "finish": "stop", "verdict": "ok"} for i in range(20)]


def benchmark(rate=1000):
    return {"completed": 256, "num_prompts": 256, "failed": 0, "duration": 32,
            "total_output_tokens": 32768, "total_input_tokens": 262144,
            "output_throughput": rate, "request_throughput": 8, "total_token_throughput": 9216}


class CandidateTests(unittest.TestCase):
    def test_fast_corrupted_candidate_loses_to_clean_candidate(self):
        corrupt = smoke()
        corrupt[0].update(verdict="degenerate", text="bul " * 100)
        candidates = [screen_candidate("fast", [benchmark(1300)], {}, corrupt, 128, legacy_screen=True),
                      screen_candidate("slower", [benchmark(1073)], {}, smoke(), 128, legacy_screen=True)]
        self.assertEqual(select(candidates)["selected"], "slower")
        self.assertFalse(select(candidates)["promotion_eligible"])

    def test_legacy_smoke_without_verdict_does_not_pass(self):
        items = smoke()
        for item in items:
            del item["verdict"]
        self.assertFalse(smoke_check(items)["pass"])

    def test_missing_quality_never_wins(self):
        candidate = screen_candidate("fast", [benchmark()], {}, None, 128, legacy_screen=True)
        self.assertIsNone(select([candidate])["selected"])

    def test_legacy_unknown_provenance_needs_explicit_opt_in(self):
        candidate = screen_candidate("legacy", [benchmark()], {}, smoke(), 128)
        self.assertFalse(candidate["screening_eligible"])
        candidate = screen_candidate("legacy", [benchmark()], {}, smoke(), 128, legacy_screen=True)
        self.assertTrue(candidate["screening_eligible"])
        self.assertTrue(candidate["requires_new_integrity_run"])

    def test_partial_fast_run_is_rejected_despite_good_smoke(self):
        bad = benchmark(99999)
        bad.update(completed=100, failed=156, total_output_tokens=12800)
        self.assertFalse(screen_candidate("bad", [bad], {}, smoke(), 128, legacy_screen=True)["screening_eligible"])

    def test_short_output_rejected_with_unknown_metadata(self):
        bad = benchmark()
        bad["total_output_tokens"] -= 1
        self.assertFalse(screen_candidate("bad", [bad], {}, smoke(), 128, legacy_screen=True)["screening_eligible"])

    def test_duplicate_prompts_cannot_manufacture_twenty_passes(self):
        self.assertFalse(smoke_check([smoke()[0]] * 20)["pass"])

    def test_live_glm_corruption_is_excluded(self):
        probe = ROOT / "results/live_20260905_host1/results/probe"
        path = probe / "glm53f_dp2tp2ep2_s384_quality20.json"
        if not path.exists():
            self.skipTest("optional live snapshot absent")
        self.assertFalse(smoke_check(json.loads(path.read_text(encoding="utf-8")))["pass"])
        good = probe / "glm53f_dp4ep4_s192_quality20.json"
        self.assertTrue(smoke_check(json.loads(good.read_text(encoding="utf-8")))["pass"])


if __name__ == "__main__":
    unittest.main()
