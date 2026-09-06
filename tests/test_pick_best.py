import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from box.pick_best import main, quality_passes


def good_smoke():
    return [dict(prompt=f"prompt {i}", text="answer", finish="stop", verdict="ok") for i in range(20)]


class PickBestTest(unittest.TestCase):
    def candidate(self, root, name, rate, smoke=None):
        directory = Path(root) / name
        directory.mkdir()
        (directory / "summary_full.tsv").write_text(
            f"tag\tlabel\tC\tout_tps\n{name}\trouter\t1024\t{rate}\n", encoding="utf-8"
        )
        if smoke is not None:
            (Path(root) / f"{name}_quality20.json").write_text(json.dumps(smoke), encoding="utf-8")
        return str(directory)

    def test_fast_corrupt_layout_loses_to_clean_layout(self):
        with tempfile.TemporaryDirectory() as root:
            bad = good_smoke()
            bad[0]["verdict"] = "degenerate"
            self.candidate(root, "fast_bad", 2100, bad)
            self.candidate(root, "clean", 1073, good_smoke())
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main([str(Path(root) / "*"), "router", "1024"])
            self.assertEqual((code, out.getvalue(), err.getvalue()), (0, "clean\n", "1073\n"))

    def test_missing_and_partial_quality_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            missing = self.candidate(root, "missing", 10000)
            partial = self.candidate(root, "partial", 9000, good_smoke()[:19])
            self.assertFalse(quality_passes(missing))
            self.assertFalse(quality_passes(partial))
            self.assertEqual(main([str(Path(root) / "*"), "router", "1024"]), 1)

    def test_duplicate_prompts_do_not_fake_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            items = good_smoke()
            items[-1] = items[0]
            directory = self.candidate(root, "duplicate", 1000, items)
            self.assertFalse(quality_passes(directory))

    def test_nonfinite_throughput_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self.candidate(root, "nan", "NaN", good_smoke())
            self.candidate(root, "inf", "inf", good_smoke())
            self.assertEqual(main([str(Path(root) / "*"), "router", "1024"]), 1)


if __name__ == "__main__":
    unittest.main()
