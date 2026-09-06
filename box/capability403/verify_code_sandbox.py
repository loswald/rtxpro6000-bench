"""Fixed, benign integration checks against the original code driver's scoring modes."""
import json
import sys
from run import SUITE, verify_suite
from sandbox import Sandbox


def main():
    verify_suite()
    sys.path.insert(0, str(SUITE))
    from families import code
    sandbox = Sandbox()
    try:
        report = sandbox.prepare()
        sandbox.patch_code_module(code)
        item = {"subfamily": "mbppplus", "test_imports": [], "test": "assert add(2, 3) == 5\nassert add(-1, 4) == 3"}
        good = code.run_candidate(item, "def add(a, b): return a + b")
        wrong = code.run_candidate(item, "def add(a, b): return a - b")
        early = code.run_candidate(item, "import sys\nsys.exit(0)")
        forged = code.run_candidate(item, "import os\nopen('_done_fake', 'w').close()\nos._exit(0)")
        assert good["passed"], good
        assert not wrong["passed"] and wrong["kind"] == "assertion", wrong
        assert not early["passed"] and early["kind"] == "early_exit", early
        assert not forged["passed"] and forged["kind"] == "early_exit", forged
        report["original_driver_checks"] = {"correct_passed": True, "wrong_rejected": True,
                                             "early_exit_rejected": True, "incorrect_completion_marker_rejected": True}
        print(json.dumps(report, indent=2))
    finally:
        sandbox.close()


if __name__ == "__main__":
    main()
