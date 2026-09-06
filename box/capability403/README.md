# Matched 403-case capability evaluation

This adapter preserves the original AIRR suite's 403 cases, source, scorers, sampling, and completion budgets. The frozen suite is a sibling directory; `build_package.py` includes it in a self-contained archive. No datasets are rebuilt or fetched. The historical manifest digest is `efb50b88d5b0aaae7f92d527bf05c761c0da6c9c153af5bde45e1add2ed3735b`.

| Family | Cases | Completion cap |
|---|---:|---:|
| Math | 80 | 32,768 |
| Code | 75 | 20,480 |
| Knowledge | 70 | 20,480 |
| Instruction following | 60 | 16,384 |
| Tools, JSON prompt mode | 70 | 8,192 |
| Long context | 48 | 6,144 |

Both arms use temperature 1, top-p 0.95, seed 20260903, concurrency 64, a 40,960-token server context, 7,260-second wall budget plus 120-second grace, 3,600-second request timeout, and three retries with the original 2/8/20-second backoff. Completion caps and reasoning settings are explicit. The baseline ledger records full request bodies; the optimized arm replays them, including the calibrated long-context prompts. The model revision must match. Sampling remains stochastic despite a fixed seed, so inspect paired regressions and repeat matched runs when differences matter.

## Preparation and execution

Linux system Python 3.12, `aiohttp`, `libseccomp2`, and root permission for filesystem isolation setup are required. Generated code always runs after dropping all user/group IDs to a unique unprivileged identity. These commands are invoked from the extracted archive's directory:

```sh
python3 capability403/run.py --check
python3 capability403/run.py --preflight
python3 capability403/verify_code_sandbox.py
python3 capability403/run.py --tag baseline --model-revision EXACT_REVISION --engine-label stock --provenance-file /path/to/stock-launch.sh --out results
python3 capability403/run.py --tag optimized --model-revision EXACT_REVISION --engine-label candidate --provenance-file /path/to/candidate-launch.sh --replay-requests results/baseline.requests.jsonl --out results
python3 capability403/compare.py results/baseline results/optimized --out results/comparison.json
```

The example endpoint is `http://127.0.0.1:8000`; `--base-urls` accepts numeric loopback endpoints. The adapter clears `EVAL_API_KEY` for this local run. Change server deployment between arms, preserving checkpoint and context window. Save engine/config/patch files through repeated `--provenance-file` options. These hashes document supplied artifacts; operators must still match them to the actual server launch.

`--check` is offline. `--preflight` and `verify_code_sandbox.py` execute only fixed, benign programs and make no model requests. Every real run repeats isolation preflight before opening the endpoint. A failed preflight aborts the run. Existing tags cannot be overwritten. The comparator rejects missing cases, infrastructure errors, mismatched source/settings/checkpoints, artifact hash changes, or changed request bodies. HTTP retries remain inside each logical request and use the original client policy.

## Code isolation

The new host blocks namespace creation. This package therefore uses a minimal chroot containing only Python's standard library and its shared-library dependencies, a separate case directory, an empty temporary directory, no host mounts, no `/proc`, and no devices. Immutable runtime files are shared through read-only hardlinks; per-case files are separate. The child closes inherited descriptors except standard input/output/error, removes environment values, drops supplementary groups and all UID/GID slots, sets `no_new_privs`, and applies a seccomp allowlist before reading candidate code. Socket, process creation, executable loading, ptrace, mount, and host-signal syscalls are denied. Hard limits cover memory, CPU, file size, open files, and process count; wall timeout kills the process group.

The fixed preflight tests host-file isolation, runtime write protection, socket/fork/exec/signal denial, unprivileged identity, clean environment, absent host paths/devices, hard limits, and stdlib operation. This is kernel/process isolation, not a virtual machine: it does not claim protection from kernel vulnerabilities or side channels. Do not interpret an isolated code score as proof of adversarial judge integrity. As in the original suite, candidate and hidden tests execute in one interpreter and introspection can inspect tests or completion markers.

## Interpretation

The original scorers are deliberately retained for historical comparability. They can rescue answers from reasoning, accept some correct answers from length-limited responses, and use custom subsets/scoring for tools and instruction following. A 20-case tripwire is insufficient to establish broad capability. `compare.py` reports each family's discordant case IDs, exact McNemar result, truncations, and reasoning rescues; aggregate equality or a nonsignificant test never certifies parity. Pair these results with the separate strict quality gate. Throughput from long generated answers is not SLO goodput or useful completed work.

Build locally with `python box/capability403/build_package.py report/priority/capability403.tar.gz`. Verify adapter behavior with `python -m unittest discover -s box/capability403 -p test_adapter.py`. Runtime Python files and every frozen suite source/data file are hashed in run artifacts.
