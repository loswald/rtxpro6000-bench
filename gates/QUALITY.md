# Quality evidence for throughput experiments

`quality_suite.py` adds a paired capability gate for **GLM-5.3-Flash**, **DeepSeek-V4-Flash**, and **Qwen3.8-Flash-Next**. Run it separately for each pinned checkpoint. It does not substitute a smaller model or interpret speed as quality. No model accuracy has been measured by adding this tooling.

The saved `quality20.py` results are corruption smoke checks, without an answer key or capability score. Seven DeepSeek result files contain 18–20 length-limited answers out of 20 at a 128-token cap. The old GSM8K run logs fail while resolving the `gptoss` served alias as a tokenizer. Those artifacts cannot establish preserved capability.

## What the gate measures

The default uses the existing [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): GSM8K CoT (math), BBH Boolean expressions (logic), and IFEval prompt-level strict accuracy (instruction following). The task names and metric filters were checked against the upstream task definitions. The actual installed harness source, task configuration, task versions, dataset document hashes, prompts and targets are recorded and must match across runs.

This is a diagnostic subset, **not an Artificial Analysis or Epoch index replication**. No coding, tool-use, multilingual, multimodal or long-context score is produced by the default suite. No result establishes universal capability preservation. Extend `tasks` with installed `generate_until` tasks exposing binary per-example metrics. Code-execution tasks are deliberately not enabled: use a separately isolated evaluator for generated code. Add held-out tool-call fixtures and representative long-context tasks before approving changes for those workloads. Keep development and final held-out evaluations separate; repeatedly choosing winners on this small subset overfits the gate.

The request cap defaults to **32,768 total generated tokens**, including reasoning according to the serving API's accounting. This is a starting configuration, not a claim about a model's native maximum. Set the intended production budget before either run, ensure input plus generation fits the server context, and increase both runs' budgets if either truncates. Match the intended production sampling, reasoning effort/budget, chat template and parser settings. Neither capture disables reasoning. `{}` for `chat_template_kwargs` preserves the server defaults; explicitly supply model-supported thinking controls for a reproducible production mode.

`generation.until` defaults to `[]`: stop at the model's end token, without the old completion-style task stops such as blank lines or `Q`, which can interrupt reasoning. This is an explicit protocol override, recorded identically on both sides; do not compare the resulting score directly to a leaderboard using different generation settings. `think_end_token` defaults to `</think>` for scoring responses that embed reasoning in final content; separate `reasoning`/`reasoning_content` fields and original final content remain in the raw captures.

## Run the pair

Use the existing isolated evaluation interpreter containing `lm_eval[api]` (for example `/workspace/venv-eval/bin/python`). The script never installs dependencies or changes the serving environment. `init` and `compare` require only Python's standard library. `run` sends requests to the URL you specify and can load task datasets; it does not launch servers or execute generated code.

Create one frozen suite per model. Fill in actual checkpoint and tokenizer revisions from the local download, and provide the prompt-format artifact actually used by the server (for example its tokenizer configuration or an exported manifest containing the template, custom tokenizer mode, reasoning parser and parser configuration). The fingerprint is an operator-supplied provenance record, not automatic proof that the deployment used those bytes. Retain the file and verify it against the launch. When changing engine kernels, keep these semantic settings fixed.

```bash
python3 gates/quality_suite.py init \
  --model-id deepseek-ai/DeepSeek-V4-Flash-0731 \
  --model-revision "$MODEL_REVISION" --tokenizer-revision "$TOKENIZER_REVISION" \
  --prompt-format /workspace/models/DeepSeek-V4-Flash-0731/tokenizer_config.json \
  --max-gen-toks 32768 --limit 200 --concurrency 8 \
  --out results/quality/deepseek-suite.json
```

For the other two candidates, use `zai-org/GLM-5.3-Flash` or `Qwen/Qwen3.8-Flash-Next-FP8` and that model's actual revisions and prompt-format artifact. Their existing served aliases are `glm53flash` and `qwen38flashnext`; DeepSeek's alias is `ds4flash`. The runner uses server-side chat formatting without trying to download a tokenizer named after the alias.

`--generation-json settings.json` accepts the complete nested generation object when creating a suite. For example, a model-supported production profile can specify temperature, top_p, chat_template_kwargs, until and reasoning_effort. Do not reuse a thinking flag from another architecture without verifying it is accepted and effective. Review the JSON before its first run. `--limit 0` uses the full task splits. Adjusting any setting requires rerunning both sides.

Capture the baseline launch, then relaunch the candidate optimization on the same hardware and workload. Each output directory must be new. Pass the corresponding real `launch.json` for review of what changed.

```bash
/workspace/venv-eval/bin/python gates/quality_suite.py run \
  --suite results/quality/deepseek-suite.json --url http://127.0.0.1:8000 \
  --served-model ds4flash --label baseline \
  --launch results/ds4flash_tp4/launch.json --out results/quality/deepseek-baseline

# Relaunch the candidate optimization, retaining the frozen evaluation settings.
/workspace/venv-eval/bin/python gates/quality_suite.py run \
  --suite results/quality/deepseek-suite.json --url http://127.0.0.1:8000 \
  --served-model ds4flash --label candidate \
  --launch results/ds4flash_tp4__candidate/launch.json --out results/quality/deepseek-candidate

python3 gates/quality_suite.py compare \
  --baseline results/quality/deepseek-baseline/quality.json \
  --candidate results/quality/deepseek-candidate/quality.json \
  --margin 0 --confidence 0.95 --out results/quality/deepseek-comparison.json
```

Each run retains its frozen suite, launch file, full harness results and samples, and one HTTP JSONL file per task. HTTP records include messages, all generation settings, raw responses, separate reasoning, finish reasons, usage, and elapsed time. Request credentials are not logged. These times are diagnostic, not a replacement for the throughput harness: quality capture adds overhead and does not enforce a serving latency SLO. Keep these artifacts private if using nonpublic task data.

## Interpret the result

- `invalid`: missing task/sample/raw answer, mismatched model/task/seed/sampling/budget/provenance, HTTP error, incomplete generation, or `finish_reason` other than `stop`. Truncated and reasoning-only answers invalidate the comparison instead of being silently discarded.
- `observed_regression`: at least one capability's observed accuracy fell beyond the predeclared margin. Other capabilities cannot compensate for that loss.
- `inconclusive`: no observed drop beyond the margin, but the sample cannot establish non-inferiority at the chosen confidence. Identical scores on a small subset at zero margin normally produce this result.
- `noninferiority_supported`: every configured capability's lower confidence bound clears the predeclared margin. This is the only passing status, and its scope is the measured protocol and task population.

The report lists paired gains, losses and changed document IDs for investigation. It uses conservative exact binomial bounds on gains and losses, with Bonferroni correction across both proportions and all tasks. The bound assumes the evaluated examples represent the task population; a prefix subset is principally a development diagnostic. Set an acceptable margin **before** examining outcomes. The default is zero; increasing the margin afterward to obtain a pass is not evidence of preserving quality. A repeat of the baseline is useful for detecting nondeterministic batch/kernel behavior even with fixed seeds. No assertion of an identical output distribution follows from deterministic accuracy agreement or an MTP/DSpark label.

Exit codes: `run` returns 0 for complete evidence, 2 for invalid evidence. `compare` returns 0 only for supported non-inferiority, 2 for invalid evidence, and 3 for regression or inconclusive evidence. Existing `gsm8k.sh` and `kv_diff.py` behavior is unchanged; their historical passes do not satisfy this paired gate.

Local verification (synthetic examples and a loopback mock server; no real model or paid API calls):

```bash
python3 -m unittest discover -s tests -p 'test_quality*.py' -v
```
