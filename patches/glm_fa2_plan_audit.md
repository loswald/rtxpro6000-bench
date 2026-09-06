# GLM FA2 planned-length experiment

This creates a diagnostic copy of the captured patched GLM backend. It leaves the input file untouched, rejects every source hash except `7a19dafb16f1a2f9ac58992ce78e4d27b8f52edf08059c387d4f32d70d0edab3`, and writes a provenance manifest alongside its output. No shared host was modified, and GPU behavior has not yet been tested. All 30 offline regressions pass in an isolated Torch `2.8.0+cpu` environment, including 12 real tensor tests for physical bounds, request ownership, conversion overflow, sentinel load masks, invalid live prefixes/tails, zero-count live rows, and exact replanning. The default local Python passes 18 and skips the 12 tensor tests because it has no Torch installation.

The corrected generated file is `analysis/glm_fa2_diagnostic/flashinfer_mla_sparse_sm90.py`, with its manifest alongside. The earlier `report/glm_fa2_diagnostic/` copy predates the current-metadata guard and must not be used. That path belongs to the benchmarking session and has been left untouched under the current ownership rules.

The check addresses one precise risk: FA2 uses host-planned lengths while the current index conversion produces exact `valid_counts`. The existing code discards those counts and clamps unused negative indices to slot zero. If a live row is overscheduled, slot zero incorrectly participates in attention. This is a demonstrated source hazard, not yet a demonstrated cause of the bad TP2/DP2/EP4 outputs.

## Prepare a separate backend copy

On the independent test host, use the captured source or an exact reproduction with the required hash. An example command is:

```bash
python patches/glm_fa2_plan_audit.py \
  --source tests/fixtures/glm_fa2_baseline.py \
  --output /workspace/glm-fa2-audit/flashinfer_mla_sparse_sm90.py
```

Use the emitted module only in a separate copy of the vendor vLLM package, or bind mount it at the matching backend path inside a dedicated test container. Point the test launch at that package copy. Preserve the original package, launch command, environment, source hash, and model revision. Reverting means selecting the original package again; the patch generator never edits it.

`tests/fixtures/glm_fa2_baseline.py` is the actual backend copied read-only from the existing host on 2026-09-05, not a replacement for AIRR's launcher. AIRR's `box/vllm_sm120_nope.py` is byte-identical to that host's patcher (SHA256 `22efaf2c645aea3097279c0e95269f047aabdc423a7ffeb6f16a4c9dedc8e1a0`). The vendor package is not stored elsewhere in AIRR. You can instead point `--source` at the reproduced vendor backend; its exact hash must match. The source fixture retains its original license header. Do not import it as a test module on a machine without the matching vLLM/Torch environment.

The current AIRR evidence still rejects the presumed fast baseline: `results/600w/probe/glm53f_dp2tp2ep2_s384_quality20.json` reports 11 OK, eight degenerate, and one wrong, while DP4/TP1's corresponding file reports 20 OK. The README's borrowed TP4 accuracy and `box/glm_perf3.sh`'s description of DP2/TP2 as fastest do not make that layout quality-qualified. Preserve the exact output quality gate when continuing those experiments. BF16 SSM and FP8 KV settings in the third sweep are separate precision-changing arms.

## Run the check arm

Configure these variables **before** starting each worker, while the marker path does not yet exist:

```bash
export GLM_FA2_AUDIT_MARKER=/workspace/glm-fa2-audit/check.ready
export GLM_FA2_AUDIT_MODE=check
export GLM_FA2_AUDIT_LOG_DIR=/workspace/results/glm_fa2_check
export GLM_FA2_AUDIT_MAX_RECORDS=256
```

Launch the exact baseline configuration with `--enforce-eager`. The instrumented backend rejects a configured marker together with graph mode at startup, even if the marker file is absent. Python checks cannot run on graph replay, so this is required for a meaningful result.

After the server is ready and dummy profiling has finished, create the marker:

```bash
touch /workspace/glm-fa2-audit/check.ready
```

Run the fixed 20 prompts sequentially, then boundary-focused requests with contexts around 2048, each kpool remainder, ragged/chunked prefills, and the actual cache page boundaries. Capture the client outputs and complete server log.

Before index conversion, the hook requires the host plan's active-row count to equal both the current `num_actual_tokens` and `query_start_loc[num_reqs]`. It checks query-start shape, a zero origin and nondecreasing offsets, callback query capacity, top-k storage, tensor dtypes/devices, physical cache geometry, and request-ID bounds. An undersized stale plan cannot hide a live query as padding. A decode-only callback with global metadata covering additional prefill rows fails with that evidence; exact mode does not silently change callback semantics. Padded converter rows with invalid request IDs are permitted only when every token index masks the block-table read.

Live request IDs must also match the sequence ownership implied by current query-start offsets. In chunks of at most 128 query rows, the check derives block-table columns in int64 and verifies each selected physical block on live rows before the converter's int32 multiplication. This prevents an oversized block ID from wrapping into an apparently valid slot. Converter address checks include all padded rows; physical block-value checks apply to live attention rows because zero-query padding never dereferences converted slots in FA2. The validation ignores unreferenced block-table padding and leaves every original index unchanged.

The pre-converter load guard mirrors Triton's division toward zero. A `-1` token sentinel has block column zero, and the pinned converter's block-table load mask excludes invalid columns but does not exclude negative tokens. Consequently, an all-`-1` padding row with request ID `-1` still fails the guard: masking the eventual output does not make that earlier read safe. A bad padding request ID is permitted only if every truncated block column falls outside the table. This follows the [pinned converter source](https://github.com/vllm-project/vllm/blob/487ecf187/vllm/v1/attention/backends/mla/sparse_utils.py) and [Triton integer-division semantics](https://triton-lang.org/main/python-api/triton-semantics.html). Whether real GLM batches exercise this hazard remains untested.

After conversion, each active attention call compares the current host plan with exact converted counts. The compact prefix must contain nonnegative indices below physical cache capacity, and its unused tail must contain exactly `-1`. The converted indices and counts must be int32 before any conversion to CPU, and every live row must have at least one visible cache entry. The first invalid or mismatched live row is recorded and raises before attention executes. Full-width lengths on metadata-confirmed zero-query padded rows are ignored; nonzero padding counts are reported separately. Counts are logged per live row, with process/rank, layer identifier, local request IDs, physical block size, and local head count.

Records are written to `GLM_FA2_AUDIT_LOG_DIR/glm_fa2_audit.<pid>.jsonl`, one file per worker. The configured limit bounds ordinary records; any terminating failure and the first eight exact corrections are always written, including corrections that begin after the ordinary log limit. Use a fresh output directory for each arm. `layer_object_id` distinguishes layers if the vendor attention object exposes no textual name. The local request IDs need correlation with the scheduler/client trace to recover prompt identity.

## Run the exact-count arm only as a diagnosis

Restart the dedicated test server using the same configuration, a fresh absent marker and log directory, and:

```bash
export GLM_FA2_AUDIT_MODE=exact
```

After startup, create its marker and rerun the same requests. When a live planned length differs from the conversion's count, the hook calls the existing FA2 `state.plan` using the exact live counts. It retains the original active row count only after the current batch metadata independently confirms it, and retains inert padded rows. It changes no model weights, top-k configuration, sparse indices, cache entries, query values, scale, or index masks. The subsequent existing slot copy and attention call are unchanged. Malformed compact prefixes, out-of-range counts or physical slots, metadata/geometry mismatches, and every all-masked live row stop the run.

An `exact_replan_applied` field identifies actual corrections. This arm introduces CPU synchronization and may replan separately per layer; it is intentionally unsuitable for throughput comparison or production deployment. If it repairs repeated bad outputs while the check arm shows length mismatches, that supplies a concrete lead for a graph-safe implementation. It does not by itself prove that every quality regression is resolved. If eager mode already fixes the failure and counts match, focus on graph lifetime or replay metadata instead.

## Interpretation and next action

| Result | Next action |
| --- | --- |
| Check fails on a count mismatch, exact mode repairs it and the affected outputs | Repair the source of stale/inexact planned lengths; build a separate graph-safe approach and rerun the full quality gate |
| Check fails on prefix layout or invalid count | Debug index conversion/indexer semantics; do not accept exact replanning as a fix |
| Check fails on active-row metadata or callback storage | Inspect builder/dispatch semantics and split decode/prefill handling; exact mode deliberately cannot repair this |
| Eager check passes and outputs recover | Investigate capture/replay, buffer ownership, and padded metadata, with graph tests |
| Eager check passes and outputs remain corrupt | Compare BF16 FA2 to a sparse-attention reference and isolate TP/EP reduction and per-rank state |
| Both layouts pass smoke | Run the complete paired capability suite; smoke coherence is insufficient for promotion |

For a clean performance measurement, return to the original module and remove all audit instrumentation from the measured arm. A future accepted fix must get its own runtime/source fingerprint, full quality result, and repeated complete serving measurements.
