#!/usr/bin/env bash
# Post-extraction fix-up for a lifted SGLang image tree. Everything here was learned the
# hard way on 3 Sept 2026; apply it to a FRESH extraction, never pip into the tree.
#   usage: bash sglfix.sh /workspace/sglimg2
set -u
IMG=${1:?image dir}
SITE=$IMG/opt/sglang/lib/python3.12/site-packages
K=$IMG/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py
STASH=/workspace/_stale_$(basename "$IMG"); mkdir -p "$STASH"
echo "== fix-up for $IMG =="

echo "-- 1. duplicated dist-info metadata: keep the highest version of each --"
# transformers' version check reads the FIRST dist-info it finds; the image ships two
# tokenizers (0.22.2 stale, 0.23.1 real) and the stale one wins, silently dropping every
# GLM model module. Same pattern for pillow and nccl.
for pkg in $(ls -d "$SITE"/*.dist-info | xargs -n1 basename | sed -E 's/-[0-9][^-]*\.dist-info$//' | sort | uniq -d); do
  for low in $(ls -d "$SITE"/"${pkg}"-*.dist-info | sort -V | head -n -1); do
    mv "$low" "$STASH/" && echo "   moved aside $(basename "$low")"
  done
done

echo "-- 2. TileLang sparse kernel: single pipeline stage on sm_120 (99 KB shared memory) --"
[ -f "$K.orig" ] || cp "$K" "$K.orig"
python3 - "$K" <<'PY'
import sys
p = sys.argv[1]; s = open(p + ".orig").read()
helper = '''
# sm_120 (RTX PRO 6000 / RTX 5090) has 99 KB of opt-in shared memory per block; the default
# block_I=64, num_stages=2 tiles need ~148 KB for D=512 bf16 and fail to launch. One stage
# keeps the warp layout (block_I=32 trips "warp_row_tiles must be greater than 16").
def _small_smem():
    try:
        import torch
        return torch.cuda.get_device_properties(0).shared_memory_per_block_optin < 120_000
    except Exception:
        return False
# Validated on GB10 (sm_121) by the DGX Spark SGLang recipe: block_I=32, num_stages=1, threads=128.
_TILE_KW = {"block_I": 32, "num_stages": 1, "threads": 128} if _small_smem() else {}

'''
i = s.index("\ndef ") + 1
lines = s[:i].rstrip("\n").split("\n")
while lines and lines[-1].startswith("@"):
    lines.pop()
head = "\n".join(lines) + "\n"
s = head + helper + s[len(head):]
old = "sm_scale=sm_scale, return_lse=return_lse\n"
assert old in s, "call site not found"
s = s.replace(old, "sm_scale=sm_scale, return_lse=return_lse, **_TILE_KW\n", 1)
open(p, "w").write(s)
import ast; ast.parse(s)
print("   patched and parses; widened call sites:", s.count("**_TILE_KW"))
PY

echo "-- 3. the editable installs hardcode /sgl-workspace --"
rm -f /sgl-workspace 2>/dev/null
ln -s "$IMG/sgl-workspace" /sgl-workspace && ls -ld /sgl-workspace | cut -c1-100 | sed 's/^/   /'

echo "-- 3b. the sm_120 code patches (each is idempotent) --"
SRT=$IMG/sgl-workspace/sglang/python/sglang/srt
HERE=$(dirname "$0")
python3 "$HERE/dsa_sm120.py" "$SRT/layers/attention/dsa_backend.py" | sed 's/^/   /'          # dense prefill on sm_120
python3 "$HERE/kda_patch.py" "$SRT/layers/attention/hybrid_linear_attn_backend.py" | sed 's/^/   /'   # Triton causal-conv knob
python3 "$HERE/pdl_patch.py" "$IMG/sgl-workspace/sglang/python/sglang/kernels/jit/utils/arch.py" | sed 's/^/   /'  # PDL off knob
python3 - "$SRT/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py" <<'PY'
import sys; p=sys.argv[1]; s=open(p).read()
old="        else:\n            # TODO(cwan): refactor other backends\n            pass\n"
new="        else:\n            # sm_120 patch: flashinfer_cutlass has no runner for this scheme; use Triton\n            self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)\n"
if old in s: open(p,"w").write(s.replace(old,new,1)); print("   MTP W8A8-FP8 MoE runner fallback: patched")
else: print("   MTP W8A8-FP8 MoE runner fallback:", "already" if "sm_120 patch" in s else "ANCHOR MISSING")
PY

echo "-- 4. sanity: cutlass DSL dir and cudnn present? --"
[ -d "$SITE/nvidia_cutlass_dsl/dsl_packages/cutlass" ] && echo "   cutlass DSL: present" || echo "   cutlass DSL: MISSING"
[ -n "$(find "$SITE/nvidia" -name 'libcudnn.so.9' 2>/dev/null | head -1)" ] && echo "   libcudnn.so.9: present" || echo "   libcudnn.so.9: MISSING"
echo "== fix-up done =="
