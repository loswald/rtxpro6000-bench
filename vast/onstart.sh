#!/usr/bin/env bash
# onstart.sh -- Vast.ai onstart script for the 4x RTX PRO 6000 Blackwell benchmark box.
#
# Handed to Vast with:  vastai create instance OFFER --image ... --ssh --direct --onstart ./vast/onstart.sh --env '-e KEY=VAL ...'
# (the CLI reads this file and uploads its CONTENTS; Vast stores it as /root/onstart.sh and runs it as root inside the
# container on every start/restart, after sshd is up, with the Vast env set: CONTAINER_ID, PUBLIC_IPADDR, GPU_COUNT,
# DATA_DIRECTORY, VAST_TCP_PORT_22, CONTAINER_API_KEY, ...).  In SSH launch mode the image ENTRYPOINT (the vLLM API
# server) is NOT started -- the harness launches engines itself.
#
# Layout on the box:  /workspace/rtxpro6000-bench = this harness (BENCH_ROOT, pushed by ./vast/sync.sh push)
#                     /workspace/models            = weights as plain directories: hf download <repo> --local-dir /workspace/models/<basename>
#                     /workspace/bench             = the box's own scratch scripts (not ours)     /workspace/tools = built helpers
#
# Tunables (pass via --env '-e KEY=VALUE'):
#   DOWNLOAD_SET      none (default). The disk is ~390 GB, so models are fetched one cell at a time by bench/prefetch.sh
#                     and deleted after their sweep (README "Disk-constrained order"). Other values: small | core | all |
#                     comma list of the keys below -- the download script refuses a model that does not fit the free disk.
#   ALLOW_ENGINE_PIP  0 (default). onstart NEVER pip-installs into the engine interpreter: the image's stale vLLM is upgraded
#                     in place with uv (vLLM main cu130 nightly wheels + flashinfer + vllm[b12x]) by bench/setup_engine.sh or by
#                     hand, and a stray 'pip install' would drag torch/vllm back to PyPI builds. 1 = allow the guarded b12x install.
#   ENGINE            auto (default: detect vllm/sglang import) | vllm | sglang
#   HARNESS_REPO      git URL to clone into $BENCH_ROOT (optional; default: wait for ./vast/sync.sh push)
#   HARNESS_WAIT_MIN  minutes to wait for $BENCH_ROOT/vast/hardware_truth.sh before giving up (default 90)
#   INSTALL_TRAIN     1 (default) -> /workspace/venv-train with unsloth for the co-tenancy cell; 0 to skip
#   INSTALL_EVAL      1 (default) -> /workspace/venv-eval with lm-eval; 0 to skip
#   SKIP_BUILDS       1 -> skip cuda-samples / nccl-tests builds (hardware_truth.sh has torch fallbacks for both)
#
# Everything is idempotent: a restart re-exports env, re-runs the (skip-if-complete) download script, and re-runs the
# hardware truth only if results/hw/decisions.env is missing. Safe to re-run on the already-upgraded instance.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
LOG=/workspace/onstart.log
mkdir -p /workspace && exec > >(tee -a "$LOG") 2>&1
echo "==== onstart $(date -u +%FT%TZ) container=${CONTAINER_ID:-?} ip=${PUBLIC_IPADDR:-?} gpus=${GPU_COUNT:-?} ===="

# ------------------------------------------------------------------ layout + env
export BENCH_ROOT="${BENCH_ROOT:-/workspace/rtxpro6000-bench}" TOOLS_DIR=/workspace/tools MODELS_DIR=/workspace/models HF_HOME=/workspace/hf
export DL_DIR=/workspace/downloads HW_DIR="${HW_DIR:-$BENCH_ROOT/results/hw}"
TOOLS_VENV=/workspace/venv-tools            # isolated venv for the hf CLI (never the engine interpreter)
mkdir -p "$BENCH_ROOT/results" "$TOOLS_DIR" "$MODELS_DIR" "$HF_HOME" "$DL_DIR"
DOWNLOAD_SET="${DOWNLOAD_SET:-none}"; ENGINE="${ENGINE:-auto}"; HARNESS_WAIT_MIN="${HARNESS_WAIT_MIN:-90}"
INSTALL_TRAIN="${INSTALL_TRAIN:-1}"; INSTALL_EVAL="${INSTALL_EVAL:-1}"; SKIP_BUILDS="${SKIP_BUILDS:-0}"
ALLOW_ENGINE_PIP="${ALLOW_ENGINE_PIP:-0}"
MARK=/workspace/.onstart_done
have() { command -v "$1" >/dev/null 2>&1; }

if [ "$ENGINE" = "auto" ]; then
  if python3 -c "import sglang" 2>/dev/null; then ENGINE=sglang; elif python3 -c "import vllm" 2>/dev/null; then ENGINE=vllm; else ENGINE=unknown; fi
fi
echo "ENGINE=$ENGINE DOWNLOAD_SET=$DOWNLOAD_SET INSTALL_TRAIN=$INSTALL_TRAIN INSTALL_EVAL=$INSTALL_EVAL"

# sm_120 baseline + NCCL baseline. NCCL_P2P_DISABLE is NEVER set here or by hardware_truth.sh (P2P works on this box and is
# faster than host staging); the hardware/decision contract lives in $HW_DIR/decisions.env, written by hardware_truth.sh.
BASE_ENV="BENCH_ROOT=$BENCH_ROOT TOOLS_DIR=$TOOLS_DIR MODELS_DIR=$MODELS_DIR HF_HOME=$HF_HOME DL_DIR=$DL_DIR HW_DIR=$HW_DIR ENGINE=$ENGINE
DOWNLOAD_SET=$DOWNLOAD_SET
VLLM_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0
NCCL_P2P_LEVEL=PHB NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8
HF_HUB_DISABLE_TELEMETRY=1 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Vast recommendation: copy the container env into /etc/environment so ssh/tmux sessions see it.
# CONTAINER_API_KEY is deliberately NOT written to any file (tmux sessions started here inherit it in memory).
{ env | grep '_' | grep -v -E '^(CONTAINER_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|PATH|LS_COLORS|_)=' ; echo "$BASE_ENV" | tr ' ' '\n'; } | awk -F= '!seen[$1]++' > /etc/environment
# profile.d exports ONLY the sm_120/NCCL baseline above.  The hardware/decision contract ($HW_DIR/decisions.env) is read by
# bench/env.sh at run time and is deliberately NOT exported into login shells (an exported, stale P2P_OK/ACS_SUSPECTED/
# NCCL_P2P_DISABLE would shadow the file); the old `source $BENCH_ROOT/env.sh` line is gone for the same reason.
{ echo '# generated by onstart.sh'; echo "$BASE_ENV" | tr ' ' '\n' | sed 's/^/export /'
  echo "# hardware decisions: $HW_DIR/decisions.env (read by bench/env.sh at run time; never exported here)"; } > /etc/profile.d/10-bench.sh
grep -q 'profile.d/10-bench.sh' /root/.bashrc 2>/dev/null || echo 'source /etc/profile.d/10-bench.sh' >> /root/.bashrc
export PATH=/usr/local/cuda/bin:$PATH

# ------------------------------------------------------------------ CUDA toolkit version (for apt package names / tags)
CUDA_VER="${CUDA_VERSION:-}"
[ -z "$CUDA_VER" ] && [ -f /usr/local/cuda/version.json ] && CUDA_VER="$(grep -o '"version" *: *"[0-9.]*"' /usr/local/cuda/version.json | head -1 | grep -o '[0-9.]*$')"
[ -z "$CUDA_VER" ] && CUDA_VER="$(python3 -c 'import torch;print(torch.version.cuda)' 2>/dev/null)"
CUDA_MAJ="${CUDA_VER%%.*}"; CUDA_MIN="$(echo "$CUDA_VER" | cut -d. -f2)"; CUDA_MAJ="${CUDA_MAJ:-13}"; CUDA_MIN="${CUDA_MIN:-0}"
echo "CUDA toolkit version guess: $CUDA_VER (apt suffix ${CUDA_MAJ}-${CUDA_MIN})"

# ------------------------------------------------------------------ apt + pip (first boot only)
if [ ! -f "$MARK" ]; then
  echo "---- apt"
  apt-get update -qq || true
  apt-get install -y -qq --no-install-recommends tmux git jq rsync pciutils cmake build-essential numactl curl wget htop iproute2 bc python3-venv ca-certificates >/dev/null || echo "WARN: some apt packages failed"
  if ! have nvcc; then
    echo "---- nvcc missing -> installing cuda-nvcc-${CUDA_MAJ}-${CUDA_MIN} (+cudart-dev, cccl) from the NVIDIA apt repo"
    if ! apt-cache policy "cuda-nvcc-${CUDA_MAJ}-${CUDA_MIN}" 2>/dev/null | grep -q Candidate; then
      . /etc/os-release; UB="ubuntu$(echo "$VERSION_ID" | tr -d .)"
      wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UB}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb && apt-get update -qq
    fi
    apt-get install -y -qq --no-install-recommends "cuda-nvcc-${CUDA_MAJ}-${CUDA_MIN}" "cuda-cudart-dev-${CUDA_MAJ}-${CUDA_MIN}" "cuda-cccl-${CUDA_MAJ}-${CUDA_MIN}" >/dev/null \
      || echo "WARN: nvcc install failed -> p2p/nccl builds will fall back / be skipped"
    [ -d "/usr/local/cuda-${CUDA_MAJ}.${CUDA_MIN}" ] && [ ! -e /usr/local/cuda ] && ln -s "/usr/local/cuda-${CUDA_MAJ}.${CUDA_MIN}" /usr/local/cuda
  fi
  # NCCL headers for nccl-tests: apt libnccl-dev if the repo has a cuda-${CUDA_MAJ} build, else reuse the pip wheel
  apt-get install -y -qq --no-install-recommends libnccl2 libnccl-dev >/dev/null 2>&1 && echo "libnccl-dev from apt" || echo "libnccl-dev not from apt (will use pip wheel)"

fi

# ------------------------------------------------------------------ engine interpreter: look, don't touch
echo "---- engine: $(python3 -c 'import vllm;print("vllm",vllm.__version__)' 2>/dev/null || python3 -c 'import sglang;print("sglang",sglang.__version__)' 2>/dev/null || echo none)  torch: $(python3 -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo -)  flashinfer: $(python3 -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null || echo -)  b12x: $(python3 -c 'import b12x;print(getattr(b12x,"__version__","present"))' 2>/dev/null || echo -)  (ALLOW_ENGINE_PIP=$ALLOW_ENGINE_PIP)"
# The image's vLLM is stale and gets upgraded IN PLACE with uv (vLLM main cu130 nightly wheels + flashinfer-python + vllm[b12x];
# see bench/setup_engine.sh). Nothing below may pip-install into that interpreter: 'pip install -U huggingface_hub' or
# 'pip install b12x' can silently pull torch/vllm back to PyPI builds. Tools live in an isolated venv instead.
if ! have hf; then
  echo "---- hf CLI missing -> isolated venv $TOOLS_VENV"
  if [ -x "$TOOLS_VENV/bin/python" ] || python3 -m venv "$TOOLS_VENV" 2>/dev/null || { have uv && uv venv -q --seed "$TOOLS_VENV"; }; then
    "$TOOLS_VENV/bin/python" -m pip install -q -U pip "huggingface_hub[cli]>=0.34" hf_transfer >/dev/null 2>&1 \
      && ln -sfn "$TOOLS_VENV/bin/hf" /usr/local/bin/hf && echo "hf CLI -> /usr/local/bin/hf ($("$TOOLS_VENV/bin/hf" version 2>/dev/null | head -1))" \
      || echo "WARN: hf CLI install into $TOOLS_VENV failed; by hand: uv tool install huggingface_hub"
  else
    echo "WARN: could not create $TOOLS_VENV (python3-venv missing?); by hand: uv tool install huggingface_hub"
  fi
fi
if [ "$ALLOW_ENGINE_PIP" = "1" ] && [ "$ENGINE" = "vllm" ] && ! python3 -c "import b12x" 2>/dev/null; then
  # Explicit opt-in only. 'vllm[b12x]' == b12x pinned by the installed vLLM; dry-run first and refuse if it would touch the engine.
  plan="$(python3 -m pip install --dry-run b12x 2>&1 | grep -i 'Would install' || true)"
  echo "pip dry-run: ${plan:-<nothing>}"
  if echo "$plan" | grep -qiE 'torch-|vllm-|flashinfer'; then
    echo "WARN: installing b12x would replace torch/vllm/flashinfer -> skipped. Use bench/setup_engine.sh."
  else
    python3 -m pip install -q b12x && echo "b12x installed: $(python3 -c 'import b12x;print(getattr(b12x,"__version__","?"))')" || echo "WARN: b12x install failed"
  fi
fi

# ------------------------------------------------------------------ tmux helper
tmux_run() { # tmux_run SESSION 'command...'  -> command is written to a script so any quoting inside is safe
  local name="$1"; shift; local f="$TOOLS_DIR/tmux-$name.sh"
  printf '#!/usr/bin/env bash\nsource /etc/profile.d/10-bench.sh 2>/dev/null\n%s\nrc=$?\necho; echo "[%s] finished rc=$rc $(date -u +%%FT%%TZ)"\nsleep infinity\n' "$*" "$name" > "$f"
  chmod +x "$f"
  tmux has-session -t "$name" 2>/dev/null && { echo "tmux session $name already running"; return 0; }
  tmux new-session -d -s "$name" "bash $f"
}
mkvenv() { # mkvenv DIR -> venv that can see the image's torch/transformers
  python3 -m venv --system-site-packages "$1" 2>/dev/null && return 0
  python3 -m pip install -q virtualenv >/dev/null 2>&1 && python3 -m virtualenv -q --system-site-packages "$1" && return 0
  have uv && uv venv --system-site-packages --python python3 "$1" && return 0
  echo "WARN: could not create venv $1"; return 1
}
export -f mkvenv have
python3 -m pip --version >/dev/null 2>&1 || python3 -m ensurepip --upgrade >/dev/null 2>&1 || apt-get install -y -qq python3-pip >/dev/null 2>&1 || true

# ------------------------------------------------------------------ build cuda-samples p2p test + nccl-tests (background)
cat > "$TOOLS_DIR/build_tools.sh" <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH
T="${TOOLS_DIR:-/workspace/tools}"; cd "$T"
CUDA_VER="$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | awk '{print $2}')"; CUDA_MAJ="${CUDA_VER%%.*}"; CUDA_MIN="$(echo "$CUDA_VER" | cut -d. -f2)"
echo "nvcc: ${CUDA_VER:-MISSING}"
if ! command -v nvcc >/dev/null; then echo "no nvcc -> cannot build p2p/nccl tests"; touch "$T/.build_done"; exit 0; fi
# --- cuda-samples: try the tag matching the toolkit, else master
if [ ! -x "$T/p2pBandwidthLatencyTest" ]; then
  rm -rf cuda-samples
  git clone -q --depth 1 --branch "v${CUDA_MAJ}.${CUDA_MIN}" https://github.com/NVIDIA/cuda-samples 2>/dev/null || git clone -q --depth 1 https://github.com/NVIDIA/cuda-samples
  CU="$(find cuda-samples -name p2pBandwidthLatencyTest.cu | head -1)"; SD="$(dirname "$CU")"
  echo "p2p sample at $SD"
  if cmake -S "$SD" -B "$SD/build" -DCMAKE_CUDA_ARCHITECTURES=120 >/dev/null 2>&1 && cmake --build "$SD/build" -j"$(nproc)" >/dev/null 2>&1; then
    cp "$(find "$SD/build" -type f -name p2pBandwidthLatencyTest | head -1)" "$T/p2pBandwidthLatencyTest"
    echo "p2pBandwidthLatencyTest built via cmake"
  else
    echo "cmake path failed -> direct nvcc"
    nvcc -O2 -arch=sm_120 -I cuda-samples/Common -I cuda-samples/cpp/Common -I cuda-samples/Common/inc "$CU" -o "$T/p2pBandwidthLatencyTest" && echo "p2pBandwidthLatencyTest built via nvcc" || echo "FAILED: p2pBandwidthLatencyTest"
  fi
fi
# --- nccl-tests (no MPI), sm_120 only, NCCL from apt or the pip wheel shipped with torch
if [ ! -x "$T/nccl-tests/build/all_reduce_perf" ]; then
  rm -rf nccl-tests; git clone -q --depth 1 https://github.com/NVIDIA/nccl-tests
  NCCL_HOME=""
  if [ -f /usr/include/nccl.h ]; then NCCL_HOME=/usr
  else
    PYN="$(python3 -c 'import nvidia.nccl,os;print(os.path.dirname(nvidia.nccl.__file__))' 2>/dev/null)"
    if [ -n "$PYN" ] && [ -f "$PYN/include/nccl.h" ]; then
      mkdir -p "$T/nccl_home/lib" && ln -sfn "$PYN/include" "$T/nccl_home/include"
      ln -sfn "$(ls "$PYN"/lib/libnccl.so.2* | head -1)" "$T/nccl_home/lib/libnccl.so" && ln -sfn "$(ls "$PYN"/lib/libnccl.so.2* | head -1)" "$T/nccl_home/lib/libnccl.so.2"
      NCCL_HOME="$T/nccl_home"
    fi
  fi
  echo "NCCL_HOME=${NCCL_HOME:-<none>}"
  make -s -C nccl-tests -j"$(nproc)" MPI=0 CUDA_HOME=/usr/local/cuda ${NCCL_HOME:+NCCL_HOME=$NCCL_HOME} NVCC_GENCODE="-gencode=arch=compute_120,code=sm_120" >/dev/null 2>&1 \
    && echo "nccl-tests built: $(ls nccl-tests/build | tr '\n' ' ')" || echo "FAILED: nccl-tests build"
  [ -n "$NCCL_HOME" ] && [ "$NCCL_HOME" != /usr ] && echo "export LD_LIBRARY_PATH=$NCCL_HOME/lib:\${LD_LIBRARY_PATH:-}" > "$T/nccl_env.sh"
fi
touch "$T/.build_done"; echo "build_tools done $(date -u +%FT%TZ)"
EOS
chmod +x "$TOOLS_DIR/build_tools.sh"
if [ "$SKIP_BUILDS" = "1" ]; then touch "$TOOLS_DIR/.build_done"; else tmux_run build "TOOLS_DIR=$TOOLS_DIR $TOOLS_DIR/build_tools.sh 2>&1 | tee $TOOLS_DIR/build.log"; fi

# ------------------------------------------------------------------ model downloads (background, sequential, resumable)
cat > "$TOOLS_DIR/download_models.sh" <<'EOS'
#!/usr/bin/env bash
# download_models.sh [SET|keys]  -- sequential hf download into $MODELS_DIR/<basename>; skips models with a .complete marker
set -uo pipefail
M="${MODELS_DIR:-/workspace/models}"; D="${DL_DIR:-/workspace/downloads}"; mkdir -p "$M" "$D"
SET="${1:-${DOWNLOAD_SET:-none}}"
declare -A REPO EXTRA SIZE_GB
# GB on disk (Hub sizes 2026-09-02). The box has ~390 GB total (373 GB free at start), so models are fetched one cell at a
# time and deleted after their sweep (README "Disk-constrained order"); a model that does not fit the free disk is skipped.
SIZE_GB[qwen38_27b_fp8]=31; SIZE_GB[gptoss120b]=65; SIZE_GB[qwen3_8b]=16; SIZE_GB[qwen38_27b]=56
SIZE_GB[qwen35_122b_fp8]=127; SIZE_GB[dsv4flash]=167; SIZE_GB[qwen38flashnext]=186; SIZE_GB[glm53flash]=328
REPO[qwen38_27b_fp8]=Qwen/Qwen3.8-27B-FP8
REPO[gptoss120b]=openai/gpt-oss-120b;            EXTRA[gptoss120b]='--exclude original/* metal/*'   # MXFP4 safetensors only (~65 GB, not 196)
REPO[qwen3_8b]=Qwen/Qwen3-8B                                                                          # train co-tenancy cell (Qwen3.8-8B does not exist on the Hub)
REPO[qwen38_27b]=Qwen/Qwen3.8-27B
REPO[qwen35_122b_fp8]=Qwen/Qwen3.5-122B-A10B-FP8
REPO[dsv4flash]=deepseek-ai/DeepSeek-V4-Flash-0731
REPO[qwen38flashnext]=Qwen/Qwen3.8-Flash-Next-FP8
REPO[glm53flash]=zai-org/GLM-5.3-Flash                                                                # 328 GB, attempt-to-load cell only
ORDER_CORE="qwen38_27b_fp8 gptoss120b qwen3_8b qwen38_27b qwen35_122b_fp8 dsv4flash qwen38flashnext"
case "$SET" in
  none)  KEYS="" ;;
  small) KEYS="qwen38_27b_fp8 gptoss120b qwen3_8b" ;;
  core)  KEYS="$ORDER_CORE" ;;
  all)   KEYS="$ORDER_CORE glm53flash" ;;
  *)     KEYS="$(echo "$SET" | tr ',' ' ')" ;;
esac
HF=hf; command -v hf >/dev/null || HF=huggingface-cli
echo "download set '$SET' -> $KEYS  (tool: $HF)"; echo -e "key\trepo\tstate\tstart_utc\tend_utc\tsize" > "$D/status.tsv"
python3 - "$M" ${KEYS} <<'PY' > "$M/models.json"
import json, sys
m = sys.argv[1]; keys = sys.argv[2:]
repos = {"qwen38_27b_fp8":"Qwen/Qwen3.8-27B-FP8","gptoss120b":"openai/gpt-oss-120b","qwen3_8b":"Qwen/Qwen3-8B","qwen38_27b":"Qwen/Qwen3.8-27B",
         "qwen35_122b_fp8":"Qwen/Qwen3.5-122B-A10B-FP8","dsv4flash":"deepseek-ai/DeepSeek-V4-Flash-0731","qwen38flashnext":"Qwen/Qwen3.8-Flash-Next-FP8","glm53flash":"zai-org/GLM-5.3-Flash"}
print(json.dumps({k: {"repo": repos[k], "path": f"{m}/{repos[k].split('/')[-1]}"} for k in keys if k in repos}, indent=2))
PY
for k in $KEYS; do
  r="${REPO[$k]:-}"; [ -z "$r" ] && { echo "unknown key $k"; continue; }
  dest="$M/${r##*/}"; mkdir -p "$dest"
  if [ -f "$dest/.complete" ]; then echo "[$k] already complete ($(du -sh "$dest" | cut -f1))"; printf '%s\t%s\tcomplete\t-\t-\t%s\n' "$k" "$r" "$(du -sh "$dest" | cut -f1)" >> "$D/status.tsv"; continue; fi
  # disk guard: expected size + 10% + 5 GB headroom must fit (partial downloads are resumable, so nothing is lost by skipping)
  need=$(( ${SIZE_GB[$k]:-0} + ${SIZE_GB[$k]:-0} / 10 + 5 )); have_gb="$(df -BG --output=avail "$M" 2>/dev/null | tail -1 | tr -dc '0-9')"
  if [ -n "$have_gb" ] && [ "$have_gb" -lt "$need" ]; then
    echo "[$k] SKIP: needs ~${need} GB free (${SIZE_GB[$k]:-?} GB + margin) but $M has ${have_gb} GB. Delete a finished model first:  rm -rf $M/<name>   (df -h $M; du -sh $M/*)"
    printf '%s\t%s\tskipped_disk\t-\t-\tavail=%sG\n' "$k" "$r" "$have_gb" >> "$D/status.tsv"; continue
  fi
  s="$(date -u +%FT%TZ)"; echo "[$k] $r -> $dest  start $s  (hf download --local-dir layout: plain directory, no HF cache blobs)"
  ok=0
  for attempt in 1 2 3 4 5 6; do
    # shellcheck disable=SC2086
    if $HF download "$r" --local-dir "$dest" --max-workers 16 ${EXTRA[$k]:-} > "$D/$k.log" 2>&1; then ok=1; break; fi
    echo "[$k] attempt $attempt failed (see $D/$k.log); retry in 30s"; sleep 30
  done
  e="$(date -u +%FT%TZ)"; sz="$(du -sh "$dest" | cut -f1)"
  if [ $ok = 1 ]; then touch "$dest/.complete"; echo "[$k] done $sz ($s -> $e)"; printf '%s\t%s\tcomplete\t%s\t%s\t%s\n' "$k" "$r" "$s" "$e" "$sz" >> "$D/status.tsv"
  else echo "[$k] FAILED after 6 attempts"; printf '%s\t%s\tfailed\t%s\t%s\t%s\n' "$k" "$r" "$s" "$e" "$sz" >> "$D/status.tsv"; fi
  df -h "$M" | tail -1
done
echo "all downloads finished $(date -u +%FT%TZ)"; touch "$D/.all_done"
EOS
chmod +x "$TOOLS_DIR/download_models.sh"
if [ "$DOWNLOAD_SET" != "none" ]; then
  tmux kill-session -t downloads 2>/dev/null || true
  tmux_run downloads "MODELS_DIR=$MODELS_DIR DL_DIR=$DL_DIR HF_HOME=$HF_HOME $TOOLS_DIR/download_models.sh '$DOWNLOAD_SET' 2>&1 | tee -a $DL_DIR/downloads.log"
fi

# ------------------------------------------------------------------ harness: clone or wait for sync.sh push, then hardware truth
cat > "$TOOLS_DIR/run_hwtruth.sh" <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
B="${BENCH_ROOT:-/workspace/rtxpro6000-bench}"; T="${TOOLS_DIR:-/workspace/tools}"; W="${HARNESS_WAIT_MIN:-90}"; HW="${HW_DIR:-$B/results/hw}"
if [ -n "${HARNESS_REPO:-}" ]; then
  if [ -d "$B/.git" ]; then git -C "$B" pull -q || true; else rm -rf "$B.tmp"; git clone -q "$HARNESS_REPO" "$B.tmp" && { mkdir -p "$B"; cp -a "$B.tmp/." "$B/"; rm -rf "$B.tmp"; }; fi
fi
n=0; until [ -f "$B/vast/hardware_truth.sh" ]; do
  n=$((n+1)); [ $n -gt $((W*3)) ] && { echo "harness not present after $W min -> run later: bash $B/vast/hardware_truth.sh"; exit 0; }
  [ $((n % 15)) -eq 1 ] && echo "waiting for harness at $B (push it with ./vast/sync.sh push) ... $(date -u +%T)"; sleep 20
done
until [ -f "$T/.build_done" ]; do echo "waiting for tool builds ..."; sleep 20; done
[ -f "$T/nccl_env.sh" ] && source "$T/nccl_env.sh"
find "$B" -type f \( -name '*.sh' -o -name '*.py' -o -name '*.env' \) -not -path '*/results/*' -exec sed -i 's/\r$//' {} +
find "$B" -name '*.sh' -not -path '*/results/*' -exec chmod +x {} +
if [ -f "$HW/decisions.env" ]; then echo "$HW/decisions.env exists; skip (delete it to re-run)"; cat "$HW/decisions.env"; exit 0; fi
HW_DIR="$HW" bash "$B/vast/hardware_truth.sh" "$HW"
EOS
chmod +x "$TOOLS_DIR/run_hwtruth.sh"
tmux_run hwtruth "BENCH_ROOT=$BENCH_ROOT HW_DIR=$HW_DIR TOOLS_DIR=$TOOLS_DIR HARNESS_WAIT_MIN=$HARNESS_WAIT_MIN HARNESS_REPO='${HARNESS_REPO:-}' $TOOLS_DIR/run_hwtruth.sh 2>&1 | tee -a $BENCH_ROOT/results/hwtruth-runner.log"

# ------------------------------------------------------------------ optional venvs: lm-eval (gates) and unsloth (train co-tenancy)
if [ ! -f "$MARK" ]; then
  if [ "$INSTALL_EVAL" = "1" ] && [ ! -x /workspace/venv-eval/bin/lm_eval ]; then
    tmux_run evalenv "mkvenv /workspace/venv-eval && /workspace/venv-eval/bin/python -m pip install -q -U pip && /workspace/venv-eval/bin/python -m pip install -q 'lm-eval[api]' requests 2>&1 | tail -3; /workspace/venv-eval/bin/lm_eval --help >/dev/null && echo 'lm-eval OK: /workspace/venv-eval/bin/lm_eval'"
  fi
  if [ "$INSTALL_TRAIN" = "1" ] && [ ! -x /workspace/venv-train/bin/python ]; then
    tmux_run trainenv "mkvenv /workspace/venv-train && /workspace/venv-train/bin/python -m pip install -q -U pip && /workspace/venv-train/bin/python -m pip install -q unsloth trl datasets hf_transfer 2>&1 | tail -3; /workspace/venv-train/bin/python -c 'import unsloth,torch;print(\"unsloth\",unsloth.__version__,\"torch\",torch.__version__,torch.version.cuda)'"
  fi
fi

touch "$MARK"
echo "==== onstart finished $(date -u +%FT%TZ). tmux sessions: $(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
echo "     logs: $LOG  $TOOLS_DIR/build.log  $DL_DIR/downloads.log  $DL_DIR/status.tsv  $BENCH_ROOT/results/hwtruth-runner.log"
echo "     engine: NOT touched by onstart (ALLOW_ENGINE_PIP=$ALLOW_ENGINE_PIP). Upgrade in place with bench/setup_engine.sh (uv, vLLM main cu130 nightly + flashinfer + vllm[b12x])."
echo "     models: bench/prefetch.sh <cell>  ->  hf download <repo> --local-dir $MODELS_DIR/<basename>;  free disk: rm -rf $MODELS_DIR/<name>  (df -h $MODELS_DIR)"
echo "     next: from your laptop  ./vast/sync.sh push ${CONTAINER_ID:-<ID>}   then   ./vast/sync.sh tmux ${CONTAINER_ID:-<ID>} hwtruth   ->  $HW_DIR/decisions.env"
