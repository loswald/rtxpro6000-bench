#!/usr/bin/env bash
# Provision a rented 4x RTX PRO 6000 Blackwell instance to serve
# GLM-5.3-Flash-NVFP4 behind the LibertAI gateway.
#
# Idempotent enough to re-run. Takes roughly 20 minutes, most of it the model
# download and the first engine start.
#
# Context that shapes every step: a rented instance is an UNPRIVILEGED container.
# There is no Docker-in-Docker, so the per-model vLLM image that carries
# `glm5_next` cannot be run. Its python tree is lifted out of the registry
# instead, which works because the instance ships the same interpreter and torch
# build as the image (python 3.12, torch 2.13.0+cu130).
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/GLM-5.3-Flash-NVFP4}"
IMAGE_TAG="${IMAGE_TAG:-glm53-flash-x86_64-cu130}"
EXTRACT="${EXTRACT:-/workspace/vllm-extract}"
SITE="${SITE:-/workspace/site}"
KERNEL_SRC="${KERNEL_SRC:-/workspace/vllm-sparse-mla-blackwell}"
TP="${TP:-4}"
MODEL_ID="${MODEL_ID:-glm-5.3-flash}"
VLLM_PORT="${VLLM_PORT:-18000}"
BACKEND_PORT="${BACKEND_PORT:-9000}"
PYDIST="$EXTRACT/usr/local/lib/python3.12/dist-packages"

say() { echo; echo "=== $* ==="; }

say "1/8 model weights"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  pip install -q huggingface_hub
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$MODEL_REPO", local_dir="$MODEL_DIR", max_workers=16)
PY
else
  echo "already present"
fi

say "2/8 vLLM python tree (glm5_next) out of the per-model image"
if [ ! -d "$PYDIST/vllm/models/glm5next" ]; then
  python3 "$(dirname "$0")/pull_vllm.py" "$IMAGE_TAG" "$EXTRACT"
else
  echo "already extracted"
fi

say "3/8 sparse-MLA kernel + plugins"
if [ ! -d "$KERNEL_SRC" ]; then
  git clone -q https://github.com/Libertai/vllm-sparse-mla-blackwell.git "$KERNEL_SRC"
fi
# a stale .so built elsewhere would shadow the installed package and fail to load
rm -f "$KERNEL_SRC"/glm53_sparse_mla/*.so
rm -rf "$KERNEL_SRC/build" "$SITE"
( cd "$KERNEL_SRC" && GLM53_ARCHS=120a MAX_JOBS=8 \
    pip install -q --no-build-isolation --no-deps --target "$SITE" . )
ls "$SITE"/glm53_sparse_mla/*.so

say "4/8 serve script"
cat > /workspace/serve.sh <<EOF
#!/usr/bin/env bash
set -x
export PYTHONPATH=$SITE:$PYDIST
export VLLM_GLM53_CUDA_SPARSE_MLA=1
export VLLM_GLM53_MOE_INPUT_SCALE=1.0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# breakable graphs measured no gain and cost KV; graphs themselves are worth 6x
# here because TP=4 over PCIe is launch/collective-latency bound, not bandwidth
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export NCCL_MIN_NCHANNELS=32
export NCCL_P2P_LEVEL=PXB
export HF_HOME=/workspace/.hf_home
cd /workspace
exec python3 -m vllm.entrypoints.openai.api_server \\
  --model $MODEL_DIR --served-model-name $MODEL_ID \\
  --host 0.0.0.0 --port $VLLM_PORT --trust-remote-code \\
  --tensor-parallel-size $TP \\
  --kv-cache-dtype fp8 --block-size 256 \\
  --max-model-len 262144 --max-num-seqs 256 --max-num-batched-tokens 8192 \\
  --gpu-memory-utilization 0.94 \\
  --moe-backend flashinfer_cutlass \\
  --tool-call-parser glm47 --reasoning-parser deepseek_r1 --enable-auto-tool-choice \\
  --enable-prompt-tokens-details \\
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
EOF
chmod +x /workspace/serve.sh

say "5/8 supervisor unit for vLLM (image's own vllm disabled)"
cat > /opt/supervisor-scripts/glm53-vllm.sh <<'EOF'
#!/bin/bash
exec /workspace/serve.sh
EOF
chmod +x /opt/supervisor-scripts/glm53-vllm.sh
cat > /etc/supervisor/conf.d/glm53-vllm.conf <<'EOF'
[program:glm53-vllm]
command=/opt/supervisor-scripts/glm53-vllm.sh
autostart=true
autorestart=true
startsecs=30
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=30
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
EOF
sed -i 's/^autostart=true/autostart=false/' /etc/supervisor/conf.d/vllm.conf 2>/dev/null || true
supervisorctl reread >/dev/null; supervisorctl update >/dev/null
supervisorctl stop vllm >/dev/null 2>&1 || true
# supervisorctl stop leaves workers reparented to init still holding VRAM
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 10

say "6/8 LibertAI gateway"
mkdir -p /opt/libertai && cd /opt/libertai
[ -d libertai-models ] || git clone -q https://github.com/libertai/libertai-models.git
cd libertai-models
python3 -m venv .venv 2>/dev/null || true
. .venv/bin/activate
# `poetry install` silently no-ops here, so install the declared deps directly
pip install -q fastapi pydantic httpx cryptography uvicorn python-dotenv python-multipart
cp -n .env.example .env 2>/dev/null || true
sed -i "s/^MODELS=.*/MODELS=$MODEL_ID/" .env
mkdir -p data
cat > "data/$MODEL_ID.json" <<EOF
{"id":"$MODEL_ID","url":"http://localhost:$VLLM_PORT","allowed_paths":["v1/completions","v1/chat/completions","completions","v1/responses","v1/messages"]}
EOF
deactivate
cat > /opt/supervisor-scripts/libertai-models.sh <<EOF
#!/bin/bash
cd /opt/libertai/libertai-models
. .venv/bin/activate
exec uvicorn src.server:app --host 127.0.0.1 --port $BACKEND_PORT
EOF
chmod +x /opt/supervisor-scripts/libertai-models.sh
cat > /etc/supervisor/conf.d/libertai-models.conf <<'EOF'
[program:libertai-models]
command=/opt/supervisor-scripts/libertai-models.sh
autostart=true
autorestart=true
startsecs=5
stopsignal=TERM
stopwaitsecs=10
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
EOF

say "7/8 Caddy / portal wiring"
cp -n /etc/portal.yaml /etc/portal.yaml.bak 2>/dev/null || true
python3 - <<PY
import yaml
d = yaml.safe_load(open('/etc/portal.yaml'))
apps = d.get('applications', {})
key = next((k for k, v in apps.items() if v.get('external_port') == 8000), None)
if key:
    apps[key]['internal_port'] = $BACKEND_PORT
yaml.safe_dump(d, open('/etc/portal.yaml', 'w'), sort_keys=False)
print('repointed', key, '-> $BACKEND_PORT')
PY
sed -i 's/^AUTH_EXCLUDE=.*/AUTH_EXCLUDE="8000"/; s/^ENABLE_HTTPS=.*/ENABLE_HTTPS="true"/' /etc/environment
grep -qE '^AUTH_EXCLUDE=' /etc/environment || echo 'AUTH_EXCLUDE="8000"' >> /etc/environment
grep -qE '^ENABLE_HTTPS=' /etc/environment || echo 'ENABLE_HTTPS="true"' >> /etc/environment
supervisorctl reread >/dev/null; supervisorctl update >/dev/null
supervisorctl restart caddy >/dev/null 2>&1 || true

say "8/8 start and wait"
supervisorctl start glm53-vllm >/dev/null 2>&1 || true
for i in $(seq 1 90); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$VLLM_PORT/v1/models || echo 000)" = "200" ]; then
    echo "vLLM ready after $((i*15))s"; break
  fi
  sleep 15
done
supervisorctl status | grep -E 'glm53-vllm|libertai-models|caddy'
curl -s -m 8 http://127.0.0.1:$VLLM_PORT/v1/models | head -c 200; echo
echo
echo "public endpoint: https://<instance-ip>:<external port for 8000>/v1  (self-signed, use -k)"
