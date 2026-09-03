#!/usr/bin/env bash
# SGLang in an isolated venv (its torch pin must not disturb the system vLLM).
set +e
log(){ echo "[$(date +%H:%M:%S)] $*"; }
VENV=/workspace/venv-sglang
log "creating $VENV"
uv venv --python 3.12 "$VENV" >/dev/null 2>&1
source "$VENV/bin/activate"
log "installing sglang (cu130 index, prerelease allowed)"
uv pip install --prerelease=allow --index-strategy unsafe-best-match --torch-backend=cu130 \
  sglang --extra-index-url https://docs.sglang.ai/whl/cu130/ > /workspace/sglang_install.log 2>&1
echo "rc=$?"
python -c "import sglang, torch; print('sglang', sglang.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)" 2>&1 | tail -2
# bench client: vllm's bench serve talks plain OpenAI, so install a client-only vllm? no - use sglang's own
python -c "import sglang.bench_serving" 2>&1 | tail -1
deactivate
