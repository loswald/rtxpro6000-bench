#!/usr/bin/env bash
# The same 403-item quality suite against the API market: each model we serve locally, run through OpenRouter's
# default routing under the vendor's own sampling recipe (lists/profiles.tsv), so the leaderboard can say whether
# the endpoint a customer gets at list price scores what the weights score on our node.
#
# The key never passes through the assistant or the chat: put it on the box yourself, one line, mode 600:
#   ssh -i ~/.ssh/id_ed25519 -p <port> root@<host> 'umask 077; cat > /workspace/.openrouter_key'   (paste, Ctrl-D)
# This script reads that file into EVAL_API_KEY for the eval client (evalsuite/common.py adds the bearer header)
# and never echoes it. Needs no GPU; runs beside the GPU work in its own tmux session.
#
#   bash or_eval.sh                 # all models below, in priority order, 45 minutes each at most
#   bash or_eval.sh glm ds          # a subset
set -u
B=/workspace/bench; R=/workspace/results; OUT=$R/eval_or; KEYFILE=${KEYFILE:-/workspace/.openrouter_key}
PROFILES=$B/lists/profiles.tsv
[ -s "$KEYFILE" ] || { echo "no key at $KEYFILE - place it first (see header)"; exit 1; }
EVAL_API_KEY=$(tr -d '\r\n' < "$KEYFILE"); export EVAL_API_KEY
[ "${#EVAL_API_KEY}" -gt 20 ] || { echo "key file looks wrong (length ${#EVAL_API_KEY})"; exit 1; }
mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
# name | OpenRouter slug | local model dir name (for the sampling profile)
MODELS="glm|z-ai/glm-5.3-flash|GLM-5.3-Flash
ds|deepseek/deepseek-v4-flash-0731|DeepSeek-V4-Flash
q27|qwen/qwen3.8-27b|Qwen3.8-27B
oss120|openai/gpt-oss-120b|gpt-oss-120b
oss20|openai/gpt-oss-20b|gpt-oss-20b
muse|meta/muse-glimmer-30b|Muse-Glimmer-30B
gemma|google/gemma-4-26b-a4b-it|gemma-4-26B-A4B
minimax|minimax/minimax-m3|MiniMax-M3"
want="${*:-glm ds q27 oss120 oss20 muse gemma minimax}"
profile_field(){ # dirname field -> the profile column for the first pattern that matches the model name
  awk -F'\t' -v n="$1" -v f="$2" '!/^#/ && NF>=3 { if (n ~ $1) { print $f; exit } }' "$PROFILES"
}
CAPS="math=32768,code=20480,knowledge=20480,ifeval=16384,tools=8192,longctx=6144"
for w in $want; do
  line=$(printf '%s\n' "$MODELS" | grep "^$w|") || { log "unknown model key $w"; continue; }
  IFS='|' read -r key slug dirname <<< "$line"
  esamp=$(profile_field "$dirname" 3)
  tag="or_${key}"
  if [ -f "$OUT/$tag.json" ] && python3 -c "import json,sys; d=json.load(open('$OUT/$tag.json')); sys.exit(0 if d.get('partial') is False else 1)" 2>/dev/null; then
    log "$tag already complete"; continue
  fi
  log "$tag: $slug via OpenRouter, sampling: ${esamp:-default reasoning recipe}"
  # shellcheck disable=SC2086
  env -u PYTHONHOME -u PYTHONPATH python3 "$B/evalsuite/run_eval.py" --tag "$tag" --base-urls https://openrouter.ai/api/v1 \
    --model "$slug" --out "$OUT" --concurrency "${OR_CONC:-24}" --time-budget "${OR_BUDGET:-2700}" \
    --reasoning --request-timeout 3600 --max-tokens 32768 --max-tokens-family "$CAPS" $esamp \
    ${OR_RESUME:+--resume} > "$OUT/$tag.log" 2>&1
  python3 -c "import json; d=json.load(open('$OUT/$tag.json')); a=d['aggregate']; print('  ', '$tag', 'n', a.get('n_scored'), 'acc', a.get('acc_micro'), 'errors', a.get('n_error'), 'partial', d.get('partial'))" 2>/dev/null || log "  $tag: no result file"
done
log "OREVAL DONE"
