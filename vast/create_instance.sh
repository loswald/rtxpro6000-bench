#!/usr/bin/env bash
# create_instance.sh -- rent one offer with the benchmark image + onstart.sh, wait for ssh, print next steps.
#
#   ./vast/create_instance.sh OFFER_ID
#   IMAGE=vllm/vllm-openai:nightly ./vast/create_instance.sh OFFER_ID
#   ENGINE=sglang IMAGE=lmsysorg/sglang:v0.5.18-cu130 DOWNLOAD_SET=qwen38_27b_fp8,gptoss120b LABEL=rtxpro6000-sglang ./vast/create_instance.sh OFFER_ID
#
# Env: IMAGE (default vllm/vllm-openai:cu130-nightly -- STALE, vLLM 0.19.2; bench/setup_engine.sh upgrades it in place with uv,
#      see README "Image tag truth"), DISK_GB=1200 (the box rented 2026-09-02 only had ~390 GB -- see README "disk budget"),
#      LABEL=rtxpro6000-bench, DOWNLOAD_SET=none (models are fetched per cell by bench/prefetch.sh), ALLOW_ENGINE_PIP=0,
#      ENGINE=auto, HARNESS_WAIT_MIN=90, INSTALL_TRAIN=1, INSTALL_EVAL=1,
#      HARNESS_REPO= (git URL; empty = you will ./vast/sync.sh push), EXTRA_PORTS="-p 8000:8000 -p 30000:30000"
# NOTE: an instance already exists and is running (2026-09-02). Only run this for a NEW box.
# The Vast API key is read by the vastai CLI from its own config file ('vastai set api-key'); never put it here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
have() { command -v "$1" >/dev/null 2>&1; }
have vastai || { echo "vastai CLI not found: pip install vastai && vastai set api-key <KEY>" >&2; exit 1; }
OFFER="${1:-}"; [ -n "$OFFER" ] || { sed -n '2,12p' "$0"; exit 2; }

IMAGE="${IMAGE:-vllm/vllm-openai:cu130-nightly}"
DISK_GB="${DISK_GB:-1200}"; LABEL="${LABEL:-rtxpro6000-bench}"
DOWNLOAD_SET="${DOWNLOAD_SET:-none}"; ENGINE="${ENGINE:-auto}"; HARNESS_WAIT_MIN="${HARNESS_WAIT_MIN:-90}"
INSTALL_TRAIN="${INSTALL_TRAIN:-1}"; INSTALL_EVAL="${INSTALL_EVAL:-1}"; HARNESS_REPO="${HARNESS_REPO:-}"
ALLOW_ENGINE_PIP="${ALLOW_ENGINE_PIP:-0}"   # onstart never pip-installs into the engine interpreter unless 1
EXTRA_PORTS="${EXTRA_PORTS:--p 8000:8000 -p 30000:30000}"

# --- image tag freshness check (Docker Hub public API; no auth). Warn if the tag is older than 30 days.
repo="${IMAGE%%:*}"; tag="${IMAGE##*:}"; [ "$repo" = "$IMAGE" ] && tag=latest
if have curl; then
  meta="$(curl -fsS --max-time 15 "https://hub.docker.com/v2/repositories/${repo}/tags/${tag}" 2>/dev/null || true)"
  if [ -z "$meta" ]; then
    echo "WARN: Docker Hub has no tag ${repo}:${tag} (or Hub unreachable). Vast will fail to pull a missing tag." >&2
  else
    upd="$(printf '%s' "$meta" | grep -o '"last_updated": *"[^"]*"' | head -1 | sed 's/.*"\([0-9-]*\)T.*/\1/')"
    echo "image ${IMAGE}: last pushed ${upd:-?}"
    if [ -n "$upd" ] && have date; then
      age_days=$(( ( $(date -u +%s) - $(date -u -d "$upd" +%s 2>/dev/null || date -u -j -f %Y-%m-%d "$upd" +%s 2>/dev/null || echo 0) ) / 86400 ))
      [ "$age_days" -gt 30 ] && echo "WARN: ${IMAGE} is ${age_days} days old. For Aug-2026 models (DeepSeek-V4, Qwen3.8, GLM-5.3) use a current tag (see README)." >&2
    fi
  fi
fi

ENV_STR="-e DOWNLOAD_SET=${DOWNLOAD_SET} -e ENGINE=${ENGINE} -e HARNESS_WAIT_MIN=${HARNESS_WAIT_MIN} -e INSTALL_TRAIN=${INSTALL_TRAIN} -e INSTALL_EVAL=${INSTALL_EVAL} -e ALLOW_ENGINE_PIP=${ALLOW_ENGINE_PIP}"
[ -n "$HARNESS_REPO" ] && ENV_STR="$ENV_STR -e HARNESS_REPO=${HARNESS_REPO}"
ENV_STR="$ENV_STR ${EXTRA_PORTS}"

echo "vastai create instance $OFFER --image $IMAGE --disk $DISK_GB --ssh --direct --label $LABEL --cancel-unavail --onstart $HERE/onstart.sh --env '$ENV_STR'"
out="$(vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --label "$LABEL" --cancel-unavail \
        --onstart "$HERE/onstart.sh" --env "$ENV_STR" 2>&1)" || { echo "$out"; exit 1; }
echo "$out"
ID="$(printf '%s' "$out" | grep -o "'new_contract': *[0-9]*" | grep -o '[0-9]*$' || true)"
[ -n "$ID" ] || ID="$(printf '%s' "$out" | grep -o '"new_contract": *[0-9]*' | grep -o '[0-9]*$' || true)"
[ -n "$ID" ] || { echo "could not parse instance id; check: vastai show instances" >&2; exit 1; }
echo "instance id: $ID   (export VAST_INSTANCE_ID=$ID)"

echo "waiting for status=running ..."
for i in $(seq 1 60); do
  st="$(vastai show instance "$ID" --raw 2>/dev/null | grep -o '"actual_status": *"[^"]*"' | head -1 | sed 's/.*: *"//;s/"$//')"
  [ "$st" = "running" ] && break
  printf '  %s  status=%s\n' "$(date +%T)" "${st:-<none>}"; sleep 15
done
url="$(vastai ssh-url "$ID" 2>/dev/null | tail -1)"
echo
echo "ssh:   ${url:-(not ready yet: vastai ssh-url $ID)}   ->  ssh -p PORT root@HOST"
echo "next:  ./vast/sync.sh push $ID          # upload the harness to /workspace/rtxpro6000-bench (onstart is waiting for it)"
echo "       ./vast/sync.sh tmux $ID hwtruth  # watch hardware truth -> results/hw/decisions.env; other sessions: build evalenv trainenv"
echo "       vastai logs $ID                  # onstart output"
echo "       vastai destroy instance $ID      # when finished -- stopped instances still bill storage"
