# shared downloader: get <repo> <dirname>; idempotent via the "done/have" lines in dl6000.log
export HF_HUB_ENABLE_HF_TRANSFER=1
get(){ d=/workspace/models/$2
  grep -qE "\] (done|have) $2( |$)" /workspace/results/dl6000.log 2>/dev/null && { echo "[$(date +%H:%M:%S)] have $2"; return; }
  echo "[$(date +%H:%M:%S)] downloading $1"
  hf download "$1" --local-dir "$d" > /workspace/dl_$2.log 2>&1 && echo "[$(date +%H:%M:%S)] done $2 ($(du -sh $d|cut -f1))" || echo "[$(date +%H:%M:%S)] FAILED $2"; }
