#!/usr/bin/env bash
# push.sh <host-file> <file> [file...]   - copy repo files to a box's /workspace/bench.
# Exists because inlining scp into `wsl bash -c "..."` has now mangled its own arguments four times:
# the port lands empty or -P is read as the identity file. One script, quoted once, no interpolation.
set -u
KEY="$HOME/.ssh/id_ed25519"
SP=/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad
A=/mnt/c/Users/ushni/Downloads/AIRR
HOSTFILE="$1"; shift
read -r _ H P _ < "$SP/$HOSTFILE"
[ -n "$P" ] || { echo "  no port in $HOSTFILE"; exit 1; }
for f in "$@"; do
  case "$f" in */*) src="$A/$f";; *) src="$A/box/$f";; esac
  dest=/workspace/bench/
  case "$f" in */lists/*) dest=/workspace/bench/lists/;; esac
  if scp -q -i "$KEY" -P "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$src" "root@$H:$dest"; then
    echo "  sent $(basename "$f") -> $dest"
  else
    echo "  FAILED $f"; exit 1
  fi
done
