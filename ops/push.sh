#!/usr/bin/env bash
# push.sh <host-file> <file> [file...]   - copy repo files to a box's /workspace/bench.
#
# Two failure modes this exists to prevent. Inlining scp into `wsl bash -c "..."` mangled its own arguments
# four times (the port arrives empty, or -P is read as the identity file; the file silently does not land and
# the next check reports the old copy as if it were new). And scp overwrites IN PLACE: bash reads a running
# script by byte offset, so overwriting one that is executing on the box makes it resume mid-line in the new
# text and die with "unexpected EOF while looking for matching quote" - which is what killed the tail of the
# DeepSeek quality sweep on 5 Sept. So every file lands under a temporary name and is renamed into place:
# a rename swaps the directory entry, and a bash that already has the old inode open keeps reading the old
# inode, untouched.
set -u
KEY="$HOME/.ssh/id_ed25519"
SP="${SP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; _p(){ case "$1" in /*) echo "$1";; *) for d in "$SP" "${SCRATCH:-}" "/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad"; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo "$(_p "$1")";; esac; }
A=/mnt/c/Users/ushni/Downloads/AIRR
HOSTFILE="$1"; shift
read -r _ H P _ < "$(_p "$HOSTFILE")"
[ -n "$P" ] || { echo "  no port in $HOSTFILE"; exit 1; }
SSH=(ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$H")
for f in "$@"; do
  case "$f" in */*) src="$A/$f";; *) src="$A/box/$f";; esac
  dest=/workspace/bench
  case "$f" in */lists/*) dest=/workspace/bench/lists;; esac
  b=$(basename "$f")
  if scp -q -i "$KEY" -P "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$src" "root@$H:$dest/.$b.push"; then
    # report (never block on) a running reader; the rename keeps it safe either way. The bracketed first
    # character keeps this grep from matching its own command line.
    pat="[b]ash $dest/$b|[p]ython3 $dest/$b"
    running=$("${SSH[@]}" "ps -eo args | grep -Ec -- '$pat'; mv -f '$dest/.$b.push' '$dest/$b'" 2>/dev/null | head -1)
    if [ "${running:-0}" -gt 0 ]; then echo "  sent $b -> $dest/  (a process is executing the old copy; it keeps its inode, the new file is in place for the next start)"
    else echo "  sent $b -> $dest/"; fi
  else
    echo "  FAILED $f"; exit 1
  fi
done
