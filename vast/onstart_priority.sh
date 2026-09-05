#!/usr/bin/env bash
# Minimal, isolated bootstrap for the separately authorized priority-model test.
# Credentials stay on the local machine. No models, upgrades or tests auto-run.
set -euo pipefail
mkdir -p /workspace/priority /workspace/models /workspace/priority/results
exec > >(tee -a /workspace/priority/bootstrap.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq tmux git curl numactl pciutils python3-venv
python3 - <<'PY'
import importlib.metadata as m, json, sys
from pathlib import Path
versions = {}
for p in ('vllm', 'torch', 'flashinfer-python', 'transformers', 'triton', 'compressed-tensors', 'b12x'):
    try: versions[p] = m.version(p)
    except m.PackageNotFoundError: versions[p] = None
Path('/workspace/priority/bootstrap_versions.json').write_text(json.dumps({'python':sys.version,'packages':versions}, indent=2))
print(json.dumps(versions, indent=2))
PY
touch /workspace/priority/bootstrap.complete
