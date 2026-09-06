#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/models/GLM-5.3-Flash-NVFP4
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
hf download RedHatAI/GLM-5.3-Flash-NVFP4 \
  --revision 36c184c6cda000a481711306df5adde42f63321a \
  --local-dir /workspace/models/GLM-5.3-Flash-NVFP4 --max-workers 8
python3 - <<'PY'
from pathlib import Path
import json, datetime
p=Path('/workspace/models/GLM-5.3-Flash-NVFP4')
index=json.loads((p/'model.safetensors.index.json').read_text())
files=sorted(set(index['weight_map'].values()))
missing=[f for f in files if not (p/f).is_file()]
if missing: raise RuntimeError(f'Missing weight shards: {missing}')
manifest={'model':'RedHatAI/GLM-5.3-Flash-NVFP4','revision':'36c184c6cda000a481711306df5adde42f63321a','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'weight_files':{f:(p/f).stat().st_size for f in files}}
(p/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
(p/'.complete').touch()
print(json.dumps({'complete':True,'files':len(files),'weight_bytes':sum(manifest['weight_files'].values())}))
PY
