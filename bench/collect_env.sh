#!/usr/bin/env bash
# =============================================================================
# bench/collect_env.sh [results_root]
#
# Software + hardware truth of the container -> results/env.json: driver, CUDA,
# vLLM / SGLang / FlashInfer / b12x / torch / triton versions (image interpreter plus
# the eval and train venvs), container image identity (as far as a container can
# see it), GPU edition / vBIOS / power limits / PCIe link, topology, CPU/RAM/disk,
# the models on disk under $MODELS_DIR, the resolved VLLM_*/NCCL_*/FLASHINFER_* env,
# and the hardware decision contract (results/hw/decisions.env via bench/env.sh:
# p2p_ok, custom_allreduce, nccl_p2p_disable, acs_suspected, pessimistic_tp,
# host_ram_gb -- the last two are null when decisions.env is missing) plus
# results/hw/hardware.json (copy: results/hardware.json) when present.
#
# Run once after hardware truth and again right before summarising.
# Env: IMAGE / IMAGE_DIGEST (from the Vast instance page or `docker manifest
#      inspect <tag>`), SGLANG_PYTHON, UNSLOTH_VENV, LMEVAL_VENV
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# bench/env.sh (decision contract, MODELS_DIR, sm_120 env) is what every launch sees
# shellcheck disable=SC1091
[[ -f "$SCRIPT_DIR/env.sh" ]] && . "$SCRIPT_DIR/env.sh"
RESULTS_ROOT="${1:-${RESULTS_ROOT:-$ROOT/results}}"
mkdir -p "$RESULTS_ROOT"
export RESULTS_ROOT OUT="$RESULTS_ROOT/env.json" ROOT
export UNSLOTH_VENV="${UNSLOTH_VENV:-/opt/unsloth-venv}" LMEVAL_VENV="${LMEVAL_VENV:-/opt/lmeval-venv}"
export MODELS_DIR="${MODELS_DIR:-/workspace/models}" HW_DECISIONS_FILE="${HW_DECISIONS_FILE:-$RESULTS_ROOT/hw/decisions.env}"

nvidia-smi -q > "$RESULTS_ROOT/nvidia-smi-q.txt" 2>&1 || true
nvidia-smi topo -m > "$RESULTS_ROOT/nvidia-smi-topo.txt" 2>&1 || true
nvidia-smi > "$RESULTS_ROOT/nvidia-smi.txt" 2>&1 || true
{ pip list 2>/dev/null || python3 -m pip list 2>/dev/null || uv pip list --system 2>/dev/null; } > "$RESULTS_ROOT/pip-list.txt" || true
# per-model-dir size + state (complete/partial) for the disk budget
: > "$RESULTS_ROOT/models-on-disk.txt"
if [[ -d "$MODELS_DIR" ]]; then
  for d in "$MODELS_DIR"/*/; do
    [[ -d "$d" ]] || continue
    d="${d%/}"
    st="unknown"; declare -F model_dir_state >/dev/null && st="$(model_dir_state "$d")"
    printf '%s\t%s\t%s\n' "$(du -sm "$d" 2>/dev/null | cut -f1)" "$st" "$d" >> "$RESULTS_ROOT/models-on-disk.txt"
  done
fi

python3 - <<'PY'
import json, os, platform, re, subprocess, sys, datetime as dt
E = os.environ; RES = E["RESULTS_ROOT"]

def sh(cmd, timeout=60):
    try: return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e: return f"<error {e!r}>"  # noqa: BLE001
def read(p, limit=20000):
    try:
        with open(p, encoding="utf-8", errors="replace") as f: return f.read(limit)
    except Exception: return None  # noqa: BLE001
def load(p):
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception: return None  # noqa: BLE001
def pyver(py, mod, attr="__version__"):
    if not os.path.exists(py) and not sh(f"command -v {py}"): return None
    try:
        r = subprocess.run([py, "-c", f"import {mod} as m; print(getattr(m, '{attr}', 'present'))"], capture_output=True, text=True, timeout=180)
    except Exception: return None  # noqa: BLE001
    return r.stdout.strip() if r.returncode == 0 else None
def venv_versions(py):
    if not os.path.exists(py): return None
    return {"python": py, **{m: pyver(py, m) for m in ("torch", "unsloth", "trl", "transformers", "triton", "bitsandbytes", "xformers", "lm_eval", "vllm")}}
def flag(v, default=None):
    if v in (None, ""): return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

# ---- GPUs -------------------------------------------------------------------------
fields = ["index", "name", "uuid", "serial", "vbios_version", "driver_version", "pci.bus_id", "compute_cap", "memory.total",
          "power.limit", "power.default_limit", "power.max_limit", "power.min_limit", "enforced.power.limit",
          "clocks.max.sm", "clocks.max.memory", "pcie.link.gen.max", "pcie.link.gen.current", "pcie.link.width.max",
          "pcie.link.width.current", "ecc.mode.current", "compute_mode", "persistence_mode", "temperature.gpu"]
gpus = []
for line in sh(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"]).splitlines():
    vals = [v.strip() for v in line.split(",")]
    if len(vals) == len(fields): gpus.append(dict(zip(fields, vals)))
smi_q = read(os.path.join(RES, "nvidia-smi-q.txt"), 400000) or ""
def grab(pat):
    m = re.search(pat, smi_q, re.M); return m.group(1).strip() if m else None
edition = {"product_name": grab(r"^\s*Product Name\s*:\s*(.+)$"), "product_brand": grab(r"^\s*Product Brand\s*:\s*(.+)$"),
           "product_architecture": grab(r"^\s*Product Architecture\s*:\s*(.+)$"), "vbios": grab(r"^\s*VBIOS Version\s*:\s*(.+)$"),
           "gsp_firmware": grab(r"^\s*GSP Firmware Version\s*:\s*(.+)$"), "board_part_number": grab(r"^\s*Board Part Number\s*:\s*(.+)$"),
           "gpu_part_number": grab(r"^\s*GPU Part Number\s*:\s*(.+)$"), "cuda_version_smi": grab(r"^\s*CUDA Version\s*:\s*(.+)$"),
           "driver_version_smi": grab(r"^\s*Driver Version\s*:\s*(.+)$"), "addressing_mode": grab(r"^\s*Addressing Mode\s*:\s*(.+)$")}

# ---- software ----------------------------------------------------------------------
py = sys.executable
try:
    r = subprocess.run([py, "-c", "import torch,json;print(json.dumps({'cuda':torch.version.cuda,'nccl':'.'.join(map(str,torch.cuda.nccl.version())) if torch.cuda.is_available() else None,'arch_list':torch.cuda.get_arch_list()}))"], capture_output=True, text=True, timeout=180)
    ti = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}
except Exception:  # noqa: BLE001
    ti = {}
versions = {"python": platform.python_version(), "torch": pyver(py, "torch"), "torch_cuda": ti.get("cuda"), "nccl": ti.get("nccl"),
            "torch_arch_list": ti.get("arch_list"), "vllm": pyver(py, "vllm"), "vllm_commit": pyver(py, "vllm", "__commit__"),
            "vllm_cli": sh("vllm --version 2>/dev/null | tail -1") or None, "b12x": pyver(py, "b12x"),
            "flashinfer": pyver(py, "flashinfer") or pyver(py, "flashinfer_python"), "transformers": pyver(py, "transformers"),
            "triton": pyver(py, "triton"), "xformers": pyver(py, "xformers"), "flash_attn": pyver(py, "flash_attn"),
            "huggingface_hub": pyver(py, "huggingface_hub"), "hf_transfer": pyver(py, "hf_transfer"),
            "cuda_nvcc": sh("nvcc --version 2>/dev/null | tail -2") or None, "uv": sh("uv --version 2>/dev/null") or None}
sgl_py = E.get("SGLANG_PYTHON") or py
versions["sglang"] = pyver(sgl_py, "sglang"); versions["sgl_kernel"] = pyver(sgl_py, "sgl_kernel")
versions["venv_eval"] = venv_versions("/workspace/venv-eval/bin/python") or venv_versions(os.path.join(E["LMEVAL_VENV"], "bin", "python"))
versions["venv_train"] = venv_versions("/workspace/venv-train/bin/python") or venv_versions(os.path.join(E["UNSLOTH_VENV"], "bin", "python"))

# ---- container / host --------------------------------------------------------------
cg = (read("/proc/self/cgroup", 4000) or "") + (read("/proc/self/mountinfo", 20000) or "")
m = re.search(r"([0-9a-f]{64})", cg)
container = {"image_tag_env": E.get("IMAGE") or E.get("VLLM_IMAGE") or E.get("IMAGE_HINT"), "image_digest_env": E.get("IMAGE_DIGEST"),
             "vast_container_label": E.get("VAST_CONTAINERLABEL"), "vast_container_id": E.get("CONTAINER_ID"),
             "vast_public_ip": E.get("PUBLIC_IPADDR"), "vast_gpu_count_env": E.get("GPU_COUNT"), "vast_data_directory": E.get("DATA_DIRECTORY"),
             "container_id_from_cgroup": m.group(1) if m else None, "hostname": platform.node(), "kernel": platform.release(),
             "os_release": read("/etc/os-release", 2000), "nvidia_driver_proc": (read("/proc/driver/nvidia/version", 1000) or "").strip() or None,
             "onstart_done": os.path.exists("/workspace/.onstart_done"), "models_json": load(os.path.join(E.get("MODELS_DIR", "/workspace/models"), "models.json")),
             "note": "a container cannot read its own image digest; pass IMAGE_DIGEST from the Vast instance page or `docker manifest inspect <tag>`"}
host = {"cpu": sh("lscpu 2>/dev/null | egrep 'Model name|Socket|Core|Thread|NUMA node\\(s\\)|CPU\\(s\\):' | head -12"), "nproc": sh("nproc"),
        "memory_free_g": sh("free -g 2>/dev/null | head -2"), "disk": sh("df -h / /workspace /root/.cache 2>/dev/null | sort -u"),
        "disk_models_dir": sh(f"df -h {E.get('MODELS_DIR', '/workspace/models')} 2>/dev/null | tail -1"),
        "shm": sh("df -h /dev/shm 2>/dev/null | tail -1"), "ulimit_n": sh("ulimit -n"), "hf_home": E.get("HF_HOME"), "models_dir": E.get("MODELS_DIR")}
models = []
for line in (read(os.path.join(RES, "models-on-disk.txt"), 200000) or "").splitlines():
    parts = line.split("\t")
    if len(parts) == 3:
        models.append({"size_mib": int(parts[0]) if parts[0].isdigit() else None, "state": parts[1], "path": parts[2]})
keys = [k for k in E if k.startswith(("VLLM_", "NCCL_", "FLASHINFER_", "TORCH_", "CUDA_", "SGLANG_", "HF_", "TRITON_", "OMP_"))
        or k in ("P2P_OK", "P2P_DISABLED", "CUSTOM_ALLREDUCE", "ACS_SUSPECTED", "PESSIMISTIC_TP", "HOST_RAM_GB",
                 "HW_DECISIONS_SOURCE", "HW_DECISIONS_FILE", "ENGINE", "BENCH_ROOT", "TOOLS_DIR", "MODELS_DIR")]
env_vars = {k: E[k] for k in sorted(keys) if not any(s in k for s in ("TOKEN", "KEY", "SECRET"))}

# ---- hardware decision contract (bench/env.sh parsed results/hw/decisions.env) ----------------
decisions = {
    "file": E.get("HW_DECISIONS_SOURCE") or E.get("HW_DECISIONS_FILE"),
    "present": E.get("HW_DECISIONS_SOURCE") not in (None, "", "missing"),
    "p2p_ok": flag(E.get("P2P_OK"), None),
    "custom_allreduce": flag(E.get("CUSTOM_ALLREDUCE"), False),
    "nccl_p2p_disable": flag(E.get("NCCL_P2P_DISABLE"), False),
    "acs_suspected": flag(E.get("ACS_SUSPECTED"), None),      # None = unknown (no decisions.env), never a silent False
    "pessimistic_tp": flag(E.get("PESSIMISTIC_TP"), None),
    "host_ram_gb": int(E["HOST_RAM_GB"]) if E.get("HOST_RAM_GB", "").isdigit() else None,
    "notes": E.get("HW_NOTES") or None,
    "raw": read(E.get("HW_DECISIONS_FILE", ""), 8000),
}
hw_path = next((p for p in (os.path.join(RES, "hw", "hardware.json"), os.path.join(RES, "hardware.json"), os.path.join(RES, "hardware", "hardware.json")) if os.path.exists(p)), None)
hardware = load(hw_path) if hw_path else None
p2p_ok = decisions["p2p_ok"]
if p2p_ok is None and hardware:
    p2p_ok = hardware.get("p2p_ok", hardware.get("P2P_OK"))

out = {"collected_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "gpu_count": len(gpus), "gpus": gpus, "gpu_edition": edition,
       "topology_matrix": read(os.path.join(RES, "nvidia-smi-topo.txt")), "nvlink_status": sh(["nvidia-smi", "nvlink", "--status"]),
       "versions": versions, "container": container, "host": host, "models_on_disk": models, "env_vars": env_vars,
       "decisions": decisions, "hardware_json_path": hw_path, "p2p_ok": p2p_ok,
       "custom_allreduce": decisions["custom_allreduce"], "nccl_p2p_disable": decisions["nccl_p2p_disable"],
       "acs_suspected": decisions["acs_suspected"], "pessimistic_tp": decisions["pessimistic_tp"],
       "p2p_status": (hardware or {}).get("p2p_status"), "p2p_reasons": (hardware or {}).get("p2p_reasons"), "hardware": hardware,
       "raw_files": {k: os.path.join(RES, v) for k, v in {"nvidia_smi_q": "nvidia-smi-q.txt", "topo": "nvidia-smi-topo.txt", "nvidia_smi": "nvidia-smi.txt",
                                                          "pip_list": "pip-list.txt", "models_on_disk": "models-on-disk.txt"}.items()}}
with open(E["OUT"], "w", encoding="utf-8") as f: json.dump(out, f, indent=2)
print(json.dumps({"gpus": [g.get("name") for g in gpus], "vbios": edition["vbios"], "driver": edition["driver_version_smi"], "cuda_smi": edition["cuda_version_smi"],
                  "power_limit_w": [g.get("power.limit") for g in gpus], "pcie": [f"gen{g.get('pcie.link.gen.current')}x{g.get('pcie.link.width.current')}" for g in gpus],
                  "vllm": versions["vllm"], "torch": versions["torch"], "flashinfer": versions["flashinfer"], "b12x": versions["b12x"], "sglang": versions["sglang"],
                  "decisions_file": decisions["file"], "p2p_ok": p2p_ok, "custom_allreduce": decisions["custom_allreduce"],
                  "nccl_p2p_disable": decisions["nccl_p2p_disable"], "acs_suspected": decisions["acs_suspected"], "pessimistic_tp": decisions["pessimistic_tp"],
                  "host_ram_gb": decisions["host_ram_gb"], "models_on_disk": len(models)}, indent=2))
print("wrote", E["OUT"])
PY
