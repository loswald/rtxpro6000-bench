import json, sys, urllib.request
CAND = [
    # replica tier: must fit ONE 96 GB card with big KV headroom
    "openai/gpt-oss-20b",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "meta-models/Muse-Glimmer-30B",
    # mid tier: 100-250 GB, TP2 or TP4
    "mistralai/Mistral-Small-4-119B-2603",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    "stepfun-ai/Step-3.7-Flash-FP8",
    "MiniMaxAI/MiniMax-M2.5",
    # GLM path
    "Libertai/GLM-5.3-Flash-NVFP4",
    "RedHatAI/GLM-5.3-Flash-NVFP4",
    "zai-org/GLM-5.3-Flash",
]
print(f"{'repo':<52} {'exists':>7} {'GB':>8} {'licence':<24} {'modified':<11}")
for r in CAND:
    try:
        d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{r}", timeout=25))
    except Exception as e:
        print(f"{r:<52} {'NO':>7} {'-':>8} {'-':<24} {type(e).__name__}")
        continue
    gb = 0.0
    for s in d.get("siblings", []):
        pass
    try:
        t = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{r}/tree/main?recursive=1", timeout=30))
        gb = sum(f.get("size", 0) or 0 for f in t if f.get("path","").endswith(".safetensors"))/1e9
    except Exception:
        gb = -1
    lic = next((t.split(":",1)[1] for t in d.get("tags", []) if t.startswith("license:")), "UNDECLARED")
    print(f"{r:<52} {'yes':>7} {gb:>8.1f} {lic:<24} {str(d.get('lastModified',''))[:10]}")