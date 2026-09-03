#!/usr/bin/env python3
"""Manifest-only probe: does each vendor image exist, how big is it, for which registry.
No layers are downloaded. Uses pull_image.py's registry code."""
import importlib.util, sys

spec = importlib.util.spec_from_file_location("pi", "/workspace/bench/pull_image.py")
pi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi)

REFS = sys.argv[1:] or [
    "ghcr.io/motiftechnologies/vllm:v0.26.0-motif3-patch1",   # Motif-3 (AA 47), vendor fork only
    "upstage/vllm-solar-open2:1.0.0",                          # Solar-Open2 (AA 37), vendor fork only
    "vllm/vllm-openai:qwen38-flash-next",                      # Qwen3.8-Flash-Next, main >= 31 Aug only
    "vllm/vllm-openai:deepseekv4-flash-vision",                # V4-Flash-Vision-Exp with vision input
]
for ref in REFS:
    try:
        reg, repo, tag = pi.parse_ref(ref)
        m = pi.manifest(reg, repo, tag)
        layers = m.get("layers", [])
        big = sorted(layers, key=lambda x: -x["size"])[:2]
        print(f"  OK   {ref}: {len(layers)} layers, {sum(l['size'] for l in layers) / 1e9:.1f} GB "
              f"(largest {', '.join(f'{l['size'] / 1e9:.1f}' for l in big)} GB)")
    except Exception as e:  # noqa: BLE001
        print(f"  MISS {ref}: {type(e).__name__} {str(e)[:120]}")
