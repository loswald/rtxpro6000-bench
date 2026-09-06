# Pinned GLM vendor-image layer plan

The public OCI manifest SHA256 was independently verified as `2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703` on 5 September 2026. The manifest and SHA256-verified config/history are preserved beside this note as `glm_vendor_oci_manifest.json` and `glm_vendor_oci_config.json`.

| One-based layer | Compressed size | Config history |
|---|---:|---|
| 1–17 | about 1.8 GB | OS, CUDA toolkit, Python/uv and build configuration |
| 18 | 4,634.87 MB | CUDA Python requirements, including Torch |
| 19 | 1,511.33 MB | FlashInfer JIT cache 0.6.17 |
| 23 | tiny | Torch version receipt |
| 24 | tiny | vLLM wheel hash receipt |
| **25** | **313.27 MB** | **Install vLLM wheel** |
| 28 | 14.48 MB | ep_kernels wheel |
| 32 | 310.83 MB | KV connectors / Mooncake |

The two tiny receipt blobs were downloaded and verified against their OCI SHA256 values. Their contents record `torch==2.13.0+cu130`, `torchaudio==2.11.0+cu130`, and `torchvision==0.28.0+cu130`; the installed vLLM wheel is `vllm-0.1.dev20051+g487ecf187-cp38-abi3-linux_x86_64.whl`, SHA256 `d9da7f045b5a48200c91bc68591edc9d30eebc117be1ce1ec550f73e406112fc`.

For the new node's existing compatible core runtime, fetch only vLLM initially:

```sh
python3 pull_glm_vendor.py \
  vllm/vllm-openai@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703 \
  /workspace/glmimg --want=vllm --layers=25 --workers=4
```

Layer 25 SHA256 is `b050c304e2b162e8fef6da142044382a744d97de71c10c3b282e7b4161c1aeb7`. This skips downloading several GB of base/toolkit/dependency content when only the vendor vLLM overlay is needed. The config history identifies the installing layer; import and binary compatibility must still be checked on the destination runtime. It does not assert that every image dependency is present or that model serving is qualified.

The updated `vast/pull_glm_vendor.py` permits explicit one-based layer ranges, downloads independent blobs concurrently, verifies manifest/blob hashes, reuses only hash-validated cache files, and extracts in original manifest order with whiteout handling. An extraction failure aborts instead of silently leaving partial packages. Its receipt records selected layers and that the output is a partial image overlay. A temporary offline fixture checked layer bounds, overlay replacement, whiteout deletion and exclusion of unrelated files. Existing managed hosts were not changed during this investigation.
