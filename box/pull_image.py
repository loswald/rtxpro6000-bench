#!/usr/bin/env python3
"""Pull the vendor vLLM build for GLM-5.3-Flash out of its Docker image without Docker.

v2: Docker Hub bearer tokens expire after a few minutes, which 401'd every layer after
the first few in v1. Now we mint a fresh token per layer, retry once on 401, and stream
each blob to disk instead of holding 8.6 GB in memory.
"""
import json, os, sys, tarfile, gzip, time, shutil, urllib.request as u, urllib.error

REPO = sys.argv[1] if len(sys.argv) > 1 else "vllm/vllm-openai"
TAG = sys.argv[2] if len(sys.argv) > 2 else "glm53-flash-x86_64-cu130"
DEST = sys.argv[3] if len(sys.argv) > 3 else "/workspace/glmimg"
ALL = "--all" in sys.argv
TMP = "/workspace/_layer_" + str(os.getpid()) + ".tmp"
ACCEPT = ",".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])
# EXTRA_WANT: extra path fragments to keep, comma-separated (editable installs put the
# real source tree outside site-packages, e.g. /sgl-workspace/sglang/python/).
EXTRA = tuple(x for x in os.environ.get("EXTRA_WANT", "").split(",") if x)
WANT = EXTRA + (("site-packages/", "dist-packages/", "/bin/python", "/lib/python3") if ("--all" in sys.argv) else ("site-packages/vllm", "dist-packages/vllm",
        "site-packages/flashinfer", "dist-packages/flashinfer",
        "site-packages/transformers", "dist-packages/transformers",
        "site-packages/compressed_tensors", "dist-packages/compressed_tensors"))


# registry selection: "ghcr.io/owner/name" -> GitHub container registry, else Docker Hub
if REPO.startswith("ghcr.io/"):
    REPO = REPO[len("ghcr.io/"):]
    REGISTRY = "https://ghcr.io"
    TOKEN_URL = f"https://ghcr.io/token?scope=repository:{REPO}:pull"
else:
    REGISTRY = "https://registry-1.docker.io"
    TOKEN_URL = (f"https://auth.docker.io/token?service=registry.docker.io"
                 f"&scope=repository:{REPO}:pull")


def token():
    return json.load(u.urlopen(TOKEN_URL, timeout=60))["token"]


def req(path, tok, accept=None):
    r = u.Request(f"{REGISTRY}/v2/{REPO}/{path}")
    r.add_header("Authorization", "Bearer " + tok)
    if accept:
        r.add_header("Accept", accept)
    return u.urlopen(r, timeout=600)


def manifest():
    tok = token()
    m = json.load(req(f"manifests/{TAG}", tok, ACCEPT))
    if "manifests" in m:
        amd = [s for s in m["manifests"]
               if s.get("platform", {}).get("architecture") == "amd64"]
        m = json.load(req(f"manifests/{amd[0]['digest']}", tok, ACCEPT))
    return m


def download(digest, size):
    """Fresh token per attempt; 401 means the token aged out mid-pull."""
    for attempt in range(4):
        try:
            resp = req(f"blobs/{digest}", token())
            got = 0
            with open(TMP, "wb") as fh:
                while True:
                    chunk = resp.read(4 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
            if size and got < size * 0.98:
                raise IOError(f"short read {got}/{size}")
            return True
        except Exception as e:
            print(f"    attempt {attempt+1} failed: {type(e).__name__} {str(e)[:90]}",
                  flush=True)
            time.sleep(3)
    return False


def main():
    m = manifest()
    layers = m.get("layers", [])
    print(f"{len(layers)} layers, {sum(l['size'] for l in layers)/1e9:.1f} GB", flush=True)
    os.makedirs(DEST, exist_ok=True)
    kept = 0
    for i, l in enumerate(layers):
        if l["size"] < 1 << 20:          # skip metadata-only layers
            continue
        print(f"  layer {i+1}/{len(layers)}  {l['size']/1e9:.2f} GB", flush=True)
        if not download(l["digest"], l["size"]):
            print("    GAVE UP on this layer", flush=True)
            continue
        try:
            opener = gzip.open if open(TMP, "rb").read(2) == b"\x1f\x8b" else open
            with opener(TMP, "rb") as raw, tarfile.open(fileobj=raw, mode="r|*") as tf:
                for mem in tf:
                    if any(w in mem.name for w in WANT):
                        try:
                            tf.extract(mem, DEST, filter="tar")
                            kept += 1
                        except Exception:
                            pass
            print(f"    kept {kept} files so far", flush=True)
        except Exception as e:
            print(f"    untar failed: {type(e).__name__} {str(e)[:110]}", flush=True)
    if os.path.exists(TMP):
        os.remove(TMP)
    print(f"EXTRACTED {kept} files into {DEST}", flush=True)
    for root, dirs, files in os.walk(DEST):
        if root.endswith("model_executor/models"):
            g = sorted(f for f in files if "glm5" in f.lower())
            print("  glm5 model files:", g or "NONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())