#!/usr/bin/env python3
"""Lift the Python tree out of a container image without Docker.

Why: several current models (GLM-5.3-Flash, Motif-3, Solar-Open2, Qwen3.8-Flash-Next) exist only in
their vendors' patched vLLM images before upstream support lands. The rented node runs inside a
container with no Docker, so we pull the image's layers over the OCI registry API and extract just
the site-/dist-packages we need into a directory that then goes on PYTHONPATH.

v3: supports Docker Hub (docker.io) and GitHub Container Registry (ghcr.io); mints a fresh bearer
token per layer (tokens expire in minutes, which 401'd v1); streams blobs to disk; only extracts
paths matching WANT; skips metadata layers.

Usage:
  pull_image.py <image-ref> <dest-dir> [--want vllm,flashinfer,...]
  e.g.  pull_image.py vllm/vllm-openai:glm53-flash-x86_64-cu130 /workspace/glmimg
        pull_image.py ghcr.io/motiftechnologies/vllm:v0.26.0-motif3-patch1 /workspace/motifimg
"""
import gzip, json, os, sys, tarfile, time, urllib.request as u, urllib.error

ACCEPT = ",".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])
DEFAULT_WANT = ["vllm", "flashinfer", "transformers", "compressed_tensors", "modelopt", "triton_kernels"]


def parse_ref(ref):
    """'ghcr.io/org/name:tag' -> (registry, repo, tag); bare 'org/name:tag' -> docker.io."""
    if "://" in ref:
        ref = ref.split("://", 1)[1]
    first, _, rest = ref.partition("/")
    if "." in first or ":" in first or first == "localhost":
        registry, path = first, rest
    else:
        registry, path = "docker.io", ref
    if "@" in path:
        repo, tag = path.split("@", 1)
    elif ":" in path.rsplit("/", 1)[-1]:
        repo, tag = path.rsplit(":", 1)
    else:
        repo, tag = path, "latest"
    if registry == "docker.io" and "/" not in repo:
        repo = "library/" + repo
    return registry, repo, tag


def endpoints(registry):
    if registry == "docker.io":
        return ("https://registry-1.docker.io",
                "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull")
    if registry == "ghcr.io":
        return ("https://ghcr.io", "https://ghcr.io/token?scope=repository:{repo}:pull")
    # generic: try the registry's own token endpoint advertised via WWW-Authenticate
    return (f"https://{registry}", None)


def token(registry, repo):
    base, tok_url = endpoints(registry)
    if tok_url is None:
        # discover from the 401 challenge
        try:
            u.urlopen(f"{base}/v2/", timeout=30)
            return None
        except urllib.error.HTTPError as e:
            ch = e.headers.get("WWW-Authenticate", "")
            realm = ch.split('realm="')[1].split('"')[0]
            service = ch.split('service="')[1].split('"')[0] if 'service="' in ch else ""
            tok_url = f"{realm}?service={service}&scope=repository:{{repo}}:pull"
    r = u.Request(tok_url.format(repo=repo))
    return json.load(u.urlopen(r, timeout=60))["token"]


def req(registry, repo, path, accept=None, tok=None):
    base, _ = endpoints(registry)
    r = u.Request(f"{base}/v2/{repo}/{path}")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    if accept:
        r.add_header("Accept", accept)
    return u.urlopen(r, timeout=900)


def manifest(registry, repo, tag):
    tok = token(registry, repo)
    m = json.load(req(registry, repo, f"manifests/{tag}", ACCEPT, tok))
    if "manifests" in m:  # multi-arch index -> pick amd64/linux
        cands = [s for s in m["manifests"]
                 if s.get("platform", {}).get("architecture") == "amd64"
                 and s.get("platform", {}).get("os", "linux") == "linux"]
        if not cands:
            raise SystemExit(f"no linux/amd64 variant; have {[s.get('platform') for s in m['manifests']]}")
        m = json.load(req(registry, repo, f"manifests/{cands[0]['digest']}", ACCEPT, tok))
    return m


def download(registry, repo, digest, size, tmp):
    for attempt in range(4):
        try:
            resp = req(registry, repo, f"blobs/{digest}", tok=token(registry, repo))
            got = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(8 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
            if size and got < size * 0.98:
                raise IOError(f"short read {got}/{size}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"    attempt {attempt + 1} failed: {type(e).__name__} {str(e)[:100]}", flush=True)
            time.sleep(3)
    return False


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) < 2:
        raise SystemExit(__doc__)
    ref, dest = args[0], args[1]
    want = opts.get("want", ",".join(DEFAULT_WANT)).split(",")
    pats = [f"{p}-packages/{w}" for w in want for p in ("site", "dist")]
    registry, repo, tag = parse_ref(ref)
    print(f"registry={registry} repo={repo} tag={tag}", flush=True)
    m = manifest(registry, repo, tag)
    layers = m.get("layers", [])
    print(f"{len(layers)} layers, {sum(l['size'] for l in layers) / 1e9:.1f} GB total", flush=True)
    os.makedirs(dest, exist_ok=True)
    tmp = os.path.join(dest, "_layer.tmp")
    kept = 0
    for i, l in enumerate(layers):
        if l["size"] < 1 << 20:
            continue
        print(f"  layer {i + 1}/{len(layers)}  {l['size'] / 1e9:.2f} GB", flush=True)
        if not download(registry, repo, l["digest"], l["size"], tmp):
            print("    GAVE UP on this layer", flush=True)
            continue
        try:
            with open(tmp, "rb") as probe:
                magic = probe.read(2)
            opener = gzip.open if magic == b"\x1f\x8b" else open
            with opener(tmp, "rb") as raw, tarfile.open(fileobj=raw, mode="r|*") as tf:
                for mem in tf:
                    if any(p in mem.name for p in pats):
                        try:
                            tf.extract(mem, dest, filter="tar")
                            kept += 1
                        except Exception:  # noqa: BLE001
                            pass
            print(f"    kept {kept} files so far", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    untar failed: {type(e).__name__} {str(e)[:120]}", flush=True)
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"EXTRACTED {kept} files into {dest}", flush=True)
    for root, dirs, files in os.walk(dest):
        if root.endswith(("site-packages", "dist-packages")):
            print("  python tree:", root, "->", sorted(d for d in dirs if not d.endswith(".dist-info"))[:12])
            ver = os.path.join(root, "vllm", "version.py")
            if os.path.exists(ver):
                print("  vllm version.py:", open(ver).read().strip()[:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
