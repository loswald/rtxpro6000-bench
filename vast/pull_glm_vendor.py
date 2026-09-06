#!/usr/bin/env python3
"""Lift the Python tree out of a container image without Docker.

Why: several current models (GLM-5.3-Flash, Motif-3, Solar-Open2, Qwen3.8-Flash-Next) exist only in
their vendors' patched vLLM images before upstream support lands. The rented node runs inside a
container with no Docker, so we pull the image's layers over the OCI registry API and extract just
the site-/dist-packages we need into a directory that then goes on PYTHONPATH.

Downloads selected layers concurrently, verifies SHA256 before extraction, and applies their
filesystem changes in manifest order. Tokens are refreshed per download. Errors are not hidden.
Only selected Python packages are lifted; this is not a complete runnable container image.

Usage:
  pull_glm_vendor.py <image-ref> <dest-dir> [--want=vllm,flashinfer,...] [--layers=18-19,25] [--workers=4]
  e.g.  pull_image.py vllm/vllm-openai:glm53-flash-x86_64-cu130 /workspace/glmimg
        pull_image.py ghcr.io/motiftechnologies/vllm:v0.26.0-motif3-patch1 /workspace/motifimg

The pinned GLM image sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703
installs vLLM in layer 25 (313 MB). With an already compatible Torch/CUDA/dependency environment,
--want=vllm --layers=25 obtains its Python and binary extension tree without downloading base layers.
Use --keep-blobs=true to keep validated layer blobs for subsequent extraction or restart.
"""
import concurrent.futures, gzip, hashlib, json, os, shutil, sys, tarfile, time, urllib.request as u, urllib.error
from pathlib import Path, PurePosixPath

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
    def get(reference):
        raw = req(registry, repo, f"manifests/{reference}", ACCEPT, tok).read()
        actual = 'sha256:' + hashlib.sha256(raw).hexdigest()
        if reference.startswith('sha256:') and reference != actual:
            raise IOError('OCI manifest checksum mismatch')
        return json.loads(raw)
    m = get(tag)
    if "manifests" in m:  # multi-arch index -> pick amd64/linux
        cands = [s for s in m["manifests"]
                 if s.get("platform", {}).get("architecture") == "amd64"
                 and s.get("platform", {}).get("os", "linux") == "linux"]
        if not cands:
            raise SystemExit(f"no linux/amd64 variant; have {[s.get('platform') for s in m['manifests']]}")
        m = get(cands[0]['digest'])
    return m


def download(registry, repo, digest, size, tmp):
    for attempt in range(4):
        try:
            resp = req(registry, repo, f"blobs/{digest}", tok=token(registry, repo))
            got = 0
            checksum = hashlib.sha256()
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(8 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    checksum.update(chunk)
                    got += len(chunk)
            if size and got != size:
                raise IOError(f"short read {got}/{size}")
            if digest != 'sha256:' + checksum.hexdigest():
                raise IOError('OCI layer checksum mismatch')
            return True
        except Exception as e:  # noqa: BLE001
            print(f"    attempt {attempt + 1} failed: {type(e).__name__} {str(e)[:100]}", flush=True)
            time.sleep(3)
    return False


def select_layers(value, count):
    if not value:
        return list(range(count))
    chosen = set()
    for part in value.split(','):
        bounds = [int(n) for n in part.split('-')]
        if len(bounds) == 1:
            start = end = bounds[0]
        elif len(bounds) == 2:
            start, end = bounds
        else:
            raise ValueError('Use one-based --layers=18-19,25')
        if start < 1 or end < start or end > count:
            raise ValueError('Layer selection is outside the manifest')
        chosen.update(range(start - 1, end))
    return sorted(chosen)


def verified_blob(registry, repo, layer, cache):
    digest, size = layer['digest'], layer['size']
    if not digest.startswith('sha256:') or len(digest) != 71:
        raise ValueError('Only SHA256 OCI blobs are supported')
    target = os.path.join(cache, digest.split(':')[1] + '.blob')
    if os.path.exists(target) and os.path.getsize(target) == size:
        checksum = hashlib.sha256()
        with open(target, 'rb') as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b''):
                checksum.update(chunk)
        if digest == 'sha256:' + checksum.hexdigest():
            print('    verified cached blob ' + digest[:24], flush=True)
            return target
    partial = target + '.part'
    if not download(registry, repo, digest, size, partial):
        raise RuntimeError('Required OCI layer download failed: ' + digest)
    os.replace(partial, target)
    print(f'    downloaded and verified {size / 1e6:.1f} MB {digest[:24]}', flush=True)
    return target


def extract_selected(path, dest, pats):
    """Apply one layer in order; reject unsafe extraction instead of hiding errors."""
    root = Path(dest).resolve()
    def wanted(name):
        return any(p in name for p in pats)
    def remove(relative):
        target = root.joinpath(*PurePosixPath(relative).parts)
        resolved = target.resolve()
        if resolved == root or not resolved.is_relative_to(root):
            raise ValueError('Unsafe OCI whiteout path')
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    kept = 0
    with open(path, 'rb') as probe:
        magic = probe.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(path, 'rb') as raw, tarfile.open(fileobj=raw, mode='r|*') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.name == '.wh..wh..opq':
                parent = root.joinpath(*name.parent.parts)
                if parent.is_dir():
                    for child in parent.iterdir():
                        relative = child.relative_to(root).as_posix()
                        if wanted(relative):
                            remove(relative)
                continue
            if name.name.startswith('.wh.'):
                deleted = str(name.with_name(name.name[4:]))
                if wanted(deleted):
                    remove(deleted)
                continue
            if wanted(member.name):
                archive.extract(member, dest, filter='data')
                kept += 1
    return kept


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
    selected = select_layers(opts.get('layers'), len(layers))
    workers = int(opts.get('workers', '4'))
    if not 1 <= workers <= 16:
        raise ValueError('--workers must be between 1 and 16')
    cache = os.path.join(dest, '_oci_blobs')
    os.makedirs(cache, exist_ok=True)
    print(f"selected layers {[i + 1 for i in selected]}, {sum(layers[i]['size'] for i in selected) / 1e9:.3f} GB, workers={workers}", flush=True)
    kept = 0
    downloaded = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for i in selected:
            layer = layers[i]
            if layer['digest'] not in pending:
                pending[layer['digest']] = pool.submit(verified_blob, registry, repo, layer, cache)
        # Downloads overlap; filesystem overlay order always follows the manifest.
        for i in selected:
            layer = layers[i]
            print(f"  extracting layer {i + 1}/{len(layers)}  {layer['size'] / 1e9:.2f} GB", flush=True)
            blob = pending[layer['digest']].result()
            downloaded.add(blob)
            kept += extract_selected(blob, dest, pats)
            print(f"    kept {kept} files so far", flush=True)
    if opts.get('keep-blobs', 'false').lower() != 'true':
        for blob in downloaded:
            os.remove(blob)
    print(f"EXTRACTED {kept} files into {dest}", flush=True)
    with open(os.path.join(dest, 'pull_manifest.json'), 'w') as receipt:
        json.dump({'image_ref': ref, 'manifest': m, 'kept_files': kept,
                   'selected_layers_one_based': [i + 1 for i in selected], 'wanted_packages': want,
                   'partial_image_overlay': len(selected) != len(layers),
                   'downloaded_layer_sha256_verified': True, 'ordered_extraction': True,
                   'parallel_download_workers': workers}, receipt, indent=2)
    for root, dirs, files in os.walk(dest):
        if root.endswith(("site-packages", "dist-packages")):
            print("  python tree:", root, "->", sorted(d for d in dirs if not d.endswith(".dist-info"))[:12])
            ver = os.path.join(root, "vllm", "version.py")
            if os.path.exists(ver):
                print("  vllm version.py:", open(ver).read().strip()[:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
