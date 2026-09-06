#!/usr/bin/env python3
"""Capture small, immutable model/config/tokenizer and runtime fingerprints.

The download receipt records a requested Hub commit and weight-file sizes. This
does not claim to re-hash all weight tensors or independently certify a provider.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path


def capture(model_dir):
    root = Path(model_dir)
    names = ('config.json', 'generation_config.json', 'model.safetensors.index.json',
             'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'chat_template.jinja', 'chat_template.json', 'download_manifest.json')
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in names if (root / name).is_file()}
    receipt = root / 'download_manifest.json'
    manifest = json.loads(receipt.read_text()) if receipt.is_file() else {}
    identity = {'model_id': manifest.get('model'), 'model_revision': manifest.get('revision'),
                'files_sha256': files}
    identity['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    versions = {}
    for name in ('vllm', 'torch', 'transformers', 'triton', 'b12x', 'compressed-tensors',
                 'flashinfer-python', 'flashinfer-jit-cache', 'flashinfer-cubin', 'huggingface-hub'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'identity': identity, 'packages': versions,
            'weight_hash_verification': 'not independently repeated; Hub download receipt captured'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(capture(args.model_dir), indent=2) + '\n', encoding='utf-8')
