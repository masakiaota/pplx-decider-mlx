"""The original BF16 checkpoints the published weights are built from, and a check of a local copy.

    python tools/convert/originals.py {flash,27b} [--original DIR] [--download] [--quick]

Without --original, the checkpoint is looked up in the Hugging Face cache at the pinned revision; --download fetches
it there first (Flash 9.6 GB, 27B 54.1 GB; both repositories are public, Apache-2.0). Every weight file is checked
by sha256 (--quick checks sizes only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from st import sha256

ORIGINAL = {
    'flash': dict(
        repo='perplexity-ai/pplx-decider-flash-v1.3', revision='d06fb3f2d97e0a52325a0d463c450cf0338f44fe',
        sha256={
            'model-00001-of-00002.safetensors': '0048d1ddf2c5754fc05540fccdbc4ac50407e5cf14a35326be01cac0756b4de6',
            'model-00002-of-00002.safetensors': '40e0c0ccb3248a59db0724ffd1ad1c81fc935d9f474e6fd52e9df32af4690608',
            'locate-span/adapter.safetensors': '37bd9b8106ed172e4e386d9bda701ee0e3a4644d1ab385f2ea80cd224824ee84',
            'locate-span/head.safetensors': '0eaa00c0953be6ee1347ee782de7da56a0dab186a886f0fe77ce5528d8ce47ee',
            'readout.safetensors': '46358a2e2639f1956561687ba261dd9b37506f878e90eb5eabc15f114532bec3',
            'tokenizer.json': '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523',
        }),
    '27b': dict(
        repo='perplexity-ai/pplx-decider-v1.3', revision='9c2ebb27470d3dec2ad7ab727727e40d142c7213',
        sha256={
            'model-00001-of-00011.safetensors': '683762f9ba120ed6980f6888d808c55812d1f3c54d79517aab7b0f5705ec7b20',
            'model-00002-of-00011.safetensors': 'b8a1694cc4ef7ae24e411adab0092e87a78abe547c97c47a8e19c57e345f1738',
            'model-00003-of-00011.safetensors': '2ee7d8d814af369542d01b590f701c5f42e22ee756235a52125232f3d6e1a9b0',
            'model-00004-of-00011.safetensors': 'e853d09c8b2dedcb3f1a28d10f406df732d54d7603f55b1e6133e56deef8d3e5',
            'model-00005-of-00011.safetensors': '87e712601578fccf91b0874c6972dd24ff1368e80d3075c0577eae956a07d637',
            'model-00006-of-00011.safetensors': 'a19864e0e8e77147ba8f0d0005dc70fca46f1521e24bcab2334a53492d2c4062',
            'model-00007-of-00011.safetensors': '7a84ac88396740b1d02e88be50c4926bea99a3ec430dcac8b26738c8f1e4727a',
            'model-00008-of-00011.safetensors': '413bd8c46e78ae179382511fdc0c0134a4db7d532f1981594b0fb6a55c0df36e',
            'model-00009-of-00011.safetensors': 'afb31b650ae67b9b271ee02a251e6ac5b775e811db0abe6084789d8920c83e6a',
            'model-00010-of-00011.safetensors': '733da075428d6eb9ff0d6fc6e07caf482d70c1c86d77e4f37238fe4721be04f4',
            'model-00011-of-00011.safetensors': 'e68776c2fd0f3a503049fce03f4612850045a618819992af564967ed089878c3',
            'locate-span/adapter.safetensors': '545deb9fc426c50749d9923858d4295f682baadbed2fbb6f88cc342f6c01972a',
            'locate-span/head.safetensors': '4c5b587c7908e49c2a35b03fc22d9b91ff22a320a70e4e0a73c7265798c484e8',
            'readout.safetensors': '46928c7cd0084289f6237a833720683c9d8ff91965706e746cc34b3ef0f12039',
            'tokenizer.json': '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523',
        }),
}
# Files of the original repository that the published folders carry unchanged.
METADATA = ['config.json', 'decision_config.json', 'joint_config.json', 'processor_config.json', 'tokenizer.json',
            'tokenizer_config.json', 'chat_template.jinja', 'readout.safetensors']


def locate(model, original=None, download=False):
    """The local folder of the original checkpoint at the pinned revision."""
    if original is not None:
        return Path(original)
    from huggingface_hub import snapshot_download
    o = ORIGINAL[model]
    return Path(snapshot_download(o['repo'], revision=o['revision'], local_files_only=not download))


def check(model, folder, quick=False):
    folder = Path(folder)
    index = json.loads((folder / 'model.safetensors.index.json').read_text())['weight_map']
    expected = ORIGINAL[model]['sha256']
    assert sorted(set(index.values())) == sorted(k for k in expected if k.startswith('model-')), 'unexpected shards'
    for name, digest in expected.items():
        path = folder / name
        if not path.exists():
            raise SystemExit(f'{path}: missing')
        if not quick and sha256(path) != digest:
            raise SystemExit(f'{path}: sha256 differs from {ORIGINAL[model]["repo"]}@{ORIGINAL[model]["revision"][:8]}')
    return folder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', choices=sorted(ORIGINAL))
    ap.add_argument('--original', type=Path)
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    folder = check(a.model, locate(a.model, a.original, a.download), a.quick)
    print(f'{a.model}: {folder} matches {ORIGINAL[a.model]["repo"]}@{ORIGINAL[a.model]["revision"][:8]}')


if __name__ == '__main__':
    main()
