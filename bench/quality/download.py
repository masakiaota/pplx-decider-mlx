"""Download the public source data of the evaluation and calibration sets into DATA/raw (about 45 MB).

    python bench/quality/download.py [--data bench/quality/data] [--refresh-vqa]

- BoolQ, SQuAD v2 and Qasper: Parquet files from the Hugging Face Hub at pinned revisions, checked by sha256.
- VQAv2: row pages from the Hugging Face dataset viewer API (questions, answers and image URLs). The images
  themselves are downloaded by build_sets.py for the rows it selects. The image URLs in these pages are signed and
  expire after a while; if build_sets.py cannot fetch an image, run this again with --refresh-vqa.
Nothing here needs a Hugging Face account. Files already present (and matching) are not downloaded again.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from common import DATA, PARQUET, USER_AGENT, VQA_PAGES, VQA_REPO, VQA_REVISION, sha256_file


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    partial = path.with_name(path.name + '.part')
    partial.write_bytes(data)
    partial.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('--refresh-vqa', action='store_true', help='download the VQAv2 row pages again (fresh image URLs)')
    a = ap.parse_args()
    raw = a.data / 'raw'
    for name, (repo, revision, filename, sha) in PARQUET.items():
        path = raw / name
        if not (path.exists() and sha256_file(path) == sha):
            print(f'{name}: {repo}@{revision[:8]} {filename}', flush=True)
            fetch(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}', path)
        if sha256_file(path) != sha:
            raise SystemExit(f'{name}: sha256 differs from the pinned file')
    info = json.loads(urllib.request.urlopen(urllib.request.Request(
        f'https://huggingface.co/api/datasets/{VQA_REPO}', headers={'User-Agent': USER_AGENT}), timeout=60).read())
    if info.get('sha') != VQA_REVISION:
        print(f'warning: {VQA_REPO} is now at {info.get("sha")}, the sets were built at {VQA_REVISION}; '
              'build_sets.py checks the image hashes', flush=True)
    for offset, length in VQA_PAGES:
        path = raw / f'vqa-rows-{offset}.json'
        if path.exists() and not a.refresh_vqa:
            continue
        query = urllib.parse.urlencode({'dataset': VQA_REPO, 'config': 'default', 'split': 'validation',
                                        'offset': offset, 'length': length})
        print(f'{path.name}: VQAv2 validation rows {offset}..{offset + length - 1}', flush=True)
        fetch(f'https://datasets-server.huggingface.co/rows?{query}', path)
        if len(json.loads(path.read_text())['rows']) != length:
            raise SystemExit(f'{path.name}: expected {length} rows')
    print(f'done: {raw}')


if __name__ == '__main__':
    main()
