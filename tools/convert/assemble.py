"""Last step: the folder that is uploaded to Hugging Face, with manifest.json.

    python tools/convert/assemble.py flash ORIGINAL OUT --main SHARED --locate LOCATE
    python tools/convert/assemble.py 27b ORIGINAL OUT --q4 Q4 --rest REST --rowq4 S4 --adapter ADAPTER

Copies (APFS clones) the converted weights into the layout decider-mlx reads (src/decider_mlx/registry.py), the
unchanged tokenizer, processor, configuration files, readout head and LICENSE of the original repository, writes
mlx_config.json (the fields the engine reads plus a description of the format), NOTICE (cards/NOTICE.template
with the original NOTICE) and README.md (the model card in cards/), and finally manifest.json: size and sha256 of
every file except README.md and manifest.json, which `decider-mlx fetch` checks after downloading.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from originals import METADATA, ORIGINAL  # noqa: E402
from st import clone, json_dump, sha256  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = {'flash': ('pplx-decider-flash-v1.3', 'masakiaota/pplx-decider-flash-v1.3-mlx'),
        '27b': ('pplx-decider-v1.3', 'masakiaota/pplx-decider-v1.3-mlx')}
KEEP = ('format', 'source_repository', 'source_revision', 'quantization', 'quantized_modules', 'quantize_gates',
        'preserved', 'weight_layout', 'mlx_version', 'shared_affine8')
LOCATE_FLASH = ['head.safetensors', 'locate_config.json', 'marker.safetensors', 'mlx_config.json',
                'model.safetensors.index.json', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']


def manifest(model, out):
    name, repo = REPO[model]
    files = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()
                   and str(p.relative_to(out)) not in ('README.md', 'manifest.json'))
    entries = {f: {'size': (out / f).stat().st_size, 'sha256': sha256(out / f)} for f in files}
    o = ORIGINAL[model]
    json_dump(out / 'manifest.json', {
        'format': 'decider-mlx-manifest-v1', 'model': name, 'repository': repo,
        'engine': 'decider-mlx (https://github.com/masakiaota/pplx-decider-mlx)',
        'base_model': {'repository': o['repo'], 'revision': o['revision'], 'license': 'apache-2.0'},
        'total_bytes': sum(e['size'] for e in entries.values()), 'files': entries})
    return len(entries)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', choices=sorted(REPO))
    ap.add_argument('original', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--main', type=Path, help='Flash: flash_shared.py output')
    ap.add_argument('--locate', type=Path, help='Flash: flash_locate.py output')
    ap.add_argument('--q4', type=Path, help='27B: quantize.py --bits 4 output (for mlx_config.json)')
    ap.add_argument('--rest', type=Path, help='27B: rest.py output')
    ap.add_argument('--rowq4', type=Path, help='27B: s4.py output')
    ap.add_argument('--adapter', type=Path, help='27B: adapter27b.py output')
    a = ap.parse_args()
    out = a.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out}: must be empty')
    files = {f: a.original / f for f in METADATA}
    if a.model == 'flash':
        files.update({f: a.main / f for f in ('model.safetensors.index.json', 'row-sums.safetensors',
                                              'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors')})
        files.update({f'locate/{f}': a.locate / f for f in LOCATE_FLASH})
        config = json.loads((a.main / 'mlx_config.json').read_text())
        config['shared_affine8'] = True
    else:
        files.update({f: a.rest / f for f in ('model.safetensors.index.json', 'model-rest.safetensors')})
        rows = sorted(a.rowq4.glob('*.safetensors'))
        assert len(rows) == 368, len(rows)
        files.update({f'rowq4/{p.name}': p for p in rows})
        files.update({'locate/locate_config.json': a.original / 'locate-span/locate_config.json',
                      'locate/head.safetensors': a.original / 'locate-span/head.safetensors',
                      'locate/adapter.safetensors': a.adapter})
        config = json.loads((a.q4 / 'mlx_config.json').read_text())
    for rel, src in files.items():
        clone(src, out / rel)
    cfg = {k: config[k] for k in KEEP if k in config}
    if a.model == '27b':   # only the group-64 modules left in model-rest (k/v); the others are in rowq4/
        index = json.loads((out / 'model.safetensors.index.json').read_text())['weight_map']
        cfg['quantized_modules'] = [n for n in cfg['quantized_modules'] if n + '.scales' in index]
        cfg['rowq4'] = {'directory': 'rowq4', 'format': 'row-wise symmetric 4-bit, signed int4 codes (s4) and one FP32 scale per row',
                        'files': len(rows)}
    json_dump(out / 'mlx_config.json', cfg)
    clone(a.original / 'LICENSE', out / 'LICENSE')
    name, repo = REPO[a.model]
    o = ORIGINAL[a.model]
    (out / 'NOTICE').write_text((HERE / 'cards/NOTICE.template').read_text().format(
        repo=repo.split('/')[1], base=o['repo'], revision=o['revision'], original=(a.original / 'NOTICE').read_text().strip()))
    clone(HERE / f'cards/{a.model}.md', out / 'README.md')
    n = manifest(a.model, out)
    print(json.dumps({'out': str(out), 'files': n}))


if __name__ == '__main__':
    main()
