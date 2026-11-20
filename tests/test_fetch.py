"""fetch.py against a local mock of the Hugging Face Hub (hub_mock.py) with a tiny synthetic repository served under
the Flash repository id: download, manifest check, cache reuse without network, corrupted and missing files, errors.
Each check runs decider-mlx in a subprocess with HF_ENDPOINT pointing at the mock. No model is needed."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from decider_mlx import fetch, registry, tips

from hub_mock import commit_of as hub_commit, start

FLASH = 'pplx-decider-flash-v1.3'
REPO = 'masakiaota/pplx-decider-flash-v1.3-mlx'


def make_repo(folder):
    folder.mkdir()
    (folder / 'sub').mkdir()
    files = {'config.json': b'{"model_type": "test"}', 'weights.safetensors': os.urandom(11 * 1024 * 1024),
             'sub/head.safetensors': os.urandom(4096), 'LICENSE': b'Apache-2.0 (test)'}
    for name, data in files.items():
        (folder / name).write_bytes(data)
    manifest = {'format': 'decider-mlx-manifest-v1', 'model': FLASH, 'repository': REPO,
                'files': {n: {'size': len(d), 'sha256': fetch.sha256(folder / n)} for n, d in files.items()}}
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    (folder / 'README.md').write_text('# test\n')
    return files


@pytest.fixture
def hub(tmp_path):
    """A mock Hub with the tiny repository, and an environment that points huggingface_hub at it with an empty cache."""
    make_repo(tmp_path / 'repo')
    server, url = start({REPO: tmp_path / 'repo'})
    env = {k: v for k, v in os.environ.items() if not k.startswith('HF_')}
    env.update(HF_ENDPOINT=url, HF_HUB_CACHE=str(tmp_path / 'cache'), HF_HUB_DISABLE_TELEMETRY='1', LC_ALL='en_US.UTF-8')
    yield server, tmp_path, env
    server.shutdown()


def python(env, code):
    return subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=120)


def cli(env, *args):
    return subprocess.run([sys.executable, '-m', 'decider_mlx.cli', *args], env=env, capture_output=True, text=True, timeout=120)


def test_fetch_verify_and_reuse(hub):
    server, tmp, env = hub
    r = cli(env, 'fetch', '--model', FLASH)
    assert r.returncode == 0, r.stderr
    root = Path(r.stdout.strip())
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()) == \
        ['LICENSE', 'README.md', 'config.json', 'manifest.json', 'sub/head.safetensors', 'weights.safetensors']
    assert 'Downloaded pplx-decider-flash-v1.3' in r.stderr and 'Tip 1/' in r.stderr and 'Verified 4 files' in r.stderr
    assert f'uv run decider-mlx serve --model {FLASH} --port 8000' in r.stderr
    assert (root / 'weights.safetensors').read_bytes() == (tmp / 'repo/weights.safetensors').read_bytes()
    pinned = registry.spec(FLASH).hf_revision
    assert root.name == hub_commit(server.repos[REPO], pinned)   # the snapshot of the resolved commit
    server.shutdown()                                            # no network from here on
    offline = env | {'HF_HUB_OFFLINE': '1'}
    r = python(offline, f'from decider_mlx import fetch; print(fetch.snapshot({FLASH!r}, download=False))')
    assert r.returncode == 0 and Path(r.stdout.strip()) == root, r.stderr
    r = cli(offline, 'fetch', '--model', FLASH)                  # already there: check only
    assert r.returncode == 0 and 'Verified 4 files' in r.stderr and 'Downloaded' not in r.stderr, r.stderr
    w = registry.weights_from_snapshot(FLASH, root)
    assert w.main == root and w.locate == root / 'locate'


def test_corrupted_file_is_reported(hub):
    _, _, env = hub
    root = Path(cli(env, 'fetch', '--model', FLASH).stdout.strip())
    blob = (root / 'sub/head.safetensors').resolve()
    data = bytearray(blob.read_bytes()); data[0] ^= 1
    blob.write_bytes(bytes(data))
    r = cli(env, 'fetch', '--model', FLASH)
    assert r.returncode == 1 and 'sub/head.safetensors' in r.stderr and 'do not match manifest.json' in r.stderr, r.stderr


def test_missing_file_is_downloaded_again(hub):
    _, _, env = hub
    root = Path(cli(env, 'fetch', '--model', FLASH).stdout.strip())
    (root / 'weights.safetensors').resolve().unlink()
    (root / 'weights.safetensors').unlink()
    r = python(env, f'from decider_mlx import fetch; print(fetch.snapshot({FLASH!r}))')
    assert r.returncode == 0 and Path(r.stdout.strip()) == root, r.stderr
    assert not fetch.missing_files(root, fetch.read_manifest(root / 'manifest.json'))


def test_not_downloaded(hub):
    _, _, env = hub
    offline = env | {'HF_HUB_OFFLINE': '1'}
    r = python(offline, f'from decider_mlx import fetch; fetch.snapshot({FLASH!r}, download=False)')
    assert 'decider-mlx fetch --model pplx-decider-flash-v1.3' in r.stderr
    r = cli(offline, 'fetch', '--model', FLASH)
    assert r.returncode == 1 and 'cannot reach Hugging Face' in r.stderr, r.stderr


def test_unknown_repository(hub):
    _, _, env = hub
    r = cli(env, 'fetch', '--model', 'pplx-decider-v1.3')        # the mock serves only the Flash repository id
    assert r.returncode == 1 and 'not available on Hugging Face' in r.stderr, r.stderr


def test_revisions_are_pinned_or_marked():
    for s in registry.MODELS.values():
        assert re.fullmatch('[0-9a-f]{40}', s.hf_revision) or s.hf_revision == 'TBD-after-upload'
        assert s.hf_repo == f'masakiaota/{s.name}-mlx'


def test_progress_text():
    p = fetch.Progress(10 * 10**9)
    p.update(2.5 * 10**9)
    assert re.search(r'2\.5 / 10\.0 GB\s+25%', p.text()) and p.bar(10) == '[##........]'
    seeded = fetch.Progress(10 * 10**9, start=9 * 10**9)    # blobs already on disk are not part of the rate
    assert seeded.done == seeded.start == 9 * 10**9 and seeded.rate == 0.0
    assert fetch._duration(3725) == '1h02m' and fetch._duration(65) == '1m05s' and fetch._duration(9) == '9s'


def test_tips_are_short_and_match_across_languages():
    assert set(tips.TIPS) == {'en', 'ja'} and len(tips.TIPS['en']) == len(tips.TIPS['ja']) >= 5
    for lang in tips.TIPS:
        for i in range(tips.count(lang)):
            for model in registry.MODELS:
                title, lines = tips.tip(i, model, lang)
                assert title and 1 <= len(lines) <= 3 and all(len(line) <= 110 for line in lines)
                assert '{' not in title.replace('{"', '') and '{model}' not in ''.join(lines)
                assert '{locate_gb}' not in ''.join(lines) and '約 ? GB' not in ''.join(lines)


def test_tip_language():
    assert tips.language({'LANG': 'ja_JP.UTF-8'}, []) == 'ja'
    assert tips.language({'LC_ALL': 'en_US.UTF-8', 'LANG': 'ja_JP.UTF-8'}, []) == 'en'
    assert tips.language({'LC_ALL': 'C', 'LANG': 'ja_JP.UTF-8'}, []) == 'ja'
    assert tips.language({}, ['ja-JP', 'en-US']) == 'ja'
    assert tips.language({'LANG': 'C.UTF-8'}, ['en-US']) == 'en'
    assert tips.language({}, []) == 'en'


def test_next_steps_name_the_model():
    for lang in tips.TIPS:
        for model in registry.MODELS:
            assert any(f'--model {model} ' in line for line in tips.next_steps(model, lang))
