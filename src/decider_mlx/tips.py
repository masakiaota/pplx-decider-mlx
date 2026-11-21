"""Usage tips shown in turn while `decider-mlx fetch` downloads the weights, and the next step shown when it is done.

Japanese when the user's language is Japanese (LC_ALL, LC_MESSAGES, LANG, then macOS AppleLanguages), English
otherwise. Only these messages are translated; errors, --help and server logs stay in English.
"""
from __future__ import annotations

import os
import subprocess
import sys

LOCATE_GB = {'pplx-decider-flash-v1.3': '3.8', 'pplx-decider-v1.3': '0.95'}

TIPS = {
    'en': (
        ('Send many requests as one array',
         'POST a JSON array of requests to /v1/decisions to get the answers back as an array in the same order.',
         'Arrays of 64 or more requests are packed into batches internally, which is faster than one at a time.'),
        ('Run a JSONL file without a server',
         'One request per line, answers in the same order:',
         'uv run decider-mlx run --model {model} --input in.jsonl --output out.jsonl'),
        ('Many questions on a long text: ?mode=isolated',
         'With ?mode=isolated the text is read once and every question is answered from it, which is faster.',
         'Answers can shift slightly with the set of questions.'),
        ('Extract matching passages (Locate)',
         'A question {"type": "locate", "instructions": "error codes"} returns verbatim spans with character offsets.',
         'The first Locate request loads extra weights (about {locate_gb} GB).'),
        ('Images too',
         'Put {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}} parts in "state".',
         'PNG, JPEG and WebP, up to 8 images per request.'),
        ('Call it from Python',
         'from decider_mlx import load; app = load("{model}"); app.decide(request)',
         'No server needed; it fits into an existing script.'),
    ),
    'ja': (
        ('複数のリクエストを配列でまとめて送る',
         '/v1/decisions にリクエストの配列を POST すると、同じ順序の配列で結果が返ります。',
         '64 件以上は内部でまとめて処理されるため、1 件ずつ送るより高速です。'),
        ('JSONL ファイルをそのまま一括処理する',
         'サーバーを起動せずに、ファイル単位で大量のリクエストを処理できます。',
         'uv run decider-mlx run --model {model} --input in.jsonl --output out.jsonl'),
        ('長い文章に複数の質問をするなら isolated モード',
         '?mode=isolated を付けると文章を 1 回だけ読み込んで全質問に答えるため高速です。',
         'なお、質問の組み合わせによって結果がわずかに変わることがあります。'),
        ('文章から該当箇所を抜き出す（Locate）',
         '{"type": "locate", "instructions": "エラーコード"} で該当箇所を文字位置付きで抽出できます。',
         '初回の Locate リクエスト時のみ、追加の重み（約 {locate_gb} GB）を読み込みます。'),
        ('画像もあわせて判定する',
         'state に {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}} を含めます。',
         'PNG、JPEG、WebP の画像を、1 リクエストにつき最大 8 枚まで渡せます。'),
        ('Python から直接呼び出す',
         'from decider_mlx import load; app = load("{model}") のあと app.decide(request) で実行できます。',
         'サーバーを立てずに、既存のスクリプトへそのまま組み込めます。'),
    ),
}

LABEL = {'en': 'Tip', 'ja': 'ヒント'}

NEXT = {
    'en': ('Ready. To start the server, run:',
           '  uv run decider-mlx serve --model {model} --port {port}',
           'For an example request, see "Quick start" in the README.'),
    'ja': ('準備が完了しました。サーバーを起動するには次のコマンドを実行してください。',
           '  uv run decider-mlx serve --model {model} --port {port}',
           'リクエストの例は README の「クイックスタート」をご覧ください。'),
}


def _is_ja(value):
    return value.lower().startswith('ja')


def language(environ=None, apple_languages=None):
    """'ja' or 'en'. The first of LC_ALL, LC_MESSAGES, LANG that is set (and not C/POSIX) decides; when none is,
    the first entry of macOS's AppleLanguages does."""
    environ = os.environ if environ is None else environ
    for key in ('LC_ALL', 'LC_MESSAGES', 'LANG'):
        value = environ.get(key, '')
        if value and value.split('.')[0] not in ('C', 'POSIX'):
            return 'ja' if _is_ja(value) else 'en'
    first = _apple_language() if apple_languages is None else (apple_languages[0] if apple_languages else '')
    return 'ja' if first and _is_ja(first) else 'en'


def _apple_language():
    if sys.platform != 'darwin':
        return ''
    try:
        out = subprocess.run(['defaults', 'read', '-g', 'AppleLanguages'], capture_output=True, text=True,
                             timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return ''
    for token in out.replace('(', ' ').replace(')', ' ').replace(',', ' ').split():
        token = token.strip('"')
        if token:
            return token
    return ''


def _fill(text, model):
    return text.replace('{model}', model).replace('{locate_gb}', LOCATE_GB.get(model, '?'))


def count(lang='en'):
    return len(TIPS[lang])


def tip(i, model, lang='en'):
    """(title, lines) of tip i (cycling), with the model's name filled in."""
    t = TIPS[lang][i % len(TIPS[lang])]
    return _fill(t[0], model), [_fill(line, model) for line in t[1:]]


def next_steps(model, lang='en'):
    """Lines shown once after `decider-mlx fetch` has finished."""
    port = 8001 if model == 'pplx-decider-v1.3' else 8000     # the ports of the README and the usage guide
    return [_fill(line, model).replace('{port}', str(port)) for line in NEXT[lang]]
