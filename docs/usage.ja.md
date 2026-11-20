# 使い方ガイド

[English](usage.md) | 日本語

`decider-mlx` の HTTP サーバー、ローカル拡張機能、JSONL 一括処理、Python API、および重み管理の詳細な仕様をまとめています。公式 Decisions API との差分一覧は [api.md](api.md)（英語）をご覧ください。

## HTTP サーバー

```sh
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
```

- **起動とバックグラウンド準備**：サーバーは起動直後に `127.0.0.1` で待ち受けを開始し、重みの読み込みが完了するまで（約 0.5 秒）は `503`（`Retry-After: 1`）を返します。その後、よく使われる入力形状の計算グラフをリクエストの合間にバックグラウンドで準備し、進捗を 1 行で表示します（例：`pplx-decider-flash-v1.3: preparing 41/101 (accepting requests)`）。準備中もリクエストは通常どおり処理され、出力結果も準備の完了前後で変わりません。
- **状態確認とオプション**：`GET /metrics` で準備状況とメモリ使用量を取得できます。指定できるオプションは `--model` と `--port` のみです。
- **公式 API 互換**：リクエスト本文は公式の [Decisions API](https://docs.perplexity.ai/api-reference/decisions-post) と同一です。`model` にはそのサーバーが起動しているモデルの正式名を指定してください（`-latest` や `perplexity/` 付き、別モデル名は `400` になります）。
- **エラー応答**：公式と同じ `{"error": {"message", "type", "code", "param"}}` 形式で返します。不正なリクエストは `400`、対象外のパスやメソッドは `404` / `405`、32 MiB を超える本文は `413`、内部エラーは `500` です。

### 画像を含むリクエスト

画像は OpenAI 形式の要素として `state` 内の任意の場所に配置できます（PNG、JPEG、WebP の base64 data URL、1 リクエスト最大 8 枚）。公式ライブラリと同様、画像は自動的にテキストの前に配置され、画像を除いた残りの `state` は送信された形式のまま読み込まれます（文字列はテキスト、オブジェクトや配列は JSON として解釈されます）。

```json
{"model": "pplx-decider-flash-v1.3",
 "state": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}},
           {"type": "text", "text": "Customer photo of the damaged parcel."}],
 "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel visibly damaged?"}}}
```

## ローカル拡張機能

公式の Decisions API にはない、ローカル実行向けの機能です。

- **配列での一括送信**：リクエスト本文を JSON 配列にして POST すると、同じ順序の配列で結果が返ります。不正な項目がある場合はその位置に `{"error": ...}` が入り、残りの項目はそのまま処理されます。64 件以上の配列は内部で 2,048 トークン単位の固定バッチに詰めて処理されるため、大量データの処理に最も高速です。
- **`?mode=isolated`**：1 リクエストにつき `state` を 1 回だけ読み込み、すべての質問にそこから回答します（学習時のレイアウト）。長い文章に対して多数の質問を行う場合に高速ですが、質問の組み合わせによって結果がわずかに動くことがあります。既定の `separate` は質問ごとに独立した系列で回答します。
- **`?temperature=`**：省略時は公式 API と同じくモデルに保存された温度（Flash：`1.3303`、27B：`1.2264`）を使います。`?temperature=1` を指定すると較正前の確率が得られます。
- **`?raw=1`**：通常の回答の代わりに、質問ごとの生の readout logit を返します。
- **Locate**：`{"type": "locate", "instructions": "error codes"}` のように指定すると、`state` の文字列から該当箇所を文字位置（開始と終了のオフセット）とスコア付きでそのまま抽出します。Locate 用の重みは初回の Locate リクエスト時に遅延読み込みされます（Flash：約 3.8 GB、27B：約 0.95 GB）。

```sh
curl -s 'localhost:8000/v1/decisions?mode=isolated&temperature=1' -H 'Content-Type: application/json' -d '[
  {"model": "pplx-decider-flash-v1.3", "state": "The museum opened in 1931.",
   "questions": {"a": {"type": "noul", "instructions": "Did it open before 1950?"}}},
  {"model": "pplx-decider-flash-v1.3", "state": "Error E401 after the key rotation.",
   "questions": {"code": {"type": "locate", "instructions": "error codes"}}}]'
```

## JSONL ファイルの一括処理

```sh
uv run decider-mlx run --model pplx-decider-flash-v1.3 --input requests.jsonl --output responses.jsonl
```

- **入力形式**：各行に 1 リクエスト（`{"state", "questions"}`、`model` は不要）を書くか、行ごとのオプションを付けた `{"id": ..., "request": {...}, "options": {"execution_mode": "isolated", "temperature": 1.0, "raw": false}}` の形式で記述します。
- **出力形式**：入力と同じ順序で `{"id", "index", "response"}` または `{"id", "index", "error": {"type", "message"}}` を 1 行ずつ出力します。
- **挙動**：64 行以上のファイルは配列バッチと同様に自動パッキングで高速処理されます。失敗した行が 1 件でもあった場合、終了コードは `1` になります。

## Python API

```python
from decider_mlx import load

app = load("pplx-decider-flash-v1.3")      # 重みの読み込みが完了すると戻ります
response = app.decide({
    "state": "The museum opened in 1931.",
    "questions": {"a": {"type": "noul", "instructions": "Did it open before 1950?"}},
})
responses = app.decide([request1, request2], execution_mode="isolated", temperature=1.0)
logits = app.decide(request1, return_logits=True)
results = app.decide_each([request1, bad_request])   # 項目ごとに応答または例外を返します
app.wait_ready()                                     # 計測用：バックグラウンド準備の完了を待ちます
```

- `decide` はスレッドセーフです。複数のスレッドから呼び出しても、モデル専用のワーカースレッドが到着順に処理します。
- 不正なリクエストには `decider_mlx.RequestError` を、質問と `state` の合計が 32,768 トークンを超える場合は `decider_mlx.LengthError` を送出します（自動切り詰めは行いません）。
- 画像はリクエスト辞書の `"images"` に、ファイルパス、data URL、PIL 画像のいずれかで渡せます。
- 内部で必要な設定を有効にするため、**`decider_mlx` は `mlx` より先に import してください**。また、ライブラリは端末には何も出力せず、標準の `logging` にのみ記録します。

## 2 つのモデルを同時に動かす

モデルごとに別々のポートでサーバーを起動します。

```sh
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
uv run decider-mlx serve --model pplx-decider-v1.3 --port 8001
```

2 つのプロセスは完全に独立しており（ロックや共有ファイルなし）、GPU を共有しながら同時に推論を実行できます（メモリは両モデルの合計容量が必要です）。

## CLI と重み管理の補足

```sh
uv run decider-mlx fetch --model pplx-decider-flash-v1.3
uv run decider-mlx fetch --model pplx-decider-v1.3
```

- **コマンドの省略**：`source .venv/bin/activate` を実行しておけば、`uv run` を省いて `decider-mlx` と直接入力できます。
- **自動取得とキャッシュ**：`fetch` は、パッケージ内に固定された revision の変換済みの重み、トークナイザ、プロセッサ、設定ファイルを各 Hugging Face リポジトリから取得し、標準キャッシュ（`~/.cache/huggingface`、または `HF_HUB_CACHE` / `HF_HOME`）に保存します。なお、`serve`、`run`、`load()` は重みが未取得の場合に自動でダウンロードするため、`fetch` の事前実行は省略することもできます。
- **進捗表示と整合性検証**：ダウンロード中は進捗と残り時間に加えて、使い方のヒントを順に表示します（システムの言語が日本語なら日本語、それ以外は英語）。取得後には全ファイルをリポジトリの `manifest.json`（sha256）と照合します。再度 `fetch` を実行すれば、いつでも整合性を再確認できます。
- **オフライン利用**：`HF_HUB_OFFLINE=1` を設定すると、ネットワーク通信を行わずローカルキャッシュのみを使用して起動します。
