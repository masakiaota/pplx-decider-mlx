# decider-mlx

[English](README.md) | 日本語

Perplexity の分類モデル Decider v1.3 を Apple silicon の Mac で速く動かす推論エンジンである。
[MLX](https://github.com/ml-explore/mlx) の上に作った。公式 Decisions API と同じ本文を HTTP で受け付け、JSONL
ファイルを処理し、Python からも使える。

| モデル（`--model`） | 元のモデル | このエンジン用の重み | 推奨メモリ |
|---|---|---|---|
| `pplx-decider-flash-v1.3` | [perplexity-ai/pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3) | [masakiaota/pplx-decider-flash-v1.3-mlx](https://huggingface.co/masakiaota/pplx-decider-flash-v1.3-mlx)（8-bit、9.6 GB） | 24 GB 以上 |
| `pplx-decider-v1.3`（27B） | [perplexity-ai/pplx-decider-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-v1.3) | [masakiaota/pplx-decider-v1.3-mlx](https://huggingface.co/masakiaota/pplx-decider-v1.3-mlx)（4-bit、16.7 GB） | 36 GB 以上 |

**非公式のプロジェクトである。Perplexity AI とは関係がなく、Perplexity AI の承認や支援も受けていない。**
このエンジンは変換した重み（Flash は 8-bit、27B は行ごとの 4-bit）を独自の INT8・W4A8 Metal カーネルで動かす。
公式の BF16 重みは動かせない。確率は元のモデルとわずかに異なる（[品質](#品質)を参照）。

Mac Studio（M5 Ultra）での値：

| | Flash | 27B |
|---|---:|---:|
| プロセス起動から受付開始まで | 0.4〜0.5 s | 0.3〜0.5 s |
| プロセス起動から最初の回答まで | 0.5 s | 0.6〜1.0 s |
| 1要求（427 トークン） | 43 ms | 220 ms |
| 20要求を1つの配列で | 0.31 s | 1.92 s |
| 長い入力（32.4K トークン） | 7.0 s | 22.9 s |
| 混在した 1,072 要求を配列で | 47 s | 236 s |
| MLX のピークメモリ（HTTP ベンチマーク後） | 14.2 GB | 23.7 GB |

Flash は、同じ HTTP の負荷で公式 Mac アプリ（Flash v1.2、q8_0 GGUF）の約2倍の速さである。

## 必要なもの

- **M5 以降**のチップを積んだ Mac。カーネルと長さの表は M5 で調整し、M5 で試験した。それより前のチップは対象外である。
- メモリ：Flash は 24 GB 以上、27B は 36 GB 以上。2つを同時に動かすなら合計が要る。
- **Xcode と Metal Toolchain** の入った macOS。インストールの途中で MLX 0.32.5 をソースからビルドする（下記）。
  `xcodebuild -showComponent MetalToolchain` が `installed` を返すこと。返さなければ
  `xcodebuild -downloadComponent MetalToolchain` を実行する。
- [uv](https://docs.astral.sh/uv/)（必要なら Python 3.14 も uv が入れる）。
- ディスク：Hugging Face のキャッシュに 9.6 GB（Flash）または 16.7 GB（27B）。

## インストール

```sh
git clone https://github.com/masakiaota/pplx-decider-mlx
cd pplx-decider-mlx
uv sync
```

`uv sync` は MLX 0.32.5 を ml-explore/mlx の commit `fc344c9`（`pyproject.toml` で固定）からビルドする。この版には
[ml-explore/mlx#4682](https://github.com/ml-explore/mlx/pull/4682)（M5 での量子化行列積の高速化）が入っているが、
まだ PyPI にない。最初のビルドは約2分かかり、ネットワークが要る。2回目以降は uv のキャッシュを使う。
`mlx==0.32.5` が PyPI に出たら、そちらに切り替える。

以下では `uv run decider-mlx` と書く。`source .venv/bin/activate` の後なら `decider-mlx` だけでよい。

## 重みのダウンロード

```sh
uv run decider-mlx fetch --model pplx-decider-flash-v1.3
uv run decider-mlx fetch --model pplx-decider-v1.3
```

`fetch` は、変換済みの重み・tokenizer・processor・設定ファイルを、モデルごとの Hugging Face リポジトリから、この
パッケージに固定した revision で取得する。置き場は Hugging Face の標準キャッシュ（`~/.cache/huggingface`、または
`HF_HUB_CACHE`・`HF_HOME`）である。進み具合と残り時間を表示し、待っている間は使い方のヒントを順に見せる。取得後に
リポジトリの `manifest.json` と全ファイルの sha256 を照合する。もう一度実行すれば、いつでも照合し直せる。

`serve`・`run`・`load()` は、重みが無ければ自動で取得する。したがってこの手順は省いてもよい。`HF_HUB_OFFLINE=1`
のときは何もダウンロードせず、キャッシュをそのまま使う。

## サーバー

```sh
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
```

サーバーはすぐに `127.0.0.1` で待ち受け、重みを読み込む（約0.5秒）までは `503`（`Retry-After: 1`）を返す。その後、
よく来る形の要求のグラフを、要求の合間に裏で準備する。状態は1行で表示する（例：
`pplx-decider-flash-v1.3: preparing 41/101 (accepting requests)`）。準備の間も要求に答え、答えは準備の有無に
左右されない。`GET /metrics` は状態とメモリ使用量を返す。

選べるのはモデルとポートだけである。メモリの上限や事前チェックは無く、MLX の既定に従う。

## HTTP

本文は公式 [Decisions API](https://docs.perplexity.ai/api-reference/decisions-post) と同じである。`model` には
そのサーバーが動かしているモデルの名前を書く。

```sh
curl -s localhost:8000/v1/decisions -H 'Content-Type: application/json' -d '{
  "model": "pplx-decider-flash-v1.3",
  "state": "My integration keeps failing with a 401 since we rotated our API key. Please help.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"billing": "Charges and refunds", "support": "Technical integration errors",
                          "sales": "Questions about buying a product"}},
    "urgent": {"type": "noul", "instructions": "Is the customer blocked right now?"},
    "frustration": {"type": "score", "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]}
  }}'
```

```json
{"model": "pplx-decider-flash-v1.3",
 "answers": {"team": {"type": "choice", "probabilities": {"billing": 0.0004, "support": 0.9992, "sales": 0.0004},
                      "choice": "support", "confidence": 0.9989},
             "urgent": {"type": "noul", "noul": 0.3847},
             "frustration": {"type": "score", "probabilities": {"0": 0.1221, "1": 0.3125, "2": 0.5492, "3": 0.0162},
                             "legend": {"0": "calm", "1": "mildly annoyed", "2": "frustrated", "3": "angry"},
                             "score": 1.4594, "confidence": 0.4271}},
 "usage": {"input_tokens": 288, "output_tokens": 0, "sequences": 3}}
```

（数値はここでは丸めてある。）エラーは公式と同じ形 `{"error": {"message", "type", "code", "param"}}` で返す。不正な要求は
400、ほかのパスやメソッドは 404・405、32 MiB を超える本文は 413、内部の失敗は 500 である。動かしているモデル以外の
名前（`-latest` や `perplexity/` 付きを含む）は 400 になる。

画像は OpenAI 形式の部品として `state` のどこに置いてもよい。公式ライブラリと同じく、画像は文章の前に置かれる：

```json
{"model": "pplx-decider-flash-v1.3",
 "state": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}},
           {"type": "text", "text": "Customer photo of the damaged parcel."}],
 "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel visibly damaged?"}}}
```

PNG・JPEG・WebP の data URL を、1要求に8枚まで送れる。画像を抜いた残りの `state` は送られたとおりに読む。文字列は
そのまま文章として、オブジェクトや配列は JSON として読む。

## ローカル拡張

公式 API に無い機能である。違いの一覧は [docs/api.md](docs/api.md) にある。

- **配列。** 本文の JSON 配列を POST すると、同じ順の配列が返る。不正な項目の位置には `{"error": ...}` が入り、
  ほかの項目には答える。64件以上の配列は内部で 2,048 トークンの固定の束に詰める。大量の処理ではこれが最も速い。
- **`?mode=isolated`。** 1要求につき state を1回だけ読み、すべての質問にそこから答える（学習時の並び）。長い state
  に多くの質問があるときは安い。ただし質問の組み合わせで答えがわずかに動く。既定の `separate` は質問ごとに別の
  系列で答える。
- **`?temperature=`。** 省略すると、公式 API と同じくモデルに保存された温度（Flash 1.3303、27B 1.2264）を使う。
  `?temperature=1` で較正前の確率になる。
- **`?raw=1`。** 答えの代わりに、質問ごとの readout の生の logit を返す。
- **Locate。** `{"type": "locate", "instructions": "error codes"}` は、state の文字列をそのまま抜き出した区間を、
  文字位置とスコア付きで返す。Locate 用の重みは最初の Locate 要求で読み込む（Flash は 3.8 GB 増える）。

```sh
curl -s 'localhost:8000/v1/decisions?mode=isolated&temperature=1' -H 'Content-Type: application/json' -d '[
  {"model": "pplx-decider-flash-v1.3", "state": "The museum opened in 1931.",
   "questions": {"a": {"type": "noul", "instructions": "Did it open before 1950?"}}},
  {"model": "pplx-decider-flash-v1.3", "state": "Error E401 after the key rotation.",
   "questions": {"code": {"type": "locate", "instructions": "error codes"}}}]'
```

## ファイルで渡す

```sh
uv run decider-mlx run --model pplx-decider-flash-v1.3 --input requests.jsonl --output responses.jsonl
```

入力の各行は1要求（`{"state", "questions"}`。`model` は要らない）か、その行だけの設定を付けた
`{"id": ..., "request": {...}, "options": {"execution_mode": "isolated", "temperature": 1.0, "raw": false}}`
である。出力の各行は `{"id", "index", "response"}` または `{"id", "index", "error": {"type", "message"}}` で、
入力と同じ順に並ぶ。64行以上のファイルは大きな配列と同じく詰めて処理する。失敗した行があれば終了コードは 1 になる。

## Python

```python
from decider_mlx import load

app = load("pplx-decider-flash-v1.3")      # 重みを読み込んだら返る
response = app.decide({"state": "The museum opened in 1931.",
                       "questions": {"a": {"type": "noul", "instructions": "Did it open before 1950?"}}})
responses = app.decide([request1, request2], execution_mode="isolated", temperature=1.0)
logits = app.decide(request1, return_logits=True)
results = app.decide_each([request1, bad_request])   # 要求ごとに応答か例外
app.wait_ready()                                   # 測定用：裏の準備が終わるまで待つ
```

`decide` はスレッド安全である。どのスレッドから呼んでも、モデルの処理担当スレッドが来た順に1つずつ処理する。
不正な要求には `decider_mlx.RequestError` を、質問と state が 32,768 トークンに収まらないときは
`decider_mlx.LengthError` を投げる（切り詰めはしない）。画像は `"images"` に、ファイルパス・data URL・PIL 画像の
いずれかで渡す。`decider_mlx` は `mlx` より先に import する。ライブラリは画面に何も出さず、`logging` にだけ書く。

## 2つのモデルを同時に使う

モデルごとにサーバーを1つ動かす：

```sh
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
uv run decider-mlx serve --model pplx-decider-v1.3 --port 8001
```

2つのプロセスは独立している（ロックも共有ファイルも無い）。同時に推論でき、GPU は共有する。プロセス間でロックを
かける方式は、自由に動かすより 11〜35% 遅かった。

## 品質

元のモデルの BF16 参照出力と、評価用に取り分けたデータで比べた：

| BF16 と同じ答え | Flash | 27B |
|---|---:|---:|
| 二値（`noul`） | 821/826 | 717/726 |
| `choice` | 222/224 | 161/164 |
| `score`（argmax） | 73/74 | 70/74 |
| Locate の区間の集合 | 305/314 | 100/114 |
| BoolQ ゲート、4つの実行条件 | 各 100/100 | 各 100/100 |

モデルごとの詳細は [docs/models.md](docs/models.md) にある。ビルドを元のモデルと比べる手順は
[bench/quality/](bench/quality/README.md) にある（英語）。

## 開発

```sh
uv sync                                         # pytest も入る
uv run pytest                                   # モデル不要の試験（API、HTTP の検査、模擬 Hub に対する fetch）
DECIDER_MLX_TEST_MODELS=pplx-decider-flash-v1.3,pplx-decider-v1.3 HF_HUB_OFFLINE=1 uv run pytest
```

`DECIDER_MLX_TEST_MODELS` を付けると、モデルを Hugging Face のキャッシュから読み、BoolQ を4つの実行条件で、
HTTP の全機能のケースとあわせて `tests/golden` とバイト単位で比べる（[tests/golden/README.md](tests/golden/README.md)）。
golden は M5 Ultra で記録した。ほかのチップや MLX のビルドでは最後の桁が変わりうる。

```sh
uv run python bench/speed.py --model pplx-decider-flash-v1.3
```

は、プロセス内での時間（単発の要求、20要求の配列、長い入力）を表示する。`tests/hub_mock.py` はフォルダをモデルの
リポジトリとして配る、Hugging Face Hub の代わりのローカルサーバーである。`fetch` をネットワーク無しで試すのに使う。

## 作り方の記録

[experiments/](experiments/README.md) は研究の記録である（英語）。モデルごとに何を試し、何を残し、何をやめたか、
その理由と測定値を書いてある（重み、INT8・W4A8 カーネル、長さのバケット、まとめ処理、起動）。
[tools/convert/](tools/convert/README.md) は、公開している重みを元の BF16 チェックポイントから作るスクリプトである（英語）。

## ライセンス

Apache-2.0（[LICENSE](LICENSE)）。Metal カーネルの一部は MLX と mxfp4-int8-m5（どちらも MIT）を元にしている
（[NOTICE](NOTICE)）。`tests/golden` の試験データには BoolQ（CC BY-SA 3.0）と SQuAD（CC BY-SA 4.0）の抜粋を含む
（[tests/golden/README.md](tests/golden/README.md)）。モデルの重みは Perplexity のモデルから派生した Apache-2.0
のものである。各 Hugging Face リポジトリの NOTICE を参照のこと。
