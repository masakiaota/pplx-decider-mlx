# decider-mlx

[English](README.md) | 日本語

**decider-mlx** は、Perplexity の分類モデル **Decider v1.3（Flash / 27B）** を Apple Silicon の Mac（M5 以降推奨。Flash は M1〜M4 でも標準 MLX フォールバックで動作）でローカル実行するための、[MLX](https://github.com/ml-explore/mlx) ベースの推論エンジンです。公式 [Decisions API](https://docs.perplexity.ai/api-reference/decisions-post) 互換の HTTP サーバーに加え、JSONL 一括処理と Python API を提供しており、既存コードの接続先を手元の Mac に向けるだけで利用できます。

- **公式 Mac アプリの約 2 倍の速度**：Flash モデルでは、公式 Mac アプリの約 2 倍の速度で応答します。さらに、本エンジン独自の配列バッチ送信を使うと、約 3.9 倍まで高速化されます。
- **27B モデルとローカル拡張に対応**：公式 Mac アプリにはない **27B モデル**も、元の BF16 の半分以下のメモリで動かせます。公式 API にない該当箇所抽出（Locate）や `isolated` モードも利用できます。
- **元のモデルに近い出力精度**：独自の INT8 と W4A8 の Metal カーネルで高速化しつつ、ホールドアウト評価で元の BF16 モデルとの高い一致率を確認しています。

> [!NOTE]
> 本プロジェクトは非公式であり、Perplexity AI とは関係がなく、同社の承認や支援も受けていません。変換済みの重み（Flash は 8-bit、27B は行単位 4-bit）を使用する設計のため公式の BF16 重みはそのまま使用できず、出力確率は元のモデルとわずかに異なります（詳細は[品質](#品質)をご覧ください）。

## 公式 Mac アプリとの比較

Flash モデルについて、同じ HTTP 負荷での処理時間を公式 Mac アプリと比較した結果です。

| 処理内容（Flash、HTTP） | 公式 Mac アプリ | decider-mlx | 速度比 |
|---|---:|---:|---:|
| 1 件（427 トークン） | 88.7 ms | 44.8 ms | 2.0 倍 |
| 20 件を 1 件ずつ送信 | 1.21 s | 0.58 s | 2.1 倍 |
| 20 件を 4 並列で送信 | 1.19 s | 0.55 s | 2.2 倍 |
| 長文入力（5.4K トークン） | 0.96 s | 0.42 s | 2.3 倍 |
| 20 件を 1 つの配列で一括送信 | 非対応 | 0.31 s ※ | 3.9 倍（1 件ずつ比） |

- 測定環境：Mac Studio（M5 Ultra）。新しく起動したサーバーに対して同一のリクエストを送信して計測しています。
- 公式 Mac アプリが配信しているモデルは Flash v1.2（`q8_0` GGUF）、`decider-mlx` が動かすモデルは v1.3 です。※印の値は公開パッケージで別途計測した数値です。
- 起動も軽量化しており、プロセス起動から約 0.5 秒で最初のリクエストに回答できます。公式アプリは 2.2 秒かかります。

## クイックスタート

### 動作環境

- **チップ**：Apple Silicon **M5 以降**推奨（**27B** は M5 以降が必須。**Flash** は M1〜M4 でも標準 MLX フォールバックで動作）
- **OS**：macOS 27 以降
- **推奨メモリ**：24 GB 以上（Flash）／ 36 GB 以上（27B）
- **ディスク空き容量**：9.6 GB 以上（Flash）／ 16.7 GB 以上（27B）
- **ツール**：[uv](https://docs.astral.sh/uv/)

### インストールとサーバー起動

```sh
git clone https://github.com/masakiaota/pplx-decider-mlx
cd pplx-decider-mlx
uv sync
uv run decider-mlx fetch --model pplx-decider-flash-v1.3   # 重み（9.6 GB）をダウンロード
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
```

> `uv sync` は PyPI 未公開の MLX 0.32.5 を本リポジトリの [GitHub Release](https://github.com/masakiaota/pplx-decider-mlx/releases/tag/mlx-0.32.5-fc344c9) から取得するため、初回は少し時間がかかることがあります（詳細は [CONTRIBUTING.md](CONTRIBUTING.md)）。

### 最初のリクエストを送る

別の端末から HTTP リクエストを送信します。

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

レスポンス例（数値は丸めています）：

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

Decider は、テキストや画像（`state`）に対する質問から、問い合わせの振り分け、緊急度の判定、文書の分類などの結果を確率やスコアで得るモデルです。基本となる質問の型は次の 3 種類です（このほかローカル拡張の `locate` が利用できます）。

- **`choice`**：選択肢ごとの確率と、最も確率の高い選択肢を返します。
- **`noul`**：「はい（true）」の確率を返します。
- **`score`**：段階ごとの確率と、段階の期待値を返します。

## モデルと性能

用途や搭載メモリに応じて 2 つのモデルを利用できます。軽快な応答や高いスループットを優先する場合は **Flash**、27B クラスの表現力が必要なタスクには **27B** をお選びください。

| 項目 | Flash | 27B |
|---|---|---|
| `--model` 指定名 | `pplx-decider-flash-v1.3` | `pplx-decider-v1.3` |
| 元のモデル | [perplexity-ai/pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3) | [perplexity-ai/pplx-decider-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-v1.3) |
| 本エンジン用の重み | [8-bit / 9.6 GB](https://huggingface.co/masakiaota/pplx-decider-flash-v1.3-mlx) | [4-bit / 16.7 GB](https://huggingface.co/masakiaota/pplx-decider-v1.3-mlx) |
| 推奨メモリ | 24 GB 以上 | 36 GB 以上 |
| プロセス起動〜受付開始 | 0.4〜0.5 s | 0.3〜0.5 s |
| プロセス起動〜初回回答 | 0.5 s | 0.6〜1.0 s |
| 1 件（427 トークン） | 43 ms | 220 ms |
| 20 件を 1 つの配列で送信 | 0.31 s | 1.92 s |
| 長文入力（32.4K トークン） | 7.0 s | 22.9 s |
| 混在した 1,072 件を配列で一括処理 | 47 s | 236 s |
| MLX ピークメモリ（HTTP ベンチマーク後） | 14.2 GB | 23.7 GB |

Mac Studio（M5 Ultra）での HTTP 計測値です。27B は元の BF16 モデルでは最大 57.1 GB のメモリを消費しますが、行単位 4-bit 量子化により HTTP ベンチマーク後で 23.7 GB（32K トークンの Locate 実行時でも 25.0 GB）まで抑えています。モデルごとの詳しい仕様や測定値は [docs/models.md](docs/models.md) をご覧ください。

### 品質

元のモデル（BF16）の参照出力とどの程度同じ判断になるかを、学習に使用していない評価データで検証しています。

| BF16 との一致率 | Flash | 27B |
|---|---:|---:|
| 二値判定（`noul`） | 821 / 826 | 717 / 726 |
| 選択（`choice`） | 222 / 224 | 161 / 164 |
| スコア（`score`、最尤段階） | 73 / 74 | 70 / 74 |
| Locate（抽出した区間の集合） | 305 / 314 | 100 / 114 |
| BoolQ ゲート（4 つの実行条件） | 各 100 / 100 | 各 100 / 100 |

ご自身の環境で元のモデルと比較する手順は [bench/quality/](bench/quality/README.md)（英語）にまとめています。

## その他の使い方

HTTP サーバーのほか、JSONL ファイルの一括処理や Python からの直接呼び出しに対応しています。

### JSONL ファイルの一括処理

```sh
uv run decider-mlx run --model pplx-decider-flash-v1.3 --input requests.jsonl --output responses.jsonl
```

### Python API

```python
from decider_mlx import load

app = load("pplx-decider-flash-v1.3")
response = app.decide({
    "state": "The museum opened in 1931.",
    "questions": {"a": {"type": "noul", "instructions": "Did it open before 1950?"}},
})
```

より詳しい使い方やオプションについては [使い方ガイド](docs/usage.ja.md) をご覧ください。

## ドキュメント

- [使い方ガイド](docs/usage.ja.md)：HTTP サーバーの詳細、画像、配列バッチや Locate などのローカル拡張、Python API、2 モデル同時起動、オフライン利用
- [API 仕様の差分](docs/api.md)：公式 Decisions API との違い一覧（英語）
- [モデルの詳細と測定値](docs/models.md)：Flash / 27B の設計、メモリ、ベンチマーク詳細（英語）
- [品質検証](bench/quality/README.md)：元の BF16 モデルとの比較手順（英語）
- [開発とテスト](CONTRIBUTING.md)：テスト実行、速度ベンチマーク、MLX 0.32.5 wheel の補足（英語）
- [設計と研究の記録](experiments/README.md) / [重み変換スクリプト](tools/convert/README.md)：最適化の検証記録と BF16 からの変換手順（英語）

## ライセンス

- 本リポジトリのコードは Apache-2.0 ライセンスです（[LICENSE](LICENSE)）。
- Metal カーネルの一部は MLX および mxfp4-int8-m5（いずれも MIT ライセンス）をベースにしています（[NOTICE](NOTICE)）。
- `tests/golden` のテストデータには BoolQ（CC BY-SA 3.0）および SQuAD（CC BY-SA 4.0）の抜粋を含みます（[tests/golden/README.md](tests/golden/README.md)）。
- 変換済みのモデル重みは Perplexity のモデルから派生した Apache-2.0 ライセンスです。詳細は各 Hugging Face リポジトリの `NOTICE` をご確認ください。
