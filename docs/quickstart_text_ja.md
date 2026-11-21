# 文章を渡すだけで探す — 最初の 10 分

自分の文章 (規程、手順書、議事録、メモ) を入れて、ふつうの言葉で質問すると、合う文章が返ってきます。
ベクトルやモデルのことを知らなくても、下の順に打てば動きます。文章は自分の機械の外に出ません。

この手順は、まっさらな環境で上から順に打って確かめました (2026-11-22、kci-vec 0.2.12、Ubuntu 24.04 の WSL2、Python 3.12)。
`kci_vec.text` は「実験的」な部品です。名前や引数は、今後の版で変わることがあります。

## 用意するもの

| | |
|---|---|
| 機械 | Linux (x86-64、Ubuntu 22.04 以降)。Windows は WSL2 の中で。macOS はまだ使えません |
| Python | 3.10 以降 (確かめたのは 3.12) |
| ディスク | 2 GB ほど (道具 1.5 GB、言語モデル 0.15〜0.7 GB) |
| ネット | 最初の 1 回だけ (道具と言語モデルを取ってくる)。そのあとは切っていても動きます (確かめるなら `export HF_HUB_OFFLINE=1` を付けて流す。ネットを見に行かなくなります) |

## 1. 入れる (1〜数分)

```sh
python3 -m venv venv
source venv/bin/activate
pip install https://github.com/KimYuki3/kci-vec-dist/releases/download/v0.2.12/kci_vec-0.2.12-py3-none-manylinux_2_35_x86_64.whl
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

- 2 行めの `source venv/bin/activate` は、端末を開き直すたびに打ちます。
- 版の番号 (`0.2.12`) は、[配布のページ](https://github.com/KimYuki3/kci-vec-dist/releases) の新しいものに読み替えてください。
- `torch` は大きいです (CPU 用で 200 MB ほど)。上の書き方だと GPU 用の部品を取ってこないので、小さく済みます。

## 2. 最初の検索

`first.py` という名前で保存します。

```python
from kci_vec.text import TextIndex

ix = TextIndex("ja", sparse="bm25")
ix.add([
    "有給休暇は入社半年後に10日付与されます。申請は3営業日前までに上長へ。",
    "経費の精算は月末締めです。領収書の原本を総務に提出してください。",
    "パスワードを忘れたときは、情報システム課の内線1234に連絡してください。",
    "会議室の予約は社内カレンダーから行います。利用は1回2時間までです。",
])
for hit in ix.search("有給は何日もらえますか", k=2):
    print(hit.id, round(hit.score, 3), hit.text)
```

```sh
python first.py
```

こう出ます (最初の 1 回は、言語モデル 150 MB を取ってくる分だけ待ちます。2 回めからは 5 秒ほどです)。

```
0 1.0 有給休暇は入社半年後に10日付与されます。申請は3営業日前までに上長へ。
3 0.226 会議室の予約は社内カレンダーから行います。利用は1回2時間までです。
```

- 左の数字 (`hit.id`) は、入れた順の番号です (0 から)。
- 真ん中 (`hit.score`) は並べるための点で、大きいほど合っています。1 件めとの差を見てください。0.226 は「ほかよりはまし」程度です。
- `Loading weights ...` のような行が出ても、故障ではありません。

## 3. 自分のファイルを入れて、保存する

`docs` というフォルダに、文章のファイル (`.txt`) を入れておきます。`build.py`:

```python
import glob, os
from kci_vec.text import TextIndex

paths = sorted(glob.glob("docs/*.txt"))
texts = [open(p, encoding="utf-8").read() for p in paths]

ix = TextIndex("ja", sparse="bm25")
ix.add(texts)                                   # 番号は、この並びの順に 0, 1, 2, ...
ix.save("my_index", key="合言葉をここに")
print(len(ix), "件を保存しました")
```

- 長い文章は、中で自動で分けて入れます。答えは文章の単位で返り、`hit.text` が一番合った部分です。
- `key=` を付けると、索引も文章の中身も暗号化して保存します。保存したフォルダを見ても、文章は読めません。
- 合言葉を忘れると開けません。合言葉そのものは保存されません。

## 4. 開き直して、質問する

`ask.py`:

```python
import glob, sys
from kci_vec.text import TextIndex

paths = sorted(glob.glob("docs/*.txt"))
ix = TextIndex.load("my_index", key="合言葉をここに")
for hit in ix.search(sys.argv[1], k=3):
    print(paths[hit.id], round(hit.score, 3))
    print("   ", hit.text.replace("\n", " ")[:80])
```

```sh
python ask.py "宿泊費の上限はいくらですか"
```

```
docs/travel.txt 1.0
    出張旅費規程  新幹線は普通車指定席を利用する。宿泊費は1泊12,000円を上限とする。 日当は1日2,500円を…
docs/rule2.txt 0.014
    第2章 就業規則  始業は9時、終業は18時とする。…
```

- 合言葉が違うと、開かずに断ります (`RuntimeError`)。
- `docs` のファイルを足したり消したりしたら、`build.py` を流し直してください (ファイルの並びと番号がずれるため)。

## よく使うこと

```python
ix.add("在宅勤務は週2日まで認められます。")        # あとから 1 件足す (番号は続きから)
ix.add(texts, ids=[101, 102, 103])               # 自分の番号を付けて入れる
ix.delete(3)                                     # 番号 3 を消す
ix.search("有給", k=5, filter=lambda i: i != 0)   # 番号で絞る (関数か、許す番号の集まり)
hit.span                                         # hit.text が、元の文章の何文字めから何文字めか
ix.texts[hit.id]                                 # 元の文章の全文
```

## つまずきやすい所

| 起きること | わけと、すること |
|---|---|
| `pip install` が「合う配布物が無い」と言う | Linux (x86-64) 用だけです。Windows は WSL2 の中で、同じ手順を |
| 最初の 1 回がとても遅い | 言語モデルを取ってきています。2 回めからは速くなります |
| 文章を入れるのが遅い | 文章をモデルに読ませるのに時間がかかります。この機械では 1 秒に 6 件ほどでした (CPU)。1 万件で 30 分ほど。入れるのは最初の 1 回だけで、保存すれば次は開くだけです |
| 入れた件数が多くて重い | 数万〜10 万件くらいまでを想定しています。本文は全部メモリに持ちます |
| 英語の文章を入れたい | `TextIndex("en", sparse="bm25")` |
| 言い換えた質問で当たりが悪い | 下の「2 つのモデルで探す」を試してください |

## 2 つのモデルで探す (既定の設定)

上の例の `sparse="bm25"` は、「意味を見るモデル」1 つと「文字の一致」で探す、速い設定です。
外すと、言葉の側にもモデルを使う既定の設定になります。

```python
ix = TextIndex("ja")          # 言葉の側もモデル。追加で pip install fugashi unidic-lite が要ります
```

| | `sparse="bm25"` | 既定 |
|---|---|---|
| 取ってくる言語モデル | 150 MB | 150 MB + 520 MB |
| 文章を入れる速さ | 速い (測った例で約 10 倍) | 遅い |
| 向いている質問 | 文章と同じ言葉で聞く。名前・型番・略語 | 言い換えて聞く短い質問 |

開発者が日本語の論文 637 件で測った例では、短い質問は既定のほうが少し良く (0.970 対 0.959)、長い質問は
`sparse="bm25"` のほうが良い (0.983 対 0.991) 結果でした。データで変わるので、迷ったら速いほうから始めてください。

## 知っておくこと

- **実験的です。** `kci_vec.text` の名前・引数・保存の形は変わることがあります。
- **言語モデルは別の人たちのものです。** 既定で使うのは `cl-nagoya/ruri-v3-30m` (Apache-2.0)、`hotchpotch/japanese-splade-v2` (MIT)、
  英語は `intfloat/multilingual-e5-small` (MIT)、`opensearch-project/opensearch-neural-sparse-encoding-v2-distill` (Apache-2.0) です
  (各モデルのページの表示。使う前に自分でも確かめてください)。
- **KCI-Vec 自体のライセンス**は、リポジトリの `LICENSE` を見てください (非商用のソース公開。商用は別の契約です)。
- 質問に文章で答える機能 (生成) は入っていません。返るのは、合う文章と、その中の一番合った部分です。
