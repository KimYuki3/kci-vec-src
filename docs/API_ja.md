# KCI-Vec の使い方 (API の説明書)

版 0.2.15 (2026-11-23)。English: [API.md](API.md)

KCI-Vec を Python と C から使う方法をまとめます。それぞれの関数が何をするか、どの設定から始めるか、守る決まりを書きます。
初めての方は、先に [最初の 10 分](quickstart_text_ja.md) をどうぞ。

```sh
pip install kci-vec          # Linux x86-64、Windows x86-64、macOS (0.2.16 から)。Python 3.10 以上
```

ソースから組み立てる場合は、このリポジトリの `clients/python` で `pip install .` です (gcc か clang が要ります)。
C から使う場合は、`kci_vec.c` の 1 ファイルとヘッダ (`kci_vec.h`、`kci_crypt.h`) をそのまま自分のプログラムに入れます。依存はありません。

使い方は 2 通りあります。

| | 使うもの | 向く場面 |
|---|---|---|
| **文章のまま渡す** | `kci_vec.text.TextIndex` | 文章を入れて、質問の文で探したい。ベクトルやモデルのことは任せたい |
| **自分でベクトルを用意する** | `kci_vec.KciVec` | 埋め込みは自分で作っている。画像など、文章以外も扱う。細かく設定したい |

目次: [1. 文章のまま渡す](#1-文章のまま渡す-textindex) · [2. 自分でベクトルを用意する](#2-自分でベクトルを用意する-kcivec) ·
[3. KciVec の関数一覧](#3-kcivec-の関数一覧) · [4. 単語の検索と合わせる](#4-単語の検索と合わせる) ·
[5. 始めるときの設定](#5-始めるときの設定) · [6. 守る決まり](#6-守る決まり) · [7. C の API](#7-c-の-api) · [8. PostgreSQL](#8-postgresql)

---

## 1. 文章のまま渡す (TextIndex)

文章を渡すと、中でベクトルに変えて索引を作ります。意味で探す側と、単語で探す側の両方を持ち、結果を合わせて返します。
モデルを動かすために、追加で次の 2 つが要ります。

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

### 入れて、探す

```python
from kci_vec.text import TextIndex

ix = TextIndex("ja", sparse="bm25")          # "ja" か "en"。最初の 1 回だけ、言語モデルを取ってきます
ix.add([
    "有給休暇は入社半年後に10日付与されます。申請は3営業日前までに上長へ。",
    "経費の精算は月末締めです。領収書の原本を総務に提出してください。",
    "会議室の予約は社内カレンダーから行います。利用は1回2時間までです。",
])                                           # 番号は 0, 1, 2 … と付きます。ids=[...] で自分の番号も渡せます

for hit in ix.search("有給は何日もらえますか", k=2):
    print(hit.id, round(hit.score, 3), hit.text)
```

`search` は `Hit` の一覧を、良い順に返します。`Hit` の中身は次のとおりです。

| 名前 | 中身 |
|---|---|
| `id` | 文章の番号 |
| `score` | 合わせた点数 (大きいほど良い) |
| `text` | 当たった部分の文章 |
| `dense_rank`、`sparse_rank` | 意味の側・単語の側での順位 (その側で見つからなかったら `None`) |
| `span` | 長い文章を分けたとき、当たった部分の位置 (始まり, 終わり) |

### 足す・消す・保存する

```python
new_id = ix.add(["在宅勤務は週2日までです。前日までに申請してください。"])[0]   # 付いた番号の一覧が返ります
ix.delete(1)                                 # 番号 1 の文章を消す。あれば True

ix.save("my_index", key="合言葉")            # フォルダに保存。key を渡すと、索引も文章も暗号化します
ix2 = TextIndex.load("my_index", key="合言葉")   # 合言葉が違えば RuntimeError
print(len(ix2), ix2.search("在宅勤務は何日まで", k=1)[0].text)
```

- `key` を渡さないと、文章は `texts.json` にそのまま書かれます。
- 中身を書き換えるには、`delete` してから `add` します。

### `TextIndex(lang="ja", mode="hybrid", ...)` の主な引数

| 引数 | 意味 |
|---|---|
| `lang` | `"ja"` か `"en"`。組み込みのモデルを選びます |
| `mode` | `"hybrid"` (意味と単語の両方、既定)、`"dense"` (意味だけ)、`"sparse"` (単語だけ) |
| `sparse="bm25"` | 単語の側を、言語モデルではなく文字の並びの数え上げにします。軽く、モデルを 1 つ取ってこなくて済みます |
| `dense=`、`sparse=` | 組み込みのモデルの代わりに、自分の関数を渡します (`sparse=` に関数を渡すときは `sparse_dim=` も) |
| `w_dense` | 合わせるときの、意味の側の重み (0〜1)。渡さなければ言語ごとの既定値 |
| `chunk`、`overlap` | この文字数より長い文章は、部分に分けて索引に入れます。`overlap` は隣の部分と重ねる文字数。`chunk=0` で分けない |
| `dedup=True` | 同じ文章が何度も出るデータ (定型文、空の欄) で、同じものを 1 度だけ持ちます |
| `keep_text=False` | 文章そのものを索引に持たせません (番号だけ返します。文章は自分で保管します) |
| `device` | モデルを動かす場所。渡さなければ CPU |
| `verbose=False` | 進み具合の表示を止めます |

### TextIndex の関数

| 関数 | |
|---|---|
| `add(texts, ids=None, batch=256)` | 文章を足す。付いた番号の一覧を返す |
| `delete(id)` | 1 件消す。あれば `True` |
| `build()` | 探せる形にする。`search` が最初に自分で呼ぶので、待つ時を自分で選びたいときだけ呼ぶ |
| `search(query, k=5, filter=None, rerank=None, select=None, ...)` | 上の説明のとおり。`filter` は、探してよい番号の集まりか、番号 → 真偽の関数 |
| `candidates(query, depth_dense=50, depth_sparse=100, filter=None)` | 両方の側が見つけたものを、合わせずにそのまま返す。自分の並べ直しの道具に渡すため |
| `tune(questions, answers, k=10)` | 正解の分かっている質問を 50 問ほど渡すと、意味の側の重みを自分の文章に合わせる。`(重み, 表)` を返す |
| `save(path, key=None)` / `TextIndex.load(path, key=None, ...)` | 上の説明のとおり |
| `len(ix)` | 入っている文章の数 |

`search` の `rerank=` には、`(質問, [文章の一覧]) → 文章ごとの点数` の関数を渡せます (自分で用意した並べ直しのモデル)。
`select=` には、`(質問, [文章の一覧]) → 役に立つ文章の位置の一覧` の関数を渡せます。どちらも、渡さなければ使いません。

---

## 2. 自分でベクトルを用意する (KciVec)

### 2 つの索引

索引は、速く探すための下ごしらえのデータです。2 種類あります。

| | **グラフ方式** (`build()`) | **区分け方式** (`build_lists()`) |
|---|---|---|
| 探し方 | 近いベクトルどうしをつないだ網を、質問に近いほうへたどる | ベクトルをかたまりに分け、質問に近いかたまりだけを調べる |
| 精度を上げるつまみ | `ef` (と `eps`) | `nprobe` と `rerank_n` |
| 速くする選択肢 | 量子化グラフ (`build_qg()`) | `set_probe_gap()` |
| float のベクトルが要るか | 並べ直しと `insert` のときだけ | いつも要る (`rerank > 0`) |
| 追加・削除・保存 | できる | できる |

既定ではコサイン類似度 (向きの近さ) で比べます。ベクトルは足したときに長さ 1 にそろえられます。
ユークリッド距離 (`metric="l2"`) と内積 (`metric="ip"`) も選べます。

どちらが合うかは、データによります。自分のデータで両方を試してください。区分け方式は索引が小さく、設定を合わせやすいです。
量子化グラフは速い代わりに、メモリを多く使います。

### 作って探す

```python
import numpy as np
from kci_vec import KciVec

rng = np.random.default_rng(0)
embeddings = rng.standard_normal((2000, 384)).astype(np.float32)   # ここは自分の埋め込みに置き換える
ids = np.arange(2000)                                              # 自分で決める整数の番号
query = embeddings[7]; queries = embeddings[:5]

v = KciVec(dim=384, rerank=200)        # rerank > 0 で float のベクトルを持っておく (区分け方式には必須)
v.add_many(ids, embeddings)            # embeddings: float32 の [n, dim]
v.build_lists()                        # 区分け方式   (グラフ方式なら v.build())

hits = v.search(query, k=10)           # [(id, score), ...] を良い順に。score はコサイン類似度
found, scores = v.search_many(queries, k=10)   # [nq, k] の配列。k 件に足りない所は -1 / -inf
```

### 属性やアクセス権で絞り込む

```python
hits = v.search(query, k=10, filter={3, 17, 42})          # この番号だけ
hits = v.search(query, k=10, filter=lambda i: i % 2 == 0) # id -> bool の関数でもよい
```

絞り込みは、探した後ではなく、探している最中にかかります。そのため、条件に合うものが少なくても、その中でいちばん近いものが返ります。

- 条件に合うものが全体のわずかなとき: 区分け方式なら `nprobe` と `rerank_n` を、グラフ方式なら `ef` を大きくします。
- ほんの数件しかないとき: 何も指定しなくて構いません。番号の集合・リスト・配列を渡し、それが十分に少なければ、索引の大きさに関係なく、`search` は通るベクトルだけと比べます。どの形の絞り込みでも、見つかった答えが `k` 個に届かないときは、通るものだけを総当たりして答え直すので、通るものを取りこぼしません (0.2.19 から。関数を渡した場合、そのときは全部のベクトルについて 1 回ずつ呼ばれます)。`search_exact(query, k, filter=...)` は、同じことを指定して行います。

### 追加・削除・詰め直し

```python
v.insert(5000, embeddings[0])   # 作った索引に足す (rerank > 0 が必要)
v.delete(5000)                  # 以後、結果に出なくなる。あれば True
v.compact()                     # 生きているベクトルだけで作り直し、場所を空ける
```

書き換えは「`delete` してから `insert`」です。大量に足すときは、`insert` を繰り返すより、`add_many` してから作り直すほうが速いです。

### 保存と読み込み (暗号化あり・なし)

```python
v.save("index.kv", key=b"a long secret")       # AES-256-CTR + AES-256-CMAC。key=None なら暗号化なし
w = KciVec.load("index.kv", key=b"a long secret", rerank=200)
```

- 鍵が違うとき、ファイルが壊れているか書き換えられているときは、`load` が `RuntimeError` になります。中身を読む前に、改ざんを見つける印を確かめます。
- `load` は、作ったときの `rerank` を知りません。もう一度渡してください。
- 区分け方式も、量子化グラフも、ファイルに入ります。読み込めば、作り直さずに使えます。

### 速いグラフ (量子化グラフ)

```python
g = KciVec(dim=384, M=8, rerank=200)
g.add_many(ids, embeddings)
g.build()                     # 先にグラフ
g.build_qg(dims_per_code=2)   # 次に量子化グラフ。使うバイト数が返る
g.set_eps(0.1)                # 止め方 (3 節を参照)
hits = g.search(query, k=10, ef=30)
```

`insert`、`compact`、`build` を呼ぶと、量子化グラフは消えます。そのあとで `build_qg()` をもう一度呼んでください。

### メモリを小さくする

探すだけの索引 (あとから足したり消したりしない) なら、要らないものを手放せます。

```python
g.half_floats()     # 並べ直し用の float を、半分の大きさ (16 ビット) にする
g.search_only()     # 量子化グラフでの検索が読まないものを手放す
hits = g.search(query, k=10, ef=30, rerank_n=100)
```

- どちらも元に戻せません。呼んだあとは、`insert`・`delete`・`save` ができなくなります。保存は、呼ぶ前に済ませてください。
- `search_only()` には、`build()` と `build_qg()` を済ませた索引が要ります。呼んだあとの `search` には `rerank_n` を渡します。
- もう 1 つの道は `float_mmap(path)` です。float のベクトルをファイルに移し、メモリには索引だけを残します。

### 同じベクトルが何度も出るデータ

```python
d = KciVec(dim=384, rerank=200, dedup=True)   # 同じベクトルは 1 度だけ持ち、番号だけを並べて持つ
d.add_many(ids, embeddings)
d.build()
print(d.dup_stats())    # 同じ行がどれだけあるか: rows、groups、rows_in_groups、largest、distinct など
d.unique = True         # 探した結果を「違うベクトルごとに 1 件」にする (既定は、同じベクトルの番号も全部返す)
```

定型文や空の欄など、同じ文章がくり返し出るデータでは、同じベクトルが大量に並びます。`dedup=True` は、そのためのものです。

### 読むだけの索引を、複数のプロセスで共有する

```python
v.save_image("index.img")                 # 暗号化しない、読むだけの形で書き出す
r = KciVec.attach("index.img", rerank=200)   # 写しを作らずに開く。同じファイルを開いたプロセスどうしでメモリを共有する
hits = r.search(query, k=10)
```

- `attach` した索引は読むだけです。足す・消す・作り直すは、できません。
- このファイルは暗号化されません。暗号化して保存するなら `save(path, key)` を使ってください。
- 書き出した機械と同じ種類の機械で開きます。

### ユークリッド (L2) 距離と内積

```python
e = KciVec(dim=384, rerank=200, metric="l2")     # 点数は距離の 2 乗。小さい順に並ぶ
p = KciVec(dim=384, rerank=400, metric="ip")     # 点数は内積。大きい順に並ぶ。ベクトルの長さはそのまま使う
```

- ほかの関数は、コサインのときと同じに使えます。
- `max_norm` (既定は 0) は、索引が受け付けるベクトルの長さの上限です。0 なら、最初に作る前に足したベクトルから決めます。
  あとで、それより長いベクトルを `insert` すると失敗します。
- 内積では、`rerank` を大きめ (400 以上) にしてください。

---

## 3. KciVec の関数一覧

### `KciVec(dim, M=16, efc=200, rerank=200, metric="cosine", max_norm=0.0, centre=False, dedup=False)`

| 引数 | 意味 |
|---|---|
| `dim` | ベクトルの長さ |
| `M`、`efc` | グラフを作るときの設定。1 点あたりのつながりの数と、作るときの探す幅。16 / 200 が無難な既定です |
| `rerank` | 既定は 200。float のベクトルを持ち、検索のたびに上位 200 件の候補を正確に比べ直す。`build_lists`、`insert`、`compact` に必要。`rerank=0` にすると 1 バイトの行だけを持つ (メモリは 1 次元あたり 5 バイト → 1 バイト、取りこぼしは増える)。0.2.16 までの既定は 0 でした |

小さな索引は総当たりで答えます。「件数 × 次元」が `exact_cells` (既定 1,000 万。1,024 次元で 1 万件、384 次元で 2.6 万件) 以下で、float のベクトルを持っている索引では、`search` は質問を全部のベクトルと比べ、答えは正確です (`ef`・`nprobe`・`rerank_n` は使われません)。`filter` をつけたときは、通る番号のベクトルだけと比べます (0.2.17 では、`filter` つきは索引の探し方のままでした)。`method="graph"` / `"lists"` を指定したときは索引の探し方のままです。`v.exact_cells = 0` で切れます。0.2.19 から、内積 (`metric="ip"`) の索引も対象です。

それより大きな索引は、自分に要る探す幅を測ります (0.2.19 から)。`build` の最後に、入っているベクトルの一部を質問にして総当たりの答えと比べ、`ef` 200 / 400 / 800 / 1600 のうち、本当の上位 10 件の 98 % を見つけられる一番狭い幅を `search` の既定にします (どれも届かなければ一番広い幅)。正確に比べ直す候補の数も、200 件で足りないときは一緒に決めます。そのため、何も指定しない `search` は、幅が要るデータでは広く (遅く) なり、要らないデータでは変わりません。結果は `save` / `load` で保たれます。一番広い設定でも 97 % に届かないとき、グラフの検索でたどり着けない行が残ったときは、`build` が警告を出します。測るのに使うのは入っているベクトルです。質問がそれと違う種類のとき (画像のベクトルを文章で探す、内積のデータの多く) は、質問をいくつか `calibrate(queries=...)` に渡してください。自分で渡した `ef` や `rerank_n` は、そのまま使われます。`build` の前に `v.auto_ef = False` とすると、測りません。
| `metric` | `"cosine"`、`"l2"`、`"ip"` (内積) |
| `max_norm` | `l2` と `ip` で、ベクトルの長さの上限。0 = 自動 |
| `centre` | `ip` だけ。ベクトルの平均を引いてから索引に入れる。順位は変わらない |
| `dedup` | 同じベクトルを 1 度だけ持つ |

### 足す

| 関数 | |
|---|---|
| `add(id, vec)` | 1 件 (float32、長さ `dim`) |
| `add_many(ids, mat)` | まとめて。`mat` は [n, dim] |
| `reserve(n)` | 先に `n` 件分の場所を取る |

### 作る

| 関数 | |
|---|---|
| `build(threads=0, fast=False)` | グラフを作る。0 = 全コア。かかった秒数を返す。区分け方式を作ってあると消えるので、両方使うなら**グラフを先に**作る。`fast=True` は、作る時間を優先した作り方 |
| `build_qg(dims_per_code=1, threads=0, code_dims=0, gate_dims=0, residual=False)` | 量子化グラフを作る (`build` か `load` のあと)。`dims_per_code` は 1 = 細かい、2 = 大きさが半分。使うバイト数を返す。残りの 3 つは、次元の高いデータで索引を小さくするための設定 (下を参照) |
| `build_lists(nlist=0, soar=0.0, threads=0, nprobe=None, ...)` | 区分け方式を作る。`nlist` 0 = 自動。`soar` 1.5 で、各ベクトルを 2 つ目のかたまりにも入れる。`nprobe` は探すときの既定値。残りの引数は 0 = 自動 |

| `calibrate(k=10, target=0.98, samples=128, threads=0, queries=None)` | この索引に要る探す幅を測り、`search` の既定にする (上の説明。`exact_cells` を超える索引では `build` が呼ぶ)。`queries`: 入っているベクトルの代わりに使う、自分の質問 (8 個以上)。`(ef, 本当の上位 k 件のうち見つかった割合)` を返す。結果は `default_ef()`、`default_rerank()`、`calibrated_recall` で読める |
| `judge(k=10, target=0.99, samples=4096, queries=None)` | (0.2.20 から) 質問ごとに探す幅を索引に決めさせる。呼んだあとは、`ef` を指定しない `search` が、少しで足りる質問には少ない手間で答える。`ef` を指定した検索は変わらない。`k`: 検索で求める件数 (2〜64)。`queries`: 入っているベクトルの代わりに使う、自分の質問 (200 個以上)。`(使うかどうか, 効き目)` を返す。使うかどうかが False のときは、この索引では効かないと測ったということで、`search` は元のまま。絞り込みなしのグラフの検索、`cosine` と `l2` が対象。保存されないので、`load` や `build` のあとは呼び直す。`judge(samples=0)` で止める |
| `choose_method(k=10, target=0.98, queries=None, samples=1000, nlist=0, threads=0)` | (0.2.20 から) グラフと区分け方式のどちらがこのデータに合うかを測り、`method` を指定しない `search` がそれを使うようにする。区分け方式がまだ無ければ作り、見本を両方の方式で探して正確な答えと比べ、`target` (本当の上位 `k` 件のうち見つかる割合) に届く設定のうち速いほうを残す。`method` と、`graph`・`lists` それぞれの設定・見つかった割合・1 問あたりのミリ秒が入った dict を返す。時間はその機械・1 スレッドで測るので、ほかの仕事が動いていないときに呼ぶ。見本は、`queries` を渡さなければ入っているベクトル。自分の質問を渡したほうが、実際に近い結果になる。保存されないので、`load` のあとは呼び直す |

`build_qg` の、索引を小さくする設定 (次元が 384 以上のデータ向け):

- `code_dims`: 符号を、ベクトルの主な方向の上位 `code_dims` 個だけから作る (偶数、8 〜 `dim` − 1)。
- `gate_dims`: `code_dims` と一緒に使う。符号をさらに短くする (偶数、8 〜 `code_dims`)。
- `residual=True`: `code_dims` と一緒に使う。メモリをいちばん小さくする形。`dims_per_code` は使われない。

どの値が合うかはデータによります。recall を測りながら決めてください。

### 探し方の設定 (変えるまで続く)

| 関数 | |
|---|---|
| `set_eps(eps)` | グラフ方式の止め方。0 = 使わない (既定)。0 より大きいと、結果の集まりを `ef` 件に保ち、今の境目から (1 + eps) 倍以内に候補がある間だけ探し続ける。小さい `ef` (k 〜 3k) と組み合わせる |
| `set_probe_gap(gap, min_probe=8)` | 区分け方式: 上位 `nprobe` 個のかたまりのうち、いちばん近いものとの点数の差が `gap` 以内のものだけを調べる (少なくとも `min_probe` 個)。0 = 使わない |
| `set_rerank("float" \| "int8")` | 区分け方式: 候補の並べ直しに float (正確、既定) を使うか、int8 (4 分の 1 の大きさ) を使うか。グラフ方式では、代わりに `search` に `rerank_n=0` を渡す |
| `set_qg_exact(on=True)` | 量子化グラフ: 見積もりを通った候補を、結果に入れる前にもう一段確かめる。次元の高いデータで試す。既定は使わない |
| `unique = True` | `dedup=True` の索引で、結果を「違うベクトルごとに 1 件」にする |

### 探す

| 関数 | |
|---|---|
| `search(query, k=10, ef=None, nprobe=None, rerank_n=None, filter=None, method="auto")` | 上位 `k` 件を `[(id, score)]` で返す。`method` は `"auto"` (区分け方式があればそれ、なければグラフ)、`"lists"`、`"graph"`。既定は `ef` = max(k, 100)、`nprobe` = 32、`rerank_n` = `rerank`。`k` は 4,096 まで |
| `search_many(queries, k=10, **kw)` | 多くの質問を順に探す。`(ids, scores)` の配列を返す |
| `search_exact(query, k=10, threads=0, filter=None)` | 総当たりで正確な答えを出す。recall の確かめや、ごく小さい索引に |

### 変える・保存する・そのほか

| 関数 | |
|---|---|
| `insert(id, vec)` | 作った索引に足す。`rerank` が無いと `RuntimeError` |
| `delete(id)` | その番号があれば `True` |
| `compact()` | 生きているベクトルだけで作り直す。その件数を返す |
| `save(path, key=None)` / `KciVec.load(path, key=None, rerank=100)` | 2 節を参照 |
| `save_image(path)` / `KciVec.attach(path, rerank=100)` | 2 節を参照 |
| `half_floats()`、`search_only()`、`float_mmap(path)` | 2 節「メモリを小さくする」を参照 |
| `dup_stats()` | 同じ行がどれだけあるかの数字 |
| `plain_bytes()` | メモリ上の大きさを種類ごとに: `codes`、`floats`、`derived`、`graph`、`ids`、`total` |
| `count`、`is_built()`、`has_lists`、`close()` | |

エラーは 2 種類です。ライブラリが断ったか失敗したときは `RuntimeError`、引数がおかしいときは `ValueError` です。

### 質問の見本から学ぶ (実験的)

質問とデータの性質が大きく違うとき (質問は文章、データは画像、など) のための関数です。実際の質問の見本と、その正解を渡して使います。
実験的な機能で、引数は変わることがあります。

| 関数 | |
|---|---|
| `learn(queries, answers, nclusters=0, list_cap=0, threads=0, pack=False)` | 質問の見本 [nq, dim] と、それぞれの正解の番号 [nq, 件数] から学ぶ。作った索引と `rerank > 0` が要る。答えがどれだけ一部のベクトルに集まっているかの割合を返す。`build` と `compact` で消え、保存されない |
| `search_learned(query, k=10, nclusters=4, per_list=1000, ef=0, ...)` | 学んだものを使って探す。`ef` が 0 より大きいと、グラフでの検索と組み合わせる |
| `vote_set(answers)`、`vote_short(ids, rows)`、`search_voted(query, voters, k=10, ...)` | 答えが一部に集まらないデータ向けの別の形 |

---

## 4. 単語の検索と合わせる

文章のまま渡す `TextIndex` (1 節) は、これを中でやっています。自分でベクトルを用意している場合は、次の部品を組み合わせます。

```python
from kci_vec.hybrid import BM25Index, hybrid_search

texts = ["KX-204B の請求書を再発行する手順", "会議室の予約は社内カレンダーから", "経費の精算は月末締め"]
tv = KciVec(dim=384, rerank=50); tv.add_many([0, 1, 2], embeddings[:3]); tv.build()

bm = BM25Index()
for i, text in enumerate(texts):
    bm.add(i, text)                    # ベクトルと同じ番号で
hits = hybrid_search(tv, bm, embeddings[0], "KX-204B 請求書", k=2)
# [(id, 合わせた点数, ベクトル側の順位, 単語側の順位)] を良い順に。見つからなかった側の順位は None
bm.save("index.bm25.json"); bm = BM25Index.load("index.bm25.json")
```

- 埋め込みでは拾いにくい型番・名前・数字を、単語の側が拾います。
- 日本語・中国語・韓国語は 2 文字ずつに区切るので、辞書は要りません。
- `BM25Index` が書き出すのは、ただの JSON です。文章が機密なら、このファイルは自分で暗号化して保管してください。

`kci_vec.sparse.KciSparse` は、単語ごとの重みを持つベクトル (疎ベクトル) のための索引です。暗号化した保存もできます。
`TextIndex` の単語の側は、これを使っています。じかに使う場合の引数は、`clients/python/kci_vec/sparse.py` の説明をご覧ください。

---

## 5. 始めるときの設定

ここに書いたのは出発点です。いちばん良い値は、データによって違います。
自分の質問を数百件用意し、`search_exact` の答えと比べて recall を測りながら、値を振って探してください。

| 目的 | まずこれで | 次に振る値 |
|---|---|---|
| 簡単に、小さい索引で | `build()`、`search(ef=100)` | `ef` 50 〜 800 |
| 速いグラフ、次元の低いデータ (100 次元くらい) | `M=32`、`build()`、`build_qg(1)`、`set_eps(0.1)`、`ef=30` | `eps` 0.05 〜 0.25、`ef` 20 〜 50 |
| 速いグラフ、文章の埋め込み (384 次元以上) | `M=8`、`build()`、`build_qg(2)`、`set_eps(0.05)`、`ef=150` | `eps` 0 〜 0.1、`ef` 100 〜 400。recall 0.99 なら `M=16` |
| 区分け方式 | `build_lists()`、`search(nprobe=64, rerank_n=200)` | `nprobe` 8 〜 512、`rerank_n` 50 〜 800 |
| 区分け方式をもっと速く | `set_probe_gap(0.25)` を足す | `gap` 0.15 〜 0.35 |
| メモリをいちばん少なく (探すだけ) | `build()`、`build_qg(...)`、`half_floats()`、`search_only()` | `build_qg` の `code_dims`・`gate_dims` |
| メモリを少なく (足す・消すもする) | `build()`、`build_lists()`、最後に `float_mmap(path)` | |

`ef`、`eps`、`nprobe`、`rerank_n` を大きくすると、recall は上がり、1 秒あたりに答えられる質問の数は減ります。
上位 100 件を取る (`k=100`) ときは、上位 10 件のときより大きい `ef` が要ります。

---

## 6. 守る決まり

- **書く人は一度に 1 人。** 探している最中の索引に、追加・挿入・削除・詰め直し・作り直しをしないでください。
- **スレッド。**
  - Python では、1 つの `KciVec` が探すための作業場所を 1 つだけ持ちます。同じ `KciVec` の `search` を、複数のスレッドから同時に呼ばないでください。
    `TextIndex` も同じです。
  - C では、スレッドごとに `KciVecSearcher` を 1 つ作ります。作り終えて変えていない索引なら、複数の searcher が同時に動けます。
- **作る順番。** `build()` が先で、`build_lists()` と `build_qg()` はあとです。あとから `build()` を呼ぶと、両方とも消えます。
- **量子化グラフ。** `insert`、`compact`、`build` で消えます。そのあとで `build_qg()` をもう一度呼んでください。
- **元に戻せない操作。** `half_floats()`、`search_only()`、`float_mmap()` のあとは、作り直しや保存ができなくなるものがあります。要るものは先に作って、先に保存してください。
- **保存したファイルの鍵。** 鍵をなくすと、ファイルは開けません。鍵を取り戻す方法はありません。
- **動く環境。**
  - 配っている部品 (wheel): AVX2 のある Linux x86-64 (AVX-512 があれば使う)、AVX2 のある Windows x86-64、macOS (Apple シリコン、AVX2 のある Intel の Mac)。
  - ソースからなら、ほかの ARM64 の機械でも組み立てられます。

---

## 7. C の API

```c
#include "kci_vec.h"

KciVec *v = kci_vec_new(dim, 16, 200);                  /* dim, M, efConstruction */
kci_vec_keep_float(v, 1);                               /* add より前に: 並べ直しと区分け方式のために float を持つ */
for (...) kci_vec_add(v, id, vec);                      /* float32[dim]。中で長さ 1 にそろえる */
kci_vec_build(v, 0, NULL);                              /* グラフ、全コア */

KciVecSearcher *s = kci_vec_searcher_new(v);            /* スレッドごとに 1 つ。質問をまたいで使い回す */
int n = kci_vec_search_rerank(s, query, 10, 100, 200, ids, scores);   /* k, ef, rerank_n */

kci_vec_save(v, "index.kv", key, keylen);               /* key が NULL でなければ暗号化 */
kci_vec_searcher_free(s);
kci_vec_free(v);
```

一通り動く例は、リポジトリの `example_kci_vec.c` にあります。

戻り値:
- 成功すると、件数か 0 を返します。
- 失敗すると負の数を返します。ふつうは `-1` です。2 つ目の理由 (「float が無い」など) を説明してある関数では `-2` もあります。
- `kci_vec_new`、`kci_vec_load`、`kci_vec_attach` は、失敗すると `NULL` を返します。

| 分類 | 関数 |
|---|---|
| 作る・足す | `kci_vec_new`、`kci_vec_new_l2`、`kci_vec_new_ip`、`kci_vec_free`、`kci_vec_reserve`、`kci_vec_add`、`kci_vec_add_i8`、`kci_vec_keep_float`、`kci_vec_set_params`、`kci_vec_set_dedup`、`kci_vec_set_ip_centre` |
| グラフ | `kci_vec_build`、`kci_vec_build_fast`、`kci_vec_search` (たまに探すとき)、`kci_vec_searcher_new` / `_free`、`kci_vec_search_r`、`kci_vec_search_rerank`、`kci_vec_search_filtered`、`kci_vec_searcher_set_eps`、`kci_vec_searcher_set_unique` |
| 量子化グラフ | `kci_vec_build_qg`、`kci_vec_build_qg_ex`、`kci_vec_build_qg_gate`、`kci_vec_build_qg_res`、`kci_vec_qg_bytes`、`kci_vec_searcher_set_qg`、`kci_vec_searcher_set_qg_exact` |
| 区分け方式 | `kci_vec_build_ivf`、`kci_vec_build_ivf_ex`、`kci_vec_build_ivf_p` (すべての設定)、`kci_vec_set_ivf_codes`、`kci_vec_set_ivf_training`、`kci_vec_search_ivf`、`kci_vec_search_ivf_filtered`、`kci_vec_searcher_set_probe_gap`、`kci_vec_searcher_set_rerank_int8` |
| 総当たり | `kci_vec_search_exact`、`kci_vec_search_exact_filtered`、`kci_vec_filter_id_bits` |
| 探す幅 | `kci_vec_calibrate`、`kci_vec_calibrate_queries`、`kci_vec_default_ef`、`kci_vec_default_rerank`、`kci_vec_set_default_ef` (`kci_vec_search_rerank` / `kci_vec_search_filtered` は `ef` に 0 を渡すと測った幅を使う)、`kci_vec_judge`、`kci_vec_judge_queries`、`kci_vec_judged`、`kci_vec_judge_off`、`kci_vec_graph_recall`、`kci_vec_unreached`、`kci_vec_searcher_set_filter_fill` |
| 変える | `kci_vec_insert`、`kci_vec_delete`、`kci_vec_forget` (そのベクトルの跡をすべて消す)、`kci_vec_compact`、`kci_vec_squeeze` |
| float | `kci_vec_half_floats`、`kci_vec_float_mmap`、`kci_vec_set_float_source` (並べ直しのときに、自分の保管先から float を取ってくる)、`kci_vec_set_float`、`kci_vec_get_float`、`kci_vec_row_id` (その行のベクトルの id) |
| 保存・読み込み | `kci_vec_save`、`kci_vec_load`、`kci_vec_save_ex` / `kci_vec_load_ex` (`KCI_VEC_SAVE_NO_FLOAT`、`KCI_VEC_SAVE_RAW_KEY`) |
| 読むだけの共有 | `kci_vec_image_save`、`kci_vec_attach`、`kci_vec_attach_mem`、`kci_vec_thaw` |
| 探すだけの索引 | `kci_vec_search_only` |
| 質問の見本から学ぶ (実験的) | `kci_vec_learn`、`kci_vec_learn_pack`、`kci_vec_has_learned`、`kci_vec_search_learned`、`kci_vec_searcher_set_learned_seeds` / `_pages` / `_hot`、`kci_vec_vote_set`、`kci_vec_vote_short`、`kci_vec_search_voted`、`kci_vec_search_voted_short` |
| メモリ | `kci_vec_protect` (ベクトルの置き場をコアダンプとスワップに出さないよう OS に頼む。できる範囲で)、`kci_vec_plain_bytes` |
| 情報 | `kci_vec_count`、`kci_vec_dim`、`kci_vec_is_built`、`kci_vec_metric`、`kci_vec_is_l2`、`kci_vec_dup_stats` |

引数、戻り値、条件の説明は、`kci_vec.h` の宣言の上のコメントにあります (並んだいくつかの関数は、まとめて 1 つの説明になっています)。
単語の側の索引 (`kci_sparse.c` / `kci_sparse.h`) も、同じ形で C から使えます。例は `example_hybrid.c` です。

---

## 8. PostgreSQL

PostgreSQL の中から呼ぶための拡張が、`pg/` にあります (試作)。組み立て方と使い方は、`pg/README.md` をご覧ください。
