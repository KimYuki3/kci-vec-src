# KCI-Vec の使い方 (API の説明書)

KCI-Vec を Python と C から使う方法をまとめます。それぞれの関数が何をするか、どの設定から始めるか、守る決まりを書きます。
English: [API.md](API.md)

- **Python**: `kci_vec` パッケージです。入れ方は 2 つあります。
  - [kci-vec-dist](https://github.com/KimYuki3/kci-vec-dist) の wheel を入れる (Linux x86-64)
  - このリポジトリの `clients/python` から組み立てる (`pip install .`。gcc か clang が要ります)
- **C**: `kci_vec.c` の 1 ファイルとヘッダ 2 つ (`kci_vec.h`、`kci_crypt.h`) で、依存はありません。すべての関数の説明は `kci_vec.h` のコメントにもあります。

目次: [1. 2 つの索引](#1-2-つの索引) · [2. Python の手順](#2-python-の手順) · [3. Python の関数一覧](#3-python-の関数一覧) ·
[4. 始めるときの設定](#4-始めるときの設定) · [5. 守る決まり](#5-守る決まり) · [6. C の API](#6-c-の-api)

---

## 1. 2 つの索引

索引 (インデックス) は、速く探すための下ごしらえのデータです。KCI-Vec には 2 種類あります。

| | **グラフ方式** (`build()`) | **区分け方式** (`build_lists()`) |
|---|---|---|
| 探し方 | 近いベクトルどうしをつないだ網 (HNSW) を、質問に近いほうへたどる | ベクトルをかたまりに分け (k-means)、質問に近いかたまりだけを調べる |
| 精度を上げるつまみ | `ef` (と `eps`) | `nprobe` と `rerank_n` |
| 速くする選択肢 | **量子化グラフ** (`build_qg()`) | **probe gap** (`set_probe_gap()`) |
| float のベクトルが要るか | 並べ直しと `insert` のときだけ | いつも要る (`rerank > 0`) |
| 追加・削除 | できる | できる |
| `save()` で保存されるか | される (量子化グラフは**されない**。[守る決まり](#5-守る決まり) を参照) | される |

どちらも、既定ではコサイン類似度 (向きの近さ) で比べます。ベクトルは追加したときに長さ 1 にそろえられます。
ユークリッド (L2) 距離 (`metric="l2"`) と内積 (`metric="ip"`) もあります。

**どちらを選ぶか**: 自分のデータで両方を試してください。

- 区分け方式は、グラフを作る必要がなく、設定も合わせやすいです。
- 量子化グラフは、ベンチマークではいちばん速く出ましたが、メモリを多く使います。

---

## 2. Python の手順

### 作って探す

```python
import numpy as np
from kci_vec import KciVec

v = KciVec(dim=384, rerank=200)        # rerank > 0 で float のベクトルを持っておく (区分け方式には必須)
v.add_many(ids, embeddings)            # ids: 自分で決める整数の番号 (uint64)、embeddings: float32 の [n, dim]
v.build_lists()                        # 区分け方式   (グラフ方式なら v.build())

hits = v.search(query, k=10)           # [(id, score), ...] を良い順に。score はコサイン類似度
ids, scores = v.search_many(queries, k=10)   # [nq, k] の配列。k 件に足りないところは -1 / -inf
```

### 属性やアクセス権で絞り込む

```python
hits = v.search(query, k=10, filter={3, 17, 42})          # この番号だけ
hits = v.search(query, k=10, filter=lambda i: i % 2 == 0) # id -> bool の関数でもよい
```

絞り込みは、探した後ではなく、探している最中にかかります。そのため、条件に合うものが少なくても、その中でいちばん近いものが返ります。

- 条件に合うものが全体のわずかなとき: 区分け方式なら `nprobe` と `rerank_n` を、グラフ方式なら `ef` を大きくします。
- ほんの数件しかないとき: `search_exact` (総当たり) で探し、自分で絞り込みます。

### 追加・削除・詰め直し

```python
v.insert(new_id, vec)    # 作った索引に足す (rerank > 0 が必要)
v.delete(old_id)         # 削除の印を付ける。以後、結果に出なくなる
v.compact()              # 生きているベクトルだけで作り直し、場所を空ける
```

更新は「`delete` してから `insert`」です。大量に足すときは、`insert` を繰り返すより、`add_many` してから `build*` で作り直すほうが速いです。

### 保存と読み込み (暗号化あり・なし)

```python
v.save("index.kv", key=b"a long secret")       # AES-256-CTR + AES-256-CMAC。key=None なら暗号化なし
w = KciVec.load("index.kv", key=b"a long secret", rerank=200)
```

- 鍵が違うとき、ファイルが壊れているか書き換えられているときは、`load` で `RuntimeError` になります。中身を読む前に、改ざん検出の印 (タグ) を確かめるためです。
- `load` は、作ったときの `rerank` を知りません。もう一度渡してください。

### 速いグラフ (量子化グラフ)

```python
v = KciVec(dim=384, M=8, rerank=200)
v.add_many(ids, embeddings)
v.build()                 # 先にグラフ
v.build_qg(dims_per_code=2)   # 次に最下層を量子化する。使うバイト数が返る
v.set_eps(0.1)            # 止め方 (下を参照)
hits = v.search(query, k=10, ef=30)
```

`load`、`insert`、`compact`、`build` の後は、`build_qg()` をもう一度呼んでください。量子化グラフは保存されず、これらを呼ぶと消えます。

### ハイブリッド検索 (単語とベクトル)

```python
from kci_vec.hybrid import BM25Index, hybrid_search

bm = BM25Index()
for i, text in zip(ids, texts):
    bm.add(i, text)                    # ベクトルと同じ番号で
hits = hybrid_search(v, bm, query_vec, "KX-204B 請求書", k=5)
# [(id, 合わせた点数, ベクトル側の順位, 単語側の順位)] を良い順に。見つからなかった側の順位は None
bm.save("index.bm25.json"); bm = BM25Index.load("index.bm25.json")
```

- 埋め込み (ベクトル) では拾いにくい型番・名前・数字を、単語の側が拾います。
- 日本語・中国語・韓国語は 2 文字ずつに区切るので、辞書は要りません。
- 単語の索引はただの JSON です。文章が機密なら、このファイルはご自分で暗号化して保管してください。

### ユークリッド (L2) 距離

```python
v = KciVec(dim=128, rerank=200, metric="l2")
```

- 点数は距離の 2 乗になり、小さい順に並びます。ほかはコサインのときと同じに使えます。
- `max_norm` (既定は 0) は、索引が受け付けるベクトルの長さの上限です。0 にすると、最初に作る前に足したベクトルから決めます。
- 後でそれよりずっと長いベクトルを `insert` すると失敗します。
- L2 のソースは、版 0.2.3 からこのリポジトリに入っています。

---

## 3. Python の関数一覧

### `KciVec(dim, M=16, efc=200, rerank=0, metric="cosine", max_norm=0.0)`

| 引数 | 意味 |
|---|---|
| `dim` | ベクトルの長さ (いくつでもよい) |
| `M`、`efc` | グラフを作るときの設定。1 点あたりの近傍の数と、作るときの探索の幅。16 / 200 が無難な既定です。384 次元の文章の埋め込みで量子化グラフを使うなら、`M` は 8 が合います |
| `rerank` | 0 より大きいと float のベクトルを持ち、正確に並べ直す候補の数の既定値になる。`build_lists`、`insert`、`compact` に必要 |
| `metric` | `"cosine"`、`"l2"`、`"ip"` (内積) |

### 足す

| 関数 | |
|---|---|
| `add(id, vec)` | 1 件 (float32、長さ `dim`) |
| `add_many(ids, mat)` | まとめて。`mat` は [n, dim] |
| `reserve(n)` | 先に `n` 件分の場所を取る (取り直しが減る) |

### 作る

| 関数 | |
|---|---|
| `build(threads=0)` | グラフを作る。0 = 全コアを使う。かかった秒数を返す。区分け方式を作ってあると消えるので、両方使うなら**グラフを先に**作る |
| `build_qg(dims_per_code=1, threads=0)` | グラフの最下層を量子化する (`build` か `load` の後)。`dims_per_code` は 1 = 細かい、2 = 大きさが半分。使うバイト数を返す |
| `build_lists(nlist=0, soar=0.0, threads=0, nprobe=None, dims_per_code=0, avq_threshold=0.0, train_iters=0, train_sample=0)` | 区分け方式を作る。`nlist` 0 = 2√n 個のかたまり。`soar` 1.5 にすると、各ベクトルを 2 つ目のかたまりにも入れる。`nprobe` は探すときの既定値。残りは符号とかたまりの作り方の設定で、0 = 自動 |

### 探し方の設定 (変えるまで続く)

| 関数 | |
|---|---|
| `set_eps(eps)` | グラフ方式: 結果の集合を `ef` 件に保ち、今の境界の (1 + eps) 倍以内に候補がある間だけ探し続ける。0 = 使わない (ふつうの HNSW)。小さい `ef` (k〜3k) と組み合わせる |
| `set_probe_gap(gap, min_probe=8)` | 区分け方式: 上位 `nprobe` 個のかたまりのうち、いちばん近いものとの点数の差が `gap` 以内のものだけを調べる (ただし少なくとも `min_probe` 個)。0 = 使わない |
| `set_rerank("float" \| "int8")` | 区分け方式: 候補の並べ直しに float (正確、既定) を使うか、int8 (4 分の 1 のバイト数) を使うか。int8 は k = 100 の文章の埋め込みなら問題ないが、glove-100 のような次元の低いデータには向かない。グラフ方式では、代わりに `search` に `rerank_n=0` を渡す |

### 探す

| 関数 | |
|---|---|
| `search(query, k=10, ef=None, nprobe=None, rerank_n=None, filter=None, method="auto")` | 上位 `k` 件を `[(id, score)]` で返す。`method` は `"auto"` (区分け方式があればそれ、なければグラフ)、`"lists"`、`"graph"`。既定は `ef` = max(k, 100)、`nprobe` = 32、`rerank_n` = `rerank` (区分け方式では少なくとも 2k)。グラフ方式では、`rerank_n` が 0 なら int8 で、0 より大きければ float で並べ直す。`k` は 4,096 まで |
| `search_many(queries, k=10, **kw)` | 多くの質問をまとめて (1 つずつ順に) 探す。`(ids, scores)` の配列を返す |
| `search_exact(query, k=10, threads=0)` | 総当たりで正確な答えを出す。recall の確認や、ごく小さい索引に |

### 変える・保存する・そのほか

| 関数 | |
|---|---|
| `insert(id, vec)` | 作った索引に足す。`rerank` が無いと `RuntimeError` |
| `delete(id)` | その番号があれば `True` |
| `compact()` | 生きているベクトルだけで作り直す。その件数を返す |
| `float_mmap(path)` | float のベクトルをファイルに移し、ディスクから読む形 (mmap) にする。RAM には int8 の行と索引だけが残る。`build()` の後に呼ぶ (グラフが要る)。区分け方式も使うなら `build_lists()` の後に呼ぶ (呼んだ後は作れない)。呼んだ後の `insert` / `compact` は `RuntimeError` |
| `save(path, key=None)` / `KciVec.load(path, key=None, rerank=100)` | 上を参照 |
| `plain_bytes()` | RAM 上のバイト数を種類ごとに: `codes`、`floats`、`derived`、`graph`、`ids`、`total` |
| `count`、`is_built()`、`has_lists`、`close()` | |

エラーは 2 種類です。ライブラリが断ったか失敗したときは `RuntimeError`、引数がおかしいときは `ValueError` です。

---

## 4. 始めるときの設定

ここに書いたのは出発点です。ANN-Benchmarks と VIBE の計測で使った設定から取りました。いちばん良い値はデータによって違います。
自分の質問を数百件用意し、`search_exact` の答えと比べて recall を測りながら、値を振って探してください。

| 目的 | まずこれで | 次に振る値 |
|---|---|---|
| 簡単に、小さい索引で | `build()`、`search(ef=100)` | `ef` 50〜800 |
| 速いグラフ、次元の低いデータ (100 次元くらい) | `M=16`、`build()`、`build_qg(1)`、`set_eps(0.15)`、`ef=30` | `eps` 0.1〜0.25、`ef` 20〜50 |
| 速いグラフ、文章の埋め込み (384 次元以上) | `M=8`、`build()`、`build_qg(2)`、`set_eps(0.1)`、`search(ef=30, rerank_n=0)` (int8 で並べ直し) | `eps` 0.05〜0.15、`ef` 20〜50。recall 0.99 なら `M=16` |
| 区分け方式 | `build_lists()`、`search(nprobe=64, rerank_n=200)` | `nprobe` 8〜512、`rerank_n` 50〜800 |
| 区分け方式をもっと速く | `set_probe_gap(0.25)` を足す | `gap` 0.15〜0.35 |
| 区分け方式、384 次元以上、高い recall | `build_lists(soar=1.5)`、`set_rerank("int8")` | `soar` あり・なしの両方 |
| RAM をいちばん少なく | `build()`、`build_lists()`、最後に `float_mmap(path)` | |

`ef`、`eps`、`nprobe`、`rerank_n` を大きくすると、recall は上がり、1 秒あたりに答えられる質問の数は減ります。

---

## 5. 守る決まり

- **書く人は一度に 1 人。** 探している最中の索引に、追加・挿入・削除・詰め直し・作り直しをしないでください。
- **スレッド (並行して動く処理)。**
  - Python では、1 つの `KciVec` オブジェクトが探索用の作業場所を 1 つだけ持ちます。同じオブジェクトの `search` を、複数のスレッドから同時に呼ばないでください。
  - C では、スレッドごとに `KciVecSearcher` を 1 つ作ります。作り終えて変更していない索引なら、複数の searcher が同時に動けます。
- **作る順番。** `build()` が先で、`build_lists()` と `build_qg()` は後です。後から `build()` を呼ぶと、両方とも消えます。
- **量子化グラフは保存されません。**
  - `load` の後は、`build_qg()` をもう一度呼んでください。
  - `insert`、`compact`、`build` でも消えます。
  - 絞り込み付きの検索は量子化グラフを使いません (ふつうのグラフで探します)。
- `float_mmap` の後は、float のベクトルが読むだけになります。要るものはすべて先に作ってください。
- 動く環境:
  - wheel は、AVX2 のある Linux x86-64 で動きます。AVX-512 があれば使います。
  - ソースからなら、macOS、Windows (MinGW)、ARM64 でも組み立てられます。

---

## 6. C の API

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

戻り値:
- 成功すると、件数か 0 を返します。
- 失敗すると負の数を返します。ふつうは `-1` です。2 つ目の理由 (「float が無い」など) を説明してある関数では `-2` もあります。
- `kci_vec_new` と `kci_vec_load` は、失敗すると `NULL` を返します。

| 分類 | 関数 |
|---|---|
| 作る・足す | `kci_vec_new`、`kci_vec_free`、`kci_vec_reserve`、`kci_vec_add`、`kci_vec_add_i8`、`kci_vec_keep_float`、`kci_vec_set_params` |
| グラフ | `kci_vec_build`、`kci_vec_search` (たまに探すとき)、`kci_vec_searcher_new` / `_free`、`kci_vec_search_r`、`kci_vec_search_rerank`、`kci_vec_search_filtered`、`kci_vec_searcher_set_eps` |
| 量子化グラフ | `kci_vec_build_qg`、`kci_vec_qg_bytes`、`kci_vec_searcher_set_qg` |
| 区分け方式 | `kci_vec_build_ivf`、`kci_vec_build_ivf_ex` (SOAR)、`kci_vec_build_ivf_p` (すべての設定)、`kci_vec_set_ivf_codes`、`kci_vec_set_ivf_training`、`kci_vec_search_ivf`、`kci_vec_search_ivf_filtered`、`kci_vec_searcher_set_probe_gap`、`kci_vec_searcher_set_rerank_int8` |
| 総当たり | `kci_vec_search_exact` |
| 変える | `kci_vec_insert`、`kci_vec_delete`、`kci_vec_forget` (そのベクトルの跡をすべて消す)、`kci_vec_compact`、`kci_vec_squeeze` |
| float | `kci_vec_float_mmap`、`kci_vec_set_float_source` (並べ直しのときに、自分の保管先から float を取ってくる)、`kci_vec_set_float`、`kci_vec_get_float` |
| 保存・読み込み | `kci_vec_save`、`kci_vec_load`、`kci_vec_save_ex` / `kci_vec_load_ex` (`KCI_VEC_SAVE_NO_FLOAT`、`KCI_VEC_SAVE_RAW_KEY`) |
| 読むだけの共有イメージ | `kci_vec_image_save`、`kci_vec_attach`、`kci_vec_attach_mem`、`kci_vec_thaw` |
| メモリ | `kci_vec_protect` (ベクトルの置き場をコアダンプとスワップに出さないよう OS に頼む。できる範囲で)、`kci_vec_plain_bytes` |
| 情報 | `kci_vec_count`、`kci_vec_dim`、`kci_vec_is_built` |

それぞれの引数、戻り値、条件のくわしい説明は、`kci_vec.h` の各宣言の上のコメントにあります。
