# KCI-Vec — ソース

**アプリに組み込む、速くて軽いベクトル検索。** C の 1 ファイル (`kci_vec.c` + `kci_vec.h`)、依存なし、サーバ不要。

*[English](README.md)*

このリポジトリは KCI-Vec の **ソース** です。評価用の wheel、速さの表とグラフ、ほかのライブラリとの比べは
**[KimYuki3/kci-vec-dist](https://github.com/KimYuki3/kci-vec-dist)** にあります。

## 入っているもの

- **int8 のベクトル (1 次元 1 バイト) の上の 2 つの索引**:
  - *区分け方式* (`kci_vec_build_ivf` / `kci_vec_search_ivf`): k-means で分けたかたまり、4 ビット符号を SIMD の表引きで
    見積もる (AVX2 で 32 個、AVX-512 で 64 個ずつ)、probe gap、SOAR (選択肢)、float か int8 での並べ直し。
  - *グラフ方式* (`kci_vec_build` / `kci_vec_search*`): 整数の内積で動く HNSW、ε 半径の探索 (`kci_vec_searcher_set_eps`)、
    最下層を「近傍の 4 ビット符号の塊」として持つ量子化グラフ (`kci_vec_build_qg`、選択肢)。
- 作った後の追加・削除・詰め直し、検索中の id への条件での絞り込み。
- 暗号化した保存 (AES-256-CTR + AES-256-CMAC。鍵違いや書き換えられたファイルは読み込まない。選択肢)、
  いくつものプロセスで共有できる読み取り専用の像。
- Python の包み (`clients/python`、ctypes + NumPy)。キーワードとベクトルを合わせる小さな BM25 付き。
- ANN-Benchmarks の部品 (`integrations/ann-benchmarks`)。

**この版に入っていないもの:** L2 (ユークリッド距離) の索引。評価用の wheel では使えます。ソースは後で公開します。
このソースは cosine だけです。

## 組み立て

```sh
sh build_kci_vec.sh          # libkci_vec.a、libkci_vec.so (または .dylib)、./example_kci_vec
./example_kci_vec
```

Linux と macOS (AVX2 / AVX-512 のある x86-64、または ARM64)。Windows は MinGW で `gcc -O3 -march=native -shared -static -static-libgcc -o libkci_vec.dll kci_vec.c`
(POSIX の部分は `kci_win_compat.h` が補います)。

## 試験

```sh
gcc -O2 -march=native -I. tests/test_qg.c kci_vec.c -lpthread -lm -o test_qg && ./test_qg
```

`tests/` の各ファイルの頭に組み立て方が書いてあります (`kci_vec.c` を自分で取り込むものもあります)。Python は
`KCI_VEC_LIB=$PWD/libkci_vec.so python3 clients/python/test_smoke.py` と `test_hybrid.py`。

## C からの使い方

```c
#include "kci_vec.h"

KciVec *v = kci_vec_new(dim, 16, 200);                 /* 次元、HNSW の M、efConstruction */
for (...) kci_vec_add(v, id, vec);                       /* float32[dim]。中で正規化 (cosine) */
kci_vec_build(v, 0, NULL);                               /* グラフ、全コア */
int n = kci_vec_search(v, query, 10, 100, ids, scores);  /* 上位 10、ef 100。点数は cosine */
kci_vec_save(v, "index.kv", key, keylen);                /* 鍵を渡せば暗号化 */
```

API の全体と、各選択肢の意味、既定値の根拠になった計測は `kci_vec.h` の注釈にあります。
`kci_vec.c` の注釈が参照している計測の記録 (`bench/results_*.txt`) は、この版には入っていません。主な数字は kci-vec-dist の README にあります。

## ライセンス

[PolyForm Noncommercial 1.0.0](LICENSE): 研究・教育・個人・その他の非商用の利用は無償です。商用の利用には別の契約が必要です。
[kci-vec-dist](https://github.com/KimYuki3/kci-vec-dist/issues) の Issue に一言書いていただければ、非公開の連絡方法をご案内します。

Required Notice: Copyright 2026 Yukihiro Kimura (https://github.com/KimYuki3)
