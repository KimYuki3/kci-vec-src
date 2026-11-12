# KCI-Vec — source

**Fast, compact vector search you embed in your application.** One C file (`kci_vec.c` + `kci_vec.h`), no
dependencies, no server.

*[日本語](README_ja.md)*

This repository holds the **source** of KCI-Vec. The evaluation wheel, the benchmark tables and charts, and the
comparison with other libraries are in **[KimYuki3/kci-vec-dist](https://github.com/KimYuki3/kci-vec-dist)**.

## What is in it

- **Two indexes over int8 vectors** (one byte per dimension):
  - *partitioned* (`kci_vec_build_ivf` / `kci_vec_search_ivf`): k-means lists, 4-bit codes scored with SIMD table
    lookups (32 at a time on AVX2, 64 on AVX-512), probe gap, optional SOAR, float or int8 re-ranking;
  - *graph* (`kci_vec_build` / `kci_vec_search*`): HNSW with integer dot products, ε-radius search
    (`kci_vec_searcher_set_eps`), and an optional quantized layer 0 (`kci_vec_build_qg`) that keeps each node's
    neighbours' 4-bit codes in one block.
- Insert, delete and compact after the build; a filter predicate on ids during search.
- Optional encrypted saves (AES-256-CTR + AES-256-CMAC; wrong keys and altered files are refused) and a read-only
  image that several processes can map.
- A Python client (`clients/python`, ctypes + NumPy) with a small BM25 index for hybrid keyword + vector ranking.
- An ANN-Benchmarks component (`integrations/ann-benchmarks`).

**Not in this release:** the L2 (Euclidean) index — it is in the evaluation wheel, and its source will follow later;
this source is cosine only.

## Build

```sh
sh build_kci_vec.sh          # libkci_vec.a, libkci_vec.so (or .dylib) and ./example_kci_vec
./example_kci_vec
```

Linux and macOS (x86-64 with AVX2 / AVX-512, or ARM64). Windows (MinGW): `gcc -O3 -march=native -shared -static -static-libgcc -o libkci_vec.dll kci_vec.c` (`kci_win_compat.h` supplies the POSIX parts).

## Tests

```sh
gcc -O2 -march=native -I. tests/test_qg.c kci_vec.c -lpthread -lm -o test_qg && ./test_qg
```

Each file in `tests/` has its build line at the top (some include `kci_vec.c` themselves). Python:
`KCI_VEC_LIB=$PWD/libkci_vec.so python3 clients/python/test_smoke.py` and `test_hybrid.py`.

## Use from C

```c
#include "kci_vec.h"

KciVec *v = kci_vec_new(dim, 16, 200);                 /* dim, HNSW M, efConstruction */
for (...) kci_vec_add(v, id, vec);                       /* float32[dim]; normalized inside (cosine) */
kci_vec_build(v, 0, NULL);                               /* graph, all cores */
int n = kci_vec_search(v, query, 10, 100, ids, scores);  /* top 10, ef 100; scores are cosine */
kci_vec_save(v, "index.kv", key, keylen);                /* encrypted when a key is given */
```

The full API, with what each option does and the measurements behind the defaults, is commented in `kci_vec.h`.
Comments in `kci_vec.c` refer to measurement records (`bench/results_*.txt`) that are not part of this release; the
figures that matter are in the kci-vec-dist README.

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free for research, education, personal and other non-commercial use.
Commercial use needs a separate license — open an issue in
[kci-vec-dist](https://github.com/KimYuki3/kci-vec-dist/issues) saying so, and a private contact will follow.

Required Notice: Copyright 2026 Yukihiro Kimura (https://github.com/KimYuki3)
