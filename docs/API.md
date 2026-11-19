# KCI-Vec API guide

How to use KCI-Vec from Python and from C: what each call does, which settings to start from, and the rules to keep.
日本語: [API_ja.md](API_ja.md)

- **Python** — the `kci_vec` package. Install the wheel from [kci-vec-dist](https://github.com/KimYuki3/kci-vec-dist)
  (Linux x86-64), or build it from `clients/python` in this repository (`pip install .`, needs gcc or clang).
- **C** — `kci_vec.h` + `kci_vec.c`, one file, no dependencies. Every function is also commented in `kci_vec.h`.

Contents: [1. The two indexes](#1-the-two-indexes) · [2. Python step by step](#2-python-step-by-step) ·
[3. Python reference](#3-python-reference) · [4. Settings to start from](#4-settings-to-start-from) ·
[5. Rules](#5-rules) · [6. C API](#6-c-api)

---

## 1. The two indexes

| | **Graph** (`build()`) | **Partitioned** (`build_lists()`) |
|---|---|---|
| How it searches | Walks a graph (HNSW) from node to nearest node | Groups the vectors (k-means) and scans only the groups nearest to the query |
| Knob for recall | `ef` (and `eps`) | `nprobe` and `rerank_n` |
| Fast option | **Quantized graph** (`build_qg()`) | **Probe gap** (`set_probe_gap()`) |
| Needs the float vectors | Only for re-ranking and `insert` | Always (`rerank > 0`) |
| Insert / delete | Yes | Yes |
| Saved with `save()` | Yes (the quantized graph is **not**; see [Rules](#5-rules)) | Yes |

Both use cosine similarity by default; the vectors are normalized when they are added. Euclidean (L2) distance (`metric="l2"`) and inner product
(`metric="ip"`) are there too.

Which to choose: try both on your data. The partitioned index needs no graph build and its build is simpler to
tune; the quantized graph gave the highest speeds in the benchmarks but takes more memory.

---

## 2. Python step by step

### Build and search

```python
import numpy as np
from kci_vec import KciVec

v = KciVec(dim=384, rerank=200)        # rerank > 0 keeps the float vectors (needed by the partitioned index)
v.add_many(ids, embeddings)            # ids: your own integer keys (uint64); embeddings: float32 [n, dim]
v.build_lists()                        # partitioned index   (or v.build() for the graph)

hits = v.search(query, k=10)           # [(id, score), ...], best first; score = cosine similarity
ids, scores = v.search_many(queries, k=10)   # arrays [nq, k]; -1 / -inf where fewer than k were found
```

### Filter by metadata or access rights

```python
hits = v.search(query, k=10, filter={3, 17, 42})          # only these ids
hits = v.search(query, k=10, filter=lambda i: i % 2 == 0) # or any function id -> bool
```

The filter is applied during the search, not after it, so a filter that keeps few vectors still returns its nearest
matches. When the filter keeps only a small share of the vectors, raise `nprobe` / `rerank_n` (partitioned) or `ef`
(graph); when it keeps only a handful, use `search_exact` and filter yourself.

### Add, delete, compact

```python
v.insert(new_id, vec)    # into the built index (needs rerank > 0)
v.delete(old_id)         # marked deleted: no longer returned
v.compact()              # rebuild over the live vectors and reclaim the space
```

An update is `delete` then `insert`. For a large batch, `add_many` + `build*` again is faster than many inserts.

### Save and load, with or without encryption

```python
v.save("index.kv", key=b"a long secret")       # AES-256-CTR + AES-256-CMAC; key=None saves in plain
w = KciVec.load("index.kv", key=b"a long secret", rerank=200)
```

A wrong key, or a damaged or altered file, raises `RuntimeError` at `load` (the tag is checked before anything is read).
`load` does not know the `rerank` you used: pass it again.

### The fast graph (quantized graph)

```python
v = KciVec(dim=384, M=8, rerank=200)
v.add_many(ids, embeddings)
v.build()                 # the graph first
v.build_qg(dims_per_code=2)   # then the quantized bottom layer; returns its size in bytes
v.set_eps(0.1)            # stop rule (see below)
hits = v.search(query, k=10, ef=30)
```

After `load`, `insert`, `compact` or `build`, call `build_qg()` again: the quantized graph is not saved and those calls
drop it.

### Hybrid search (words + vectors)

```python
from kci_vec.hybrid import BM25Index, hybrid_search

bm = BM25Index()
for i, text in zip(ids, texts):
    bm.add(i, text)                    # the same ids as the vectors
hits = hybrid_search(v, bm, query_vec, "KX-204B invoice", k=5)
# [(id, fused_score, vector_rank, word_rank)], best first; a rank is None if that side did not find the id
bm.save("index.bm25.json"); bm = BM25Index.load("index.bm25.json")
```

Product codes, names and numbers that embeddings miss are found by the word side. Japanese, Chinese and Korean are
split into two-character pieces, so no dictionary is needed. The word index is plain JSON: if the text is
sensitive, keep that file encrypted yourself.

### Euclidean (L2) distance

```python
v = KciVec(dim=128, rerank=200, metric="l2")
```

Scores are then squared distances, smallest first. Everything else works as for cosine. `max_norm` (default 0) is
the largest vector length the index will accept; 0 takes it from the vectors added before the first build, and a
later `insert` of a much longer vector fails. The L2 source is in this repository from version 0.2.3.

---

## 3. Python reference

### `KciVec(dim, M=16, efc=200, rerank=0, metric="cosine", max_norm=0.0)`

| argument | meaning |
|---|---|
| `dim` | vector length (any length) |
| `M`, `efc` | graph build settings: neighbours per node and the build-time search width. 16 / 200 are good defaults; 8 suits 384-dim text embeddings with the quantized graph |
| `rerank` | > 0 keeps the float vectors and is the default shortlist re-scored exactly. Needed for `build_lists`, `insert`, `compact` |
| `metric` | `"cosine"`, `"l2"` or `"ip"` (inner product) |

### Adding

| method | |
|---|---|
| `add(id, vec)` | one vector (float32, length `dim`) |
| `add_many(ids, mat)` | many at once; `mat` is [n, dim] |
| `reserve(n)` | make room for `n` vectors first (fewer reallocations) |

### Building

| method | |
|---|---|
| `build(threads=0)` | the graph. 0 = all cores. Returns the seconds taken. Drops the partitioned index if one was built, so build the graph **first** if you want both |
| `build_qg(dims_per_code=1, threads=0)` | quantized bottom layer of the graph (after `build` or `load`). `dims_per_code` 1 = finer, 2 = half the size. Returns its bytes |
| `build_lists(nlist=0, soar=0.0, threads=0, nprobe=None, dims_per_code=0, avq_threshold=0.0, train_iters=0, train_sample=0)` | the partitioned index. `nlist` 0 = 2√n groups. `soar` 1.5 puts each vector in a second group too. `nprobe` sets the default for searches. The rest change how the codes and the groups are trained; 0 = automatic |

### Search settings (kept until changed)

| method | |
|---|---|
| `set_eps(eps)` | graph: keep the result set at `ef` and go on while a candidate is within (1 + eps) of the current bound. 0 = off (plain HNSW). Use with a small `ef` (k to 3k) |
| `set_probe_gap(gap, min_probe=8)` | partitioned: of the best `nprobe` groups, scan only those whose score is within `gap` of the nearest, but at least `min_probe`. 0 = off |
| `set_rerank("float" \| "int8")` | partitioned index: re-score the shortlist with the floats (exact, default) or the int8 rows (a quarter of the bytes; fine for text embeddings at k = 100, not for low-dimensional data such as glove-100). For the graph, pass `rerank_n=0` to `search` instead |

### Searching

| method | |
|---|---|
| `search(query, k=10, ef=None, nprobe=None, rerank_n=None, filter=None, method="auto")` | top `k` as `[(id, score)]`. `method`: `"auto"` (partitioned if built, else graph), `"lists"`, `"graph"`. `ef` defaults to max(k, 100), `nprobe` to 32, `rerank_n` to `rerank` (at least 2k for the partitioned index). Graph: `rerank_n` 0 re-scores the result with the int8 rows, > 0 with the floats. `k` up to 4,096 |
| `search_many(queries, k=10, **kw)` | the same for many queries (one after another); returns `(ids, scores)` arrays |
| `search_exact(query, k=10, threads=0)` | brute force: exact answer, for checking recall or for tiny indexes |

### Changing, saving, other

| method | |
|---|---|
| `insert(id, vec)` | add to the built index. Raises `RuntimeError` without `rerank` |
| `delete(id)` | returns `True` if the id was there |
| `compact()` | rebuild over live vectors; returns their count |
| `float_mmap(path)` | move the float vectors to a file mapped from disk, so RAM holds only the int8 rows and the index. Call it after `build()` (it needs the graph), and after `build_lists()` if you want the partitioned index too: lists cannot be built after it. Then `insert` / `compact` raise `RuntimeError` |
| `save(path, key=None)` / `KciVec.load(path, key=None, rerank=100)` | see above |
| `plain_bytes()` | bytes in RAM by kind: `codes`, `floats`, `derived`, `graph`, `ids`, `total` |
| `count`, `is_built()`, `has_lists`, `close()` | |

Errors are `RuntimeError` (the library refused or failed) or `ValueError` (a bad argument).

---

## 4. Settings to start from

These are starting points taken from the settings used in the ANN-Benchmarks and VIBE runs. The best values depend on
your data: measure recall against `search_exact` on a few hundred of your own queries and sweep.

| goal | start with | then sweep |
|---|---|---|
| Simple, small index | `build()`, `search(ef=100)` | `ef` 50–800 |
| Fast graph, low-dim data (≈100 dims) | `M=16`, `build()`, `build_qg(1)`, `set_eps(0.15)`, `ef=30` | `eps` 0.1–0.25, `ef` 20–50 |
| Fast graph, text embeddings (384+ dims) | `M=8`, `build()`, `build_qg(2)`, `set_eps(0.1)`, `search(ef=30, rerank_n=0)` (int8 re-score) | `eps` 0.05–0.15, `ef` 20–50; `M=16` for recall 0.99 |
| Partitioned | `build_lists()`, `search(nprobe=64, rerank_n=200)` | `nprobe` 8–512, `rerank_n` 50–800 |
| Partitioned, faster | add `set_probe_gap(0.25)` | `gap` 0.15–0.35 |
| Partitioned, 384+ dims, high recall | `build_lists(soar=1.5)`, `set_rerank("int8")` | with and without `soar` |
| Least RAM | `build()`, `build_lists()`, then `float_mmap(path)` | |

Larger `ef`, `eps`, `nprobe`, `rerank_n` give higher recall and fewer queries per second.

---

## 5. Rules

- **One writer at a time.** Do not add, insert, delete, compact or build while searching the same index.
- **Threads.** In Python, one `KciVec` object holds one search context: do not call `search` on the same object from
  several threads at once. In C, make one `KciVecSearcher` per thread; several searchers may run at once on a built
  index that is not being changed.
- **Order of builds.** `build()` first, then `build_lists()` / `build_qg()`. A later `build()` drops both.
- **The quantized graph is not saved.** After `load`, call `build_qg()` again. `insert`, `compact` and `build` drop
  it too, and filtered search does not use it (it falls back to the plain graph).
- `float_mmap` makes the float store read-only: build everything you need before it.
- The wheel runs on Linux x86-64 with AVX2 (AVX-512 is used when present). The source also builds on macOS and
  Windows (MinGW) and on ARM64.

---

## 6. C API

```c
#include "kci_vec.h"

KciVec *v = kci_vec_new(dim, 16, 200);                  /* dim, M, efConstruction */
kci_vec_keep_float(v, 1);                               /* before any add: keep floats for re-rank / lists */
for (...) kci_vec_add(v, id, vec);                      /* float32[dim], normalized inside */
kci_vec_build(v, 0, NULL);                              /* graph, all cores */

KciVecSearcher *s = kci_vec_searcher_new(v);            /* one per thread, reused across queries */
int n = kci_vec_search_rerank(s, query, 10, 100, 200, ids, scores);   /* k, ef, rerank_n */

kci_vec_save(v, "index.kv", key, keylen);               /* encrypted when key != NULL */
kci_vec_searcher_free(s);
kci_vec_free(v);
```

Return values: counts or 0 on success; negative on error (`-1`; `-2` where a function documents a second reason, such
as "no float store"). `kci_vec_new` / `kci_vec_load` return `NULL` on failure.

| group | functions |
|---|---|
| Create, add | `kci_vec_new`, `kci_vec_free`, `kci_vec_reserve`, `kci_vec_add`, `kci_vec_add_i8`, `kci_vec_keep_float`, `kci_vec_set_params` |
| Graph | `kci_vec_build`, `kci_vec_search` (one-off), `kci_vec_searcher_new` / `_free`, `kci_vec_search_r`, `kci_vec_search_rerank`, `kci_vec_search_filtered`, `kci_vec_searcher_set_eps` |
| Quantized graph | `kci_vec_build_qg`, `kci_vec_qg_bytes`, `kci_vec_searcher_set_qg` |
| Partitioned | `kci_vec_build_ivf`, `kci_vec_build_ivf_ex` (SOAR), `kci_vec_build_ivf_p` (all options), `kci_vec_set_ivf_codes`, `kci_vec_set_ivf_training`, `kci_vec_search_ivf`, `kci_vec_search_ivf_filtered`, `kci_vec_searcher_set_probe_gap`, `kci_vec_searcher_set_rerank_int8` |
| Exact | `kci_vec_search_exact` |
| Change | `kci_vec_insert`, `kci_vec_delete`, `kci_vec_forget` (erase every trace of a vector), `kci_vec_compact`, `kci_vec_squeeze` |
| Floats | `kci_vec_float_mmap`, `kci_vec_set_float_source` (fetch floats from your own store at re-rank time), `kci_vec_set_float`, `kci_vec_get_float` |
| Save, load | `kci_vec_save`, `kci_vec_load`, `kci_vec_save_ex` / `kci_vec_load_ex` (`KCI_VEC_SAVE_NO_FLOAT`, `KCI_VEC_SAVE_RAW_KEY`) |
| Shared read-only image | `kci_vec_image_save`, `kci_vec_attach`, `kci_vec_attach_mem`, `kci_vec_thaw` |
| Memory | `kci_vec_protect` (keep vector buffers out of core dumps and swap, best effort), `kci_vec_plain_bytes` |
| Info | `kci_vec_count`, `kci_vec_dim`, `kci_vec_is_built` |

The comment above each declaration in `kci_vec.h` gives its arguments, return values and conditions in full.
