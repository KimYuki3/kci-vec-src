# Using KCI-Vec (API guide)

Version 0.2.15 (2026-11-23). 日本語: [API_ja.md](API_ja.md)

How to use KCI-Vec from Python and from C: what each function does, which settings to start from, and the rules to keep.
If you are new, read [the first ten minutes](quickstart_text.md) first.

```sh
pip install kci-vec          # Linux x86-64, Windows x86-64, macOS (from 0.2.16). Python 3.10 or later
```

To build from source, run `pip install .` in `clients/python` of this repository (needs gcc or clang).
From C, add the single file `kci_vec.c` and its headers (`kci_vec.h`, `kci_crypt.h`) to your program. There are no dependencies.

There are two ways to use it.

| | What you use | When it fits |
|---|---|---|
| **Hand over plain text** | `kci_vec.text.TextIndex` | You want to put texts in and search with a question, and leave vectors and models to the library |
| **Bring your own vectors** | `kci_vec.KciVec` | You make the embeddings yourself, handle things other than text (images), or want the fine settings |

Contents: [1. Plain text](#1-plain-text-textindex) · [2. Your own vectors](#2-your-own-vectors-kcivec) ·
[3. KciVec reference](#3-kcivec-reference) · [4. Combining with word search](#4-combining-with-word-search) ·
[5. Settings to start from](#5-settings-to-start-from) · [6. Rules](#6-rules) · [7. C API](#7-c-api) · [8. PostgreSQL](#8-postgresql)

---

## 1. Plain text (TextIndex)

You hand over texts; they are turned into vectors and indexed inside. The index has a side that searches by meaning and a
side that searches by words, and returns the merged result. To run the models you also need:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

### Put texts in and search

```python
from kci_vec.text import TextIndex

ix = TextIndex("en", sparse="bm25")          # "en" or "ja". The very first run fetches the language model
ix.add([
    "Paid leave: 10 days are granted after six months of service. Apply three business days ahead.",
    "Expense reports close at the end of the month. Hand the original receipts to general affairs.",
    "Meeting rooms are booked in the company calendar, for up to two hours at a time.",
])                                           # numbered 0, 1, 2 ...; pass ids=[...] for your own numbers

for hit in ix.search("how many days of paid leave do I get?", k=2):
    print(hit.id, round(hit.score, 3), hit.text)
```

`search` returns a list of `Hit`, best first.

| Field | Meaning |
|---|---|
| `id` | the text's number |
| `score` | the merged score (larger is better) |
| `text` | the part that matched |
| `dense_rank`, `sparse_rank` | its rank on the meaning side and on the word side (`None` if that side did not find it) |
| `span` | for a long text cut into parts: where the matched part is (start, end) |

### Add, delete, save

```python
new_id = ix.add(["Remote work is allowed up to two days a week. Apply by the day before."])[0]   # returns the numbers given
ix.delete(1)                                 # remove text 1; True if it was there

ix.save("my_index", key="a passphrase")      # into a folder. With key, the index and the texts are encrypted
ix2 = TextIndex.load("my_index", key="a passphrase")   # a wrong passphrase raises RuntimeError
print(len(ix2), ix2.search("how many days of remote work?", k=1)[0].text)
```

- Without `key` the texts are written to `texts.json` as they are.
- To change a text, `delete` it and `add` the new one.

### Main arguments of `TextIndex(lang="ja", mode="hybrid", ...)`

| Argument | Meaning |
|---|---|
| `lang` | `"ja"` or `"en"`: which built-in models to use |
| `mode` | `"hybrid"` (meaning and words, the default), `"dense"` (meaning only), `"sparse"` (words only) |
| `sparse="bm25"` | the word side counts character sequences instead of running a language model. Lighter, and one model fewer to fetch |
| `dense=`, `sparse=` | your own functions instead of the built-in models (with a `sparse=` function also give `sparse_dim=`) |
| `w_dense` | the meaning side's weight in the merge (0..1). Default: the preset for the language |
| `chunk`, `overlap` | texts longer than this many characters are indexed in parts; `overlap` is how many characters neighbouring parts share. `chunk=0`: never cut |
| `dedup=True` | for data where the same text repeats (boilerplate, empty fields): keep each one once |
| `keep_text=False` | do not keep the texts in the index (it returns numbers only; you keep the texts) |
| `device` | where the models run. Default: CPU |
| `verbose=False` | no progress messages |

### TextIndex functions

| Function | |
|---|---|
| `add(texts, ids=None, batch=256)` | add texts; returns the numbers given |
| `delete(id)` | remove one; `True` if it was there |
| `build()` | make the index searchable. `search` calls it by itself the first time; call it yourself only to choose when the waiting happens |
| `search(query, k=5, filter=None, rerank=None, select=None, ...)` | as above. `filter`: the set of numbers that may be returned, or a function number -> bool |
| `candidates(query, depth_dense=50, depth_sparse=100, filter=None)` | what each side found, unmerged, for a re-ranker of your own |
| `tune(questions, answers, k=10)` | given about 50 questions whose right answers you know, fits the meaning side's weight to your texts. Returns `(weight, table)` |
| `save(path, key=None)` / `TextIndex.load(path, key=None, ...)` | as above |
| `len(ix)` | how many texts are in it |

`search(rerank=...)` takes a function `(question, [texts]) -> one score per text` (a re-ranking model of your own).
`search(select=...)` takes a function `(question, [texts]) -> the positions of the useful texts`. Neither is used unless given.

---

## 2. Your own vectors (KciVec)

### Two indexes

An index is the prepared data that makes search fast. There are two kinds.

| | **Graph** (`build()`) | **Partitioned lists** (`build_lists()`) |
|---|---|---|
| How it searches | walks a net of vectors joined to their near ones, toward the query | splits the vectors into partitions and scans only those near the query |
| Knobs for accuracy | `ef` (and `eps`) | `nprobe` and `rerank_n` |
| Option for speed | quantized graph (`build_qg()`) | `set_probe_gap()` |
| Needs the float vectors | only for the exact pass and `insert` | always (`rerank > 0`) |
| Add, delete, save | yes | yes |

By default vectors are compared by cosine similarity, and are scaled to length 1 when added.
Euclidean distance (`metric="l2"`) and inner product (`metric="ip"`) are available too.

Which one suits depends on the data: try both on yours. The lists give a smaller index and are easy to tune.
The quantized graph is fast and uses more memory.

### Build and search

```python
import numpy as np
from kci_vec import KciVec

rng = np.random.default_rng(0)
embeddings = rng.standard_normal((2000, 384)).astype(np.float32)   # replace with your own embeddings
ids = np.arange(2000)                                              # integer numbers of your choice
query = embeddings[7]; queries = embeddings[:5]

v = KciVec(dim=384, rerank=200)        # rerank > 0 keeps the float vectors (required for the lists)
v.add_many(ids, embeddings)            # embeddings: float32 [n, dim]
v.build_lists()                        # partitioned lists   (for the graph: v.build())

hits = v.search(query, k=10)           # [(id, score), ...], best first; score is the cosine similarity
found, scores = v.search_many(queries, k=10)   # arrays [nq, k]; -1 / -inf where fewer than k were found
```

### Filter by attribute or access right

```python
hits = v.search(query, k=10, filter={3, 17, 42})          # only these numbers
hits = v.search(query, k=10, filter=lambda i: i % 2 == 0) # or a function id -> bool
```

The filter applies during the search, not after it, so even when few items pass, the nearest among them are returned.

- When only a small share passes: raise `nprobe` and `rerank_n` (lists) or `ef` (graph).
- When only a handful pass: nothing to set. Given a set, list or array of numbers that is small enough, `search` compares the query with the vectors that pass, whatever the size of the index. With any filter, a search that finds fewer than `k` answers is answered again by brute force over the ones that pass, so none of them is left out (since 0.2.19; a function is then called once per vector). `search_exact(query, k, filter=...)` does the same on request.

### Insert, delete, compact

```python
v.insert(5000, embeddings[0])   # add to a built index (needs rerank > 0)
v.delete(5000)                  # no longer returned; True if it was there
v.compact()                     # rebuild from the live vectors and free the space
```

To change a vector, `delete` then `insert`. For many additions, `add_many` and a rebuild are faster than repeated `insert`.

### Save and load (encrypted or not)

```python
v.save("index.kv", key=b"a long secret")       # AES-256-CTR + AES-256-CMAC; key=None: not encrypted
w = KciVec.load("index.kv", key=b"a long secret", rerank=200)
```

- A wrong key, a damaged file or an altered file makes `load` raise `RuntimeError`. The tamper check runs before anything is read.
- `load` does not know the `rerank` the index was made with: pass it again.
- The lists and the quantized graph are in the file. A loaded index uses them without building again.

### The fast graph (quantized graph)

```python
g = KciVec(dim=384, M=8, rerank=200)
g.add_many(ids, embeddings)
g.build()                     # the graph first
g.build_qg(dims_per_code=2)   # then the quantized graph; returns the bytes it takes
g.set_eps(0.1)                # the stop rule (see section 3)
hits = g.search(query, k=10, ef=30)
```

`insert`, `compact` and `build` drop the quantized graph. Call `build_qg()` again after them.

### Using less memory

An index that is only searched (nothing added or deleted later) can give up what it does not need.

```python
g.half_floats()     # the floats for the exact pass become 16-bit: half the size
g.search_only()     # gives up what a quantized-graph search does not read
hits = g.search(query, k=10, ef=30, rerank_n=100)
```

- Neither can be undone. After them `insert`, `delete` and `save` no longer work: save before you call them.
- `search_only()` needs an index with `build()` and `build_qg()` done. Afterwards pass `rerank_n` to `search`.
- The other way is `float_mmap(path)`: the float vectors move to a file and only the index stays in memory.

### Data where the same vector repeats

```python
d = KciVec(dim=384, rerank=200, dedup=True)   # identical vectors are kept once, with their numbers beside them
d.add_many(ids, embeddings)
d.build()
print(d.dup_stats())    # how many rows are copies: rows, groups, rows_in_groups, largest, distinct ...
d.unique = True         # one result per distinct vector (default: every number of an identical vector is returned)
```

Boilerplate and empty fields give many identical vectors. `dedup=True` is for such data.

### A read-only index shared by several processes

```python
v.save_image("index.img")                    # written in a read-only form, not encrypted
r = KciVec.attach("index.img", rerank=200)   # opened without copying; processes that open the same file share its memory
hits = r.search(query, k=10)
```

- An attached index is read-only: nothing can be added, deleted or rebuilt.
- This file is not encrypted. For an encrypted file use `save(path, key)`.
- Open it on the same kind of machine that wrote it.

### Euclidean (L2) distance and inner product

```python
e = KciVec(dim=384, rerank=200, metric="l2")     # scores are squared distances, smallest first
p = KciVec(dim=384, rerank=400, metric="ip")     # scores are inner products, largest first; vectors keep their length
```

- The other functions work as with cosine.
- `max_norm` (default 0) is the longest vector the index accepts. 0: taken from the vectors added before the first build.
  A longer vector inserted later is refused.
- For inner product use a larger `rerank` (400 or more).

---

## 3. KciVec reference

### `KciVec(dim, M=16, efc=200, rerank=200, metric="cosine", max_norm=0.0, centre=False, dedup=False)`

| Argument | Meaning |
|---|---|
| `dim` | the length of the vectors |
| `M`, `efc` | graph build settings: links per point and the search width while building. 16 / 200 is a safe default |
| `rerank` | 200 by default: the float vectors are kept and every search scores its best 200 candidates exactly. Needed for `build_lists`, `insert`, `compact`. `rerank=0` keeps the one-byte rows only (5 bytes a dimension → 1, and a lower recall). Up to 0.2.16 the default was 0 |

A small index is answered by brute force. While count × dim is at most `exact_cells` (10,000,000 by default: 10,000 vectors of 1,024 dimensions, or 26,000 of 384) and the index holds the float vectors, `search` compares the query with every vector and the answer is exact (`ef`, `nprobe` and `rerank_n` are not used). With a `filter` it compares the vectors whose numbers pass (in 0.2.17 a filtered search still used the index). With `method="graph"` / `"lists"` the index's own search is used. `v.exact_cells = 0` turns it off. Since 0.2.19 this covers inner product (`metric="ip"`) too.

A larger index measures the search width it needs (since 0.2.19). At the end of `build`, a sample of the stored vectors is used as questions and compared with brute force; the narrowest of `ef` 200 / 400 / 800 / 1600 that finds 98 % of the true 10 nearest becomes the default of `search` (the widest when none does), together with the number of candidates scored exactly when 200 are not enough. So a `search` with nothing given is wider, and slower, on data that needs it, and unchanged on data that does not. `save` / `load` keep the result. `build` warns when even the widest setting stays below 97 %, and when rows are left that the graph search cannot reach. The sample is made of stored vectors: when your questions are of another kind (text questions over image vectors, most inner-product data), pass some of them to `calibrate(queries=...)`. An `ef` or `rerank_n` you give is used as given. `v.auto_ef = False` before `build` turns the measurement off.
| `metric` | `"cosine"`, `"l2"`, `"ip"` (inner product) |
| `max_norm` | for `l2` and `ip`: the longest vector accepted. 0 = automatic |
| `centre` | `ip` only: subtract the mean of the vectors before indexing. The ranking does not change |
| `dedup` | keep identical vectors once |

### Adding

| Function | |
|---|---|
| `add(id, vec)` | one vector (float32, length `dim`) |
| `add_many(ids, mat)` | many; `mat` is [n, dim] |
| `reserve(n)` | reserve room for `n` vectors |

### Building

| Function | |
|---|---|
| `build(threads=0, fast=False)` | build the graph. 0 = all cores. Returns the seconds taken. It drops the lists, so if you want both, build **the graph first**. `fast=True`: a build that favours build time |
| `build_qg(dims_per_code=1, threads=0, code_dims=0, gate_dims=0, residual=False)` | build the quantized graph (after `build` or `load`). `dims_per_code`: 1 = fine, 2 = half the size. Returns the bytes it takes. The last three are for a smaller index on high-dimensional data (below) |
| `build_lists(nlist=0, soar=0.0, threads=0, nprobe=None, ...)` | build the partitioned lists. `nlist` 0 = automatic. `soar` 1.5 puts every vector in a second partition too. `nprobe` is the default for searches. The remaining arguments: 0 = automatic |

| `calibrate(k=10, target=0.98, samples=128, threads=0, queries=None)` | measure the search width this index needs and make it the default of `search` (see above; `build` calls it for an index above `exact_cells`). `queries`: 8 or more of your own questions instead of stored vectors. Returns `(ef, the share of the true k nearest found)`; `default_ef()`, `default_rerank()` and `calibrated_recall` give the result back |
| `judge(k=10, target=0.99, samples=4096, queries=None)` | (since 0.2.20) let the index decide the search width for each query: after it, a `search` without `ef` can answer the queries that need little with less work; a search with an explicit `ef` is unchanged. `k`: the number of results your searches ask for (2 to 64). `queries`: 200 or more of your own questions instead of stored vectors. Returns `(in_use, gain)`; `in_use` False means it was measured not to help on this index and `search` stays as it was. For graph searches without a filter, `cosine` and `l2`. Not saved: call it again after `load` or `build`. `judge(samples=0)` turns it off |
| `choose_method(k=10, target=0.98, queries=None, samples=1000, nlist=0, threads=0)` | (since 0.2.20) measure which of the two ways of searching suits this data, the graph or the partitioned lists, and make it what `search` does when no `method` is given. Builds the lists if they are not there, searches a sample both ways against the exact answer, and keeps the faster setting that reaches `target` (the share of the true `k` nearest found). Returns a dict with `method` and, for `graph` and `lists`, the setting, its recall and its milliseconds a query. The times are taken on this machine with one thread, so run it when the machine is quiet. The sample is made of stored vectors unless you give `queries`; with your own questions the result is closer to what you will see. Not saved: call it again after `load` |

The `build_qg` settings for a smaller index (for data of 384 dimensions and up):

- `code_dims`: the codes are made from the top `code_dims` principal directions only (even, 8 .. `dim` − 1).
- `gate_dims`: with `code_dims`; shortens the codes further (even, 8 .. `code_dims`).
- `residual=True`: with `code_dims`; the form that uses the least memory. `dims_per_code` is ignored.

Which values suit depends on the data. Decide while measuring recall.

### Search settings (stay until changed)

| Function | |
|---|---|
| `set_eps(eps)` | the graph's stop rule. 0 = off (default). Above 0 the result set is kept at `ef` entries and the search goes on only while a candidate lies within (1 + eps) times the current boundary. Use it with a small `ef` (k .. 3k) |
| `set_probe_gap(gap, min_probe=8)` | lists: of the best `nprobe` partitions, scan only those whose score is within `gap` of the nearest one's (at least `min_probe`). 0 = off |
| `set_rerank("float" \| "int8")` | lists: score the candidates with the floats (exact, default) or with int8 rows (a quarter of the size). For the graph, pass `rerank_n=0` to `search` instead |
| `set_qg_exact(on=True)` | quantized graph: check the candidates that pass the estimate once more before they enter the result. Worth trying on high-dimensional data. Off by default |
| `unique = True` | on a `dedup=True` index: one result per distinct vector |

### Searching

| Function | |
|---|---|
| `search(query, k=10, ef=None, nprobe=None, rerank_n=None, filter=None, method="auto")` | the top `k` as `[(id, score)]`. `method`: `"auto"` (the lists if built, else the graph), `"lists"`, `"graph"`. Defaults: `ef` = max(k, 100), `nprobe` = 32, `rerank_n` = `rerank`. `k` up to 4,096 |
| `search_many(queries, k=10, **kw)` | many queries, one after another. Returns the arrays `(ids, scores)` |
| `search_exact(query, k=10, threads=0, filter=None)` | the exact answer by brute force. For checking recall, or for tiny indexes |

### Changing, saving and the rest

| Function | |
|---|---|
| `insert(id, vec)` | add to a built index. `RuntimeError` without `rerank` |
| `delete(id)` | `True` if that number was there |
| `compact()` | rebuild from the live vectors. Returns how many |
| `save(path, key=None)` / `KciVec.load(path, key=None, rerank=100)` | see section 2 |
| `save_image(path)` / `KciVec.attach(path, rerank=100)` | see section 2 |
| `half_floats()`, `search_only()`, `float_mmap(path)` | see "Using less memory" in section 2 |
| `dup_stats()` | how many rows are copies of another |
| `plain_bytes()` | bytes in memory by kind: `codes`, `floats`, `derived`, `graph`, `ids`, `total` |
| `count`, `is_built()`, `has_lists`, `close()` | |

There are two kinds of error: `RuntimeError` when the library refused or failed, `ValueError` for a bad argument.

### Learning from a query sample (experimental)

For data whose queries are of a different nature than the stored vectors (text queries over images, for instance).
You give a sample of real queries and their right answers. Experimental: the arguments may change.

| Function | |
|---|---|
| `learn(queries, answers, nclusters=0, list_cap=0, threads=0, pack=False)` | learns from a query sample [nq, dim] and the numbers of each query's answers [nq, count]. Needs a built index and `rerank > 0`. Returns how far the answers concentrate on few vectors. Dropped by `build` and `compact`; not saved |
| `search_learned(query, k=10, nclusters=4, per_list=1000, ef=0, ...)` | search with what was learned. With `ef` above 0 it is combined with the graph search |
| `vote_set(answers)`, `vote_short(ids, rows)`, `search_voted(query, voters, k=10, ...)` | another form, for data whose answers do not concentrate |

---

## 4. Combining with word search

`TextIndex` (section 1) does this inside. If you bring your own vectors, combine these parts.

```python
from kci_vec.hybrid import BM25Index, hybrid_search

texts = ["How to reissue the invoice for KX-204B", "Meeting rooms are booked in the company calendar", "Expense reports close at month end"]
tv = KciVec(dim=384, rerank=50); tv.add_many([0, 1, 2], embeddings[:3]); tv.build()

bm = BM25Index()
for i, text in enumerate(texts):
    bm.add(i, text)                    # the same numbers as the vectors
hits = hybrid_search(tv, bm, embeddings[0], "KX-204B invoice", k=2)
# [(id, merged score, rank on the vector side, rank on the word side)], best first; None for a side that did not find it
bm.save("index.bm25.json"); bm = BM25Index.load("index.bm25.json")
```

- The word side catches part numbers, names and figures that embeddings miss.
- Japanese, Chinese and Korean are cut into character pairs, so no dictionary is needed.
- `BM25Index` writes plain JSON. If the texts are confidential, encrypt that file yourself.

`kci_vec.sparse.KciSparse` is an index for vectors that hold a weight per word (sparse vectors), with encrypted saves.
The word side of `TextIndex` uses it. For direct use, see the descriptions in `clients/python/kci_vec/sparse.py`.

---

## 5. Settings to start from

These are starting points. The best values differ from data to data.
Prepare a few hundred of your own queries, compare with the answers of `search_exact`, and vary the values while measuring recall.

| Goal | Start with | Then vary |
|---|---|---|
| Simple, small index | `build()`, `search(ef=100)` | `ef` 50 .. 800 |
| Fast graph, low dimensions (about 100) | `M=32`, `build()`, `build_qg(1)`, `set_eps(0.1)`, `ef=30` | `eps` 0.05 .. 0.25, `ef` 20 .. 50 |
| Fast graph, text embeddings (384 dimensions and up) | `M=8`, `build()`, `build_qg(2)`, `set_eps(0.05)`, `ef=150` | `eps` 0 .. 0.1, `ef` 100 .. 400. For recall 0.99: `M=16` |
| Partitioned lists | `build_lists()`, `search(nprobe=64, rerank_n=200)` | `nprobe` 8 .. 512, `rerank_n` 50 .. 800 |
| Faster lists | add `set_probe_gap(0.25)` | `gap` 0.15 .. 0.35 |
| Least memory (search only) | `build()`, `build_qg(...)`, `half_floats()`, `search_only()` | `code_dims`, `gate_dims` of `build_qg` |
| Less memory (with adds and deletes) | `build()`, `build_lists()`, then `float_mmap(path)` | |

Raising `ef`, `eps`, `nprobe` or `rerank_n` raises recall and lowers the number of queries answered per second.
Taking the top 100 (`k=100`) needs a larger `ef` than taking the top 10.

---

## 6. Rules

- **One writer at a time.** Do not add, insert, delete, compact or rebuild an index while it is being searched.
- **Threads.**
  - In Python one `KciVec` has a single work area for searching. Do not call `search` on the same `KciVec` from several threads at once.
    The same holds for `TextIndex`.
  - In C, make one `KciVecSearcher` per thread. On an index that is built and not being changed, several searchers can run at once.
- **Build order.** `build()` first, `build_lists()` and `build_qg()` after. A later `build()` drops both.
- **Quantized graph.** Dropped by `insert`, `compact` and `build`. Call `build_qg()` again after them.
- **One-way operations.** After `half_floats()`, `search_only()` or `float_mmap()` some rebuilds and saves no longer work. Build and save what you need first.
- **The key of a saved file.** Without the key the file cannot be opened. There is no way to recover a key.
- **Where it runs.**
  - The distributed wheels: Linux x86-64 with AVX2 (AVX-512 where present), Windows x86-64 with AVX2, macOS (Apple silicon; Intel Macs with AVX2).
  - From source it also builds on other ARM64 machines.

---

## 7. C API

```c
#include "kci_vec.h"

KciVec *v = kci_vec_new(dim, 16, 200);                  /* dim, M, efConstruction */
kci_vec_keep_float(v, 1);                               /* before add: keep the floats for the exact pass and the lists */
for (...) kci_vec_add(v, id, vec);                      /* float32[dim]; scaled to length 1 inside */
kci_vec_build(v, 0, NULL);                              /* the graph, all cores */

KciVecSearcher *s = kci_vec_searcher_new(v);            /* one per thread; reuse it across queries */
int n = kci_vec_search_rerank(s, query, 10, 100, 200, ids, scores);   /* k, ef, rerank_n */

kci_vec_save(v, "index.kv", key, keylen);               /* encrypted when key is not NULL */
kci_vec_searcher_free(s);
kci_vec_free(v);
```

A complete example is `example_kci_vec.c` in the repository.

Return values:
- On success, a count or 0.
- On failure, a negative number, usually `-1`; `-2` where a function documents a second reason ("no floats", for instance).
- `kci_vec_new`, `kci_vec_load` and `kci_vec_attach` return `NULL` on failure.

| Group | Functions |
|---|---|
| Create, add | `kci_vec_new`, `kci_vec_new_l2`, `kci_vec_new_ip`, `kci_vec_free`, `kci_vec_reserve`, `kci_vec_add`, `kci_vec_add_i8`, `kci_vec_keep_float`, `kci_vec_set_params`, `kci_vec_set_dedup`, `kci_vec_set_ip_centre` |
| Graph | `kci_vec_build`, `kci_vec_build_fast`, `kci_vec_search` (for the occasional search), `kci_vec_searcher_new` / `_free`, `kci_vec_search_r`, `kci_vec_search_rerank`, `kci_vec_search_filtered`, `kci_vec_searcher_set_eps`, `kci_vec_searcher_set_unique` |
| Quantized graph | `kci_vec_build_qg`, `kci_vec_build_qg_ex`, `kci_vec_build_qg_gate`, `kci_vec_build_qg_res`, `kci_vec_qg_bytes`, `kci_vec_searcher_set_qg`, `kci_vec_searcher_set_qg_exact` |
| Partitioned lists | `kci_vec_build_ivf`, `kci_vec_build_ivf_ex`, `kci_vec_build_ivf_p` (every setting), `kci_vec_set_ivf_codes`, `kci_vec_set_ivf_training`, `kci_vec_search_ivf`, `kci_vec_search_ivf_filtered`, `kci_vec_searcher_set_probe_gap`, `kci_vec_searcher_set_rerank_int8` |
| Brute force | `kci_vec_search_exact`, `kci_vec_search_exact_filtered`, `kci_vec_filter_id_bits` |
| Search width | `kci_vec_calibrate`, `kci_vec_calibrate_queries`, `kci_vec_default_ef`, `kci_vec_default_rerank`, `kci_vec_set_default_ef` (`kci_vec_search_rerank` / `kci_vec_search_filtered` with `ef` 0 use the calibrated width), `kci_vec_judge`, `kci_vec_judge_queries`, `kci_vec_judged`, `kci_vec_judge_off`, `kci_vec_graph_recall`, `kci_vec_unreached`, `kci_vec_searcher_set_filter_fill` |
| Change | `kci_vec_insert`, `kci_vec_delete`, `kci_vec_forget` (removes every trace of that vector), `kci_vec_compact`, `kci_vec_squeeze` |
| Floats | `kci_vec_half_floats`, `kci_vec_float_mmap`, `kci_vec_set_float_source` (fetch the floats from your own store for the exact pass), `kci_vec_set_float`, `kci_vec_get_float`, `kci_vec_row_id` (the id of the vector in a row) |
| Save, load | `kci_vec_save`, `kci_vec_load`, `kci_vec_save_ex` / `kci_vec_load_ex` (`KCI_VEC_SAVE_NO_FLOAT`, `KCI_VEC_SAVE_RAW_KEY`) |
| Read-only sharing | `kci_vec_image_save`, `kci_vec_attach`, `kci_vec_attach_mem`, `kci_vec_thaw` |
| Search-only index | `kci_vec_search_only` |
| Learning from a query sample (experimental) | `kci_vec_learn`, `kci_vec_learn_pack`, `kci_vec_has_learned`, `kci_vec_search_learned`, `kci_vec_searcher_set_learned_seeds` / `_pages` / `_hot`, `kci_vec_vote_set`, `kci_vec_vote_short`, `kci_vec_search_voted`, `kci_vec_search_voted_short` |
| Memory | `kci_vec_protect` (asks the OS to keep the vectors out of core dumps and swap, as far as it can), `kci_vec_plain_bytes` |
| Information | `kci_vec_count`, `kci_vec_dim`, `kci_vec_is_built`, `kci_vec_metric`, `kci_vec_is_l2`, `kci_vec_dup_stats` |

The arguments, return values and conditions are described in the comments above the declarations in `kci_vec.h` (some neighbouring functions share one description).
The word-side index (`kci_sparse.c` / `kci_sparse.h`) is used from C in the same way; see `example_hybrid.c`.

---

## 8. PostgreSQL

An extension for calling it from inside PostgreSQL is in `pg/` (a prototype). See `pg/README.md` for how to build and use it.
