# kci-vec (Python)

Embedded Python binding for the kci_vec vector engine — in-process, no server, nothing
leaves the machine. Two ways to search (an int8 HNSW graph, and partitioned lists with
4-bit table lookup), exact or int8 re-rank, metadata filters, incremental updates, and
snapshots encrypted at rest (AES-256-CTR + AES-256-CMAC).

```sh
pip install .          # compiles the C engine into the wheel (needs gcc/clang; set CC to override)
```
```python
from kci_vec import KciVec
import numpy as np
v = KciVec(dim=384, rerank=100)          # keep the floats: exact re-rank, inserts, and the lists need them
v.add_many(ids, embeddings)              # float32 [n, dim]; ids are your own uint64 keys
v.build_lists()                          # the fast path (nlist = 2*sqrt(n)); v.build() builds the graph instead
v.search(query_vec, k=5)                 # [(id, cosine), ...] best first
v.search(query_vec, k=5, nprobe=64, rerank_n=200)        # more lists / longer shortlist = higher recall
v.search(query_vec, k=5, filter={12, 57, 901})           # only these ids (metadata, ACL); or filter=lambda id: ...
ids, scores = v.search_many(queries, k=5)                # [nq, k] arrays
v.set_rerank("int8")                      # 1/4 of the re-rank bytes; fine for text embeddings at k=100, not for CLIP
v.insert(new_id, vec); v.delete(old_id); v.compact()
v.save("index.kv", key=b"secret")        # encrypted + authenticated; wrong key or altered file -> load raises
w = KciVec.load("index.kv", key=b"secret")
```

- `search(..., method="graph")` forces the graph (`ef` controls recall there); `"lists"` the lists; `"auto"` picks the lists when built.
- `build_lists(soar=1.5)` puts every vector in a second list (SOAR): helps at high recall on 384+ dims, costs memory; measure.
- Hybrid search (`kci_vec.hybrid`): a small BM25 word index over the same chunks (Latin words, Japanese as
  character bigrams — no analyser needed) fused with the vector ranking by reciprocal rank, so exact codes,
  names and numbers are found too:
  ```python
  from kci_vec.hybrid import BM25Index, hybrid_search
  bm = BM25Index(); bm.add(i, text) ...; bm.save("index.bm25.json")
  hybrid_search(v, bm, query_vec, "KX-204B invoice", k=5, filter=allowed)   # [(id, fused, vec_rank, word_rank)]
  ```
- Tests without a model: `KCI_VEC_LIB=/path/libkci_vec.so python clients/python/test_smoke.py`, `test_hybrid.py`.

Full guide: [docs/API.md](../../docs/API.md) (日本語: [docs/API_ja.md](../../docs/API_ja.md)). Benchmark numbers: the repository root README.
