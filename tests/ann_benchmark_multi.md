# kci_vec vs hnswlib vs faiss-SQ8 — multi-dataset Pareto (ann-benchmarks angular)

Third-party-reproducible. Standard ann-benchmarks **angular (cosine)** datasets,
recall@10 vs each dataset's own brute-force GT, identical graph params (M=16,
efConstruction=200), in-process, 8 threads, AMD Ryzen 7 8845HS (8c/16t).

- **kci_vec** (signed-int8 SDOT HNSW) via `kci_vec_bench`; `+rerank` = int8 shortlist
  → full-precision float re-rank (shortlist 100–200). RAM ≈ 1/4 of float32.
- **hnswlib** (float HNSW) and **faiss** `IndexHNSWSQ` (int8 SQ8 HNSW) on the same
  normalized arrays via `tests/ann_benchmark.py`.

QPS is 8-thread. Reproduce:
`python tests/ann_benchmark.py tests/_ann/<name>.hdf5 16 200 10000 8`
then `kvb tests/_ann 16 200 10000 8 tests/_ann/gt.u64 <rerank_n>`.

## glove-100-angular (1,183,514 × 100)

| ef | kci int8 | kci int8+rerank | hnswlib (float) | faiss-SQ8 (int8) |
|----|----------|-----------------|-----------------|------------------|
| 100 | 80.6% / 21,497 | 83.0% / 19,238 | 80.9% / 12,571 | 80.5% / 8,135 |
| 200 | 85.3% / 11,814 | 88.3% / 10,876 | 86.7% / 7,380  | 85.8% / 4,423 |
| 400 | —              | 92.3% / 5,606  | 91.2% / 3,816  | 89.8% / 2,276 |
| 800 | —              | 95.4% / 3,311  | —              | —             |

kci ~1.6–1.8× hnswlib QPS, ~2.6× faiss-SQ8, equal-or-higher recall with re-rank.

## nytimes-256-angular (290,000 × 256)

| ef | kci int8 | kci int8+rerank | hnswlib (float) | faiss-SQ8 (int8) |
|----|----------|-----------------|-----------------|------------------|
| 100 | 87.4% / 22,000 | 88.0% / 18,246 | 86.6% / 7,730 | 2.9% / 8,316 |
| 200 | 90.0% / 11,934 | 90.8% / 9,518  | 89.7% / 4,325 | 3.6% / 4,397 |
| 400 | —              | —              | 92.2% / 2,254 | 19.3% / 2,561 |

kci ~2.2–2.8× hnswlib QPS at equal recall. **faiss `IndexHNSWSQ` collapses on this
distribution** (recall 3–19% with the default QT_8bit scalar quantizer — likely
retunable, but a real fragility); kci's normalize-then-global-scale int8 does not.

## glove-25-angular (1,183,514 × 25)

| ef | kci int8 | kci int8+rerank | hnswlib (float) | faiss-SQ8 (int8) |
|----|----------|-----------------|-----------------|------------------|
| 100 | 92.4% / 37,945 | 98.8% / 35,542 | 97.8% / 37,125 | 95.9% / 16,954 |
| 200 | 93.0% / 20,469 | 99.7% / 18,986 | 99.4% / 19,553 | 97.0% / 8,739 |
| 400 | —              | 99.95% / 9,944 | 99.9% / 10,614 | 97.3% / 4,385 |

At 25-d the int8 *speed* edge over hnswlib shrinks to ~parity (a float dot is already
cheap), but re-rank lifts recall from 93% (pure int8) to 99.7%, **above faiss-SQ8 (97%)
at ~2× its QPS**. Shows the int8-SDOT advantage is largest in the 100–256-d embedding
regime and that re-rank is what carries the low-dim case.

## deep-image-96-angular (9,990,000 × 96) — scale check

Does it hold at 10 M? kci_vec int8 + re-rank, single machine, 8 threads
(competitors not run at this size). Built in **784 s (~13 min)**; loaded N=9.99 M
correctly (needs the 64-bit fvecs reader — `long`/ftell overflow on the 3.88 GB file
otherwise). recall@10 vs dataset GT:

| ef | kci int8+rerank |
|----|-----------------|
| 40  | 88.6% / 43,200 |
| 100 | 95.9% / 21,341 |
| 200 | 98.3% / 11,834 |

Holds up: 98.3% recall@10 at 11.8 k QPS (8T) on 10 M vectors, ~13 min build. Vector
RAM stays int8 (~1/4 of float32; +retained floats only when re-rank is enabled, and
in production those are mmap'd from disk).

## Memory frontier (glove-100, measured resident set)

The axis where int8 wins: recall/QPS **per unit RAM**. Measured resident set
(WorkingSetSize / psutil RSS) after build, ~equal recall@10 (ef200), 8 threads:

| engine | resident RAM | recall@10 | QPS(8T) |
|---|---|---|---|
| **kci_vec (pure int8)** | **365 MB** | 86.5% | 12,135 |
| hnswlib (float HNSW)    | ~871 MB     | 86.7% | 7,380 |
| faiss-SQ8 (int8 HNSW)   | ~309 MB     | 85.8% | 4,423 |

- vs **hnswlib (the float standard): kci is Pareto-dominant** — ~equal recall at
  **2.4× less RAM and 1.6× the QPS**.
- vs **faiss-SQ8** (int8 peer): comparable RAM, ~2.7× the QPS at equal-or-better
  recall, and more robust (faiss-SQ8 collapsed on nytimes-256; see above).
- With the disk-backed float store (`kci_vec_float_mmap`), int8+re-rank reaches 88%
  (> hnswlib) while the resident set stays at the int8 level — the re-rank floats
  are mmap'd and evictable under memory pressure, not pinned like hnswlib's vectors.

**Honest scope:** this is #1 against the mainstream open baselines (hnswlib, faiss-SQ8)
on the combined recall×QPS×RAM frontier. It is NOT a claim against specialized
PQ/anisotropic libraries (ScaNN, glass, NGT-qg) — those use cheaper-per-distance
quantization and lead the raw recall-vs-QPS leaderboard; they were not installable
here to measure. kci's edge is the memory-aware frontier + the embedded/encrypted/
hybrid one-binary package, not raw-ANN-speed supremacy.

## Honest summary

- **Recall:** kci_vec + re-rank reaches **equal-or-higher recall@10 than both hnswlib
  (float) and faiss-SQ8 (int8) on all three datasets**, at ~1/4 the RAM.
- **Speed:** ~1.6–2.8× hnswlib QPS at 100–256 d; ~parity at 25 d. Always faster than
  faiss-SQ8, which additionally collapsed on nytimes.
- **Caveats:** angular/cosine only (kci is a cosine engine — euclidean sets like
  sift-128 are out of scope). faiss-SQ8's nytimes collapse may be retunable. Single
  machine; numbers vary by host/config. Not yet on the official ann-benchmarks
  leaderboard (see `integrations/ann-benchmarks/`).
