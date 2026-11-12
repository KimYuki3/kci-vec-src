# kci_vec — ann-benchmarks integration

Files to add kci_vec to [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
so its results land on the public leaderboard's recall/QPS Pareto plots.

## Layout

In an ann-benchmarks checkout, create `ann_benchmarks/algorithms/kci_vec/` and place:

```
ann_benchmarks/algorithms/kci_vec/
  __init__.py        # empty
  module.py          # this dir's module.py  (BaseANN subclass, ctypes binding)
  config.yml         # this dir's config.yml
  Dockerfile         # this dir's Dockerfile
  kci_vec.c          # copy from repo root
  kci_vec.h          # copy from repo root
```

(The Dockerfile `COPY`s `kci_vec.c`/`kci_vec.h`, builds `libkci_vec.so`, and the
module loads it via ctypes — in-process, no server, so latency is the library's,
matching how hnswlib/faiss are measured.)

## Run

```sh
python install.py --algorithm kci_vec          # builds the docker image
python run.py --algorithm kci_vec --dataset glove-100-angular
python run.py --algorithm kci_vec --dataset nytimes-256-angular
python run.py --algorithm kci_vec --dataset glove-25-angular
python plot.py --dataset glove-100-angular     # recall vs QPS
```

## Notes

- **Angular/cosine only.** kci_vec L2-normalizes internally and ranks by inner
  product; euclidean datasets (sift-128, etc.) are intentionally not configured.
- Two algorithms. `kci_vec` (class `KciVec`) is the int8 HNSW graph: run groups
  `int8-M16` / `int8-M32` (pure SDOT, ~1/4 RAM) and `int8-rerank-M16` /
  `int8-rerank-M32` (int8 shortlist + float re-rank, highest recall — this retains
  float vectors); query arg = ef. `kci_vec-ivf` (class `KciVecIvf`, 2026-11-01) is
  the partitioned 4-bit table-lookup index: run groups `ivf-1024` / `ivf-2048` /
  `ivf-4096` (nlist); query args = nprobe x rerank. One setting per group:
  ann-benchmarks passes each dict in `arg_groups` as a separate constructor argument.
- Checked with `run.py --local` (no Docker) on glove-100-angular, 2026-11-01: both
  classes build and write result files.
- Submitting to the upstream leaderboard is an external pull request to
  erikbern/ann-benchmarks; do that step yourself (it runs in their CI). These files
  are everything that PR needs.

## Local sanity numbers (not the official run)

See `../../tests/ann_benchmark_multi.md` for measured glove-100 / nytimes-256 /
glove-25 results vs hnswlib and faiss on one machine.
