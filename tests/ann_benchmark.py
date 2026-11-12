#!/usr/bin/env python3
"""Reproducible ANN bench on a standard ann-benchmarks angular (cosine) dataset.

* hnswlib (float HNSW) * faiss (IndexHNSWSQ = int8 SQ8 HNSW)

Win or lose, the numbers are real and third-party-reproducible (anyone can re-run with the same public hdf5)."""
import sys, os, time, struct, json
import numpy as np

def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)

def write_fvecs(path, x):
    n, d = x.shape
    with open(path, "wb") as f:
        hdr = struct.pack("<i", d)
        for i in range(n):
            f.write(hdr); f.write(x[i].tobytes())

def main():
    h5 = sys.argv[1]
    M   = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    efc = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    nq  = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    T   = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    K = 10
    EFS = [10, 20, 40, 80, 100, 200, 400]
    import h5py
    with h5py.File(h5, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test  = np.array(f["test"],  dtype=np.float32)
        gt    = np.array(f["neighbors"], dtype=np.int64)[:, :K]
    N, dim = train.shape
    nq = min(nq, test.shape[0])
    train = l2norm(train); test = l2norm(test[:nq]); gt = gt[:nq]
    print(f"dataset={os.path.basename(h5)} N={N} dim={dim} nq={nq} M={M} efc={efc} T={T}")

    outdir = os.path.join(os.path.dirname(h5))
    base_p = os.path.join(outdir, "sift_base.fvecs")   # kci_vec_bench reads these names
    qry_p  = os.path.join(outdir, "sift_query.fvecs")
    gt_p   = os.path.join(outdir, "gt.u64")
    if not (os.path.exists(base_p) and os.path.getsize(base_p) == 4*N*(dim+1)):
        t=time.time(); write_fvecs(base_p, train); print(f"wrote base.fvecs {time.time()-t:.0f}s")
    write_fvecs(qry_p, test)
    gt[:, :K].astype(np.uint64).tofile(gt_p)   # row indices into base == kci ids
    print(f"wrote {qry_p}, {gt_p} (kci ids = row index)")

    def recall(pred):  # pred: nq x K row indices
        hit = sum(len(set(pred[i]) & set(gt[i])) for i in range(nq))
        return 100.0 * hit / (nq * K)

    rows = []

    # ---- hnswlib (float HNSW) ----
    try:
        import hnswlib
        h = hnswlib.Index(space="cosine", dim=dim)
        t=time.time(); h.init_index(max_elements=N, ef_construction=efc, M=M)
        h.add_items(train, np.arange(N), num_threads=T); hb=time.time()-t
        for ef in EFS:
            h.set_ef(max(ef, K))
            t=time.time(); lab,_=h.knn_query(test, k=K, num_threads=T); qt=time.time()-t
            rows.append(("hnswlib(float)", ef, recall(lab), nq/qt, hb))
        print(f"hnswlib build {hb:.1f}s")
    except Exception as e:
        print("hnswlib FAILED:", e)

    # ---- faiss IndexHNSWSQ (int8 SQ8 HNSW) ----
    try:
        import faiss
        faiss.omp_set_num_threads(T)
        idx = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, M)
        idx.hnsw.efConstruction = efc
        t=time.time(); idx.train(train); idx.add(train); fb=time.time()-t
        for ef in EFS:
            idx.hnsw.efSearch = max(ef, K)
            t=time.time(); _,lab=idx.search(test, K); qt=time.time()-t
            rows.append(("faiss-SQ8(int8)", ef, recall(lab), nq/qt, fb))
        print(f"faiss build {fb:.1f}s")
    except Exception as e:
        print("faiss FAILED:", e)

    print("\n| engine | ef | recall@10 | QPS(%dT) | build s |" % T)
    print("|---|---|---|---|---|")
    for nm, ef, rc, qps, bt in rows:
        print(f"| {nm} | {ef} | {rc:.2f}% | {qps:.0f} | {bt:.1f} |")

    rep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ann_benchmark.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write(f"# ANN bench — {os.path.basename(h5)} (angular/cosine, dataset GT)\n\n")
        f.write(f"- N={N} dim={dim} nq={nq} · M={M} efc={efc} · {T} threads · recall@10 vs dataset brute-force GT\n\n")
        f.write(f"| engine | ef | recall@10 | QPS({T}T) | build s |\n|---|---|---|---|---|\n")
        for nm, ef, rc, qps, bt in rows:
            f.write(f"| {nm} | {ef} | {rc:.2f}% | {qps:.0f} | {bt:.1f} |\n")
        f.write("\nkci_vec (run separately, in-process C harness):\n\n")
        f.write(f"```\n./kvb tests/_ann {M} {efc} {nq} {T} tests/_ann/gt.u64\n```\n")
    print("\nreport ->", rep)
    print(f"\nNow run kci_vec (in-process):  kvb {outdir} {M} {efc} {nq} {T} {gt_p} [rerank_n]")

if __name__ == "__main__":
    main()
