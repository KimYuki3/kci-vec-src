#!/usr/bin/env python3
"""Convert an ann-benchmarks angular .hdf5 to fvecs + u64 GT for kci_vec_bench,
WITHOUT running the (slow at 10M) hnswlib/faiss competitors. Use for large-scale
"does it hold up" runs: prep here, then run kci_vec_bench on the output dir.

usage: python tests/ann_prep.py tests/_ann/deep-image-96-angular.hdf5 [nq]
"""
import sys, os, struct, time
import numpy as np

def write_fvecs(path, x, chunk=200000):
    n, d = x.shape
    hdr = struct.pack("<i", d)
    with open(path, "wb") as f:
        for s in range(0, n, chunk):
            blk = x[s:s+chunk]
            buf = bytearray()
            for i in range(blk.shape[0]):
                buf += hdr; buf += blk[i].tobytes()
            f.write(buf)

def main():
    h5 = sys.argv[1]
    nq = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    K = 10
    import h5py
    with h5py.File(h5, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test  = np.array(f["test"],  dtype=np.float32)
        gt    = np.array(f["neighbors"], dtype=np.int64)[:, :K]
    n, dim = train.shape
    nq = min(nq, test.shape[0])
    nrm = np.linalg.norm(train, axis=1, keepdims=True); nrm[nrm == 0] = 1; train /= nrm
    nrm = np.linalg.norm(test, axis=1, keepdims=True);  nrm[nrm == 0] = 1; test /= nrm
    train = train.astype(np.float32); test = test[:nq].astype(np.float32); gt = gt[:nq]
    d = os.path.dirname(h5)
    print(f"{os.path.basename(h5)}: N={n} dim={dim} nq={nq}")
    t = time.time(); write_fvecs(os.path.join(d, "sift_base.fvecs"), train); print(f"base {time.time()-t:.0f}s")
    write_fvecs(os.path.join(d, "sift_query.fvecs"), test)
    gt.astype(np.uint64).tofile(os.path.join(d, "gt.u64"))
    print(f"wrote sift_base.fvecs / sift_query.fvecs / gt.u64 in {d}")
    print(f"run: kvb {d} 16 200 {nq} 8 {os.path.join(d,'gt.u64')} 100")

if __name__ == "__main__":
    main()
