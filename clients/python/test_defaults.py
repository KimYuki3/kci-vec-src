"""The defaults favour the right answer: KciVec(dim) keeps the floats and re-ranks on them, and a small index is
answered by brute force.
Run: KCI_VEC_LIB=.../libkci_vec.so python test_defaults.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from kci_vec import KciVec

rng = np.random.default_rng(7)
def truth(X, Q, metric, k):
    out = []
    Xd = X.astype(np.float64)
    if metric == "cosine": Xd = Xd / np.linalg.norm(Xd, axis=1, keepdims=True)
    for q in Q.astype(np.float64):
        s = -(Xd @ q) if metric == "cosine" else ((Xd - q) ** 2).sum(1)
        out.append(np.argsort(s, kind="stable")[:k].tolist())
    return out
def ids(hits): return [i for i, _ in hits]
ok = True
def check(name, cond, extra=""):
    global ok
    print(("ok   " if cond else "FAIL ") + name + (" " + extra if extra else "")); ok = ok and bool(cond)

N, D, K = 30000, 64, 20                       # random points in 64 dims: the graph misses some of the true top 20
X = rng.normal(size=(N, D)).astype(np.float32); Q = rng.normal(size=(200, D)).astype(np.float32)
for metric in ("cosine", "l2"):
    gt = truth(X, Q, metric, K)
    v = KciVec(dim=D, metric=metric)          # nothing else given
    v.add_many(np.arange(N, dtype=np.int64), X); v.build()
    check(f"{metric}: KciVec(dim) keeps the floats", v.rerank == 200 and v.lib.kci_vec_has_floats(v._v) == 1)
    got = [ids(v.search(q, k=K)) for q in Q]
    check(f"{metric}: a small index answers exactly (every query, in order)", got == gt,
          f"({sum(a == b for a, b in zip(got, gt))} of {len(Q)})")
    # the score is the metric's own: cosine, or squared distance
    q = Q[0]; i, s = v.search(q, k=1)[0]
    want = float(X[i] @ q / np.linalg.norm(X[i]) / np.linalg.norm(q)) if metric == "cosine" else float(((X[i] - q) ** 2).sum())
    check(f"{metric}: its score is the {metric} value", abs(s - want) <= 1e-3 * max(1.0, abs(want)), f"({s:.5f} vs {want:.5f})")
    # the index's own search is still there, and it is the one that misses
    g = [ids(v.search(q, k=K, method="graph")) for q in Q]
    rec_g = np.mean([len(set(a) & set(b)) for a, b in zip(g, gt)]) / K
    v.exact_cells = 0
    g0 = [ids(v.search(q, k=K)) for q in Q]
    check(f"{metric}: method='graph' and exact_cells=0 give the graph's answer", g == g0 and 0.9 < rec_g < 1.0, f"(recall {rec_g:.4f})")
    v.exact_cells = N * D - 1                 # one number too many: the graph
    check(f"{metric}: above exact_cells the graph answers", [ids(v.search(q, k=K)) for q in Q] == g)
    v.exact_cells = N * D
    check(f"{metric}: at exact_cells, exact", [ids(v.search(q, k=K)) for q in Q] == gt)
    # a filter keeps the filtered search (it honours the filter); a deleted id is not returned by the exact path
    f = v.search(Q[1], k=K, filter=lambda i: i % 2 == 0)
    check(f"{metric}: a filter is honoured", len(f) == K and all(i % 2 == 0 for i, _ in f))
    dead = gt[2][0]; v.delete(dead)
    after = ids(v.search(Q[2], k=K))
    check(f"{metric}: a deleted id is gone, the rest is still exact", dead not in after and after[:K - 1] == gt[2][1:])

# rerank=0 is the small index: no floats, the int8 search as before, and no brute force (it would rank by int8)
v = KciVec(dim=D, rerank=0); v.add_many(np.arange(N, dtype=np.int64), X); v.build()
gt = truth(X, Q, "cosine", K)
r0 = np.mean([len(set(ids(v.search(q, k=K))) & set(b)) for q, b in zip(Q, gt)]) / K
check("rerank=0: no floats, the int8 search", v.lib.kci_vec_has_floats(v._v) == 0 and not v._small_exact() and 0.5 < r0 < 1.0, f"(recall {r0:.4f})")

# inner product keeps the index's search (and the default re-rank)
v = KciVec(dim=D, metric="ip"); v.add_many(np.arange(N, dtype=np.int64), X); v.build()
check("ip: not answered by brute force, searches work", not v._small_exact() and len(v.search(Q[0], k=K)) == K)

# a saved small index: load() finds the floats in the file and answers exactly too
import tempfile
v = KciVec(dim=D); v.add_many(np.arange(N, dtype=np.int64), X); v.build()
p = os.path.join(tempfile.mkdtemp(), "t.kvec"); v.save(p, key=b"k")
w = KciVec.load(p, key=b"k")
check("load: a small saved index answers exactly", [ids(w.search(q, k=K)) for q in Q] == gt)

print("ALL OK" if ok else "FAILED"); sys.exit(0 if ok else 1)
