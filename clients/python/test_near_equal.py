"""Rows that are nearly the same vector come back in the right order, and no row is left that the graph search cannot
reach. The truth is counted in float64 from the rows given.
    python clients/python/test_near_equal.py"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from kci_vec import KciVec

fails = 0
def check(name, ok, note=""):
    global fails
    print(("ok   " if ok else "FAIL ") + name, note)
    fails += 0 if ok else 1

rng = np.random.default_rng(7)
G, S, D, K = 20, 300, 64, 10
C = rng.uniform(-0.5, 0.5, (G, D))
X = (np.repeat(C, S, axis=0) + rng.uniform(-0.0005, 0.0005, (G * S, D))).astype(np.float32)
Q = X[(np.arange(100) * 2654435761) % (G * S)]
X64 = X.astype(np.float64)

def truth(metric):
    out = []
    for q in Q.astype(np.float64):
        if metric == "l2":
            d = ((X64 - q) ** 2).sum(axis=1)
        else:
            d = 1.0 - (X64 @ q) / (np.linalg.norm(X64, axis=1) * np.linalg.norm(q))
        out.append(set(np.argsort(d, kind="stable")[:K].tolist()))
    return out

for metric, lo_exact, lo_graph in (("cosine", 0.98, 0.93), ("l2", 0.90, 0.88)):
    gt = truth(metric)
    v = KciVec(dim=D, metric=metric)
    v.add_many(np.arange(G * S, dtype=np.int64), X)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always"); v.build()
    check(f"{metric}: build() does not warn about rows it cannot reach", not any("reach" in str(w.message) for w in wl))
    r = sum(len(g & {i for i, _ in v.search_exact(q, k=K)}) for q, g in zip(Q, gt)) / (K * len(Q))
    check(f"{metric}: search_exact against the truth in float64", r >= lo_exact, f"({r:.3f})")
    r = sum(len(g & {i for i, _ in v.search(q, k=K)}) for q, g in zip(Q, gt)) / (K * len(Q))
    check(f"{metric}: search() on a small index", r >= lo_exact, f"({r:.3f})")
    v = KciVec(dim=D, metric=metric); v.exact_cells = 0; v.auto_ef = False
    v.add_many(np.arange(G * S, dtype=np.int64), X); v.build()
    r = sum(len(g & {i for i, _ in v.search(q, k=K, ef=4 * S, rerank_n=4 * S)}) for q, g in zip(Q, gt)) / (K * len(Q))
    check(f"{metric}: the graph search at a width of four groups", r >= lo_graph, f"({r:.3f})")

print("ALL OK" if fails == 0 else f"FAILED ({fails})")
sys.exit(1 if fails else 0)
