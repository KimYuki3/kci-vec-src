"""judge(): the search width judged query by query; search() without ef then uses it."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from kci_vec import KciVec

ok = True
def check(name, cond, extra=""):
    global ok
    print(("ok   " if cond else "FAIL ") + name, extra)
    ok = ok and bool(cond)

# half of the vectors in tight groups (easy queries), half without structure (hard ones): both kinds of query occur
N, D, K, NC = 60000, 48, 10, 300
rng = np.random.default_rng(4)
cen = 2.0 * rng.standard_normal((NC, D)).astype(np.float32)
def make(n):
    a = rng.standard_normal((n, D)).astype(np.float32)
    g = cen[rng.integers(0, NC, n)] + 0.05 * rng.standard_normal((n, D)).astype(np.float32)
    return np.where((np.arange(n) % 2 == 1)[:, None], g, a).astype(np.float32)
X, Q = make(N), make(400)
for metric in ("cosine", "l2"):
    if metric == "cosine":
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True); gt = [set(np.argsort(-(Xn @ q), kind="stable")[:K].tolist()) for q in Q]
    else:
        xn = (X * X).sum(axis=1); gt = [set(np.argsort(xn - 2 * (X @ q), kind="stable")[:K].tolist()) for q in Q]
    def recall(v, **kw):
        return sum(len(g & {i for i, _ in v.search(q, k=K, method="graph", **kw)}) for q, g in zip(Q, gt)) / (K * len(Q))
    v = KciVec(dim=D, metric=metric); v.exact_cells = 0; v.add_many(np.arange(N, dtype=np.int64), X); v.build()
    r_before, r_wide = recall(v), recall(v, ef=1600)
    on, gain = v.judge(k=K, target=0.99)
    check(f"{metric}: judge() returns (in_use, gain)", isinstance(on, bool) and 0.3 < gain < 100, f"(in use {on}, gain {gain:.2f}x)")
    r_after = recall(v)
    check(f"{metric}: an explicit ef is searched as before", recall(v, ef=1600) == r_wide)
    if on:
        check(f"{metric}: search() without ef is judged and stays near the widest search", r_after >= r_wide - 0.03,
              f"(before {r_before:.4f}, judged {r_after:.4f}, ef 1600 {r_wide:.4f})")
    else:
        check(f"{metric}: rule left off, search() as before", r_after == r_before)
    on2, _ = v.judge(k=K, queries=Q)
    check(f"{metric}: judge(queries=...) runs", isinstance(on2, bool))
    off, g0 = v.judge(samples=0)
    check(f"{metric}: judge(samples=0) forgets the rule", off is False and recall(v) == r_before)
    v.judge(k=K); v.build()
    check(f"{metric}: build() drops the rule", v.lib.kci_vec_judged(v._v) == 0)
try:
    v = KciVec(dim=D, metric="ip"); v.exact_cells = 0; v.add_many(np.arange(5000, dtype=np.int64), X[:5000]); v.build(); v.judge()
    check("ip: refused", False)
except RuntimeError:
    check("ip: refused", True)
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
