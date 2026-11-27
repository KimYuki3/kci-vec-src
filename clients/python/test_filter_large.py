"""Filters that allow few ids, on an index too large for the small-index brute force."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from kci_vec import KciVec

ok = True
def check(name, cond, extra=""):
    global ok
    print(("ok   " if cond else "FAIL ") + name, extra)
    ok = ok and bool(cond)

N, D, K = 60000, 200, 10          # 12,000,000 numbers: above exact_cells, so search() uses the graph
rng = np.random.default_rng(5)
X = rng.standard_normal((N, D)).astype(np.float32)
Q = rng.standard_normal((30, D)).astype(np.float32)
ids = lambda r: [i for i, _ in r]

def truth(metric, q, allowed, k):
    a = np.array(sorted(allowed), dtype=np.int64)
    if metric == "l2":
        s = ((X[a] - q) ** 2).sum(1); o = np.argsort(s, kind="stable")
    else:
        s = (X[a] @ q) / np.linalg.norm(X[a], axis=1); o = np.argsort(-s, kind="stable")
    return [int(a[i]) for i in o[:k]]

for metric in ("cosine", "l2"):
    v = KciVec(dim=D, metric=metric); v.add_many(np.arange(N, dtype=np.int64), X); v.build()
    check(f"{metric}: the index is not a small one", not v._small_exact())
    three = [set(int(x) for x in rng.choice(N, 3, replace=False)) for _ in Q]
    got = [ids(v.search(q, k=K, filter=s)) for q, s in zip(Q, three)]
    check(f"{metric}: a set of 3 ids returns the 3, best first", got == [truth(metric, q, s, K) for q, s in zip(Q, three)])
    many = np.sort(rng.choice(N, 2000, replace=False))                    # 400,000 numbers: brute force over the set
    got = [ids(v.search(q, k=K, filter=many)) for q in Q]
    check(f"{metric}: an array of 2,000 ids: the exact top-{K} among them", got == [truth(metric, q, many, K) for q in Q])
    got = [ids(v.search(q, k=K, filter=list(s))) for q, s in zip(Q[:5], three)]
    check(f"{metric}: a list works as a set", [sorted(g) for g in got] == [sorted(s) for s in three[:5]])
    # a callable: the count is not known; the graph finds fewer than k and the index answers again by brute force
    got = [ids(v.search(q, k=K, filter=lambda i, s=s: i in s)) for q, s in zip(Q[:5], three)]
    check(f"{metric}: a callable passing 3 ids returns the 3", got == [truth(metric, q, s, K) for q, s in zip(Q[:5], three)])
    v.filter_fill = False; v._s = None
    short = sum(len(v.search(q, k=K, ef=50, filter=lambda i, s=s: i in s)) < 3 for q, s in zip(Q[:5], three))   # ef given: build() calibrated a wide one
    check(f"{metric}: with filter_fill off the graph leaves ids out (the test can fail)", short > 0, f"({short} of 5 queries)")
    v.filter_fill = True; v._s = None
    half = set(range(0, N, 2))
    r = v.search(Q[0], k=K, filter=half, method="graph")
    check(f"{metric}: method='graph' with a set: {K} answers, all allowed", len(r) == K and all(i % 2 == 0 for i, _ in r))
    check(f"{metric}: an empty set -> []", v.search(Q[0], k=K, filter=set()) == [] and v.search_exact(Q[0], k=K, filter=[]) == [])
    check(f"{metric}: ids outside the index only -> []", v.search(Q[0], k=K, filter={N + 5, N + 6}) == [])
    check(f"{metric}: search_exact with a set", ids(v.search_exact(Q[1], k=K, filter=three[1])) == truth(metric, Q[1], three[1], K))

# ids too large for a bitmap: the Python callback, and still every allowed id
big = 1 << 40
v = KciVec(dim=D); v.add_many(np.arange(N, dtype=np.int64) + big, X); v.build()
s = {big + 11, big + 4040, big + 59999}
check("ids above the bitmap limit: a set of 3 returns the 3", sorted(ids(v.search(Q[0], k=K, filter=s))) == sorted(s))

print("ALL OK" if ok else "FAILED"); sys.exit(0 if ok else 1)
