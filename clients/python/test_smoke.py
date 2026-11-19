"""Smoke test of the Python binding, no model needed (random clustered vectors).
    KCI_VEC_LIB=/path/libkci_vec.so python clients/python/test_smoke.py
Checks: graph and lists search agree with brute force, filters return only allowed ids, int8 re-rank, search_many,
insert / delete, encrypted save + load (and that a wrong key is refused)."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kci_vec import KciVec

rng = np.random.default_rng(1)
n, d = 20000, 64
centres = rng.standard_normal((64, d)).astype(np.float32)
X = centres[rng.integers(0, 64, n)] + 0.6 * rng.standard_normal((n, d)).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
Q = X[rng.integers(0, n, 100)] + 0.3 * rng.standard_normal((100, d)).astype(np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
ids = np.arange(n, dtype=np.int64) * 10          # caller ids need not be 0..n-1
truth = [set((ids[np.argsort(-(X @ q))[:10]]).tolist()) for q in Q]

def recall(v, **kw):
    hit = 0
    for q, t in zip(Q, truth):
        hit += len(set(i for i, _ in v.search(q, k=10, **kw)) & t)
    return hit / (10 * len(Q))

fails = 0
def check(cond, msg):
    global fails
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails += 1

v = KciVec(dim=d, rerank=100)
v.add_many(ids, X)
v.build()                                       # graph
r_graph = recall(v, method="graph", ef=100)
check(r_graph >= 0.9, f"graph recall@10 = {r_graph:.3f}")
v.build_lists(nlist=100)                        # lists (fast path)
r_lists = recall(v, nprobe=20)
check(r_lists >= 0.9, f"lists recall@10 = {r_lists:.3f}")
check(v.has_lists, "has_lists after build_lists")

# filters: only even caller ids (set) and only ids < 50,000 (callable)
even = set(int(i) for i in ids[::2])
res = [v.search(q, k=10, nprobe=40, filter=even) for q in Q]
check(all(i in even for r in res for i, _ in r), "lists + set filter returns only allowed ids")
check(np.mean([len(r) for r in res]) >= 9.5, "lists + set filter still fills k")
res = [v.search(q, k=10, filter=lambda i: i < 50000, method="graph") for q in Q]
check(all(i < 50000 for r in res for i, _ in r), "graph + callable filter returns only allowed ids")

# int8 re-rank: close to float
v.set_rerank("int8"); r_i8 = recall(v, nprobe=20); v.set_rerank("float")
check(r_i8 >= 0.9, f"int8 re-rank recall@10 = {r_i8:.3f} (float {r_lists:.3f}; 64 dims is where int8 loses most)")

# search_many
I, S = v.search_many(Q[:5], k=10, nprobe=20)
check(I.shape == (5, 10) and S.shape == (5, 10) and (I >= 0).all(), "search_many shapes")
check(all(set(I[j].tolist()) == set(i for i, _ in v.search(Q[j], k=10, nprobe=20)) for j in range(5)), "search_many == search")

# insert + delete
new_id = 10 * n + 7
v.insert(new_id, Q[0])
check(any(i == new_id for i, _ in v.search(Q[0], k=3)), "inserted vector is found")
v.delete(new_id)
check(not any(i == new_id for i, _ in v.search(Q[0], k=3)), "deleted vector is not returned")

# encrypted save / load
path = os.path.join(tempfile.gettempdir(), "kci_vec_smoke.kv")
check(v.save(path, key=b"smoke-key"), "save (encrypted)")
w = KciVec.load(path, key=b"smoke-key")
check(w.count == v.count and w.has_lists, f"load: count {w.count}, lists kept")
check(recall(w, nprobe=20) >= 0.9, "loaded index searches")
try:
    KciVec.load(path, key=b"wrong-key"); check(False, "wrong key must be refused")
except RuntimeError:
    check(True, "wrong key refused")
os.remove(path)

g = KciVec(d, M=16, efc=100, rerank=50)
g.add_many(ids, X); g.build()
g.build_qg(dims_per_code=2, code_dims=32); g.set_qg_exact(2)
img = os.path.join(tempfile.gettempdir(), "kci_vec_smoke.kvimg")
check(g.save_image(img), "save_image")
a = KciVec.attach(img, rerank=50); a.set_qg_exact(2)
check(a.count == g.count and a.qg and a.is_built(), f"attach: count {a.count}, quantized graph {a.qg}")
check(all(g.search(q, k=10, ef=80, method="graph") == a.search(q, k=10, ef=80, method="graph") for q in Q[:30]), "attached index answers as the saved one")
ra, rg = recall(a, ef=120, method="graph"), recall(g, ef=120, method="graph")
check(ra == rg and ra >= 0.5,   # short codes on clustered random data: about 0.62
      f"attached index: recall {ra:.3f}, the saved index {rg:.3f}")
check(a.delete(int(ids[0])) is False, "delete on an attached index is refused")
try:
    KciVec.attach(img + ".missing"); check(False, "a missing image must be refused")
except RuntimeError:
    check(True, "missing image refused")
a.close(); g.close(); os.remove(img)
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
