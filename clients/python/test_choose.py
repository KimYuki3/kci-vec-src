"""choose_method(): measure whether the graph or the lists suit the data; search() then follows it."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from kci_vec import KciVec

ok = True
def check(name, cond, extra=""):
    global ok
    print(("ok   " if cond else "FAIL ") + name, extra)
    ok = ok and bool(cond)

N, D, K = 60000, 64, 10
rng = np.random.default_rng(21)
cen = rng.standard_normal((400, D)).astype(np.float32)
X = (cen[rng.integers(0, 400, N)] + 0.35 * rng.standard_normal((N, D))).astype(np.float32)
Q = (cen[rng.integers(0, 400, 200)] + 0.35 * rng.standard_normal((200, D))).astype(np.float32)
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
gt = [set(np.argsort(-(Xn @ q), kind="stable")[:K].tolist()) for q in Q]
def recall(v, **kw):
    return sum(len(g & {i for i, _ in v.search(q, k=K, **kw)}) for q, g in zip(Q, gt)) / (K * len(Q))

v = KciVec(dim=D); v.exact_cells = 0; v.add_many(np.arange(N, dtype=np.int64), X); v.build()
check("no lists before", not v.has_lists)
r = v.choose_method(k=K, target=0.95)
check("returns the method and both measurements", r["method"] in ("graph", "lists") and {"ef", "recall", "ms"} <= set(r["graph"]) and {"nprobe", "recall", "ms"} <= set(r["lists"]), str(r))
check("the lists were built for the comparison", v.has_lists)
best = r[r["method"]]; other = r["lists" if r["method"] == "graph" else "graph"]
check("the kept setting reaches the target, or is the better of two that do not",
      best["recall"] >= 0.95 or best["recall"] >= other["recall"])
# the lists are asked for 0.005 more than the target when stored vectors stand in for the questions
gok, lok = r["graph"]["recall"] >= 0.95, r["lists"]["recall"] >= 0.955
check("when both reach what is asked of them the faster is kept", not (gok and lok) or best["ms"] <= other["ms"])
check("lists that do not reach target + 0.005 are not kept when the graph reaches the target", not (gok and not lok) or r["method"] == "graph")
a = [v.search(q, k=K) for q in Q[:50]]
kw = dict(method="graph", ef=best["ef"], rerank_n=max(K, best["ef"])) if r["method"] == "graph" else dict(method="lists", nprobe=best["nprobe"])
b = [v.search(q, k=K, **kw) for q in Q[:50]]
check("search() without a method follows the choice and its setting", a == b, str(kw))
check("its recall on other questions is near what was measured", recall(v) >= best["recall"] - 0.05, f"({recall(v):.4f} against {best['recall']:.4f})")
check("an explicit method still wins", recall(v, method="graph", ef=400) > 0.5 and recall(v, method="lists", nprobe=64) > 0.5)
r2 = v.choose_method(k=K, target=0.95, queries=Q)
check("choose_method(queries=...) runs", r2["method"] in ("graph", "lists"))
v._prefer = "graph"; v._graph_rn = 24; v.lib.kci_vec_set_default_ef(v._v, 24)
check("a kept graph width below the rerank is searched with a shortlist of that size",
      [v.search(q, k=K) for q in Q[:20]] == [v.search(q, k=K, method="graph", ef=24, rerank_n=24) for q in Q[:20]]
      and recall(v) <= recall(v, method="graph", ef=400) and recall(v, ef=24) == recall(v, method="graph", ef=200))
check("a graph preference is followed although the lists are built", [v.search(q, k=K) for q in Q[:20]] == [v.search(q, k=K, method="graph") for q in Q[:20]])
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
