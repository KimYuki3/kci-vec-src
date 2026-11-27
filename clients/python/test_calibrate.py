"""build() measures the search width a large index needs; save / load keep it."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from kci_vec import KciVec

ok = True
def check(name, cond, extra=""):
    global ok
    print(("ok   " if cond else "FAIL ") + name, extra)
    ok = ok and bool(cond)

N, D, K = 60000, 200, 10          # 12,000,000 numbers: above exact_cells; independent noise in 200 dims is hard for a graph
rng = np.random.default_rng(9)
X = rng.standard_normal((N, D)).astype(np.float32)
Q = rng.standard_normal((100, D)).astype(np.float32)
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
gt = [set(np.argsort(-(Xn @ q), kind="stable")[:K].tolist()) for q in Q]
def recall(v, **kw):
    return sum(len(g & {i for i, _ in v.search(q, k=K, **kw)}) for q, g in zip(Q, gt)) / (K * len(Q))

v = KciVec(dim=D); v.add_many(np.arange(N, dtype=np.int64), X); v.build()
ef = v.default_ef()
r_def, r_200 = recall(v), recall(v, ef=200)
check("build() calibrated the index", ef in (200, 400, 800, 1600), f"(ef {ef})")
check("hard data gets a wider search than 200", ef > 200)
check("search() with nothing given uses it: recall above the old default's", r_def > r_200 + 0.02, f"(ef 200: {r_200:.4f}, default: {r_def:.4f})")
check("an explicit ef still wins", abs(recall(v, ef=ef) - r_def) < 1e-9 and recall(v, ef=200) == r_200)
e2, rec = v.calibrate()
check("calibrate() returns (ef, recall) and repeats", e2 == ef and (rec >= 0.98 or ef == 1600), f"(sample recall {rec:.4f})")
p = os.path.join(tempfile.mkdtemp(), "c.kvec"); v.save(p, key=b"k")
w = KciVec.load(p, key=b"k", rerank=200)
check("load() restores the width", w.default_ef() == ef and abs(recall(w) - r_def) < 0.01)

v = KciVec(dim=D); v.auto_ef = False; v.add_many(np.arange(N, dtype=np.int64), X); v.build()
check("auto_ef = False: not calibrated, the old default", v.default_ef() == 100 and recall(v) < r_def - 0.02)

v = KciVec(dim=D); v.add_many(np.arange(2000, dtype=np.int64), X[:2000]); v.build()
check("a small index is not calibrated (it answers by brute force)", v.default_ef() == 100 and v._small_exact())
v = KciVec(dim=D, rerank=0); v.add_many(np.arange(N, dtype=np.int64), X); v.build()
check("rerank=0 (no floats): not calibrated, build() does not fail", v.default_ef() == 100)
try:
    v.calibrate(); check("calibrate() without floats raises", False)
except RuntimeError:
    check("calibrate() without floats raises", True)
v = KciVec(dim=D, metric="ip"); v.add_many(np.arange(N, dtype=np.int64), X); v.build()
check("ip: calibrated too", v.default_ef() in (200, 400, 800, 1600) and len(v.search(Q[0], k=K)) == K, f"(ef {v.default_ef()})")
gi = [set(np.argsort(-(X @ q), kind="stable")[:K].tolist()) for q in Q]
ri = lambda **kw: sum(len(g & {i for i, _ in v.search(q, k=K, **kw)}) for q, g in zip(Q, gi)) / (K * len(Q))
r_self = ri(); e_q, m_q = v.calibrate(queries=Q); r_q = ri()
check("ip: calibrate(queries=...) measures the questions themselves", abs(m_q - r_q) < 0.02 and v.calibrated_recall == m_q, f"(stored vectors as questions: {r_self:.4f}; own questions: ef {e_q}, measured {m_q:.4f}, got {r_q:.4f})")
try:
    v.calibrate(queries=Q[:3]); check("calibrate(queries) with 3 questions raises", False)
except RuntimeError:
    check("calibrate(queries) with 3 questions raises", True)
Xo = (rng.standard_normal((30000, 64)) + 3.0).astype(np.float32); Qo = rng.standard_normal((100, 64)).astype(np.float32)
v = KciVec(dim=64, metric="ip"); v.add_many(np.arange(30000, dtype=np.int64), Xo); v.build()
go = [np.argsort(-(Xo @ q), kind="stable")[:K].tolist() for q in Qo]
check("ip: a small index gives numpy's top 10 in order", v._small_exact() and all([i for i, _ in v.search(q, k=K)] == g for q, g in zip(Qo, go)))
few = {7, 12345, 29998}
check("ip: a set of 3 ids returns the 3, best first", all([i for i, _ in v.search(q, k=K, filter=few)] == sorted(few, key=lambda i: -float(Xo[i] @ q)) for q in Qo[:20]))
# build() says so when the calibration stays far below its target
import warnings
Xh = rng.standard_normal((60000, 256)).astype(np.float32)
v = KciVec(dim=256, M=4, efc=20); v.add_many(np.arange(60000, dtype=np.int64), Xh)
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always"); v.build()
check("build() warns when the measured recall is far below the target", any("will miss some" in str(x.message) for x in wl) and v.calibrated_recall < 0.97, f"(measured {v.calibrated_recall:.4f} at ef {v.default_ef()})")
v = KciVec(dim=D); v.add_many(np.arange(N, dtype=np.int64), X)
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always"); v.build()
check("no warning when the target is reached", len(wl) == 0, f"(measured {v.calibrated_recall:.4f})")

G = 48; cen = rng.standard_normal((60, G)).astype(np.float32)
Xg = (cen[np.arange(24000) % 60] + 0.02 * rng.standard_normal((24000, G))).astype(np.float32)
v = KciVec(dim=G); v.exact_cells = 0; v.add_many(np.arange(24000, dtype=np.int64), Xg); v.build(); v.calibrate()    # exact_cells 0: build() leaves a small index alone
Qg = (cen[np.arange(50) % 60] + 0.02 * rng.standard_normal((50, G))).astype(np.float32)
Xn = Xg / np.linalg.norm(Xg, axis=1, keepdims=True)
gtg = [set(np.argsort(-(Xn @ q), kind="stable")[:K].tolist()) for q in Qg]
rg = lambda **kw: sum(len(g & {i for i, _ in v.search(q, k=K, **kw)}) for q, g in zip(Qg, gtg)) / (K * len(Qg))
r_wide, r_200 = rg(), rg(rerank_n=200)
check("groups of 400: the calibration widens the float pass", v.default_rerank() == v.default_ef() > 200, f"(ef {v.default_ef()}, rerank {v.default_rerank()})")
check("groups of 400: search() with nothing given uses it", r_wide >= 0.95 and r_wide > r_200 + 0.1, f"(rerank 200: {r_200:.4f}, default: {r_wide:.4f})")

print("ALL OK" if ok else "FAILED"); sys.exit(0 if ok else 1)
