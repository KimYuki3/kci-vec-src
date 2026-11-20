"""Sparse index + hybrid search smoke test, no model needed.
    KCI_VEC_LIB=/path/libkci_vec.so KCI_SPARSE_LIB=/path/libkci_sparse.so python clients/python/test_sparse.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kci_vec import KciVec
from kci_vec.sparse import KciSparse, hybrid_search_sparse, seal, unseal
from kci_vec.hybrid import mix, rrf, choose_weight

fails = 0
def check(cond, msg):
    global fails
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails += 1

def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False

rng = np.random.default_rng(7)
DIM, N = 500, 3000

# rows in CSR form, and the same rows as a dense matrix for the all-pairs answer
dense = np.zeros((N, DIM), dtype=np.float32)
indptr, indices, data = [0], [], []
for d in range(N):
    idx = np.unique((rng.random(rng.integers(5, 40)) ** 3 * DIM).astype(np.int64))
    val = (0.05 + 3 * rng.random(len(idx))).astype(np.float32)
    dense[d, idx] = val
    indices.extend(idx); data.extend(val); indptr.append(len(indices))
ids = np.arange(N) + 10_000

sp = KciSparse(DIM)
check(sp.add_csr(ids, indptr, indices, data) == N and len(sp) == N, "add_csr")
check(raises(RuntimeError, lambda: sp.search([1], [1.0])), "search before build is refused")
sp.build(threads=2)

bad = 0
for _ in range(50):
    qi = np.unique(rng.integers(0, DIM, 12)); qv = (0.1 + rng.random(len(qi))).astype(np.float32)
    q = np.zeros(DIM, dtype=np.float32); q[qi] = qv
    ref = dense @ q
    order = np.argsort(-ref, kind="stable")[:20]
    got = sp.search(qi, qv, k=20)
    if len(got) != 20: bad += 1; continue
    for (gid, gs), d in zip(got, order):
        if abs(gs - ref[d]) > 1e-3 * (1 + abs(ref[d])): bad += 1; break
    approx = sp.search(qi, qv, k=20, budget=2000, rerank=60)
    for gid, gs in approx:
        if abs(gs - ref[gid - 10_000]) > 1e-3 * (1 + abs(ref[gid - 10_000])): bad += 1; break
check(bad == 0, "exact search equals all-pairs; budgeted search returns exact scores")

check(raises(RuntimeError, lambda: sp.search_blocks([1], [1.0])), "search_blocks before build_blocks is refused")
sp.build_blocks(list_len=10**6, block_size=8, summary_mass=1.0, threads=2)
bad = 0
for _ in range(30):
    qi = np.unique(rng.integers(0, DIM, 12)); qv = (0.1 + rng.random(len(qi))).astype(np.float32)
    ex = sp.search(qi, qv, k=20); bl = sp.search_blocks(qi, qv, k=20, query_terms=0, heap_factor=1.0)
    if len(ex) != len(bl) or any(abs(a[1] - b[1]) > 1e-3 * (1 + abs(a[1])) for a, b in zip(ex, bl)): bad += 1
check(bad == 0, "block search with nothing cut equals the exact search")
sp.build_blocks(threads=2); sp.build_neighbours(10, threads=2)
h0 = h1 = 0
for _ in range(30):
    qi = np.unique(rng.integers(0, DIM, 12)); qv = (0.1 + rng.random(len(qi))).astype(np.float32)
    ex = {i for i, _ in sp.search(qi, qv, k=20)}
    h0 += len(ex & {i for i, _ in sp.search_blocks(qi, qv, k=20, query_terms=3, heap_factor=0.8)})
    h1 += len(ex & {i for i, _ in sp.search_blocks(qi, qv, k=20, query_terms=3, heap_factor=0.8, neighbours=10)})
check(h1 >= h0 > 0, f"neighbours do not lower the recall ({h0} -> {h1} of 600)")
check(raises(ValueError, lambda: sp.search_blocks([1], [1.0], heap_factor=0.0)), "heap_factor 0 refused")
sp.drop_lists()
check(raises(RuntimeError, lambda: sp.search([1], [1.0])) and len(sp.search_blocks([1], [1.0], k=3)) > 0, "after drop_lists: search refused, search_blocks works")
check(raises(ValueError, lambda: sp.search_blocks([1], [1.0], heap_factor=4.5)) and len(sp.search_blocks([1], [1.0], k=3, heap_factor=1.5)) > 0,
      "heap_factor above 1 works up to 4")
import os, tempfile
_tmp = os.path.join(tempfile.mkdtemp(), "drop.kcs")
sp.save(_tmp); sp2 = KciSparse.load(_tmp)
before = sp2.search_blocks([1, 5], [1.0, 0.5], k=5, query_terms=0, heap_factor=1.0)
check(sp2.has_rows, "has_rows before drop_rows")
sp2.drop_rows()
after = sp2.search_blocks([1, 5], [1.0, 0.5], k=5, query_terms=0, heap_factor=1.0)
check(not sp2.has_rows and len(after) == len(before) > 0 and all(abs(a[1] - b[1]) <= 2e-3 * (abs(b[1]) + 1) for a, b in zip(after, before)),
      "after drop_rows: search_blocks gives the same scores to within the 16-bit weights")
check(raises(RuntimeError, lambda: sp2.drop_rows()) and raises(RuntimeError, lambda: sp2.build_blocks()), "after drop_rows: building again is refused")
sp2.save(_tmp); sp3 = KciSparse.load(_tmp)
check(not sp3.has_rows and sp3.search_blocks([1, 5], [1.0, 0.5], k=5, query_terms=0, heap_factor=1.0) == after, "an index without rows saves and loads to the same")
del sp2, sp3; os.remove(_tmp)
sp.build(threads=2)

# one row by hand, a repeat summed
check(sp.add(99, [3, 3, 7], [1.0, 2.0, 0.5]) == N + 1, "add one row")
check(raises(RuntimeError, lambda: sp.search([3], [1.0])), "search after add is refused until build")
sp.build()
top = sp.search([3], [100.0], k=N + 1)
check(any(i == 99 and abs(s - 300.0) < 1e-3 for i, s in top), "repeat summed (3: 1 + 2) and found")
check(sp.delete(99) == 1 and sp.delete(99) == 0 and len(sp) == N, "delete")
check(all(i != 99 for i, _ in sp.search([3], [100.0], k=N + 1)), "deleted row not returned (no rebuild)")

# filter: a set, and a callable
qi = np.unique(rng.integers(0, DIM, 12)); qv = (0.1 + rng.random(len(qi))).astype(np.float32)
full = sp.search(qi, qv, k=N)
even = [h for h in full if h[0] % 2 == 0][:15]
check(sp.search(qi, qv, k=15, filter=lambda i: i % 2 == 0) == even, "filter (callable): the allowed part of the full ranking")
pick = {full[3][0], full[40][0], full[7][0]}
check([i for i, _ in sp.search(qi, qv, k=15, filter=pick)] == [h[0] for h in full if h[0] in pick], "filter (set of ids)")
check(sp.search(qi, qv, k=15) == full[:15], "no filter left behind on the searcher")

# insert into a built index: found at once, with the exact score; a rebuild keeps it
check(raises(RuntimeError, lambda: KciSparse(8).insert(1, [1], [1.0])), "insert before build is refused")
n0 = len(sp)
check(sp.insert(777, [3, 9], [50.0, 1.0]) == n0 + 1 and sp.pending == 1, "insert, pending")
top = sp.search([3], [1.0], k=1)
check(top == [(777, 50.0)], f"inserted row found at once: {top}")
sp.build(threads=2)
check(sp.pending == 0 and sp.search([3], [1.0], k=1) == [(777, 50.0)], "still first after the rebuild")
check(sp.delete(777) == 1 and len(sp) == n0, "delete the inserted row")

# bad input
check(raises(ValueError, lambda: sp.add(1, [DIM], [1.0])), "dimension >= dim refused")
check(raises(ValueError, lambda: sp.add(1, [-1], [1.0])), "negative dimension refused")
check(raises(ValueError, lambda: sp.add(1, [1], [float("nan")])), "NaN refused")
check(raises(ValueError, lambda: sp.add(1, [1, 2], [1.0])), "length mismatch refused")
check(raises(ValueError, lambda: sp.add(1, [1.5], [1.0])), "non-integer dimension refused")
check(raises(ValueError, lambda: sp.add_csr([1, 2], [0, 1], [1], [1.0])), "bad indptr refused")
check(raises(ValueError, lambda: sp.search([1], [1.0], k=0)), "k 0 refused")
check(len(sp) == N and sp.search([], [], k=5) == [], "refused calls changed nothing; empty query gives nothing")
check(sp.nbytes > 0, "nbytes")

# hybrid: the dense side knows the topic, the sparse side knows the code word; only the merge finds the document
TOP, PER, VD = 12, 30, 64
centre = rng.standard_normal((TOP, VD)).astype(np.float32)
centre /= np.linalg.norm(centre, axis=1, keepdims=True)
def about(t):
    v = centre[t] + 0.08 * rng.standard_normal(VD).astype(np.float32)
    return v / np.linalg.norm(v)
vec = KciVec(dim=VD)
hs = KciSparse(PER)
for t in range(TOP):
    for j in range(PER):
        vec.add(t * 100 + j, about(t))
        hs.add(t * 100 + j, [j], [1.0])
vec.build(); hs.build()
first = both = 0
for t in range(TOP):
    for j in range(0, PER, 3):
        hits = hybrid_search_sparse(vec, hs, about(t), [j], [1.0], k=5, depth=PER)
        first += hits[0][0] == t * 100 + j
        both += hits[0][2] is not None and hits[0][3] is not None
nq = TOP * (PER // 3)
check(first == nq and both == nq, f"hybrid puts the wanted document first: {first}/{nq}")
hits = hybrid_search_sparse(vec, hs, about(2), [5], [1.0], k=5, depth=PER, filter=lambda i: i // 100 != 2)
check(len(hits) > 0 and all(h[0] // 100 != 2 for h in hits), "hybrid with a filter: no result from the excluded topic")
hm = hybrid_search_sparse(vec, hs, about(0), [4], [1.0], k=3, depth=PER, method="mix")
assert len(hm) == 3 and hm[0][1] >= hm[1][1] >= hm[2][1], hm
only_sparse = hybrid_search_sparse(vec, hs, about(0), [4], [1.0], k=3, depth=PER, w_vec=0.0)
check([h[0] for h in only_sparse] == [i for i, _ in hs.search([4], [1.0], k=3)], "w_vec 0 gives the sparse order")

import tempfile
tmp = os.path.join(tempfile.mkdtemp(), "hs.kcs")
hs.build_blocks(); hs.build_neighbours(5)
for key in (None, b"secret"):
    hs.save(tmp, key=key)
    h2 = KciSparse.load(tmp, key=key)
    check(len(h2) == len(hs) and h2.dim == hs.dim, "load: count and dim")
    check(all(h2.search([j], [1.0], k=5) == hs.search([j], [1.0], k=5) for j in range(0, PER, 4)), "load: search gives the same")
    check(all(h2.search_blocks([j], [1.0], k=5, neighbours=5) == hs.search_blocks([j], [1.0], k=5, neighbours=5) for j in range(0, PER, 4)),
          "load: search_blocks gives the same")
    h2.close()
for bad in (b"wrong", None):
    try:
        KciSparse.load(tmp, key=bad); check(False, "load with the wrong / no key must fail")
    except RuntimeError:
        pass
os.remove(tmp)

sp.close(); hs.close()
la, lb = [(1, 0.9), (2, 0.8), (3, 0.5), (4, 0.1)], [(3, 30.0), (5, 12.0), (1, 10.0)]
m = mix([la, lb], k=8)
assert [i for i, _, _ in m] == [3, 1, 2, 5, 4], m
assert abs(m[0][1] - 1.5) < 1e-9 and abs(m[2][1] - 0.875) < 1e-9 and m[0][2] == [3, 1] and m[3][2] == [None, 2], m
assert [i for i, _, _ in mix([la, lb], k=3, weights=[1.0, 0.2])] == [1, 2, 3]
# side a is right on every query, side b is noise: the chosen weight must end on side a's order
qa = [[(10 * q + j, 1.0 - 0.1 * j) for j in range(5)] for q in range(6)]
qb = [[(10 * q + 4 - j, 5.0 - j) for j in range(5)] for q in range(6)]
w, table = choose_weight(qa, qb, [{10 * q} for q in range(6)], k=3)
assert w >= 0.6 and table[0][0] == 0 and table[-1][0] == 1 and table[-1][1] == 1.0 and table[0][1] == 0.0, (w, table)
w2, _ = choose_weight(qb, qa, [{10 * q} for q in range(6)], k=3, method="rrf")
assert w2 <= 0.4, w2
# prefer=: a weight that is as good as the best is kept; one that clearly loses is left
w3, _ = choose_weight(qa, qb, [{10 * q} for q in range(6)], k=3, prefer=0.9)
assert w3 == 0.9, w3
w4, _ = choose_weight(qa, qb, [{10 * q} for q in range(6)], k=3, prefer=0.0)
assert w4 >= 0.5, w4
try:
    choose_weight(qa, qb[:2], [{0}] * 6); raise SystemExit("choose_weight took lists of different length")
except ValueError:
    pass
print("ok   mix / choose_weight")

blob = seal("有給休暇 paid leave".encode("utf-8"), "pass phrase")
assert len(blob) == len("有給休暇 paid leave".encode("utf-8")) + 40 and b"paid" not in blob
assert unseal(blob, "pass phrase").decode("utf-8") == "有給休暇 paid leave" and unseal(seal(b"", b"k"), b"k") == b""
for bad in (lambda: unseal(blob, "pass phrasf"), lambda: unseal(blob[:-1], "pass phrase"), lambda: unseal(blob[:5] + b"X" + blob[6:], "pass phrase"),
            lambda: unseal(b"short", "pass phrase")):
    try:
        bad(); raise SystemExit("unseal accepted a wrong key or a damaged block")
    except RuntimeError:
        pass
print("ok   seal / unseal")

print("test_sparse.py:", "all OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
