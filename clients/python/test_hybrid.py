"""Hybrid (BM25 + vectors) smoke test, no model needed.
    KCI_VEC_LIB=/path/libkci_vec.so python clients/python/test_hybrid.py"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kci_vec import KciVec
from kci_vec.hybrid import BM25Index, tokenize, rrf, hybrid_search

fails = 0
def check(cond, msg):
    global fails
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails += 1

# tokenizer: Latin words lower-cased, product codes kept, CJK as bigrams
t = tokenize("Order KX-204B for Tanaka. 血圧の記録を確認")
check("kx-204b" in t and "tanaka" in t, f"latin tokens kept: {t[:4]}")
check("血圧" in t and "圧の" in t and "記録" in t, "CJK bigrams")
check(tokenize("薬") == ["薬"], "single CJK char kept")

docs = {
    0: "Annual leave: full-time staff accrue 20 paid days per year.",
    1: "Order KX-204B shipped to Tanaka on Monday; invoice INV-7781.",
    2: "Security: customer data must never leave the premises.",
    3: "田中さんの血圧の記録: 朝 128/82、夜 135/88。次回は 11 月 20 日。",
    4: "Expenses: meals are reimbursed up to 50 per day.",
}
bm = BM25Index()
for i, d in docs.items(): bm.add(i, d)
check(len(bm) == 5, "5 docs indexed")
check(bm.search("KX-204B", k=3)[0][0] == 1, "exact product code finds its chunk")
check(bm.search("INV-7781 invoice", k=1)[0][0] == 1, "invoice number")
check(bm.search("血圧 記録", k=1)[0][0] == 3, "Japanese bigram query")
check(bm.search("田中", k=2)[0][0] == 3, "Japanese name")
check(bm.search("zzz-nothing", k=3) == [], "unknown term -> empty")
check(bm.search("staff leave", k=5, allowed={2, 4}) == [] or all(i in {2, 4} for i, _ in bm.search("staff leave", k=5, allowed={2, 4})), "allowed set respected")
bm.remove(1)
check(bm.search("KX-204B", k=3) == [] and len(bm) == 4, "remove")
bm.add(1, docs[1])
p = os.path.join(tempfile.gettempdir(), "kci_hybrid_smoke.json"); bm.save(p); bm2 = BM25Index.load(p); os.remove(p)
check(bm2.search("Tanaka", k=1)[0][0] == 1 and len(bm2) == 5, "save / load")

# RRF: an id ranked first by one side and absent from the other still fuses high
f = rrf([[(7, 0.9), (8, 0.8), (9, 0.7)], [(9, 12.0), (5, 3.0)]], k=4)
check(f[0][0] == 9 and f[0][2] == [3, 1], f"rrf puts the id ranked by both first: {f[0]}")

# hybrid with a real vector index: random vectors, so the exact-term side must carry the code query
rng = np.random.default_rng(3); d = 32
X = rng.standard_normal((5, d)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
v = KciVec(dim=d, rerank=10); v.add_many(range(5), X); v.build_lists(nlist=2)
q = rng.standard_normal(d).astype(np.float32); q /= np.linalg.norm(q)
h = hybrid_search(v, bm, q, "KX-204B invoice", k=3)
check(h[0][0] == 1 and h[0][3] == 1, f"hybrid: exact code wins the fusion: {h[0]}")
check(all(len(x) == 4 for x in h), "hybrid result shape (id, fused, vec_rank, bm25_rank)")
h2 = hybrid_search(v, bm, q, "KX-204B", k=3, filter={0, 2, 3})
check(all(i in {0, 2, 3} for i, *_ in h2), "hybrid honours a set filter on both sides")
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
