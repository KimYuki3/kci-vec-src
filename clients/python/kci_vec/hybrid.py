"""Hybrid search: word matching (BM25) fused with the vector search, all in-process.

Embeddings miss exact things — product codes, names, numbers, rare words. A small BM25 index over the same
chunks catches them; the two rankings are fused by reciprocal rank (RRF), which needs no score scaling.

    from kci_vec import KciVec
    from kci_vec.hybrid import BM25Index, hybrid_search
    bm = BM25Index()
    bm.add(0, "Annual leave: 20 paid days ...")          # same ids as the vectors
    ...
    hits = hybrid_search(vec_index, bm, query_vec, "paid leave days", k=5)   # [(id, fused, vec_rank, bm25_rank)]
    bm.save("index.bm25.json"); bm = BM25Index.load("index.bm25.json")

Tokens: Latin / digit runs lower-cased as words; runs of CJK characters (Japanese, Chinese, Korean) as character
bigrams (plus the single character for 1-char runs), so no morphological analyser is needed. Pure Python / numpy;
tens of thousands of chunks search in milliseconds. The text index is stored as plain JSON — keep it beside the
encrypted vector snapshot inside your own encrypted store when the text is sensitive.
"""
import json
import math
import re
import unicodedata
from collections import defaultdict

_WORD = re.compile(r"[0-9A-Za-z_][0-9A-Za-z_\-\.]*|[぀-ヿ㐀-鿿豈-﫿ｦ-ﾟ]+")


def tokenize(text):
    """Lower-cased words for Latin / digits; character bigrams for CJK runs."""
    out = []
    for m in _WORD.finditer(unicodedata.normalize("NFKC", text)):
        t = m.group(0)
        if ord(t[0]) < 0x3000:
            out.append(t.lower().strip(".-"))
        elif len(t) == 1:
            out.append(t)
        else:
            out.extend(t[i:i + 2] for i in range(len(t) - 1))
    return [t for t in out if t]


class BM25Index:
    """BM25 (k1, b) over tokenized texts keyed by your ids (the same ids as the vectors)."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.post = defaultdict(dict)   # token -> {id: tf}
        self.dl = {}                    # id -> token count
        self.avgdl = 0.0

    def add(self, id, text):
        id = int(id)
        if id in self.dl:
            self.remove(id)
        toks = tokenize(text)
        self.dl[id] = len(toks)
        for t in toks:
            self.post[t][id] = self.post[t].get(id, 0) + 1
        self.avgdl = sum(self.dl.values()) / max(1, len(self.dl))

    def remove(self, id):
        id = int(id)
        if id not in self.dl:
            return False
        for t in list(self.post):
            self.post[t].pop(id, None)
            if not self.post[t]:
                del self.post[t]
        del self.dl[id]
        self.avgdl = sum(self.dl.values()) / max(1, len(self.dl))
        return True

    def __len__(self):
        return len(self.dl)

    def search(self, text, k=10, allowed=None):
        """[(id, bm25 score)], best first. allowed: optional set of ids to restrict to."""
        N = len(self.dl)
        if N == 0:
            return []
        scores = defaultdict(float)
        for t in set(tokenize(text)):
            plist = self.post.get(t)
            if not plist:
                continue
            idf = math.log(1.0 + (N - len(plist) + 0.5) / (len(plist) + 0.5))
            for id, tf in plist.items():
                if allowed is not None and id not in allowed:
                    continue
                dl = self.dl[id]
                scores[id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(id, float(s)) for id, s in top]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"k1": self.k1, "b": self.b, "dl": self.dl,
                       "post": {t: p for t, p in self.post.items()}}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        bm = cls(d.get("k1", 1.5), d.get("b", 0.75))
        bm.dl = {int(k): v for k, v in d["dl"].items()}
        for t, p in d["post"].items():
            bm.post[t] = {int(k): v for k, v in p.items()}
        bm.avgdl = sum(bm.dl.values()) / max(1, len(bm.dl))
        return bm


def rrf(rankings, k=10, c=60.0, weights=None):
    """Reciprocal rank fusion of several [(id, score)] lists: fused(id) = sum_i w_i / (c + rank_i(id)).
    Returns [(id, fused, [rank_i or None ...])], best first."""
    weights = weights or [1.0] * len(rankings)
    fused, ranks = defaultdict(float), defaultdict(lambda: [None] * len(rankings))
    for i, (lst, w) in enumerate(zip(rankings, weights)):
        for r, (id, _) in enumerate(lst):
            fused[id] += w / (c + r + 1)
            ranks[id][i] = r + 1
    top = sorted(fused.items(), key=lambda x: -x[1])[:k]
    return [(id, float(s), ranks[id]) for id, s in top]


def hybrid_search(vec, bm25, query_vec, query_text, k=10, depth=None, filter=None, w_vec=1.0, w_text=1.0, **search_kw):
    """Vector search (kci_vec.KciVec) + BM25 fused by RRF. depth: candidates taken from each side (default 4k).
    filter: a set of allowed ids (applied to both sides) or a callable id -> bool (vector side; BM25 filtered after).
    Returns [(id, fused_score, vec_rank, bm25_rank)], best first; a rank is None when that side did not return the id."""
    depth = depth or 4 * k
    vhits = vec.search(query_vec, k=depth, filter=filter, **search_kw)
    allowed = filter if isinstance(filter, (set, frozenset)) else None
    thits = bm25.search(query_text, k=depth, allowed=allowed)
    if callable(filter):
        thits = [(i, s) for i, s in thits if filter(i)]
    return [(id, s, r[0], r[1]) for id, s, r in rrf([vhits, thits], k=k, weights=[w_vec, w_text])]
