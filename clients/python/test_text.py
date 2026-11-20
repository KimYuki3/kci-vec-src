"""kci_vec.text.TextIndex with two toy models (no torch needed): the plumbing, not the search quality .

    KCI_VEC_LIB=... KCI_SPARSE_LIB=... python test_text.py"""
import os, shutil, sys, tempfile, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kci_vec.text import TextIndex, Hit

DIM, VOC = 48, 4000
fails = 0


def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    fails += not ok


def words(t): return t.lower().replace(".", " ").replace("?", " ").split()


def dense(texts, is_query):                     # a bag of words thrown into 48 dims: texts sharing words are close
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in words(t):
            rng = np.random.default_rng(zlib.crc32(w.encode()))
            out[i] += rng.standard_normal(DIM).astype(np.float32)
        n = np.linalg.norm(out[i]); out[i] = out[i] / n if n > 0 else out[i]
        if n == 0: out[i, 0] = 1.0
    return out


def sparse(texts, is_query):                    # one dimension per word, weight = how often
    rows = []
    for t in texts:
        c = {}
        for w in words(t): c[zlib.crc32(w.encode()) % VOC] = c.get(zlib.crc32(w.encode()) % VOC, 0) + 1.0
        ks = sorted(c)
        rows.append((np.array(ks, dtype=np.uint32), np.array([c[k] for k in ks], dtype=np.float32)))
    return rows


topics = ["paid leave days holiday", "expense report receipt travel", "password reset login account", "server backup restore disk",
          "invoice payment bank transfer", "meeting room booking calendar"]
texts = [f"{topics[i % 6]} note {i} filler{i % 17} extra{i % 5}" for i in range(600)]

ix = TextIndex(dense=dense, sparse=sparse, sparse_dim=VOC, lang="toy", verbose=False)
ids = ix.add(texts)
check(ids == list(range(600)) and len(ix) == 600, "add numbers the texts 0, 1, 2, ...")
h = ix.search("how do I reset my password", k=5)
check(len(h) == 5 and isinstance(h[0], Hit) and all(x.id % 6 == 2 for x in h), "search builds by itself and finds the password texts")
check(h[0].text == texts[h[0].id] and h[0].score >= h[1].score >= h[4].score, "hits carry the text and come best first")
check(any(x.dense_rank for x in h) and any(x.sparse_rank for x in h), "both sides contribute ranks")
h = ix.search("note 123 filler4", k=3)
check(h[0].id == 123, "a text is found by its own rare words")
h = ix.search("server backup", k=5, filter=lambda i: i >= 300)
check(len(h) == 5 and all(x.id >= 300 and x.id % 6 == 3 for x in h), "filter by a function")
h = ix.search("server backup", k=5, filter={3, 9, 15})
check(sorted(x.id for x in h)[:3] == [3, 9, 15][:len(h)] or {x.id for x in h} <= {3, 9, 15}, "filter by a set of ids")

# adding after the first search, own ids, a clash, delete
new = ix.add(["quarterly zebra budget forecast", "zebra crossing safety"], ids=[9000, 9001])
check(new == [9000, 9001] and len(ix) == 602, "add after searching, with own ids")
h = ix.search("zebra budget", k=2)
check(h[0].id == 9000 and {x.id for x in h} == {9000, 9001}, "texts added later are found")
check(ix.add("one more zebra text") == [9002], "a single string, numbered after the largest id")
try:
    ix.add(["again"], ids=[9000]); check(False, "an id already there is refused")
except ValueError:
    check(True, "an id already there is refused")
try:
    ix.add([b"bytes"]); check(False, "a text that is not a string is refused")
except TypeError:
    check(True, "a text that is not a string is refused")
check(ix.delete(9001) and not ix.delete(9001) and len(ix) == 602, "delete, once")
check(all(x.id != 9001 for x in ix.search("zebra crossing safety", k=5)), "a deleted text is not returned")

# the weight from judged questions. Here the right answers are what the sparse side finds (rare words), the dense toy
# model is the weaker side: from the middle the weight must not move towards dense, and a weight at the dense end
# (clearly losing) must be left
qs = [f"note {i} filler{i % 17}" for i in range(0, 120, 3)]; ans = [{i} for i in range(0, 120, 3)]
w, table = ix.tune(qs, ans, k=3)
check(0.0 <= w <= 0.4 and len(table) == 11 and ix.w_dense == w and max(v for _, v in table) > 0.9, f"tune keeps or lowers a middle weight (w_dense {w})")
ix.w_dense = 1.0
w1, table = ix.tune(qs, ans, k=3)
check(w1 < 1.0 and table[-1][1] < max(v for _, v in table), f"tune leaves a weight that clearly loses (1.0 -> {w1})")
ix.w_dense = w

from kci_vec.text import split_text
ja = "".join(f"これは{i}番目の文です。" for i in range(60)); sp = split_text(ja, 100)
check(split_text("short", 100) == [(0, 5)] and split_text(ja, 0) == [(0, len(ja))], "split_text: a short text, or size 0, is one part")
check(len(sp) > 3 and sp[0][0] == 0 and sp[-1][1] == len(ja) and all(e - a <= 100 for a, e in sp) and
      all(ja[e - 1] == "。" for a, e in sp) and all(sp[i][0] < sp[i + 1][0] <= sp[i][1] for i in range(len(sp) - 1)) and
      any(sp[i + 1][0] < sp[i][1] for i in range(len(sp) - 1)), f"split_text: parts end at sentence ends, overlap, cover the text ({len(sp)} parts)")
sp0 = split_text(ja, 100, overlap=0); raw = split_text("x" * 250, 100)
check(all(sp0[i + 1][0] == sp0[i][1] for i in range(len(sp0) - 1)) and raw[0] == (0, 100) and len(raw) == 3 and raw[2][1] == 250 and
      all(raw[i + 1][0] == raw[i][1] for i in range(2)) and min(e - a for a, e in raw) >= 50,
      "split_text: overlap 0 joins end to start; no place to cut = cut at the size, the last two parts about equal")
en = split_text("One sentence here. " * 30, 100)
check(all(e - a <= 100 for a, e in en) and all("One sentence here. " * 30 == ("One sentence here. " * 30)[:a] + ("One sentence here. " * 30)[a:] for a, e in en) and
      all(("One sentence here. " * 30)[e - 1] == "." for a, e in en[:-1]), "split_text: English sentences")
long_a = " ".join(f"{topics[0]} alpha{i}." for i in range(8)) + " " + " ".join(f"{topics[3]} beta{i}." for i in range(8)) + " " + " ".join(f"quokka wombat gamma{i}." for i in range(8))
long_b = " ".join(f"{topics[4]} delta{i}." for i in range(20))
cx = TextIndex(dense=dense, sparse=sparse, sparse_dim=VOC, lang="toy", verbose=False, chunk=120)
cx.add(texts[:200]); cx.add([long_a, long_b], ids=[5000, 5001])
check(len(cx) == 202 and len(cx._rows) > 206 and len(cx._doc_rows[5000]) >= 3, f"a long text becomes several parts, the index still counts texts ({len(cx._rows)} parts)")
h = cx.search("quokka wombat", k=5)
check(h[0].id == 5000 and [x.id for x in h].count(5000) == 1 and "quokka" in h[0].text and len(h[0].text) <= 120 and
      long_a[h[0].span[0]:h[0].span[1]] == h[0].text and cx.texts[5000] == long_a, "a search answers with the text once, hit.text = the part that matched, span = where")
h = cx.search("server backup restore disk beta3", k=3)
check(h[0].id == 5000 and "beta3" in h[0].text, "another part of the same text is found by its own words")
h = cx.search("paid leave", k=3)
check(h[0].span == (0, len(cx.texts[h[0].id])) or h[0].id == 5000, "a short text is one part: span = the whole text")
check({x.id for x in cx.search("invoice payment delta", k=5, filter={5001, 4})} <= {5001, 4} and
      cx.search("invoice payment delta", k=5, filter=lambda i: i == 5001)[0].id == 5001 and cx.search("invoice", k=5, filter=set()) == [], "filters are about texts, not parts")
cw, _ = cx.tune(["quokka wombat", "invoice payment delta3"], [{5000}, {5001}], k=3)
check(0.0 <= cw <= 1.0, "tune counts texts, not parts")
ctmp = tempfile.mkdtemp(prefix="kci_text_c_")
try:
    before = [(x.id, x.span, x.text) for x in cx.search("quokka wombat", k=4)]
    cx.save(os.path.join(ctmp, "c"), key="k")
    c2 = TextIndex.load(os.path.join(ctmp, "c"), key="k", dense=dense, sparse=sparse, verbose=False)
    check([(x.id, x.span, x.text) for x in c2.search("quokka wombat", k=4)] == before and c2.chunk == 120 and len(c2) == 202, "save / load keeps the parts")
    check(c2.add("z" * 50 + ". " + "newword " * 40) == [5002] and c2.search("newword", k=1)[0].id == 5002 and len(c2._doc_rows[5002]) > 1, "a loaded index cuts new long texts too")
    check(cx.delete(5000) and all(x.id != 5000 for x in cx.search("quokka wombat", k=5)) and 5000 not in cx._doc_rows and len(cx) == 201, "delete removes every part of a text")
    import json as _json
    o = TextIndex(dense=dense, sparse=sparse, sparse_dim=VOC, lang="toy", verbose=False); o.add(texts[:100]); o.save(os.path.join(ctmp, "o"))
    want = [(x.id, x.text) for x in o.search("expense receipt", k=5)]
    mp = os.path.join(ctmp, "o", "meta.json"); m = _json.load(open(mp))
    for f in ("rows", "row_next", "chunk", "overlap"): m.pop(f)
    m["format"] = 1; _json.dump(m, open(mp, "w"))
    o1 = TextIndex.load(os.path.join(ctmp, "o"), dense=dense, sparse=sparse, verbose=False)
    got = o1.search("expense receipt", k=5)
    check([(x.id, x.text) for x in got] == want and got[0].span == (0, len(got[0].text)) and o1.add("later text") == [100] and o1.delete(3), "an index saved before texts were cut into parts still loads")
finally:
    shutil.rmtree(ctmp, ignore_errors=True)

dx = TextIndex(dense=dense, sparse=sparse, sparse_dim=VOC, lang="toy", verbose=False, chunk=120, dense_chunk=400)
dx.add(texts[:200]); dx.add([long_a, long_b], ids=[5000, 5001])
nd5 = sum(1 for r in dx._doc_rows[5000] if dx._side[r] == "d"); ns5 = sum(1 for r in dx._doc_rows[5000] if dx._side[r] == "s")
check(dx.dense_chunk == 400 and dx.chunk == 120 and 1 < nd5 < ns5 and all(dx._side[r] == "b" for d in range(200) for r in dx._doc_rows[d]),
      f"a long text: {nd5} dense parts, {ns5} sparse parts; a short text is one row on both sides")
check(dx._nside["d"] == 200 + nd5 + sum(1 for r in dx._doc_rows[5001] if dx._side[r] != "s") and dx._nside["s"] > dx._nside["d"], "rows counted per side")
h = dx.search("quokka wombat", k=5)
check(h[0].id == 5000 and [x.id for x in h].count(5000) == 1 and h[0].dense_rank and h[0].sparse_rank and long_a[h[0].span[0]:h[0].span[1]] == h[0].text and "quokka" in h[0].text,
      "found once, by both sides, with a part that holds the words")
dtmp = tempfile.mkdtemp(prefix="kci_text_d_")
try:
    before = [(x.id, x.span, round(x.score, 5)) for x in dx.search("server backup restore disk beta3", k=4)]
    dx.save(os.path.join(dtmp, "d"))
    d2 = TextIndex.load(os.path.join(dtmp, "d"), dense=dense, sparse=sparse, verbose=False)
    check([(x.id, x.span, round(x.score, 5)) for x in d2.search("server backup restore disk beta3", k=4)] == before and d2.dense_chunk == 400 and d2._nside == dx._nside,
          "save / load keeps which side holds which part")
    check(d2.delete(5000) and all(x.id != 5000 for x in d2.search("quokka wombat", k=5)) and d2._nside["d"] == dx._nside["d"] - nd5 and d2._nside["s"] == dx._nside["s"] - ns5,
          "delete removes the parts of both sides")
    import json as _json2
    o2 = TextIndex(dense=dense, sparse=sparse, sparse_dim=VOC, lang="toy", verbose=False, chunk=120)      # one part size, as format 2 had
    o2.add(texts[:100]); o2.add([long_a], ids=[5000]); o2.save(os.path.join(dtmp, "o2"))
    want2 = [(x.id, x.span) for x in o2.search("quokka wombat", k=3)]
    mp = os.path.join(dtmp, "o2", "meta.json"); m = _json2.load(open(mp))
    check(m["format"] == 3 and all(e[4] == "b" for e in m["rows"]), "one part size: every row is on both sides")
    m["format"] = 2; m.pop("dense_chunk"); m["rows"] = [e[:4] for e in m["rows"]]; _json2.dump(m, open(mp, "w"))
    d3 = TextIndex.load(os.path.join(dtmp, "o2"), dense=dense, sparse=sparse, verbose=False)
    check(d3.dense_chunk == d3.chunk == 120 and [(x.id, x.span) for x in d3.search("quokka wombat", k=3)] == want2 and d3._nside == o2._nside,
          "a format-2 save (one part size, no side) still loads and answers the same")
finally:
    shutil.rmtree(dtmp, ignore_errors=True)

# one side only
d = TextIndex(dense=dense, mode="dense", lang="toy", verbose=False); d.add(texts[:100])
s = TextIndex(sparse=sparse, sparse_dim=VOC, mode="sparse", lang="toy", verbose=False); s.add(texts[:100])
hd, hs = d.search("expense receipt", k=3), s.search("expense receipt", k=3)
check(all(x.id % 6 == 1 and x.sparse_rank is None for x in hd) and all(x.id % 6 == 1 and x.dense_rank is None for x in hs), "mode dense / sparse")
check(TextIndex(dense=dense, sparse=sparse, sparse_dim=VOC, lang="toy", verbose=False).search("anything") == [], "an empty index returns nothing")

# save / load
tmp = tempfile.mkdtemp(prefix="kci_text_")
try:
    before = [(x.id, round(x.score, 5)) for x in ix.search("invoice payment", k=8)]
    p = os.path.join(tmp, "plain")
    check(ix.save(p) is True and os.path.exists(os.path.join(p, "texts.json")), "save without a key writes the texts")
    a = TextIndex.load(p, dense=dense, sparse=sparse, verbose=False)
    after = a.search("invoice payment", k=8)
    check([(x.id, round(x.score, 5)) for x in after] == before and after[0].text == texts[after[0].id] and len(a) == len(ix) and a.w_dense == ix.w_dense,
          "load gives the same answers, texts and weight")
    check(a.add("fresh text about zebra invoices") == [9003] and a.search("fresh zebra invoices", k=1)[0].id == 9003, "a loaded index takes more texts")
    k = os.path.join(tmp, "locked")
    check(ix.save(k, key="correct horse") is True and not os.path.exists(os.path.join(k, "texts.json")) and os.path.exists(os.path.join(k, "texts.kct")),
          "save with a key writes the texts encrypted, no plain copy")
    raw = b"".join(open(os.path.join(k, f), "rb").read() for f in os.listdir(k))
    check(b"zebra" not in raw and b"password" not in raw and b"filler" not in raw, "no text in any file of the encrypted save")
    b = TextIndex.load(k, key="correct horse", dense=dense, sparse=sparse, verbose=False)
    hb = b.search("invoice payment", k=8)
    check([(x.id, round(x.score, 5)) for x in hb] == before and hb[0].text == texts[hb[0].id], "load with the key: same answers and the texts")
    blob = bytearray(open(os.path.join(k, "texts.kct"), "rb").read()); blob[len(blob) // 2] ^= 1
    open(os.path.join(k, "texts.kct"), "wb").write(bytes(blob))
    try:
        TextIndex.load(k, key="correct horse", dense=dense, sparse=sparse, verbose=False); check(False, "changed encrypted texts are refused")
    except RuntimeError:
        check(True, "changed encrypted texts are refused")
    b2 = TextIndex.load(k, key="correct horse", texts={i: "mine " + t for i, t in enumerate(texts)}, dense=dense, sparse=sparse, verbose=False)
    check(b2.search("invoice payment", k=1)[0].text.startswith("mine "), "load with texts= uses those (the damaged file is not read)")
    nt = TextIndex(dense=dense, sparse=sparse, sparse_dim=VOC, lang="toy", keep_text=False, verbose=False); nt.add(texts[:50])
    k2 = os.path.join(tmp, "notext")
    check(nt.save(k2, key="correct horse") is False and not os.path.exists(os.path.join(k2, "texts.kct")) and
          TextIndex.load(k2, key="correct horse", dense=dense, sparse=sparse, verbose=False).search("paid leave", k=1)[0].text is None, "keep_text=False: no texts written, none returned")
    for bad, what in ((dict(key="wrong horse"), "a wrong key is refused"), (dict(), "an encrypted index without a key is refused")):
        try:
            TextIndex.load(k, dense=dense, sparse=sparse, verbose=False, **bad); check(False, what)
        except RuntimeError:
            check(True, what)
    ix.save(p, key="now locked")
    check(not os.path.exists(os.path.join(p, "texts.json")), "saving with a key over a plain save removes the plain texts")
    ix.save(p)
    check(not os.path.exists(os.path.join(p, "texts.kct")) and os.path.exists(os.path.join(p, "texts.json")), "and the other way round")
    try:
        TextIndex.load(os.path.join(tmp, "nothing"), verbose=False); check(False, "a folder that is not an index is refused")
    except RuntimeError:
        check(True, "a folder that is not an index is refused")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("test_text.py:", "all OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
