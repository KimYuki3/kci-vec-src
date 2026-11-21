"""kci_vec.text — search your texts by handing over the texts.

    from kci_vec.text import TextIndex

    ix = TextIndex("ja")                               # "ja" Japanese, "en" English
    ix.add(["有給休暇は入社半年後に10日付与されます。", "経費の精算は月末締めです。", ...])
    for hit in ix.search("有給は何日もらえる?", k=3):
        print(hit.id, round(hit.score, 3), hit.text)
    ix.save("my_index")                                # later: ix = TextIndex.load("my_index")

What happens inside: every text is turned into two vectors by two language models that run on this machine
(nothing is sent anywhere after the models have been downloaded once): a dense one (the meaning as a whole)
and a sparse one (which words matter, including related words). A question is searched in both indexes (KciVec
and KciSparse) and the two result lists are merged by score (hybrid.mix).

Needs, besides kci_vec: torch and transformers (pip install torch transformers); for "ja" also fugashi and
unidic-lite (the sparse model's word splitter). The first use downloads the models (a few hundred MB). Turning
texts into vectors is the slow part: on a laptop CPU 4 to 8 texts a second, so 10,000 texts take 20 to 40
minutes; a question takes about 0.07 s. add() can be called again later; it prints its progress.

Long texts: a text longer than the models read at once is cut into overlapping parts at sentence ends
(TextIndex(chunk=) characters; "ja" 350, "en" 1000; 0 = never cut) and every part is indexed. The dense model
reads further, so its parts are longer (dense_chunk=; "ja" 900, "en" 2000). A search still answers with texts:
each text once, with the part that matched best as hit.text and its place in the text as hit.span; the whole
text is ix.texts[hit.id]. With your own models nothing is cut unless you give chunk=.

Without a second model: TextIndex("ja", sparse="bm25") uses character-pair BM25 for the word side (CharBM25)
instead of the sparse language model: nothing to download for that side, and names, codes and terms no model
has seen are matched as written. dedup=True keeps texts that repeat once on the dense side.

Your own models: TextIndex(dense=f, sparse=g, sparse_dim=n) with f(texts, is_query) -> array [len(texts), dim]
and g(texts, is_query) -> [(indices, values), ...]; mode="dense" or mode="sparse" uses one side only.

Saving: save(dir) writes the two indexes, the texts and a small meta.json. Without a key everything is
readable (the texts as texts.json). With a key (a passphrase) the two indexes and the texts are encrypted
(texts.kct; AES-256-CTR + CMAC with a key from PBKDF2, all in the C library) and a wrong key or a changed file
is refused. meta.json stays readable either way: it holds the ids, the model names and the settings, no text.

Licences of the built-in models (as their model pages on the Hugging Face hub state; check again before you
rely on it): "ja": cl-nagoya/ruri-v3-30m Apache-2.0, hotchpotch/japanese-splade-v2 MIT.
"en": intfloat/multilingual-e5-small MIT, opensearch-project/opensearch-neural-sparse-encoding-v2-distill
Apache-2.0."""
import io, json, math, os, re, sys, time, unicodedata, zlib
from collections import namedtuple, Counter
import numpy as np

from . import KciVec
from .sparse import KciSparse, seal, unseal
from .hybrid import mix, choose_weight, tokenize

Hit = namedtuple("Hit", "id score text dense_rank sparse_rank span")
Cand = namedtuple("Cand", "id text span dense_rank dense_score sparse_rank sparse_score")
# the words of a question that count for the phrase match: runs of kanji, of katakana, or of Latin letters / digits, 2 or longer
_TERMS = re.compile("[\u4e00-\u9fff\u3005]{2,}|[\u30a1-\u30fa\u30fc]{2,}|[0-9A-Za-z][0-9A-Za-z_.\\-]+")


PRESETS = {
    "ja": dict(dense="cl-nagoya/ruri-v3-30m", q_prefix="検索クエリ: ", d_prefix="検索文書: ",
               sparse="hotchpotch/japanese-splade-v2", w_dense=0.4, chunk=350, dense_chunk=900),
    "en": dict(dense="intfloat/multilingual-e5-small", q_prefix="query: ", d_prefix="passage: ",
               sparse="opensearch-project/opensearch-neural-sparse-encoding-v2-distill", zero_special=True, w_dense=0.4, chunk=1000, dense_chunk=2000),
}
BLOCKS_FROM = 50000        # from this many texts on the sparse side uses the approximate (block) search


_ENDS = "\u3002\uff01\uff1f!?\n"          # a part may end after these (Japanese full stop, ! ?, a line break)


def split_text(text, size, overlap=None):
    """Cut a long text into parts of at most `size` characters: [(start, end), ...], text[start:end] is a part."""
    n = len(text); size = int(size)
    if size <= 0 or n <= size: return [(0, n)]
    overlap = size // 8 if overlap is None else max(0, min(int(overlap), size // 2))
    def is_end(i):
        c = text[i - 1]
        return c in _ENDS or (c in ".;:" and text[i:i + 1].isspace())
    out, start = [], 0
    while n - start > size:
        rem = n - start; lim = start + size
        if rem < 2 * size - overlap:                    # the last two parts: about equal, not a full one and a scrap
            lim = start + min(size, (rem + overlap + 1) // 2 + size // 8)
        half, end = start + (lim - start) // 2, 0
        for i in range(lim, half, -1):
            if is_end(i): end = i; break
        if not end:
            for i in range(lim, half, -1):
                if text[i - 1].isspace(): end = i; break
        if not end: end = lim
        out.append((start, end))
        nxt = end
        for i in range(max(end - overlap, start + 1), end):
            if is_end(i): nxt = i; break
        start = nxt
    out.append((start, n))
    return out


def _torch():
    try:
        import torch, transformers  # noqa: F401
    except ImportError as e:
        raise ImportError("kci_vec.text needs torch and transformers for its built-in models: "
                          "pip install torch transformers  (or pass your own dense= / sparse= functions)") from e
    return torch


def _device(torch, device):
    """"cuda" / "mps" / "auto" (a GPU when torch sees one): NOT tried by the author, who has no such machine at hand;
    the code path is the usual .to(device)."""
    if device in (None, "cpu"): return "cpu"
    if device == "auto":
        if torch.cuda.is_available(): return "cuda"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"
    return device


class CharBM25:
    """The sparse side without a model: BM25 over words (Latin, digits) and character pairs (Japanese, Chinese,
    Korean), the tokens of kci_vec.hybrid.tokenize, hashed into `dim` slots. For names, codes and terms a language
    model has never seen (drug names, abbreviations, part numbers), and for when no second model should run. A
    text becomes (slot, tf part of BM25); a question becomes (slot, idf), so their inner product is the BM25 score
    and the same KciSparse index (and its encryption) serves. The counts it keeps (how many texts hold each slot,
    the mean length) are saved with the index. Deleting a text does not take its counts back; they follow the
    texts added. TextIndex(lang, sparse="bm25") uses it."""
    name = "char-bm25"

    def __init__(self, dim=1 << 18, k1=1.5, b=0.75):
        self.dim, self.k1, self.b = int(dim), float(k1), float(b)
        self.df = np.zeros(self.dim, dtype=np.uint32); self.n = 0; self.sum_len = 0

    def _slots(self, text):
        return [zlib.crc32(tok.encode("utf-8")) % self.dim for tok in tokenize(text)]

    def __call__(self, texts, is_query):
        rows = []
        for tx in texts:
            sl = self._slots(tx)
            if is_query:
                u = sorted(set(sl)); n = max(self.n, 1)
                rows.append((np.array(u, dtype=np.uint32),
                             np.array([math.log(1.0 + (n - float(self.df[s]) + 0.5) / (float(self.df[s]) + 0.5)) for s in u], dtype=np.float32)))
                continue
            c = Counter(sl); L = len(sl)
            self.n += 1; self.sum_len += L
            u = sorted(c); avg = self.sum_len / self.n if self.n else 1.0
            for s in u: self.df[s] += 1
            norm = self.k1 * (1.0 - self.b + self.b * (L / avg if avg > 0 else 1.0))
            rows.append((np.array(u, dtype=np.uint32), np.array([c[s] * (self.k1 + 1.0) / (c[s] + norm) for s in u], dtype=np.float32)))
        return rows

    def state(self):
        buf = io.BytesIO(); nz = np.nonzero(self.df)[0].astype(np.uint32)
        np.savez(buf, slots=nz, df=self.df[nz], head=np.array([self.dim, self.n, self.sum_len], dtype=np.int64), kb=np.array([self.k1, self.b]))
        return buf.getvalue()

    def load_state(self, raw):
        z = np.load(io.BytesIO(raw)); head = z["head"]
        if int(head[0]) != self.dim: raise RuntimeError("CharBM25: the saved counts are for another number of slots")
        self.df[:] = 0; self.df[z["slots"]] = z["df"]; self.n = int(head[1]); self.sum_len = int(head[2])
        self.k1, self.b = float(z["kb"][0]), float(z["kb"][1])


def _batches(texts, batch):
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))     # similar lengths share a batch
    for b in range(0, len(order), batch):
        yield order[b:b + batch]


class HFDense:
    """A sentence-embedding model from the Hugging Face hub: mean of the token vectors, length 1."""
    def __init__(self, name, q_prefix="", d_prefix="", max_tokens=512, batch=16, device=None):
        torch = _torch()
        from transformers import AutoModel, AutoTokenizer
        self.name, self.qp, self.dp, self.max_tokens, self.batch = name, q_prefix, d_prefix, max_tokens, batch
        self.tok = AutoTokenizer.from_pretrained(name); self.model = AutoModel.from_pretrained(name).eval()
        self.dim = int(self.model.config.hidden_size); self.torch = torch
        self.device = _device(torch, device); self.model.to(self.device)

    def __call__(self, texts, is_query):
        torch = self.torch; out = np.zeros((len(texts), self.dim), dtype=np.float32)
        pre = self.qp if is_query else self.dp
        with torch.inference_mode():
            for ids in _batches(texts, self.batch):
                enc = self.tok([pre + texts[i] for i in ids], padding=True, truncation=True, max_length=self.max_tokens, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out[ids] = torch.nn.functional.normalize(v, dim=1).cpu().numpy()
        return out


class HFSplade:
    """A SPLADE-type model: per text, log(1 + relu(.)) of the word scores, the largest over the tokens."""
    def __init__(self, name, max_tokens=256, max_query_tokens=64, batch=16, zero_special=False, device=None):
        torch = _torch()
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.name, self.max_tokens, self.max_query_tokens, self.batch = name, max_tokens, max_query_tokens, batch
        try:
            self.tok = AutoTokenizer.from_pretrained(name)
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(f"the tokenizer of {name} needs more packages ({e}); for Japanese: pip install fugashi unidic-lite") from e
        self.model = AutoModelForMaskedLM.from_pretrained(name).eval().float()   # some models are stored in 16 bits
        self.dim = int(self.model.config.vocab_size); self.torch = torch
        # zero_special: the weights of [CLS], [SEP], ... are set to 0 (what the opensearch models' cards ask for)
        self.special = list(self.tok.all_special_ids) if zero_special else []
        self.device = _device(torch, device); self.model.to(self.device)

    def __call__(self, texts, is_query):
        torch = self.torch; rows = [None] * len(texts)
        with torch.inference_mode():
            for ids in _batches(texts, self.batch):
                enc = self.tok([texts[i] for i in ids], padding=True, truncation=True,
                               max_length=self.max_query_tokens if is_query else self.max_tokens, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                logits.masked_fill_(enc["attention_mask"].unsqueeze(-1) == 0, float("-inf"))
                w = torch.log1p(torch.relu(logits.max(dim=1).values))
                if self.special: w[:, self.special] = 0
                w = w.cpu()
                for j, i in enumerate(ids):
                    nz = torch.nonzero(w[j]).squeeze(1)
                    rows[i] = (nz.numpy().astype(np.uint32), w[j, nz].numpy().astype(np.float32))
        return rows


class TextIndex:
    """Texts in, the best matching texts out. See the module text for an example.

    lang: "ja" or "en" — which built-in models to use (PRESETS). mode: "hybrid" (both, merged), "dense", "sparse".
    dense / sparse: your own functions instead of the built-in models (see the module text); sparse_dim: the size
    of your sparse function's vocabulary. w_dense: the dense side's weight in the merge, 0..1 (the sparse side gets
    the rest); None = the preset's. tune() picks it from questions whose answers you know. chunk: texts longer
    than this many characters are cut into parts (split_text; None = the preset's, 0 = never), overlap: how many
    characters two neighbouring parts may share (None = chunk // 8)."""

    def __init__(self, lang="ja", mode="hybrid", dense=None, sparse=None, sparse_dim=None, w_dense=None, keep_text=True, verbose=True,
                 chunk=None, overlap=None, dense_chunk=None, dedup=False, device=None, phrase=False, phrase_depth=1000):
        if mode not in ("hybrid", "dense", "sparse"):
            raise ValueError("mode must be 'hybrid', 'dense' or 'sparse'")
        # dedup=True: texts that repeat (boilerplate, empty fields) are kept once on the dense side (kci_vec_set_dedup).
        # device: where the built-in models run (None = CPU; "auto" / "cuda" / "mps": see _device, not tried by the author).
        if isinstance(sparse, str):
            if sparse != "bm25": raise ValueError("sparse= as a word can only be 'bm25'")
            sparse = CharBM25()
        self.dedup, self.device = bool(dedup), device
        # in which the question's own words stand as written come first (_phrase_order). Needs the texts (keep_text).
        self.phrase, self.phrase_depth = bool(phrase), int(phrase_depth)
        if lang not in PRESETS and (dense is None or sparse is None) and not (mode == "dense" and dense) and not (mode == "sparse" and sparse):
            raise ValueError(f"lang must be one of {sorted(PRESETS)} (or pass your own dense= and sparse= functions)")
        self.lang, self.mode, self.keep_text, self.verbose = lang, mode, keep_text, verbose
        self._preset = PRESETS.get(lang, {})
        self._dense_fn, self._sparse_fn, self._sparse_dim = dense, sparse, sparse_dim
        if sparse is not None and sparse_dim is None and not hasattr(sparse, "dim"):
            raise ValueError("with your own sparse= function give sparse_dim= (the size of its vocabulary)")
        self.w_dense = float(w_dense if w_dense is not None else self._preset.get("w_dense", 0.4))
        if not 0.0 <= self.w_dense <= 1.0:
            raise ValueError("w_dense must be between 0 and 1")
        self.chunk = int(chunk if chunk is not None else self._preset.get("chunk", 0)); self.overlap = overlap
        # of 1000 characters. So the dense side has its own, longer part size. With chunk= given and dense_chunk= not,
        self.dense_chunk = int(dense_chunk if dense_chunk is not None else (self._preset.get("dense_chunk", self.chunk) if chunk is None else self.chunk))
        if self.chunk < 0 or self.dense_chunk < 0 or (overlap is not None and int(overlap) < 0): raise ValueError("chunk, dense_chunk and overlap must be >= 0")
        self._side = {}; self._nside = {"d": 0, "s": 0}              # row -> "b" both sides, "d" dense only, "s" sparse only; rows per side
        self.vec = None; self.sps = None
        self.texts = {}; self._ids = set(); self._next = 0
        self._rows = {}; self._doc_rows = {}; self._row_next = 0        # row -> (text id, start, end); text id -> rows
        self._built = False; self._blocks = False; self._added_since = 0

    # ---- the two models, loaded when first needed
    def _dense(self):
        if self._dense_fn is None:
            p = self._preset; self._say(f"loading the dense model {p['dense']} ...")
            self._dense_fn = HFDense(p["dense"], p["q_prefix"], p["d_prefix"], device=self.device)
        return self._dense_fn

    def _sparse(self):
        if self._sparse_fn is None:
            p = self._preset; self._say(f"loading the sparse model {p['sparse']} ...")
            self._sparse_fn = HFSplade(p["sparse"], zero_special=p.get("zero_special", False), device=self.device)
        if self._sparse_dim is None:
            self._sparse_dim = int(self._sparse_fn.dim)
        return self._sparse_fn

    def _say(self, s):
        if self.verbose:
            print(s, file=sys.stderr, flush=True)

    def __len__(self): return len(self._ids)

    # ---- adding
    def add(self, texts, ids=None, batch=256):
        """Add texts (a list of strings, or one string). ids: your own whole numbers (>= 0), one per text; without
        them the texts are numbered 0, 1, 2, ... in the order they come. Returns the ids. Can be called again at any
        time, also after searching. An id that is already in the index is refused (delete it first)."""
        if isinstance(texts, str): texts = [texts]
        texts = list(texts)
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("add: every text must be a string")
        if ids is None:
            ids = list(range(self._next, self._next + len(texts)))
        else:
            ids = [int(i) for i in ids]
            if len(ids) != len(texts): raise ValueError("add: one id per text")
            if any(i < 0 for i in ids) or len(set(ids)) != len(ids): raise ValueError("add: ids must be distinct whole numbers >= 0")
        clash = [i for i in ids if i in self._ids]
        if clash:
            raise ValueError(f"add: id {clash[0]} is already in the index" + (f" (and {len(clash) - 1} more)" if len(clash) > 1 else ""))
        flat = []                                               # (row, text id, start, end, the part, side)
        two = self.mode == "hybrid" and self.dense_chunk != self.chunk
        for d, tx in zip(ids, texts):
            sp = split_text(tx, self.dense_chunk if self.mode == "dense" else self.chunk, self.overlap)
            dn = split_text(tx, self.dense_chunk, self.overlap) if two else sp
            parts = [(a, e, "b") for a, e in sp] if dn == sp else [(a, e, "d") for a, e in dn] + [(a, e, "s") for a, e in sp]
            for a, e, side in parts:
                flat.append((self._row_next + len(flat), d, a, e, tx if (a, e) == (0, len(tx)) else tx[a:e], side))
        whole = dict(zip(ids, texts)); t0 = time.time()
        for b in range(0, len(flat), batch):
            fb = flat[b:b + batch]
            fd = [f for f in fb if f[5] != "s"] if self.mode != "sparse" else []
            fs = [f for f in fb if f[5] != "d"] if self.mode != "dense" else []
            if fd:
                tb = [f[4] for f in fd]; ib = [f[0] for f in fd]
                dv = np.ascontiguousarray(self._dense()(tb, False), dtype=np.float32)
                if dv.ndim != 2 or len(dv) != len(tb): raise ValueError("the dense function must return one row per text")
                if self.vec is None: self.vec = KciVec(dv.shape[1], M=16, efc=200, rerank=200, dedup=self.dedup)
                if self._built:
                    for i, v in zip(ib, dv): self.vec.insert(i, v)
                else:
                    self.vec.add_many(ib, dv)
            if fs:
                tb = [f[4] for f in fs]; ib = [f[0] for f in fs]
                rows = self._sparse()(tb, False)
                if len(rows) != len(tb): raise ValueError("the sparse function must return one (indices, values) per text")
                if self.sps is None: self.sps = KciSparse(self._sparse_dim)
                for i, (ix, val) in zip(ib, rows):
                    if len(ix) == 0: continue                       # nothing to match on: the dense side still has it
                    (self.sps.insert if self._built else self.sps.add)(i, ix, val)
            for r, d, a, e, _, side in fb:
                self._rows[r] = (d, a, e); self._doc_rows.setdefault(d, []).append(r); self._row_next = max(self._row_next, r + 1)
                self._side[r] = side
                if side != "s": self._nside["d"] += 1
                if side != "d": self._nside["s"] += 1
                if d not in self._ids:
                    self._ids.add(d)
                    if self.keep_text: self.texts[d] = whole[d]
            if len(flat) > batch:
                done = b + len(fb); el = time.time() - t0
                self._say(f"  added {done} / {len(flat)}" + (" parts" if len(flat) > len(texts) else "") + f"  ({el:.0f} s, about {(len(flat) - done) * el / done:.0f} s to go)")
        if ids: self._next = max(self._next, max(ids) + 1)
        self._added_since += len(ids)
        return ids

    def delete(self, id):
        """Remove one text. Returns True when it was there."""
        id = int(id)
        if id not in self._ids: return False
        for r in self._doc_rows.pop(id, []):
            side = self._side.pop(r, "b")
            if self.vec is not None and side != "s": self.vec.delete(r); self._nside["d"] -= 1
            if self.sps is not None and side != "d": self.sps.delete(r); self._nside["s"] -= 1
            self._rows.pop(r, None)
        self._ids.discard(id); self.texts.pop(id, None)
        return True

    def build(self):
        """Make the indexes searchable. search() calls this by itself the first time; call it yourself to choose when
        the waiting happens (a second or two per 10,000 texts)."""
        if not self._ids: return
        if self.vec is not None: self.vec.build()
        if self.sps is not None:
            self.sps.build()
            if len(self._ids) >= BLOCKS_FROM:
                self.sps.build_blocks(); self.sps.build_neighbours(20); self._blocks = True
        self._built = True; self._added_since = 0

    # ---- searching
    def _row_filter(self, filter):
        if filter is None: return None
        rows = self._rows
        if callable(filter): return lambda r: bool(filter(rows[r][0]))
        return [r for d in filter for r in self._doc_rows.get(int(d), ())]

    def _texts_of(self, lst, depth):
        """a list of parts, best first -> (the list of texts, each once at its best part; text id -> that part)"""
        out, part = [], {}
        for r, s in lst:
            d = self._rows[r][0]
            if d not in part:
                part[d] = r
                if len(out) < depth: out.append((d, s))
        return out, part

    def _lists(self, query, depth, filter=None):
        """the two result lists as texts: (dense list, sparse list, dense parts, sparse parts); None = side not used"""
        dl = sl = None; dp = sp = {}
        getd = depth * 2 if self._nside["d"] > len(self._ids) else depth    # some texts are in parts on that side: look deeper
        gets = depth * 2 if self._nside["s"] > len(self._ids) else depth
        filter = self._row_filter(filter)
        if filter is not None and not callable(filter) and len(filter) == 0:
            return ([] if self.mode != "sparse" and self.vec is not None else None), ([] if self.mode != "dense" and self.sps is not None else None), {}, {}
        if self.mode != "sparse" and self.vec is not None:
            qv = np.ascontiguousarray(self._dense()([query], True), dtype=np.float32)[0]
            dl, dp = self._texts_of(self.vec.search(qv, k=getd, ef=max(200, getd), filter=filter), depth)
        if self.mode != "dense" and self.sps is not None:
            ix, val = self._sparse()([query], True)[0]
            use_phrase = self.phrase and self.keep_text and bool(self.texts)
            if use_phrase: gets = max(gets, self.phrase_depth)
            if len(ix) == 0: sl = []
            elif self._blocks: sl = self.sps.search_blocks(ix, val, k=gets, query_terms=8, heap_factor=1.15, neighbours=20, filter=filter)
            else: sl = self.sps.search(ix, val, k=gets, filter=filter)
            if use_phrase and sl: sl = self._phrase_order(query, sl)
            sl, sp = self._texts_of(sl, depth)
        return dl, sl, dp, sp

    def _phrase_order(self, query, lst):
        """the word side's parts, those first in which the question's words stand as written. A word = a run of kanji, of
        katakana, or of Latin letters / digits, 2 characters or longer. A part's phrase score = the sum, over the
        question's words it contains verbatim, of log(1 + parts looked at / parts that contain the word): a word few
        parts contain counts more. Order: phrase score, then the BM25 score. The scores given on are phrase score +
        BM25 / (largest BM25 + 1), so that the merge with the dense side keeps this order. Why: character-pair BM25
        ranks by pairs, and a part can collect many of a question's pairs without holding any of its words."""
        terms = sorted(set(m.group(0).lower() for m in _TERMS.finditer(unicodedata.normalize("NFKC", query))))
        if not terms: return lst
        parts = []
        for r, s in lst:
            d, a, e = self._rows[r]; tx = self.texts.get(d)
            parts.append(unicodedata.normalize("NFKC", tx[a:e]).lower() if tx is not None else "")
        has = [[tm in px for px in parts] for tm in terms]
        n = len(lst); w = [math.log(1.0 + n / c) if c else 0.0 for c in (sum(h) for h in has)]
        top = max(s for _, s in lst) + 1.0
        scored = [(sum(w[j] for j in range(len(terms)) if has[j][i]) + lst[i][1] / top, lst[i][0]) for i in range(n)]
        scored.sort(key=lambda x: -x[0])
        return [(r, s) for s, r in scored]

    def candidates(self, query, depth_dense=50, depth_sparse=100, filter=None):
        """Everything either side found, unmerged, for a re-ranker of your own: [Cand(id, text, span, dense_rank,
        dense_score, sparse_rank, sparse_score)], the word side's texts first (in its order), then those only the
        dense side found. A side that did not find a text gives None for its rank and score. text / span: the part
        that side matched (the word side's when both did). Nothing is cut off and no weight is applied: which
        candidates are worth keeping is for the re-ranker to say."""
        if not isinstance(query, str): raise TypeError("candidates: the question must be a string")
        if not self._ids: return []
        if not self._built: self.build()
        dl, sl, dp, sp = self._lists(query, max(int(depth_dense), int(depth_sparse), 1), filter)
        dl = (dl or [])[:max(int(depth_dense), 0)]; sl = (sl or [])[:max(int(depth_sparse), 0)]
        dpos = {i: (r + 1, s) for r, (i, s) in enumerate(dl)}; spos = {i: (r + 1, s) for r, (i, s) in enumerate(sl)}
        out = []
        for i in [i for i, _ in sl] + [i for i, _ in dl if i not in spos]:
            row = sp[i] if i in spos else dp[i]
            _, a, e = self._rows[row]; tx = self.texts.get(i)
            if e is None: e = len(tx) if tx is not None else None
            dr, ds = dpos.get(i, (None, None)); sr, ss = spos.get(i, (None, None))
            out.append(Cand(i, None if tx is None else tx[a:e], (a, e), dr, ds, sr, ss))
        return out

    def _hit(self, d, score, dr, sr, dp, sp):
        r = dp[d] if dr is not None and (sr is None or dr <= sr) else sp[d]     # the part from the side that ranks it higher
        _, a, e = self._rows[r]; tx = self.texts.get(d)
        if e is None: e = len(tx) if tx is not None else None
        return Hit(d, score, None if tx is None else tx[a:e], dr, sr, (a, e))

    def search(self, query, k=5, depth=100, filter=None, rerank=None, depth_dense=50, depth_sparse=100):
        """The k best texts for a question: [Hit(id, score, text, dense_rank, sparse_rank, span)], best first. rerank: a
        function (question, [texts]) -> one score per text, larger = better (a cross-encoder of your own)."""
        if not isinstance(query, str): raise TypeError("search: the question must be a string")
        if not self._ids: return []
        if not self._built: self.build()
        if rerank is not None:
            if not self.keep_text: raise RuntimeError("search: rerank= needs the texts (keep_text=True)")
            cs = self.candidates(query, depth_dense, depth_sparse, filter)
            if not cs: return []
            sc = [float(x) for x in rerank(query, [c.text for c in cs])]
            if len(sc) != len(cs): raise ValueError("the rerank function must return one score per text")
            order = sorted(range(len(cs)), key=lambda j: -sc[j])[:k]
            return [Hit(cs[j].id, sc[j], cs[j].text, cs[j].dense_rank, cs[j].sparse_rank, cs[j].span) for j in order]
        depth = max(int(depth), int(k))
        dl, sl, dp, sp = self._lists(query, depth, filter)
        if dl is None or sl is None:
            one = dl if sl is None else sl
            return [self._hit(i, s, r + 1 if sl is None else None, r + 1 if dl is None else None, dp, sp) for r, (i, s) in enumerate((one or [])[:k])]
        if self.w_dense >= 1.0:   merged = [(i, s, [r + 1, None]) for r, (i, s) in enumerate(dl[:k])]
        elif self.w_dense <= 0.0: merged = [(i, s, [None, r + 1]) for r, (i, s) in enumerate(sl[:k])]
        else:                     merged = mix([dl, sl], k=k, weights=[self.w_dense, 1.0 - self.w_dense])
        return [self._hit(i, s, r[0], r[1], dp, sp) for i, s, r in merged]

    def tune(self, questions, answers, k=10, depth=100):
        """Check the dense side's weight against questions whose right answers you know, and move it when another weight
        is clearly better for your texts. answers[i]: the ids of the right texts for questions[i] (a set / list, or
        {id: grade}). 50 questions or more; with few, the present weight is kept unless it clearly loses
        (hybrid.choose_weight, prefer=). Look at the table: if one end is far above the other, the weak side's model
        does not suit your texts, and a better model for that side helps more than any weight."""
        if self.mode != "hybrid": raise RuntimeError("tune: only for mode='hybrid'")
        if not self._built: self.build()
        la, lb = [], []
        for q in questions:
            dl, sl, _, _ = self._lists(q, depth); la.append(dl or []); lb.append(sl or [])
        w, table = choose_weight(la, lb, list(answers), k=k, method="mix", prefer=round(self.w_dense, 1))
        self.w_dense = float(w)
        return self.w_dense, table

    # ---- saving
    def save(self, path, key=None):
        """Write the index into the folder `path`. key (a passphrase, str or bytes): the two index files and the texts
        are encrypted. Without a key the texts go to texts.json as they are. Returns True when the texts were written
        (False with keep_text=False)."""
        if not self._ids: raise RuntimeError("save: the index is empty")
        if not self._built: self.build()
        kb = key.encode("utf-8") if isinstance(key, str) else key
        os.makedirs(path, exist_ok=True)
        if self.vec is not None and not self.vec.save(os.path.join(path, "dense.kvec"), kb): raise RuntimeError("save: writing dense.kvec failed")
        if self.sps is not None: self.sps.save(os.path.join(path, "sparse.kcs"), kb)       # raises when it fails
        sp_state = os.path.join(path, "sparse_state.bin")           # what the sparse function has counted (CharBM25): with the key, encrypted
        if os.path.exists(sp_state): os.remove(sp_state)
        has_state = self._sparse_fn is not None and hasattr(self._sparse_fn, "state")
        if has_state:
            raw_state = self._sparse_fn.state()
            with open(sp_state, "wb") as f: f.write(raw_state if kb is None else seal(raw_state, kb))
        tp, cp = os.path.join(path, "texts.json"), os.path.join(path, "texts.kct")
        with_text = bool(self.keep_text)
        for old in (tp, cp):                                    # never leave an older copy (a plain one least of all)
            if os.path.exists(old): os.remove(old)
        if with_text:
            raw = json.dumps({str(i): t for i, t in self.texts.items()}, ensure_ascii=False).encode("utf-8")
            if kb is None:
                with open(tp, "wb") as f: f.write(raw)
            else:
                with open(cp, "wb") as f: f.write(seal(raw, kb))
        meta = dict(format=3, lang=self.lang, mode=self.mode, w_dense=self.w_dense, blocks=self._blocks, next=self._next,
                    ids=sorted(self._ids), texts=with_text, encrypted=kb is not None,
                    dense=getattr(self._dense_fn, "name", None) or self._preset.get("dense"),
                    sparse=getattr(self._sparse_fn, "name", None) or self._preset.get("sparse"), sparse_dim=self._sparse_dim,
                    chunk=self.chunk, dense_chunk=self.dense_chunk, overlap=self.overlap, row_next=self._row_next,
                    sparse_state=has_state, dedup=self.dedup, phrase=self.phrase, phrase_depth=self.phrase_depth,
                    rows=[[r, d, a, e, self._side.get(r, "b")] for r, (d, a, e) in sorted(self._rows.items())])
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f: json.dump(meta, f)
        return with_text

    @classmethod
    def load(cls, path, key=None, texts=None, dense=None, sparse=None, verbose=True, device=None):
        """Open a saved index. key: the passphrase it was saved with (a wrong one is refused). texts: {id: text} to use
        instead of the saved ones (or when none were saved). dense / sparse: your own functions, when the index was
        made with them."""
        try:
            with open(os.path.join(path, "meta.json"), encoding="utf-8") as f: meta = json.load(f)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"load: {path} is not a saved TextIndex ({e})") from e
        if meta.get("format") not in (1, 2, 3): raise RuntimeError("load: made by a newer version")
        kb = key.encode("utf-8") if isinstance(key, str) else key
        if meta.get("encrypted") and kb is None: raise RuntimeError("load: this index was saved with a key; pass key=")
        if sparse is None and meta.get("sparse") == CharBM25.name: sparse = CharBM25(dim=int(meta.get("sparse_dim") or (1 << 18)))
        ix = cls(meta["lang"], mode=meta["mode"], dense=dense, sparse=sparse, sparse_dim=meta.get("sparse_dim"),
                 w_dense=meta["w_dense"], verbose=verbose, chunk=meta.get("chunk", 0), overlap=meta.get("overlap"),
                 dense_chunk=meta.get("dense_chunk", meta.get("chunk", 0)), dedup=bool(meta.get("dedup", False)), device=device,
                 phrase=bool(meta.get("phrase", False)), phrase_depth=int(meta.get("phrase_depth", 1000)))
        if meta.get("sparse_state"):
            if not hasattr(ix._sparse_fn, "load_state"): raise RuntimeError("load: this index needs the sparse function it was made with (its counts were saved)")
            try:
                with open(os.path.join(path, "sparse_state.bin"), "rb") as f: raw_state = f.read()
            except OSError as e:
                raise RuntimeError(f"load: sparse_state.bin is missing ({e})") from e
            ix._sparse_fn.load_state(raw_state if kb is None else unseal(raw_state, kb))
        if meta["mode"] != "sparse": ix.vec = KciVec.load(os.path.join(path, "dense.kvec"), key=kb, rerank=200)
        if meta["mode"] != "dense":  ix.sps = KciSparse.load(os.path.join(path, "sparse.kcs"), key=kb)
        ix._ids = set(int(i) for i in meta["ids"]); ix._next = int(meta["next"]); ix._blocks = bool(meta["blocks"]); ix._built = True
        for ent in (meta["rows"] if "rows" in meta else [[i, i, 0, None] for i in ix._ids]):
            r, d, a, e = ent[:4]; side = ent[4] if len(ent) > 4 else "b"
            ix._rows[int(r)] = (int(d), int(a), e if e is None else int(e)); ix._doc_rows.setdefault(int(d), []).append(int(r))
            ix._side[int(r)] = side
            if side != "s" and meta["mode"] != "sparse": ix._nside["d"] += 1
            if side != "d" and meta["mode"] != "dense": ix._nside["s"] += 1
        ix._row_next = int(meta.get("row_next", ix._next))
        if texts is not None:
            ix.texts = {int(i): t for i, t in dict(texts).items()}
        elif meta.get("texts") and kb is not None:
            with open(os.path.join(path, "texts.kct"), "rb") as f: ix.texts = {int(i): t for i, t in json.loads(unseal(f.read(), kb).decode("utf-8")).items()}
        elif meta.get("texts"):
            with open(os.path.join(path, "texts.json"), encoding="utf-8") as f: ix.texts = {int(i): t for i, t in json.load(f).items()}
        else:
            ix.keep_text = False
        return ix

    def __repr__(self):
        return f"TextIndex({self.lang!r}, mode={self.mode!r}, {len(self)} texts, w_dense={self.w_dense})"
