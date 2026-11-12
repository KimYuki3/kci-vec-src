"""kci_vec — embedded Python binding for the kci_vec vector engine.

In-process, no server: this loads the kci_vec C shared library via ctypes and exposes a small Pythonic API.
Two ways to search (an int8 HNSW graph, and partitioned lists with 4-bit table lookup — the fast one), float /
int8 re-rank, metadata filtering, incremental update, and snapshots encrypted at rest (AES-256-CTR +
AES-256-CMAC) — all inside your process, no data leaving the machine.

    from kci_vec import KciVec
    import numpy as np
    v = KciVec(dim=384, rerank=200)         # keep floats for re-rank
    v.add_many(ids, embeddings)             # float32 [n, dim]
    v.build_lists()                         # fast path (lists); or v.build() for the graph
    for doc_id, score in v.search(query_vec, k=5):
        ...
    ids, scores = v.search_many(queries, k=5)             # [nq, k] arrays
    v.search(query_vec, k=5, filter=lambda id: id in allowed)   # metadata / ACL
    v.save("index.kv", key=b"secret")       # encrypted at rest
    w = KciVec(dim=128, rerank=200, metric="l2")   # Euclidean: scores are |q - x|^2, smallest first
    v2 = KciVec.load("index.kv", key=b"secret")

Build the library first:
    gcc -O3 -maes -mavx2 -mfma -shared -static -static-libgcc -o libkci_vec.dll kci_vec.c   # Windows
    sh build_kci_vec.sh                                                                     # POSIX (libkci_vec.so/.dylib)
Set KCI_VEC_LIB to the .dll/.so/.dylib, or place it next to this file / repo root."""
import ctypes as C
import os
import sys
import numpy as np

c_f32p = C.POINTER(C.c_float)
c_u64p = C.POINTER(C.c_uint64)
_FILTER = C.CFUNCTYPE(C.c_int, C.c_uint64, C.c_void_p)   # int filt(uint64 id, void *ctx)


class _IvfParams(C.Structure):
    _fields_ = [("nlist", C.c_int), ("threads", C.c_int), ("soar", C.c_float), ("code_dims", C.c_int), ("lowrank", C.c_int)]


def _find_lib():
    if os.environ.get("KCI_VEC_LIB"):
        return os.environ["KCI_VEC_LIB"]
    names = ["libkci_vec.dll", "libkci_vec.so", "libkci_vec.dylib"] if sys.platform == "win32" \
        else ["libkci_vec.so", "libkci_vec.dylib", "libkci_vec.dll"]
    here = os.path.dirname(os.path.abspath(__file__))
    # is compiled): libkci_vec_avx512.so needs AVX-512 BW + VNNI, libkci_vec_avx2.so needs AVX2 + FMA. Chosen by the
    # CPU's flags; a plain libkci_vec.so next to this file (built for this machine) still wins when present.
    if sys.platform.startswith("linux") and not os.path.exists(os.path.join(here, "libkci_vec.so")):
        try:
            flags = open("/proc/cpuinfo").read().split("flags")[1].split("\n")[0].split()
        except Exception:
            flags = []
        for n, need in (("libkci_vec_avx512.so", ("avx512bw", "avx512_vnni")), ("libkci_vec_avx2.so", ("avx2", "fma"))):
            p = os.path.join(here, n)
            if os.path.exists(p) and all(f in flags for f in need):
                return p
    # package dir first (pip-installed bundled lib), then repo root(s), then cwd
    roots = [here, os.path.join(here, "..", ".."),
             os.path.join(here, "..", "..", "..", ".."), os.getcwd()]
    for r in roots:
        for n in names:
            p = os.path.abspath(os.path.join(r, n))
            if os.path.exists(p):
                return p
    raise FileNotFoundError("libkci_vec not found; set KCI_VEC_LIB or build it (see module docstring)")


def _bind(lib):
    s = lib
    s.kci_vec_new.restype = C.c_void_p;            s.kci_vec_new.argtypes = [C.c_int, C.c_int, C.c_int]
    s.kci_vec_free.argtypes = [C.c_void_p]
    s.kci_vec_keep_float.argtypes = [C.c_void_p, C.c_int]
    s.kci_vec_reserve.argtypes = [C.c_void_p, C.c_uint32]
    s.kci_vec_add.restype = C.c_int64;             s.kci_vec_add.argtypes = [C.c_void_p, C.c_uint64, c_f32p]
    s.kci_vec_build.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_double)]
    s.kci_vec_build_ivf_p.restype = C.c_int;       s.kci_vec_build_ivf_p.argtypes = [C.c_void_p, C.POINTER(_IvfParams)]
    s.kci_vec_searcher_new.restype = C.c_void_p;   s.kci_vec_searcher_new.argtypes = [C.c_void_p]
    s.kci_vec_searcher_free.argtypes = [C.c_void_p]
    s.kci_vec_searcher_set_rerank_int8.argtypes = [C.c_void_p, C.c_int]
    if hasattr(s, "kci_vec_searcher_set_probe_gap"):
        s.kci_vec_searcher_set_probe_gap.restype = C.c_int
        s.kci_vec_searcher_set_probe_gap.argtypes = [C.c_void_p, C.c_float, C.c_int]
    if hasattr(s, "kci_vec_searcher_set_eps"):
        s.kci_vec_searcher_set_eps.restype = C.c_int
        s.kci_vec_searcher_set_eps.argtypes = [C.c_void_p, C.c_float]
    if hasattr(s, "kci_vec_build_qg"):
        s.kci_vec_build_qg.restype = C.c_int;      s.kci_vec_build_qg.argtypes = [C.c_void_p, C.c_int, C.c_int]
        s.kci_vec_qg_bytes.restype = C.c_int64;    s.kci_vec_qg_bytes.argtypes = [C.c_void_p]
        s.kci_vec_searcher_set_qg.restype = C.c_int; s.kci_vec_searcher_set_qg.argtypes = [C.c_void_p, C.c_int]
    s.kci_vec_search_r.restype = C.c_int;          s.kci_vec_search_r.argtypes = [C.c_void_p, c_f32p, C.c_int, C.c_int, c_u64p, c_f32p]
    s.kci_vec_search_rerank.restype = C.c_int;     s.kci_vec_search_rerank.argtypes = [C.c_void_p, c_f32p, C.c_int, C.c_int, C.c_int, c_u64p, c_f32p]
    s.kci_vec_search_filtered.restype = C.c_int;   s.kci_vec_search_filtered.argtypes = [C.c_void_p, c_f32p, C.c_int, C.c_int, C.c_int, _FILTER, C.c_void_p, c_u64p, c_f32p]
    s.kci_vec_search_ivf.restype = C.c_int;        s.kci_vec_search_ivf.argtypes = [C.c_void_p, c_f32p, C.c_int, C.c_int, C.c_int, c_u64p, c_f32p]
    s.kci_vec_search_ivf_filtered.restype = C.c_int
    s.kci_vec_search_ivf_filtered.argtypes = [C.c_void_p, c_f32p, C.c_int, C.c_int, C.c_int, _FILTER, C.c_void_p, c_u64p, c_f32p]
    s.kci_vec_search_exact.restype = C.c_int;      s.kci_vec_search_exact.argtypes = [C.c_void_p, c_f32p, C.c_int, C.c_int, c_u64p, c_f32p]
    s.kci_vec_insert.restype = C.c_int64;          s.kci_vec_insert.argtypes = [C.c_void_p, C.c_uint64, c_f32p]
    s.kci_vec_delete.restype = C.c_int;            s.kci_vec_delete.argtypes = [C.c_void_p, C.c_uint64]
    s.kci_vec_compact.restype = C.c_uint32;        s.kci_vec_compact.argtypes = [C.c_void_p]
    s.kci_vec_float_mmap.restype = C.c_int;        s.kci_vec_float_mmap.argtypes = [C.c_void_p, C.c_char_p]
    s.kci_vec_save.restype = C.c_int;              s.kci_vec_save.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int]
    s.kci_vec_load.restype = C.c_void_p;           s.kci_vec_load.argtypes = [C.c_char_p, C.c_void_p, C.c_int]
    s.kci_vec_count.restype = C.c_uint32;          s.kci_vec_count.argtypes = [C.c_void_p]
    if hasattr(s, "kci_vec_plain_bytes"):
        s.kci_vec_plain_bytes.restype = C.c_int
        s.kci_vec_plain_bytes.argtypes = [C.c_void_p, C.POINTER(C.c_uint64)]
    s.kci_vec_dim.restype = C.c_int;               s.kci_vec_dim.argtypes = [C.c_void_p]
    if hasattr(s, "kci_vec_set_ivf_training"):
        s.kci_vec_set_ivf_training.restype = C.c_int; s.kci_vec_set_ivf_training.argtypes = [C.c_void_p, C.c_int, C.c_uint32]
    if hasattr(s, "kci_vec_set_ivf_codes"):
        s.kci_vec_set_ivf_codes.restype = C.c_int; s.kci_vec_set_ivf_codes.argtypes = [C.c_void_p, C.c_int, C.c_float]
    if hasattr(s, "kci_vec_new_l2"):
        s.kci_vec_new_l2.restype = C.c_void_p;     s.kci_vec_new_l2.argtypes = [C.c_int, C.c_int, C.c_int, C.c_float]
        s.kci_vec_is_l2.restype = C.c_int;         s.kci_vec_is_l2.argtypes = [C.c_void_p]
    s.kci_vec_is_built.restype = C.c_int;          s.kci_vec_is_built.argtypes = [C.c_void_p]
    return s


_LIB = None
def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _bind(C.CDLL(_find_lib()))
    return _LIB


def _f32(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(c_f32p)


def _make_filter(filt):
    """filt: None | callable(id) -> bool | a set / list / numpy array of allowed ids. Returns a C callback or None."""
    if filt is None:
        return None
    if callable(filt):
        return _FILTER(lambda i, _ctx: 1 if filt(int(i)) else 0)
    allowed = filt if isinstance(filt, (set, frozenset)) else set(int(x) for x in np.asarray(filt).ravel())
    return _FILTER(lambda i, _ctx: 1 if int(i) in allowed else 0)


class KciVec:
    """One index."""

    def __init__(self, dim, M=16, efc=200, rerank=0, metric="cosine", max_norm=0.0, _handle=None):
        self.lib = _lib()
        self.dim = dim
        self.rerank = rerank
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        if _handle is not None:
            self._v = _handle
            self.metric = "l2" if hasattr(self.lib, "kci_vec_is_l2") and self.lib.kci_vec_is_l2(_handle) else "cosine"
        else:
            self.metric = metric
            if metric == "l2":
                if not hasattr(self.lib, "kci_vec_new_l2"):
                    raise RuntimeError("this libkci_vec has no L2 (a cosine-only build, or one from before 2026-11-08)")
                self._v = self.lib.kci_vec_new_l2(dim, M, efc, float(max_norm))
            else:
                self._v = self.lib.kci_vec_new(dim, M, efc)
            if not self._v:
                raise RuntimeError("kci_vec_new failed (dim must be > 0; l2: max_norm >= 0)")
            if rerank:
                self.lib.kci_vec_keep_float(self._v, 1)
        self._s = None
        self._lists = False          # lists built (fast path)?
        self._dirty = False
        self.nprobe = 32
        self.rerank_mode = "float"
        self.probe_gap = (0.0, 8)
        self.eps = 0.0
        self.qg = False
        self._ob = (C.c_uint64 * 4096)()
        self._os = (C.c_float * 4096)()

    # ---- ingest ----
    def reserve(self, n): self.lib.kci_vec_reserve(self._v, int(n))

    def add(self, id, vec):
        _, p = _f32(np.reshape(vec, -1))
        return self.lib.kci_vec_add(self._v, int(id), p)

    def add_many(self, ids, mat):
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        n, d = mat.shape
        assert d == self.dim, f"dim mismatch {d} != {self.dim}"
        self.reserve(n + 1)
        base = mat.ctypes.data_as(C.c_void_p).value
        for i, vid in enumerate(ids):
            self.lib.kci_vec_add(self._v, int(vid), C.cast(base + i * d * 4, c_f32p))

    def build(self, threads=0):
        """Build the int8 HNSW graph. Returns the build seconds."""
        secs = C.c_double(0.0)
        self.lib.kci_vec_build(self._v, threads, C.byref(secs))
        self.qg = False              # a rebuild drops the quantized graph
        self._searcher(reset=True)
        return secs.value

    def build_qg(self, dims_per_code=1, threads=0):
        """Quantized graph lay layer 0 out as blocks of the neighbours' 4-bit codes, so a hop reads one block and scores
        every neighbour with table lookups. Dropped by insert() / compact() / build(); call again after. Returns the
        bytes it takes."""
        if not hasattr(self.lib, "kci_vec_build_qg"):
            raise RuntimeError("this libkci_vec has no kci_vec_build_qg (rebuild the library)")
        if self.lib.kci_vec_build_qg(self._v, int(dims_per_code), int(threads)) != 0:
            raise RuntimeError("kci_vec_build_qg failed (no graph yet? dims_per_code must be 1 or 2; out of memory?)")
        self.qg = True
        self._searcher(reset=True)
        return int(self.lib.kci_vec_qg_bytes(self._v))

    def build_lists(self, nlist=0, soar=0.0, threads=0, nprobe=None, dims_per_code=0, avq_threshold=0.0,
                    train_iters=0, train_sample=0):
        """Build the partitioned lists (4-bit table lookup) — the fast path. Build the graph first if you want both (the
        graph build renumbers). dims_per_code / avq_threshold how the residuals are coded, 0 = by the dims (one dim
        per code and 0.55 up to 32 dims, else pairs and 0.2); see kci_vec_set_ivf_codes. train_iters / train_sample
        k-means iterations (0 = 10) and sample size (0 = 250,000; -1 = all vectors) for the centroids."""
        if not self.rerank:
            raise RuntimeError("build_lists needs rerank=... (retained floats)")
        if dims_per_code or avq_threshold:
            if not hasattr(self.lib, "kci_vec_set_ivf_codes"):
                raise RuntimeError("this libkci_vec has no kci_vec_set_ivf_codes (built before 2026-11-08)")
            if self.lib.kci_vec_set_ivf_codes(self._v, int(dims_per_code), float(avq_threshold)) != 0:
                raise ValueError("dims_per_code must be 0, 1 or 2 and avq_threshold in [0, 0.98]")
        if train_iters or train_sample:
            if not hasattr(self.lib, "kci_vec_set_ivf_training"):
                raise RuntimeError("this libkci_vec has no kci_vec_set_ivf_training (built before 2026-11-08)")
            smp = 0xFFFFFFFF if train_sample < 0 else int(train_sample)
            if self.lib.kci_vec_set_ivf_training(self._v, int(train_iters), smp) != 0:
                raise ValueError("train_iters must be 0..1000")
        p = _IvfParams(int(nlist), int(threads), float(soar), 0, 0)
        if self.lib.kci_vec_build_ivf_p(self._v, C.byref(p)) != 0:
            raise RuntimeError("kci_vec_build_ivf_p failed")
        self._lists = True
        if nprobe: self.nprobe = int(nprobe)
        self._searcher(reset=True)

    # ---- query ----
    def set_rerank(self, mode):
        """'float' (exact, default) or 'int8' (1/4 of the bytes; fine for text embeddings at k=100, not for CLIP)."""
        if mode not in ("float", "int8"):
            raise ValueError("mode must be 'float' or 'int8'")
        self.rerank_mode = mode
        if self._s:
            self.lib.kci_vec_searcher_set_rerank_int8(self._s, 1 if mode == "int8" else 0)

    def set_probe_gap(self, gap, min_probe=8):
        """Lists: of the best nprobe, scan only those whose centroid score is within `gap` (cosine) of the nearest one's,
        at least min_probe."""
        if not hasattr(self.lib, "kci_vec_searcher_set_probe_gap"):
            raise RuntimeError("this libkci_vec has no kci_vec_searcher_set_probe_gap (rebuild the library)")
        self.probe_gap = (float(gap), int(min_probe))
        if self._s:
            self.lib.kci_vec_searcher_set_probe_gap(self._s, self.probe_gap[0], self.probe_gap[1])

    def set_eps(self, eps):
        """Graph: exploration radius (NGT-qg's stop rule). eps > 0 keeps the result set at `ef` entries (use a small ef,
        k..3k) and explores while the next candidate is within (1+eps) x the ef-th best cosine distance. Try
        0.08-0.25."""
        if not hasattr(self.lib, "kci_vec_searcher_set_eps"):
            raise RuntimeError("this libkci_vec has no kci_vec_searcher_set_eps (rebuild the library)")
        eps = float(eps)
        if eps != eps:
            raise ValueError("eps must be a number (NaN given)")
        self.eps = eps if eps > 0 else 0.0
        if self._s:
            self.lib.kci_vec_searcher_set_eps(self._s, self.eps)

    def _searcher(self, reset=False):
        if (reset or self._dirty) and self._s:
            self.lib.kci_vec_searcher_free(self._s); self._s = None
        self._dirty = False
        if not self._s:
            self._s = self.lib.kci_vec_searcher_new(self._v)
            if self.rerank_mode == "int8":
                self.lib.kci_vec_searcher_set_rerank_int8(self._s, 1)
            if self.probe_gap[0] > 0:
                self.lib.kci_vec_searcher_set_probe_gap(self._s, self.probe_gap[0], self.probe_gap[1])
            if self.eps > 0:
                self.lib.kci_vec_searcher_set_eps(self._s, self.eps)
            if self.qg:
                self.lib.kci_vec_searcher_set_qg(self._s, 1)
        return self._s

    def _query(self, q, k, ef, nprobe, rn, cb, method):
        s = self._searcher()
        use_lists = self._lists if method == "auto" else (method == "lists")
        if use_lists:
            if cb is not None:
                return self.lib.kci_vec_search_ivf_filtered(s, q, k, nprobe, rn, cb, None, self._ob, self._os)
            return self.lib.kci_vec_search_ivf(s, q, k, nprobe, rn, self._ob, self._os)
        if cb is not None:
            return self.lib.kci_vec_search_filtered(s, q, k, ef, rn, cb, None, self._ob, self._os)
        if rn and rn > 0:
            return self.lib.kci_vec_search_rerank(s, q, k, ef, rn, self._ob, self._os)
        return self.lib.kci_vec_search_r(s, q, k, ef, self._ob, self._os)

    def search(self, query, k=10, ef=None, nprobe=None, rerank_n=None, filter=None, method="auto"):
        """A selective filter needs a larger nprobe / rerank_n to find k matches."""
        k = int(k); ef = ef or max(k, 100); nprobe = int(nprobe or self.nprobe)
        rn = self.rerank if rerank_n is None else rerank_n
        if method == "lists" or (method == "auto" and self._lists):
            rn = max(int(rn or 0), 2 * k)
        _, q = _f32(np.reshape(query, -1))
        cb = _make_filter(filter)
        n = self._query(q, k, ef, nprobe, int(rn or 0), cb, method)
        return [(int(self._ob[i]), float(self._os[i])) for i in range(max(n, 0))]

    def search_many(self, queries, k=10, **kw):
        """Many queries at once. Returns (ids, scores): int64 / float32 arrays [nq, k], -1 / -inf (l2: +inf) where fewer than k."""
        Q = np.ascontiguousarray(queries, dtype=np.float32)
        if Q.ndim == 1: Q = Q[None, :]
        fill = np.inf if self.metric == "l2" else -np.inf
        ids = np.full((len(Q), k), -1, dtype=np.int64); scores = np.full((len(Q), k), fill, dtype=np.float32)
        for i, q in enumerate(Q):
            for j, (vid, sc) in enumerate(self.search(q, k=k, **kw)):
                ids[i, j] = vid; scores[i, j] = sc
        return ids, scores

    def search_exact(self, query, k=10, threads=0):
        """Brute force (ground truth / tiny indexes)."""
        _, q = _f32(np.reshape(query, -1))
        n = self.lib.kci_vec_search_exact(self._v, q, int(k), int(threads), self._ob, self._os)
        return [(int(self._ob[i]), float(self._os[i])) for i in range(max(n, 0))]

    # ---- incremental update (single-writer) ----
    def insert(self, id, vec):
        _, p = _f32(np.reshape(vec, -1))
        r = self.lib.kci_vec_insert(self._v, int(id), p)
        if r == -2:
            raise RuntimeError("insert requires rerank=… (retained floats)")
        if r < 0:
            raise RuntimeError("insert failed" + (" (l2: longer or much shorter than the vectors at the build?)" if self.metric == "l2" else ""))
        self._dirty = True
        self.qg = False
        return r

    def delete(self, id): return self.lib.kci_vec_delete(self._v, int(id)) == 0
    def compact(self):
        n = self.lib.kci_vec_compact(self._v); self.qg = False; self._searcher(reset=True); return n
    def float_mmap(self, path): return self.lib.kci_vec_float_mmap(self._v, path.encode()) == 0

    # ---- persistence (encrypted at rest if key given: AES-256-CTR + AES-256-CMAC) ----
    def save(self, path, key=None):
        kp = (C.c_char_p(key), len(key)) if key else (None, 0)
        return self.lib.kci_vec_save(self._v, path.encode(), kp[0], kp[1]) == 0

    @classmethod
    def load(cls, path, key=None, rerank=100):
        """(see the header for this function)"""
        lib = _lib()
        kp = (C.c_char_p(key), len(key)) if key else (None, 0)
        h = lib.kci_vec_load(path.encode(), kp[0], kp[1])
        if not h:
            raise RuntimeError("load failed (wrong key, damaged file, or file missing)")
        v = cls(lib.kci_vec_dim(h), _handle=h, rerank=rerank)
        # the file carries the lists when they were built; a probe tells which path is available
        try:
            s = lib.kci_vec_searcher_new(h); z = np.zeros(v.dim, dtype=np.float32)
            v._lists = lib.kci_vec_search_ivf(s, z.ctypes.data_as(c_f32p), 1, 1, 1, v._ob, v._os) >= 0
            lib.kci_vec_searcher_free(s)
        except Exception:
            v._lists = False
        return v

    def plain_bytes(self):
        """Plaintext bytes of the index in RAM, by kind (kci_vec_plain_bytes): codes, floats, derived (centroids,
        codebooks), graph, ids, total. Allocated sizes; the last query in a searcher is not counted."""
        out = (C.c_uint64 * 6)()
        if not hasattr(self.lib, "kci_vec_plain_bytes") or self.lib.kci_vec_plain_bytes(self._v, out) != 0:
            raise RuntimeError("kci_vec_plain_bytes not available")
        return dict(zip(("codes", "floats", "derived", "graph", "ids", "total"), (int(x) for x in out)))
    @property
    def count(self): return self.lib.kci_vec_count(self._v)
    def is_built(self): return self.lib.kci_vec_is_built(self._v) != 0
    @property
    def has_lists(self): return self._lists

    def close(self):
        if getattr(self, "_s", None): self.lib.kci_vec_searcher_free(self._s); self._s = None
        if getattr(self, "_v", None): self.lib.kci_vec_free(self._v); self._v = None
    def __del__(self):
        try: self.close()
        except Exception: pass
