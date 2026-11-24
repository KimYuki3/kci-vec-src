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
    u = KciVec(dim=128, rerank=400, metric="ip")   # inner product: scores are q.x, largest first
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
    # The Windows wheel carries one library, libkci_vec_avx2.dll (AVX2 + FMA + AES, no other DLLs needed). A CPU
    # without AVX2 would stop inside the DLL with an illegal instruction and no message, so the OS is asked first; a
    # libkci_vec.dll next to this file (built for this machine: pip install . from the source) still wins.
    # KCI_VEC_SKIP_CPU_CHECK=1 skips the question (an old Windows that does not know it answers no).
    if sys.platform == "win32" and not os.path.exists(os.path.join(here, "libkci_vec.dll")):
        p = os.path.join(here, "libkci_vec_avx2.dll")
        if os.path.exists(p):
            if not os.environ.get("KCI_VEC_SKIP_CPU_CHECK"):
                try:
                    has = bool(C.windll.kernel32.IsProcessorFeaturePresent(40))     # PF_AVX2_INSTRUCTIONS_AVAILABLE
                except Exception:
                    has = True
                if not has:
                    raise RuntimeError("kci_vec: this CPU (or this Windows) does not report AVX2, which the distributed "
                                       "library needs. Build from the source for this machine (pip install . in "
                                       "clients/python), or set KCI_VEC_SKIP_CPU_CHECK=1 if the CPU does have AVX2.")
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
    if hasattr(s, "kci_vec_half_floats"):
        s.kci_vec_half_floats.restype = C.c_int; s.kci_vec_half_floats.argtypes = [C.c_void_p]
    if hasattr(s, "kci_vec_build_qg_res"):
        s.kci_vec_build_qg_res.restype = C.c_int; s.kci_vec_build_qg_res.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]
    if hasattr(s, "kci_vec_search_only"):
        s.kci_vec_search_only.restype = C.c_int; s.kci_vec_search_only.argtypes = [C.c_void_p]
    s.kci_vec_reserve.argtypes = [C.c_void_p, C.c_uint32]
    s.kci_vec_add.restype = C.c_int64;             s.kci_vec_add.argtypes = [C.c_void_p, C.c_uint64, c_f32p]
    s.kci_vec_build.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_double)]
    if hasattr(s, "kci_vec_build_fast"):
        s.kci_vec_build_fast.restype = C.c_int;    s.kci_vec_build_fast.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_double)]
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
    if hasattr(s, "kci_vec_build_qg_ex"):
        s.kci_vec_build_qg_ex.restype = C.c_int; s.kci_vec_build_qg_ex.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]
    if hasattr(s, "kci_vec_image_save"):
        s.kci_vec_image_save.restype = C.c_int;    s.kci_vec_image_save.argtypes = [C.c_void_p, C.c_char_p]
        s.kci_vec_attach.restype = C.c_void_p;     s.kci_vec_attach.argtypes = [C.c_char_p]
    if hasattr(s, "kci_vec_build_qg_gate"):
        s.kci_vec_build_qg_gate.restype = C.c_int; s.kci_vec_build_qg_gate.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
    if hasattr(s, "kci_vec_searcher_set_qg_exact"):
        s.kci_vec_searcher_set_qg_exact.restype = C.c_int; s.kci_vec_searcher_set_qg_exact.argtypes = [C.c_void_p, C.c_int]
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
    if hasattr(s, "kci_vec_set_dedup"):             # identical vectors kept once
        s.kci_vec_set_dedup.restype = C.c_int;     s.kci_vec_set_dedup.argtypes = [C.c_void_p, C.c_int]
        s.kci_vec_searcher_set_unique.restype = C.c_int; s.kci_vec_searcher_set_unique.argtypes = [C.c_void_p, C.c_int]
    if hasattr(s, "kci_vec_dup_stats"):             # exact copies among the rows
        s.kci_vec_dup_stats.restype = C.c_int;     s.kci_vec_dup_stats.argtypes = [C.c_void_p, C.POINTER(C.c_uint64)]
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
    if hasattr(s, "kci_vec_new_ip"):
        s.kci_vec_new_ip.restype = C.c_void_p;     s.kci_vec_new_ip.argtypes = [C.c_int, C.c_int, C.c_int, C.c_float]
    if hasattr(s, "kci_vec_learn"):
        s.kci_vec_learn.restype = C.c_int
        s.kci_vec_learn.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
        s.kci_vec_has_learned.restype = C.c_int;   s.kci_vec_has_learned.argtypes = [C.c_void_p]
        if hasattr(s, "kci_vec_learn_pack"):
            s.kci_vec_learn_pack.restype = C.c_int; s.kci_vec_learn_pack.argtypes = [C.c_void_p]
        for f in ("kci_vec_searcher_set_learned_seeds", "kci_vec_searcher_set_learned_pages", "kci_vec_searcher_set_learned_hot"):
            if hasattr(s, f):
                getattr(s, f).restype = C.c_int; getattr(s, f).argtypes = [C.c_void_p, C.c_int]
        s.kci_vec_search_learned.restype = C.c_int
        s.kci_vec_search_learned.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
    if hasattr(s, "kci_vec_set_ip_centre"):
        s.kci_vec_set_ip_centre.restype = C.c_int; s.kci_vec_set_ip_centre.argtypes = [C.c_void_p, C.c_int]
    if hasattr(s, "kci_vec_metric"):
        s.kci_vec_metric.restype = C.c_int;        s.kci_vec_metric.argtypes = [C.c_void_p]
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

    def __init__(self, dim, M=16, efc=200, rerank=0, metric="cosine", max_norm=0.0, centre=False, _handle=None, dedup=False):
        self.lib = _lib()
        self._unique = False
        self.dim = dim
        self.rerank = rerank
        if metric not in ("cosine", "l2", "ip"):
            raise ValueError("metric must be 'cosine', 'l2' or 'ip'")
        if _handle is not None:
            self._v = _handle
            if hasattr(self.lib, "kci_vec_metric"):
                self.metric = ("cosine", "l2", "ip")[self.lib.kci_vec_metric(_handle)]
            else:
                self.metric = "l2" if hasattr(self.lib, "kci_vec_is_l2") and self.lib.kci_vec_is_l2(_handle) else "cosine"
        else:
            self.metric = metric
            if metric == "l2":
                if not hasattr(self.lib, "kci_vec_new_l2"):
                    raise RuntimeError("this libkci_vec has no L2 (a cosine-only build, or one from before 2026-11-08)")
                self._v = self.lib.kci_vec_new_l2(dim, M, efc, float(max_norm))
            elif metric == "ip":
                if not hasattr(self.lib, "kci_vec_new_ip"):
                    raise RuntimeError("this libkci_vec has no inner product (a cosine-only build, or one from before 2026-11-16)")
                self._v = self.lib.kci_vec_new_ip(dim, M, efc, float(max_norm))
                if centre and self._v:
                    if not hasattr(self.lib, "kci_vec_set_ip_centre") or self.lib.kci_vec_set_ip_centre(self._v, 1) != 0:
                        raise RuntimeError("centre=True is not available in this libkci_vec")
            else:
                self._v = self.lib.kci_vec_new(dim, M, efc)
            if not self._v:
                raise RuntimeError("kci_vec_new failed (dim must be > 0; l2 / ip: max_norm >= 0)")
            if rerank:
                self.lib.kci_vec_keep_float(self._v, 1)
            # dedup=True: identical vectors are kept once, their ids beside the row (kci_vec_set_dedup). For data with
            # texts that repeat: see dup_stats(). Searches still answer with every id; .unique = True gives one id per
            if dedup:
                if not hasattr(self.lib, "kci_vec_set_dedup") or self.lib.kci_vec_set_dedup(self._v, 1) != 0:
                    raise RuntimeError("dedup=True is not available in this libkci_vec; rebuild it from the current kci_vec.c")
        self._s = None
        self._lists = False          # lists built (fast path)?
        self._dirty = False
        self.nprobe = 32
        self.rerank_mode = "float"
        self.probe_gap = (0.0, 8)
        self.eps = 0.0
        self.qg_exact = False
        self.qg = False
        self.attached = False
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

    def build(self, threads=0, fast=False):
        """Build the int8 HNSW graph."""
        secs = C.c_double(0.0)
        if fast and not hasattr(self.lib, "kci_vec_build_fast"):
            raise RuntimeError("this libkci_vec has no kci_vec_build_fast (built before 2026-11-20); rebuild it, or use build()")
        rc = (self.lib.kci_vec_build_fast if fast else self.lib.kci_vec_build)(self._v, threads, C.byref(secs))
        if fast and rc != 0:
            raise RuntimeError("kci_vec_build_fast failed (out of memory, or no vectors)")
        self.qg = False              # a rebuild drops the quantized graph
        self._searcher(reset=True)
        return secs.value

    def build_qg(self, dims_per_code=1, threads=0, code_dims=0, gate_dims=0, residual=False):
        """Quantized graph lay layer 0 out as blocks of the neighbours' 4-bit codes, so a hop reads one block and scores
        every neighbour with table lookups. Dropped by insert() / compact() / build(); call again after. Returns the
        bytes it takes. code_dims > 0 the codes cover the top code_dims principal directions only (even, 8..dim-1), a
        smaller block per node."""
        if not hasattr(self.lib, "kci_vec_build_qg"):
            raise RuntimeError("this libkci_vec has no kci_vec_build_qg (rebuild the library)")
        if residual:
            if not hasattr(self.lib, "kci_vec_build_qg_res"):
                raise RuntimeError("this libkci_vec has no kci_vec_build_qg_res (rebuild the library)")
            if self.lib.kci_vec_build_qg_res(self._v, int(code_dims), int(gate_dims), int(threads)) != 0:
                raise RuntimeError("kci_vec_build_qg_res failed (no graph yet? code_dims must be even, 8..dim-1, gate_dims even, 8..code_dims; out of memory?)")
        elif int(gate_dims):
            if not hasattr(self.lib, "kci_vec_build_qg_gate"):
                raise RuntimeError("this libkci_vec has no kci_vec_build_qg_gate (rebuild the library)")
            if self.lib.kci_vec_build_qg_gate(self._v, int(dims_per_code), int(code_dims), int(gate_dims), int(threads)) != 0:
                raise RuntimeError("kci_vec_build_qg_gate failed (no graph yet? gate_dims must be even, 8..code_dims; out of memory?)")
        elif int(code_dims):
            if not hasattr(self.lib, "kci_vec_build_qg_ex"):
                raise RuntimeError("this libkci_vec has no kci_vec_build_qg_ex (rebuild the library)")
            if self.lib.kci_vec_build_qg_ex(self._v, int(dims_per_code), int(code_dims), int(threads)) != 0:
                raise RuntimeError("kci_vec_build_qg_ex failed (no graph yet? code_dims must be even, 8..dim-1; out of memory?)")
        elif self.lib.kci_vec_build_qg(self._v, int(dims_per_code), int(threads)) != 0:
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

    def set_qg_exact(self, on=True):
        """Worth trying at high dimensions with dims_per_code 2. Off by default. on=2: with build_qg(code_dims=...) score
        them with short int8 rows (code_dims bytes) instead; less to read per hop, a little coarser, so search with a
        wider rerank_n. Without code_dims the same as True."""
        if not hasattr(self.lib, "kci_vec_searcher_set_qg_exact"):
            raise RuntimeError("this libkci_vec has no kci_vec_searcher_set_qg_exact (rebuild the library)")
        self.qg_exact = 2 if (on == 2 and on is not True) else (1 if on else 0)
        if self._s:
            self.lib.kci_vec_searcher_set_qg_exact(self._s, self.qg_exact)

    @property
    def unique(self):
        """With dedup=True: True = a search answers with one id per distinct vector (k results are k different
        vectors); False (the default) = every id, the ids of one vector one after the other."""
        return self._unique
    @unique.setter
    def unique(self, on):
        self._unique = bool(on)
        if getattr(self, "_s", None): self._searcher(reset=True)

    def _searcher(self, reset=False):
        if (reset or self._dirty) and self._s:
            self.lib.kci_vec_searcher_free(self._s); self._s = None
        self._dirty = False
        if not self._s:
            self._s = self.lib.kci_vec_searcher_new(self._v)
            if self._unique and hasattr(self.lib, "kci_vec_searcher_set_unique"):
                self.lib.kci_vec_searcher_set_unique(self._s, 1)
            if self.rerank_mode == "int8":
                self.lib.kci_vec_searcher_set_rerank_int8(self._s, 1)
            if self.probe_gap[0] > 0:
                self.lib.kci_vec_searcher_set_probe_gap(self._s, self.probe_gap[0], self.probe_gap[1])
            if self.eps > 0:
                self.lib.kci_vec_searcher_set_eps(self._s, self.eps)
            if self.qg:
                self.lib.kci_vec_searcher_set_qg(self._s, 1)
            if self.qg_exact:
                self.lib.kci_vec_searcher_set_qg_exact(self._s, int(self.qg_exact))
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

    def learn(self, queries, answers, nclusters=0, list_cap=0, threads=0, pack=False):
        """Learn lists from a query sample, for data whose queries come from another distribution than the vectors and
        whose answers concentrate on few vectors (e.g. the queries and keys of an attention head). queries: float32
        [nq, dim]; answers: ids [nq, per_query], the (true or approximate) nearest of each query. Needs a built index
        (build() or build_lists()) and rerank > 0."""
        if not hasattr(self.lib, "kci_vec_learn"):
            raise RuntimeError("this libkci_vec has no kci_vec_learn (from before 2026-11-16)")
        Qs = np.ascontiguousarray(queries, dtype=np.float32); A = np.ascontiguousarray(answers, dtype=np.uint64)
        if Qs.ndim != 2 or A.ndim != 2 or len(Qs) != len(A) or Qs.shape[1] != self.dim:
            raise ValueError("queries must be [nq, dim] and answers [nq, per_query]")
        top5 = C.c_float(0.0)
        if self.lib.kci_vec_learn(self._v, Qs.ctypes.data, len(Qs), A.ctypes.data, A.shape[1], int(nclusters), int(list_cap),
                                  int(threads), C.byref(top5)) != 0:
            raise RuntimeError("kci_vec_learn failed (index not built? no answer id is in the index? out of memory?)")
        if pack:
            if not hasattr(self.lib, "kci_vec_learn_pack") or self.lib.kci_vec_learn_pack(self._v) != 0:
                raise RuntimeError("kci_vec_learn_pack failed (out of memory, or a libkci_vec without it)")
        return float(top5.value)

    def search_learned(self, query, k=10, nclusters=4, per_list=1000, ef=0, seeds=0, pages=0, hot=0):
        """Top-k through the learned lists: the nclusters (1..64) nearest query clusters, the first per_list vectors of
        each. ef = 0: those are scored exactly and the best k returned (fast; no fallback for a query unlike the sample).
        ef > 0: they start the graph search (needs build()), which completes what the lists miss.
        ef < 0 (after learn(pack=True)): the lists' int8 copies are scanned and only the -ef best (150 - 500) scored exactly.
        seeds > 0 (ef > 0, packed): only the `seeds` best of the int8 scan start the graph (0: every vector of the prefixes).
        pages > 0: the clusters are chosen in two stages, through the nearest `pages` groups of them (cheaper when
        nclusters is small; 0: compare with every cluster).
        hot > 0: also the first `hot` of the rows most often an answer over the whole sample (learn() keeps up to a
        quarter of the index); for the last answers the nearest lists miss, when the answers are concentrated."""
        _, q = _f32(np.reshape(query, -1))
        for f, x in (("kci_vec_searcher_set_learned_seeds", seeds), ("kci_vec_searcher_set_learned_pages", pages),
                     ("kci_vec_searcher_set_learned_hot", hot)):
            if hasattr(self.lib, f):
                if getattr(self.lib, f)(self._searcher(), int(x)) != 0:
                    raise ValueError(f + ": negative value")
            elif x:
                raise RuntimeError(f + ": this libkci_vec does not have it")
        n = self.lib.kci_vec_search_learned(self._searcher(), q, int(k), int(nclusters), int(per_list), int(ef), self._ob, self._os)
        if n == -2:
            raise RuntimeError("nothing learned (call learn(); build() and compact() drop it; ef < 0 needs learn(pack=True))")
        if n < 0:
            raise RuntimeError("search_learned failed (rerank=... needed; ef > 0 needs build())")
        return [(int(self._ob[i]), float(self._os[i])) for i in range(n)]

    def vote_set(self, answers):
        """For data whose queries come from another distribution and whose answers are NOT concentrated (learn() returns
        a low share): keep the answers of a query sample, answers = ids [nq, per_query], and an int8 copy of every
        vector (dim bytes each). search_voted() then takes the numbers (0 .. nq-1) of the sample queries nearest to a
        query, which the caller finds, for instance with a second KciVec holding the sample queries. Needs a built
        index and rerank > 0. build() and compact() drop it; it is not saved."""
        if not hasattr(self.lib, "kci_vec_vote_set"):
            raise RuntimeError("this libkci_vec has no kci_vec_vote_set (from before 2026-11-17)")
        A = np.ascontiguousarray(answers, dtype=np.uint64)
        if A.ndim != 2:
            raise ValueError("answers must be [nq, per_query]")
        self.lib.kci_vec_vote_set.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_int]
        if self.lib.kci_vec_vote_set(self._v, A.ctypes.data, A.shape[0], A.shape[1]) != 0:
            raise RuntimeError("kci_vec_vote_set failed (index not built? rerank=... needed; out of memory?)")

    def vote_short(self, ids, rows):
        """Optional, after vote_set(): one short vector per stored vector (rows float32 [count, sdim], in the order of
        ids), chosen by the caller so that short query . short row orders the candidates like the score. Kept as int8;
        search_voted(short_query=...) ranks the candidates by it first. For high dimensions."""
        I = np.ascontiguousarray(ids, dtype=np.uint64); R = np.ascontiguousarray(rows, dtype=np.float32)
        if R.ndim != 2 or I.ndim != 1 or len(I) != len(R):
            raise ValueError("rows must be [count, sdim] and ids [count]")
        if not hasattr(self.lib, "kci_vec_vote_short"):
            raise RuntimeError("this libkci_vec has no kci_vec_vote_short (from before 2026-11-17)")
        self.lib.kci_vec_vote_short.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32, C.c_int]
        if self.lib.kci_vec_vote_short(self._v, I.ctypes.data, R.ctypes.data, len(I), R.shape[1]) != 0:
            raise RuntimeError("kci_vec_vote_short failed (vote_set() first; every id of the index exactly once; sdim 1..4096)")
        self._vote_sdim = R.shape[1]

    def search_voted(self, query, voters, k=10, rerank=200, short_query=None, keep=1000):
        """Top-k among the answers of the sample queries numbered in `voters` (see vote_set): the union is scanned on
        int8 copies and the `rerank` best (at most 4096) are scored exactly. With short_query (after vote_short) the
        union is ranked by the short codes first and only the `keep` best go on. No fallback for a query unlike the sample."""
        _, q = _f32(np.reshape(query, -1))
        V = np.ascontiguousarray(voters, dtype=np.uint32).reshape(-1)
        if len(V) == 0:
            return []
        k = min(int(k), len(self._ob))
        if short_query is None:
            self.lib.kci_vec_search_voted.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
            n = self.lib.kci_vec_search_voted(self._searcher(), q, V.ctypes.data, len(V), k, int(rerank), self._ob, self._os)
        else:
            qs = np.ascontiguousarray(np.reshape(short_query, -1), dtype=np.float32)
            if len(qs) != getattr(self, "_vote_sdim", -1):
                raise ValueError("short_query must have the sdim given to vote_short()")
            self.lib.kci_vec_search_voted_short.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
            n = self.lib.kci_vec_search_voted_short(self._searcher(), q, qs.ctypes.data, int(keep), V.ctypes.data, len(V), k, int(rerank), self._ob, self._os)
        if n == -2:
            raise RuntimeError("nothing set (call vote_set(); build() and compact() drop it; short_query needs vote_short())")
        if n < 0:
            raise RuntimeError("search_voted failed")
        return [(int(self._ob[i]), float(self._os[i])) for i in range(n)]

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
            raise RuntimeError("insert failed" + (" (l2: longer or much shorter than the vectors at the build?)" if self.metric == "l2" else
                                                " (ip: longer than the longest vector at the build?)" if self.metric == "ip" else ""))
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
        v.qg = hasattr(lib, "kci_vec_qg_bytes") and lib.kci_vec_qg_bytes(h) > 0
        # the file carries the lists when they were built; a probe tells which path is available
        try:
            s = lib.kci_vec_searcher_new(h); z = np.zeros(v.dim, dtype=np.float32)
            v._lists = lib.kci_vec_search_ivf(s, z.ctypes.data_as(c_f32p), 1, 1, 1, v._ob, v._os) >= 0
            lib.kci_vec_searcher_free(s)
        except Exception:
            v._lists = False
        return v

    def save_image(self, path):
        """Write the index as a read-only image (kci_vec_image_save): a flat, plaintext file that KciVec.attach() opens
        without parsing or copying. It carries the rows, the floats, the lists, and for an index with a graph the
        graph, the quantized graph (build_qg) and the halves (half_floats). Not encrypted: for an encrypted file use
        save(path, key). For the machine that wrote it (byte order). Returns True, or raises."""
        if not hasattr(self.lib, "kci_vec_image_save"):
            raise RuntimeError("this libkci_vec has no kci_vec_image_save (rebuild the library)")
        if self.lib.kci_vec_image_save(self._v, os.fsencode(path)) != 0:
            raise RuntimeError("kci_vec_image_save failed (cannot write the file, or the floats are mapped from a file)")
        return True

    @classmethod
    def attach(cls, path, rerank=100):
        """Anything that writes (add, insert, delete, build, build_qg...) fails or is refused. rerank: the default
        shortlist for search, as in load(). Missing, damaged or foreign file -> RuntimeError."""
        lib = _lib()
        if not hasattr(lib, "kci_vec_attach"):
            raise RuntimeError("this libkci_vec has no kci_vec_attach (rebuild the library)")
        h = lib.kci_vec_attach(os.fsencode(path))
        if not h:
            raise RuntimeError("attach failed (file missing, not an image, or damaged)")
        v = cls(lib.kci_vec_dim(h), _handle=h, rerank=rerank)
        v.attached = True
        try:
            s = lib.kci_vec_searcher_new(h); z = np.zeros(v.dim, dtype=np.float32)
            v._lists = lib.kci_vec_search_ivf(s, z.ctypes.data_as(c_f32p), 1, 1, 1, v._ob, v._os) >= 0
            lib.kci_vec_searcher_free(s)
        except Exception:
            v._lists = False
        v.qg = hasattr(lib, "kci_vec_qg_bytes") and lib.kci_vec_qg_bytes(h) > 0
        return v

    def half_floats(self):
        """Turn the retained floats into 16-bit halves (kci_vec_half_floats): half the memory for them, and
        search_rerank's float pass reads half as much (cosine scores within 1e-4 of the floats'; inner-product / L2
        scores are off by the same relative amount). One way; afterwards the index is as one without floats for
        everything but search_rerank (no insert, a save has no floats). Call after build() and build_qg()."""
        if not hasattr(self.lib, "kci_vec_half_floats"):
            raise RuntimeError("this libkci_vec has no kci_vec_half_floats (rebuild the library)")
        if self.lib.kci_vec_half_floats(self._v) != 0:
            raise RuntimeError("kci_vec_half_floats failed (needs a built index with floats kept in RAM)")

    def search_only(self):
        """Give up what a quantized-graph search does not read — the int8 rows and the plain graph's layer-0 links
        (kci_vec_search_only) — for an index that will only be searched. Needs build(), build_qg() and an exact pass
        (half_floats(), or rerank > 0 at creation)."""
        if not hasattr(self.lib, "kci_vec_search_only"):
            raise RuntimeError("this libkci_vec has no kci_vec_search_only (rebuild the library)")
        if self.lib.kci_vec_search_only(self._v) != 0:
            raise RuntimeError("kci_vec_search_only failed (needs a built index with a quantized graph and halves or floats; no lists)")
        self._searcher(reset=True)

    def plain_bytes(self):
        """Plaintext bytes of the index in RAM, by kind (kci_vec_plain_bytes): codes, floats, derived (centroids,
        codebooks), graph, ids, total. Allocated sizes; the last query in a searcher is not counted."""
        out = (C.c_uint64 * 6)()
        if not hasattr(self.lib, "kci_vec_plain_bytes") or self.lib.kci_vec_plain_bytes(self._v, out) != 0:
            raise RuntimeError("kci_vec_plain_bytes not available")
        return dict(zip(("codes", "floats", "derived", "graph", "ids", "total"), (int(x) for x in out)))
    @property
    def count(self): return self.lib.kci_vec_count(self._v)

    def dup_stats(self):
        """How many rows are exact copies of another row (a built index; kci_vec_dup_stats): a dict with rows, groups (of
        2 or more identical rows), rows_in_groups, largest, groups_over_m0 (larger than the layer-0 neighbour list)
        and distinct."""
        if not hasattr(self.lib, "kci_vec_dup_stats"):
            raise RuntimeError("this libkci_vec has no kci_vec_dup_stats; rebuild it from the current kci_vec.c")
        out = (C.c_uint64 * 8)()
        if self.lib.kci_vec_dup_stats(self._v, out) != 0:
            raise RuntimeError("dup_stats: the index is not built (or its int8 rows were given up: search_only)")
        return dict(rows=out[0], groups=out[1], rows_in_groups=out[2], largest=out[3], groups_over_m0=out[4], distinct=out[5],
                    ids_beside_rows=out[6], rows_with_company=out[7])     # the last two: with dedup=True
    def is_built(self): return self.lib.kci_vec_is_built(self._v) != 0
    @property
    def has_lists(self): return self._lists

    def close(self):
        if getattr(self, "_s", None): self.lib.kci_vec_searcher_free(self._s); self._s = None
        if getattr(self, "_v", None): self.lib.kci_vec_free(self._v); self._v = None
    def __del__(self):
        try: self.close()
        except Exception: pass
