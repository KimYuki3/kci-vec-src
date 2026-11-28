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
import warnings
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
    if hasattr(s, "kci_vec_search_exact_filtered"):
        s.kci_vec_search_exact_filtered.restype = C.c_int
        s.kci_vec_search_exact_filtered.argtypes = [C.c_void_p, c_f32p, C.c_int, C.c_int, _FILTER, C.c_void_p, c_u64p, c_f32p]
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
    if hasattr(s, "kci_vec_calibrate"):
        s.kci_vec_calibrate.restype = C.c_int
        s.kci_vec_calibrate.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_float, C.c_int, C.c_int, C.POINTER(C.c_float)]
        s.kci_vec_default_ef.restype = C.c_int;     s.kci_vec_default_ef.argtypes = [C.c_void_p]
        s.kci_vec_set_default_ef.restype = C.c_int; s.kci_vec_set_default_ef.argtypes = [C.c_void_p, C.c_int]
    if hasattr(s, "kci_vec_searcher_set_filter_fill"):
        s.kci_vec_searcher_set_filter_fill.restype = C.c_int; s.kci_vec_searcher_set_filter_fill.argtypes = [C.c_void_p, C.c_int]
    if hasattr(s, "kci_vec_default_rerank"):
        s.kci_vec_default_rerank.restype = C.c_int; s.kci_vec_default_rerank.argtypes = [C.c_void_p]
    if hasattr(s, "kci_vec_judge"):
        s.kci_vec_judge.restype = C.c_int
        s.kci_vec_judge.argtypes = [C.c_void_p, C.c_int, C.c_float, C.c_int, C.POINTER(C.c_float)]
        s.kci_vec_judge_queries.restype = C.c_int
        s.kci_vec_judge_queries.argtypes = [C.c_void_p, c_f32p, C.c_int, C.c_int, C.c_float, C.POINTER(C.c_float)]
        s.kci_vec_judged.restype = C.c_int;         s.kci_vec_judged.argtypes = [C.c_void_p]
        s.kci_vec_judge_off.restype = C.c_int;      s.kci_vec_judge_off.argtypes = [C.c_void_p]
    if hasattr(s, "kci_vec_calibrate_queries"):
        s.kci_vec_calibrate_queries.restype = C.c_int
        s.kci_vec_calibrate_queries.argtypes = [C.c_void_p, c_f32p, C.c_int, C.c_int, C.c_int, C.c_float, C.c_int, C.POINTER(C.c_float)]
        s.kci_vec_unreached.restype = C.c_uint32;   s.kci_vec_unreached.argtypes = [C.c_void_p]
    if hasattr(s, "kci_vec_has_floats"):
        s.kci_vec_has_floats.restype = C.c_int;    s.kci_vec_has_floats.argtypes = [C.c_void_p]
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


class _IdBits(C.Structure):
    _fields_ = [("bits", C.c_void_p), ("nbits", C.c_uint64)]


_BITS_MAX = 1 << 27


def _id_bits(filt):
    """a set / list / array of allowed ids as (bitmap uint64[], nbits, count), or None when the ids do not fit a
    bitmap (negative, not whole numbers, or >= _BITS_MAX)."""
    if isinstance(filt, (set, frozenset)):
        try:
            a = np.fromiter(filt, dtype=np.int64, count=len(filt))
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        a = np.asarray(filt).ravel()
        if a.dtype.kind not in "iu":
            return None
    if a.size == 0:
        return np.zeros(1, dtype=np.uint64), 0, 0
    if int(a.min()) < 0 or int(a.max()) >= _BITS_MAX:
        return None
    nbits = int(a.max()) + 1
    flags = np.zeros(((nbits + 63) // 64) * 64, dtype=np.uint8)
    flags[a.astype(np.int64, copy=False)] = 1
    bits = np.ascontiguousarray(np.packbits(flags, bitorder="little").view("<u8").astype(np.uint64, copy=False))
    return bits, nbits, int(flags.sum())


class KciVec:
    """One index."""

    def __init__(self, dim, M=16, efc=200, rerank=200, metric="cosine", max_norm=0.0, centre=False, _handle=None, dedup=False):
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
        self.exact_cells = 10_000_000  # search(): brute force on the floats while count x dim <= this
        self.calibrated_recall = None  # what the last calibrate() measured
        self.auto_ef = True            # build() measures the search width this data needs (see calibrate)
        self.filter_fill = True        # a filtered search that finds fewer than k answers is answered again by brute force
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
        if rc == 0 and hasattr(self.lib, "kci_vec_unreached") and self.lib.kci_vec_unreached(self._v):
            warnings.warn("kci_vec: %d rows cannot be reached by the graph search after this build (search_exact still "
                          "finds them)" % self.lib.kci_vec_unreached(self._v), RuntimeWarning, stacklevel=2)
        if (rc == 0 and self.auto_ef and hasattr(self.lib, "kci_vec_calibrate") and self.rerank
                and hasattr(self.lib, "kci_vec_has_floats") and self.lib.kci_vec_has_floats(self._v) == 1 and not self._small_exact()):
            ef, rec = self.calibrate(threads=threads)
            if rec < 0.98 - 0.01:
                warnings.warn("kci_vec: at the widest search tried (ef %d) the index finds %.1f %% of the true 10 nearest of "
                              "its own vectors; search() will miss some. search_exact() is exact; a larger M at creation "
                              "or ef / rerank_n at search raise it." % (ef, 100.0 * rec), RuntimeWarning, stacklevel=2)
        return secs.value

    def calibrate(self, k=10, target=0.98, samples=128, threads=0, queries=None):
        """Measure the search width (ef) this index needs and make it the default of search().
        `samples` stored vectors are used as questions against brute force, and the smallest of ef 200 / 400 / 800 /
        1600 that finds the share `target` of the true k nearest is kept (the widest when none does). build() calls
        this for an index above exact_cells unless auto_ef is False; save() / load() keep the result. Returns
        (ef, the share found at it); the share is also kept in calibrated_recall. Needs the floats (rerank > 0).
        queries: your own questions (8 or more rows of dim floats; at most 4,096 are used) instead of stored vectors,
        for questions of another kind than the stored vectors."""
        if not hasattr(self.lib, "kci_vec_calibrate"):
            raise RuntimeError("this libkci_vec has no kci_vec_calibrate (built before 2026-11-26); rebuild it")
        r = C.c_float(0.0)
        if queries is not None:
            if not hasattr(self.lib, "kci_vec_calibrate_queries"):
                raise RuntimeError("this libkci_vec has no kci_vec_calibrate_queries (built before 2026-11-28); rebuild it")
            Qa = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, self.dim)
            ef = self.lib.kci_vec_calibrate_queries(self._v, Qa.ctypes.data_as(c_f32p), len(Qa), int(k), int(self.rerank or 0), float(target), int(threads), C.byref(r))
        else:
            ef = self.lib.kci_vec_calibrate(self._v, int(k), int(self.rerank or 0), float(target), int(samples), int(threads), C.byref(r))
        if ef < 0:
            raise RuntimeError("calibrate needs a built graph with the float vectors kept (rerank > 0)" + (", and 8 or more queries" if queries is not None else ""))
        self.calibrated_recall = float(r.value)
        return ef, float(r.value)

    def judge(self, k=10, target=0.99, samples=4096, queries=None):
        """Let the index decide the search width for each query. After judge(), search() without ef can answer the
        queries that need little with less work; a search with an explicit ef is unchanged.
        The rule is prepared from `samples` stored vectors, or from your own `queries` (200 or more rows; use them
        when your questions do not look like the stored vectors). k: the number of results your searches ask for
        (2..64). target: the share of a wide search's answer to keep.
        Returns (in_use, gain): in_use False means the rule was measured not to help on this index and search()
        stays as it was; gain is the measured ratio of work against a fixed width.
        For graph searches of up to k results without a filter; not for metric "ip". Not saved: call it again after
        load() or build(). judge(samples=0) forgets the rule."""
        if not hasattr(self.lib, "kci_vec_judge"):
            raise RuntimeError("this libkci_vec has no kci_vec_judge (built before 2026-11-28); rebuild it")
        if queries is None and samples == 0:
            self.lib.kci_vec_judge_off(self._v); return False, 0.0
        g = C.c_float(0.0)
        if queries is not None:
            Qa = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, self.dim)
            r = self.lib.kci_vec_judge_queries(self._v, Qa.ctypes.data_as(c_f32p), len(Qa), int(k), float(target), C.byref(g))
        else:
            r = self.lib.kci_vec_judge(self._v, int(k), float(target), int(samples), C.byref(g))
        if r < 0:
            raise RuntimeError("judge: needs a built graph with the floats kept (rerank > 0), cosine or l2, 400 vectors or more, k 2..64, 200 queries or more")
        return bool(r), float(g.value)

    def _use_lists(self, method):
        """True when a search with this method goes through the lists."""
        return method == "lists" or (method == "auto" and self._lists and getattr(self, "_prefer", "lists") != "graph")

    def choose_method(self, k=10, target=0.98, queries=None, samples=1000, nlist=0, threads=0):
        """Measure which way of searching suits this data, the graph or the lists, and make it what search() does when
        no method is given. The lists are built if they are not there (build_lists(nlist, threads=threads)); then
        `samples` stored vectors, or your own `queries`, are searched both ways at growing widths and compared with
        the exact answer, and the faster of the two settings that reach `target` (recall@k) is kept: the method, and
        its ef or nprobe. If neither reaches the target, the one with the higher recall is kept.
        Returns a dict: 'method', and for 'graph' / 'lists' the setting, its recall and its milliseconds a query.
        Times are taken here, on this machine, one thread: run it when the machine is quiet. Needs the floats
        (rerank > 0) and a built graph. Stored vectors stand in for real questions only roughly: give `queries` when
        you have them, or a target a little above what you need. Not saved: call it again after load(). build() and
        calibrate() set the width again."""
        import time
        if (not self.rerank or self.lib.kci_vec_is_built(self._v) == 0
                or (hasattr(self.lib, "kci_vec_has_floats") and self.lib.kci_vec_has_floats(self._v) != 1)):
            raise RuntimeError("choose_method: needs a built graph with the floats kept (rerank > 0)")
        k1 = int(k) + 1
        if queries is not None:
            Qa = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, self.dim)[:int(samples) if samples else None]
        else:
            if not hasattr(self.lib, "kci_vec_get_float"):
                raise RuntimeError("choose_method: this libkci_vec cannot give stored vectors back; pass queries")
            self.lib.kci_vec_get_float.restype = C.c_int; self.lib.kci_vec_get_float.argtypes = [C.c_void_p, C.c_uint32, c_f32p]
            n = int(self.count); rows = np.unique(np.random.default_rng(12345).integers(0, n, size=int(samples)))
            Qa = np.zeros((len(rows), self.dim), dtype=np.float32); got = 0
            for r in rows:
                if self.lib.kci_vec_get_float(self._v, int(r), Qa[got].ctypes.data_as(c_f32p)) == 0: got += 1
            Qa = Qa[:got]
        if len(Qa) < 8:
            raise RuntimeError("choose_method: 8 or more questions are needed")
        # the best exact hit of a stored vector is itself (or a copy of it): out of the truth, and never counted
        truth = [self.search_exact(q, k=k1, threads=threads) for q in Qa]
        own = [t[0][0] if queries is None and t else None for t in truth]
        truth = [{i for i, _ in t[:int(k)]} if o is None else {i for i, _ in t} - {o} for t, o in zip(truth, own)]
        if not self._lists:
            self.build_lists(nlist=nlist, threads=threads)
        def measure(**kw):
            best = 1e9; hit = 0
            for rep in range(2):
                t0 = time.perf_counter(); res = [self.search(q, k=k1, **kw) for q in Qa]; best = min(best, time.perf_counter() - t0)
            hit = sum(len(t & ({i for i, _ in r[:int(k)]} if o is None else {i for i, _ in r} - {o})) for t, r, o in zip(truth, res, own))
            return hit / max(1, sum(len(t) for t in truth)), 1000.0 * best / len(Qa)
        out = {}
        need = {"graph": target, "lists": target + (min(0.005, (1.0 - target) / 2) if queries is None else 0.0)}
        for name, key, ladder, kw in (("graph", "ef", (16, 24, 32, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600), dict(method="graph")),
                                      ("lists", "nprobe", (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512), dict(method="lists"))):
            pick = None
            if name == "graph" and queries is None and self.metric != "ip" and hasattr(self.lib, "kci_vec_graph_recall"):
                efs = [x for x in ladder if k1 <= x <= int(self.count) // 4]
                ea = (C.c_int * max(1, len(efs)))(*efs); ra = (C.c_float * max(1, len(efs)))()
                self.lib.kci_vec_graph_recall.restype = C.c_int
                self.lib.kci_vec_graph_recall.argtypes = [C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_int), C.c_int, C.c_int, C.POINTER(C.c_float)]
                if efs and self.lib.kci_vec_graph_recall(self._v, int(k), len(Qa), ea, len(efs), int(threads), ra) >= 8:
                    j = next((j for j in range(len(efs)) if ra[j] >= target), len(efs) - 1)
                    _, ms = measure(ef=efs[j], rerank_n=efs[j], **kw); out[name] = (efs[j], float(ra[j]), ms)
                    continue
            for x in ladder:
                r, ms = measure(**{key: x}, **(dict(kw, rerank_n=max(x, k1)) if name == "graph" else kw)); pick = (x, r, ms)
                if r >= need[name]: break
            out[name] = pick
        g, l = out["graph"], out["lists"]
        g_ok, l_ok = g[1] >= need["graph"], l[1] >= need["lists"]
        if g_ok != l_ok: use = "graph" if g_ok else "lists"
        elif g_ok: use = "graph" if g[2] <= l[2] else "lists"
        else: use = "graph" if g[1] >= l[1] else "lists"
        self._prefer = use; self._graph_rn = 0
        if use == "lists": self.nprobe = int(l[0])
        elif hasattr(self.lib, "kci_vec_set_default_ef") and self.lib.kci_vec_set_default_ef(self._v, int(g[0])) == 0:
            self._graph_rn = int(g[0])
        return {"method": use, "graph": {"ef": g[0], "recall": g[1], "ms": g[2]}, "lists": {"nprobe": l[0], "recall": l[1], "ms": l[2]}}

    def default_ef(self):
        """The ef search() uses when none is given: the calibrated width, or 100 when the index was never calibrated."""
        d = self.lib.kci_vec_default_ef(self._v) if hasattr(self.lib, "kci_vec_default_ef") else 0
        return d if d > 0 else 100

    def default_rerank(self):
        """The shortlist calibrate() found necessary when it is larger than `rerank` (many nearly equal vectors need
        the whole search width re-scored on the floats); 0 when `rerank` is enough. search() uses it when rerank_n is
        not given and ef is not below the calibrated width."""
        return self.lib.kci_vec_default_rerank(self._v) if hasattr(self.lib, "kci_vec_default_rerank") else 0

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
            if not self.filter_fill and hasattr(self.lib, "kci_vec_searcher_set_filter_fill"):
                self.lib.kci_vec_searcher_set_filter_fill(self._s, 0)
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

    def _filter(self, filt):
        """A callable, or ids that do not fit a bitmap: a Python callback, count None."""
        if filt is None:
            return None, None, None, None
        if not callable(filt) and hasattr(self.lib, "kci_vec_filter_id_bits"):
            b = _id_bits(filt)
            if b is not None:
                bits, nbits, m = b
                st = _IdBits(bits.ctypes.data, nbits)
                return C.cast(self.lib.kci_vec_filter_id_bits, _FILTER), C.cast(C.pointer(st), C.c_void_p), m, (bits, st)
        return _make_filter(filt), None, None, None

    def _query(self, q, k, ef, nprobe, rn, cb, method, ctx=None):
        s = self._searcher()
        use_lists = self._use_lists(method)
        if use_lists:
            if cb is not None:
                return self.lib.kci_vec_search_ivf_filtered(s, q, k, nprobe, rn, cb, ctx, self._ob, self._os)
            return self.lib.kci_vec_search_ivf(s, q, k, nprobe, rn, self._ob, self._os)
        if cb is not None:
            return self.lib.kci_vec_search_filtered(s, q, k, ef, rn, cb, ctx, self._ob, self._os)
        if rn and rn > 0:
            return self.lib.kci_vec_search_rerank(s, q, k, ef, rn, self._ob, self._os)
        return self.lib.kci_vec_search_r(s, q, k, ef, self._ob, self._os)

    def search(self, query, k=10, ef=None, nprobe=None, rerank_n=None, filter=None, method="auto"):
        """A selective filter needs a larger nprobe / rerank_n to find the best k matches.
        Filters that allow few ids: given a set / array, when (allowed ids) x dim <= exact_cells the query is compared
        with every allowed vector, whatever the size of the index, and the answer is exact. With any filter, a search
        that finds fewer than k answers is answered again by brute force, so no allowed id is left out when fewer than
        k exist (a callable is then called once per vector). Both need the floats (rerank > 0).
        Small indexes: while count x dim <= exact_cells and the floats are kept (rerank > 0), a search with method
        'auto' compares the query with every vector (with a filter: every vector whose id passes) and the answer is
        exact; ef / nprobe / rerank_n are not used then. method='graph' / 'lists' or exact_cells = 0 gives the index's
        own search.
        Larger indexes: when neither ef nor rerank_n is given, the width and the shortlist that build() measured for
        this data are used (see calibrate)."""
        k = int(k)
        if method == "auto" and self._small_exact() and (filter is None or hasattr(self.lib, "kci_vec_search_exact_filtered")):
            return self.search_exact(query, k=k, threads=1, filter=filter)
        cb, ctx, m, _keep = self._filter(filter)
        if m == 0:
            return []
        if (method == "auto" and m is not None and m * self.dim <= (self.exact_cells or 0) and self._exact_ok()
                and hasattr(self.lib, "kci_vec_search_exact_filtered")):
            _, q = _f32(np.reshape(query, -1))
            n = self.lib.kci_vec_search_exact_filtered(self._v, q, k, 1, cb, ctx, self._ob, self._os)
            return [(int(self._ob[i]), float(self._os[i])) for i in range(max(n, 0))]
        jd = (ef is None and cb is None and self.rerank and not self._use_lists(method)
              and hasattr(self.lib, "kci_vec_judged") and self.lib.kci_vec_judged(self._v) == 1)
        ef_given = ef; ef = ef or max(k, self.default_ef()); nprobe = int(nprobe or self.nprobe)
        rn = self.rerank if rerank_n is None else rerank_n
        if rerank_n is None and rn:
            wide = self.default_rerank()
            if wide > rn and ef >= wide:
                rn = wide
            g_rn = getattr(self, "_graph_rn", 0)
            if g_rn and ef_given is None and not self._use_lists(method) and self.default_ef() == g_rn:
                rn = max(k, g_rn)
        if self._use_lists(method):
            rn = max(int(rn or 0), 2 * k)
        _, q = _f32(np.reshape(query, -1))
        n = self._query(q, k, 0 if jd else ef, nprobe, int(rn or 0), cb, method, ctx)
        return [(int(self._ob[i]), float(self._os[i])) for i in range(max(n, 0))]

    def _small_exact(self):
        """True when search() should answer by brute force: built, full floats in the index, and count x dim within
        exact_cells."""
        return self._exact_ok() and self.lib.kci_vec_count(self._v) * self.dim <= self.exact_cells

    def _exact_ok(self):
        """True when brute force gives this index's scores: exact_cells not 0, built, full floats kept."""
        if not self.exact_cells or not hasattr(self.lib, "kci_vec_has_floats"):
            return False
        return self.lib.kci_vec_has_floats(self._v) == 1 and (self.lib.kci_vec_is_built(self._v) != 0 or self._lists)

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

    def search_exact(self, query, k=10, threads=0, filter=None):
        """Brute force (ground truth / tiny indexes). filter: as in search(); only the ids that pass are compared, so
        the answer is the exact top-k under the filter (fewer than k when fewer ids pass)."""
        _, q = _f32(np.reshape(query, -1))
        cb, ctx, m, _keep = self._filter(filter)
        if m == 0:
            return []
        if cb is not None:
            if not hasattr(self.lib, "kci_vec_search_exact_filtered"):
                raise RuntimeError("this libkci_vec has no kci_vec_search_exact_filtered (built before 2026-11-26); rebuild it")
            n = self.lib.kci_vec_search_exact_filtered(self._v, q, int(k), 1, cb, ctx, self._ob, self._os)
            return [(int(self._ob[i]), float(self._os[i])) for i in range(max(n, 0))]
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
