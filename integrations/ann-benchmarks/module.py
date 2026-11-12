"""ann-benchmarks module for kci_vec: the int8 HNSW graph (KciVec) and the partitioned 4-bit table-lookup index
(KciVecIvf, .

Drop this into ann-benchmarks as `ann_benchmarks/algorithms/kci_vec/` together with `Dockerfile` and
`config.yml`.

Metric: angular/cosine, and euclidean through kci_vec_new_l2 when the library has it (the evaluation wheel;
the source release is cosine only). KciVec: build params M / efConstruction, query param ef; rerank_n = ef
when `rerank` is set (int8 shortlist re-scored with the retained float vectors). KciVecIvf: build param nlist;
query params nprobe (lists scanned) and rerank (shortlist re-scored in float)."""
import ctypes as C
import os
import numpy as np
from ..base.module import BaseANN

def _lib_path():
    """KCI_VEC_LIB, else the library bundled in an installed `kci_vec` wheel the distributed, cosine-only build picks
    libkci_vec_avx512.so / libkci_vec_avx2.so by the CPU), else libkci_vec.so on the loader path."""
    if os.environ.get("KCI_VEC_LIB"):
        return os.environ["KCI_VEC_LIB"]
    try:
        from kci_vec import _find_lib
        return _find_lib()
    except Exception:
        return "libkci_vec.so"
_LIB = _lib_path()


def _load():
    lib = C.CDLL(_LIB)
    lib.kci_vec_new.restype = C.c_void_p
    lib.kci_vec_new.argtypes = [C.c_int, C.c_int, C.c_int]
    if hasattr(lib, "kci_vec_new_l2"):
        lib.kci_vec_new_l2.restype = C.c_void_p
        lib.kci_vec_new_l2.argtypes = [C.c_int, C.c_int, C.c_int, C.c_float]
    lib.kci_vec_free.argtypes = [C.c_void_p]
    lib.kci_vec_keep_float.argtypes = [C.c_void_p, C.c_int]
    lib.kci_vec_reserve.argtypes = [C.c_void_p, C.c_uint32]
    lib.kci_vec_add.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p]
    lib.kci_vec_add.restype = C.c_int64
    lib.kci_vec_build.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_double)]
    lib.kci_vec_build_ivf.argtypes = [C.c_void_p, C.c_int, C.c_int]
    lib.kci_vec_searcher_new.restype = C.c_void_p
    lib.kci_vec_searcher_new.argtypes = [C.c_void_p]
    lib.kci_vec_searcher_free.argtypes = [C.c_void_p]
    lib.kci_vec_search_r.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
    lib.kci_vec_search_rerank.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
    lib.kci_vec_search_ivf.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
    if hasattr(lib, "kci_vec_set_ivf_training"):
        lib.kci_vec_set_ivf_training.argtypes = [C.c_void_p, C.c_int, C.c_uint32]
    if hasattr(lib, "kci_vec_searcher_set_probe_gap"):
        lib.kci_vec_searcher_set_probe_gap.argtypes = [C.c_void_p, C.c_float, C.c_int]
    if hasattr(lib, "kci_vec_searcher_set_eps"):
        lib.kci_vec_searcher_set_eps.argtypes = [C.c_void_p, C.c_float]
    if hasattr(lib, "kci_vec_build_qg"):
        lib.kci_vec_build_qg.argtypes = [C.c_void_p, C.c_int, C.c_int]
        lib.kci_vec_searcher_set_qg.argtypes = [C.c_void_p, C.c_int]
    return lib


class _Base(BaseANN):
    def _check_metric(self, metric):
        if metric not in ("angular", "cosine", "euclidean"):
            raise NotImplementedError("kci_vec: angular / cosine / euclidean only")
        self._l2 = metric == "euclidean"

    def _new_index(self, X, M, efc):
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, dim = X.shape
        if self._l2:
            if not hasattr(self._lib, "kci_vec_new_l2"):
                raise RuntimeError("this libkci_vec has no kci_vec_new_l2")
            v = self._lib.kci_vec_new_l2(dim, M, efc, 0.0)
        else:
            v = self._lib.kci_vec_new(dim, M, efc)
        if not v:
            raise RuntimeError("kci_vec_new failed")
        self._lib.kci_vec_keep_float(v, 1)
        self._lib.kci_vec_reserve(v, n + 1)
        base, row = X.ctypes.data, dim * 4
        for i in range(n):
            self._lib.kci_vec_add(v, i, base + i * row)
        self._dim = dim
        self._out_id = (C.c_uint64 * 4096)()
        self._out_sc = (C.c_float * 4096)()
        return v

    def batch_query(self, X, n):
        # threads would share one searcher and the one output buffer at the same time. The queries one after another.
        self.res = [self.query(q, n) for q in X]

    def _ids(self, got):
        return np.frombuffer(self._out_id, dtype=np.uint64, count=got).astype(np.int64)

    def freeIndex(self):
        if getattr(self, "_s", None):
            self._lib.kci_vec_searcher_free(self._s)
            self._s = None
        if getattr(self, "_v", None):
            self._lib.kci_vec_free(self._v)
            self._v = None


class KciVec(_Base):
    """int8 HNSW graph, optional float re-rank of the ef-sized shortlist."""

    def __init__(self, metric, method_param):
        self._check_metric(metric)
        self._lib = _load()
        self._M = int(method_param["M"])
        self._efc = int(method_param["efConstruction"])
        self._rerank = int(method_param.get("rerank", 0))
        self._ef = self._efc
        self._eps = 0.0
        self._qg = int(method_param.get("qg", 0))
        self._v = self._s = None

    def fit(self, X):
        self._v = self._new_index(X, self._M, self._efc)
        secs = C.c_double(0.0)
        if self._lib.kci_vec_build(self._v, 0, C.byref(secs)) != 0:
            raise RuntimeError("kci_vec_build failed")
        if self._qg and self._lib.kci_vec_build_qg(self._v, self._qg, 0) != 0:
            raise RuntimeError("kci_vec_build_qg failed")
        self._s = self._lib.kci_vec_searcher_new(self._v)
        if self._qg and self._lib.kci_vec_searcher_set_qg(self._s, 1) != 0:
            raise RuntimeError("kci_vec_searcher_set_qg failed")

    def set_query_arguments(self, ef, eps=0.0):
        self._ef = int(ef); self._eps = float(eps)
        if self._s is not None and hasattr(self._lib, "kci_vec_searcher_set_eps"):
            self._lib.kci_vec_searcher_set_eps(self._s, self._eps)

    def query(self, q, n):
        q = np.ascontiguousarray(q, dtype=np.float32)
        ef = max(self._ef, n)
        if self._rerank:
            got = self._lib.kci_vec_search_rerank(self._s, q.ctypes.data, n, ef, ef, self._out_id, self._out_sc)
        else:
            got = self._lib.kci_vec_search_r(self._s, q.ctypes.data, n, ef, self._out_id, self._out_sc)
        return self._ids(got)

    def __str__(self):
        return f"KciVec(M={self._M},efC={self._efc},ef={self._ef},eps={self._eps},rerank={'ef' if self._rerank else 0},qg={self._qg})"


class KciVecIvf(_Base):
    """partitioned lists + 4-bit table lookup, float re-rank of the best `rerank`."""

    def __init__(self, metric, method_param):
        self._check_metric(metric)
        self._lib = _load()
        self._nlist = int(method_param["nlist"])
        self._train = int(method_param.get("train", 0))
        self._nprobe, self._rr = 1, 100
        self._v = self._s = None

    def fit(self, X):
        self._v = self._new_index(X, 16, 200)
        if self._train:
            if not hasattr(self._lib, "kci_vec_set_ivf_training"):
                raise RuntimeError("this libkci_vec has no kci_vec_set_ivf_training")
            self._lib.kci_vec_set_ivf_training(self._v, 30, 0xFFFFFFFF)
        if self._lib.kci_vec_build_ivf(self._v, self._nlist, 0) != 0:
            raise RuntimeError("kci_vec_build_ivf failed")
        self._s = self._lib.kci_vec_searcher_new(self._v)

    def set_query_arguments(self, nprobe, rerank, gap=0.0):
        self._nprobe, self._rr, self._gap = int(nprobe), int(rerank), float(gap)
        if self._gap > 0 and not hasattr(self._lib, "kci_vec_searcher_set_probe_gap"):
            raise RuntimeError("this libkci_vec has no kci_vec_searcher_set_probe_gap")
        if hasattr(self._lib, "kci_vec_searcher_set_probe_gap"):
            self._lib.kci_vec_searcher_set_probe_gap(self._s, self._gap, 8)

    def query(self, q, n):
        q = np.ascontiguousarray(q, dtype=np.float32)
        got = self._lib.kci_vec_search_ivf(self._s, q.ctypes.data, n, self._nprobe, max(self._rr, n),
                                           self._out_id, self._out_sc)
        return self._ids(got)

    def __str__(self):
        g = f",gap={self._gap}" if getattr(self, "_gap", 0) > 0 else ""
        t = ",train=all30" if getattr(self, "_train", 0) else ""
        return f"KciVecIvf(nlist={self._nlist}{t},nprobe={self._nprobe},rerank={self._rr}{g})"
