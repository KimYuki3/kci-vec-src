"""Sparse vectors (word weights: SPLADE, BM25 and the like) and hybrid search with the dense index.

A sparse vector is given as two arrays: the dimension numbers that are present and their weights. Similarity
is the inner product. Use the same id for the same document as in the dense KciVec.

Needs the library libkci_sparse (built from kci_sparse.c by build_kci_vec.sh, or by pip from the repo tree).
Set KCI_SPARSE_LIB to the .dll/.so/.dylib, or place it next to this file / the repo root. A KciSparse is not
for use from several threads at once (it holds one search context). New. No save / load yet."""
import ctypes as C
import os
import sys
import numpy as np

from .hybrid import rrf, mix, choose_weight

c_f32p = C.POINTER(C.c_float)
c_u32p = C.POINTER(C.c_uint32)
c_u64p = C.POINTER(C.c_uint64)
_FILTER = C.CFUNCTYPE(C.c_int, C.c_uint64, C.c_void_p)   # int filt(uint64 id, void *ctx)


def _make_filter(filt):
    """filt: None | callable(id) -> bool | a set / list / numpy array of allowed ids. Returns a C callback or None."""
    if filt is None:
        return None
    if callable(filt):
        return _FILTER(lambda i, _ctx: 1 if filt(int(i)) else 0)
    allowed = filt if isinstance(filt, (set, frozenset)) else set(int(x) for x in np.asarray(filt).ravel())
    return _FILTER(lambda i, _ctx: 1 if int(i) in allowed else 0)


class _BlockParams(C.Structure):
    _fields_ = [("struct_size", C.c_uint), ("list_len", C.c_int), ("block_size", C.c_int), ("summary_mass", C.c_float), ("threads", C.c_int),
                ("cluster_terms", C.c_int), ("centre_terms", C.c_int), ("head_mass", C.c_float), ("global_cut", C.c_int)]


def _find_lib():
    if os.environ.get("KCI_SPARSE_LIB"):
        return os.environ["KCI_SPARSE_LIB"]
    names = ["libkci_sparse.dll", "libkci_sparse.so", "libkci_sparse.dylib"] if sys.platform == "win32" \
        else ["libkci_sparse.so", "libkci_sparse.dylib", "libkci_sparse.dll"]
    here = os.path.dirname(os.path.abspath(__file__))
    for r in (here, os.path.join(here, "..", ".."), os.path.join(here, "..", "..", "..", ".."), os.getcwd()):
        for n in names:
            p = os.path.abspath(os.path.join(r, n))
            if os.path.exists(p):
                return p
    raise FileNotFoundError("libkci_sparse not found; set KCI_SPARSE_LIB or build it (see module docstring)")


_LIB = None
def _lib():
    global _LIB
    if _LIB is None:
        s = C.CDLL(_find_lib())
        s.kci_sparse_new.restype = C.c_void_p;           s.kci_sparse_new.argtypes = [C.c_uint32]
        s.kci_sparse_free.restype = None;                s.kci_sparse_free.argtypes = [C.c_void_p]
        s.kci_sparse_add.restype = C.c_int64;            s.kci_sparse_add.argtypes = [C.c_void_p, C.c_uint64, c_u32p, c_f32p, C.c_int]
        s.kci_sparse_insert.restype = C.c_int64;         s.kci_sparse_insert.argtypes = [C.c_void_p, C.c_uint64, c_u32p, c_f32p, C.c_int]
        s.kci_sparse_pending.restype = C.c_uint32;       s.kci_sparse_pending.argtypes = [C.c_void_p]
        s.kci_sparse_delete.restype = C.c_int;           s.kci_sparse_delete.argtypes = [C.c_void_p, C.c_uint64]
        s.kci_sparse_build.restype = C.c_int;            s.kci_sparse_build.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_double)]
        s.kci_sparse_searcher_new.restype = C.c_void_p;  s.kci_sparse_searcher_new.argtypes = [C.c_void_p]
        s.kci_sparse_searcher_free.restype = None;       s.kci_sparse_searcher_free.argtypes = [C.c_void_p]
        s.kci_sparse_search.restype = C.c_int
        s.kci_sparse_search.argtypes = [C.c_void_p, c_u32p, c_f32p, C.c_int, C.c_int, C.c_uint32, C.c_int, c_u64p, c_f32p]
        s.kci_sparse_build_blocks.restype = C.c_int
        s.kci_sparse_build_blocks.argtypes = [C.c_void_p, C.POINTER(_BlockParams), C.POINTER(C.c_double)]
        s.kci_sparse_build_knn.restype = C.c_int
        s.kci_sparse_build_knn.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_float, C.c_int, C.POINTER(C.c_double)]
        s.kci_sparse_drop_lists.restype = C.c_int;       s.kci_sparse_drop_lists.argtypes = [C.c_void_p]
        s.kci_sparse_has_blocks.restype = C.c_int;       s.kci_sparse_has_blocks.argtypes = [C.c_void_p]
        s.kci_sparse_drop_rows.restype = C.c_int;        s.kci_sparse_drop_rows.argtypes = [C.c_void_p]
        s.kci_sparse_has_rows.restype = C.c_int;         s.kci_sparse_has_rows.argtypes = [C.c_void_p]
        s.kci_sparse_knn.restype = C.c_int;              s.kci_sparse_knn.argtypes = [C.c_void_p]
        s.kci_sparse_search_blocks.restype = C.c_int
        s.kci_sparse_search_blocks.argtypes = [C.c_void_p, c_u32p, c_f32p, C.c_int, C.c_int, C.c_int, C.c_float, C.c_int, c_u64p, c_f32p]
        s.kci_sparse_searcher_set_filter.restype = None
        s.kci_sparse_searcher_set_filter.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
        s.kci_sparse_count.restype = C.c_uint32;         s.kci_sparse_count.argtypes = [C.c_void_p]
        s.kci_sparse_is_built.restype = C.c_int;         s.kci_sparse_is_built.argtypes = [C.c_void_p]
        s.kci_sparse_bytes.restype = C.c_int64;          s.kci_sparse_bytes.argtypes = [C.c_void_p]
        s.kci_sparse_dim.restype = C.c_uint32;           s.kci_sparse_dim.argtypes = [C.c_void_p]
        s.kci_sparse_save.restype = C.c_int;             s.kci_sparse_save.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_int]
        s.kci_sparse_load.restype = C.c_void_p;          s.kci_sparse_load.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
        _LIB = s
    return _LIB


def _pairs(indices, values, dim):
    """Check one sparse vector and return it as contiguous uint32 / float32 arrays."""
    idx = np.asarray(indices)
    val = np.ascontiguousarray(values, dtype=np.float32)
    if idx.ndim != 1 or val.ndim != 1 or len(idx) != len(val):
        raise ValueError("indices and values must be two 1-d arrays of the same length")
    if len(idx):
        if idx.dtype.kind not in "iu":
            raise ValueError("indices must be integers")
        if int(idx.min()) < 0 or int(idx.max()) >= dim:
            raise ValueError(f"a dimension number is outside 0..{dim - 1}")
        if not np.isfinite(val).all():
            raise ValueError("a weight is NaN or infinite")
    return np.ascontiguousarray(idx, dtype=np.uint32), val


class KciSparse:
    """One sparse index over a vocabulary of `dim` dimensions."""

    def __init__(self, dim):
        self.lib = _lib()
        if not (0 < int(dim) < 2 ** 32):
            raise ValueError("dim must be between 1 and 2^32 - 1")
        self.dim = int(dim)
        self._s = self.lib.kci_sparse_new(self.dim)
        if not self._s:
            raise MemoryError("kci_sparse_new failed")
        self._searcher = None

    def close(self):
        if getattr(self, "_searcher", None):
            self.lib.kci_sparse_searcher_free(self._searcher)
            self._searcher = None
        if getattr(self, "_s", None):
            self.lib.kci_sparse_free(self._s)
            self._s = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return int(self.lib.kci_sparse_count(self._s))

    @property
    def nbytes(self):
        return int(self.lib.kci_sparse_bytes(self._s))

    def add(self, id, indices, values):
        """Add one row. A dimension given twice is summed."""
        idx, val = _pairs(indices, values, self.dim)
        r = self.lib.kci_sparse_add(self._s, int(id), idx.ctypes.data_as(c_u32p), val.ctypes.data_as(c_f32p), len(idx))
        if r < 0:
            raise RuntimeError("kci_sparse_add failed")
        return int(r)

    def insert(self, id, indices, values):
        """Add one row to a built index without rebuilding; searches find it at once. Every search scores all
        rows inserted since the last build, so call build() again once `pending` is in the thousands."""
        idx, val = _pairs(indices, values, self.dim)
        r = self.lib.kci_sparse_insert(self._s, int(id), idx.ctypes.data_as(c_u32p), val.ctypes.data_as(c_f32p), len(idx))
        if r < 0:
            raise RuntimeError("kci_sparse_insert failed (call build() first)")
        return int(r)

    @property
    def pending(self):
        """Rows inserted since the last build."""
        return int(self.lib.kci_sparse_pending(self._s))

    def add_csr(self, ids, indptr, indices, data):
        """Add many rows held in CSR form (as in scipy.sparse.csr_matrix: m.indptr, m.indices, m.data).
        Row i is ids[i] with pairs indptr[i] .. indptr[i+1]. Nothing is added if the input is refused."""
        ids = np.asarray(ids)
        indptr = np.asarray(indptr, dtype=np.int64)
        if ids.ndim != 1 or len(indptr) != len(ids) + 1:
            raise ValueError("indptr must have len(ids) + 1 entries")
        idx, val = _pairs(indices, data, self.dim)
        if len(indptr) and (indptr[0] != 0 or indptr[-1] != len(idx) or (np.diff(indptr) < 0).any()):
            raise ValueError("indptr does not describe indices / data")
        pi, pv = idx.ctypes.data, val.ctypes.data
        add, s = self.lib.kci_sparse_add, self._s
        for i in range(len(ids)):
            a, n = int(indptr[i]), int(indptr[i + 1] - indptr[i])
            if add(s, int(ids[i]), C.cast(pi + 4 * a, c_u32p), C.cast(pv + 4 * a, c_f32p), n) < 0:
                raise RuntimeError(f"kci_sparse_add failed at row {i} (rows before it were added)")
        return len(self)

    def delete(self, id):
        """Remove the row(s) added under this id. No rebuild needed. Returns how many were removed."""
        r = self.lib.kci_sparse_delete(self._s, int(id))
        if r < 0:
            raise RuntimeError("kci_sparse_delete failed")
        return int(r)

    def save(self, path, key=None):
        """Write the index to `path` with everything built so far, so load() needs no build. key: bytes or None."""
        kp = (bytes(key), len(key)) if key else (None, 0)
        if self.lib.kci_sparse_save(self._s, os.fsencode(path), *kp) != 0:
            raise RuntimeError("kci_sparse_save failed")

    @classmethod
    def load(cls, path, key=None):
        """Read an index written by save(). Wrong key, damaged or altered file, a file saved without a key when
        one is given (or the other way round) -> RuntimeError."""
        lib = _lib()
        kp = (bytes(key), len(key)) if key else (None, 0)
        p = lib.kci_sparse_load(os.fsencode(path), *kp)
        if not p:
            raise RuntimeError("load failed (wrong key, damaged file, or file missing)")
        self = cls.__new__(cls)
        self.lib = lib
        self._s = p
        self._searcher = None
        self.dim = int(lib.kci_sparse_dim(p))
        return self

    def build(self, threads=0):
        """Build the search structure over everything added. Returns the seconds it took."""
        if self._searcher:
            self.lib.kci_sparse_searcher_free(self._searcher)
            self._searcher = None
        secs = C.c_double(0)
        if self.lib.kci_sparse_build(self._s, int(threads), C.byref(secs)) != 0:
            raise MemoryError("kci_sparse_build failed")
        return secs.value

    def _with_filter(self, filter, call):
        """Run one search with the filter set on the searcher, and take it off again whatever happens."""
        cb = _make_filter(filter)
        if cb is None:
            return call()
        self.lib.kci_sparse_searcher_set_filter(self._searcher, C.cast(cb, C.c_void_p), None)
        try:
            return call()
        finally:
            self.lib.kci_sparse_searcher_set_filter(self._searcher, None, None)

    def search(self, indices, values, k=10, budget=0, rerank=None, filter=None):
        """Top k by inner product, as [(id, score)], best first. budget 0 = exact; a positive budget caps the work
        per query (larger = closer to exact) and the best `rerank` rows found (default 3k) are scored exactly.
        filter: a set / list of allowed ids or a callable id -> bool; only those rows are returned."""
        if not self.lib.kci_sparse_is_built(self._s):
            raise RuntimeError("call build() first (rows were added since the last build)")
        if k <= 0 or budget < 0:
            raise ValueError("k must be positive and budget must not be negative")
        if self._searcher is None:
            self._searcher = self.lib.kci_sparse_searcher_new(self._s)
            if not self._searcher:
                raise MemoryError("kci_sparse_searcher_new failed")
        idx, val = _pairs(indices, values, self.dim)
        ids = np.empty(k, dtype=np.uint64)
        sc = np.empty(k, dtype=np.float32)
        n = self._with_filter(filter, lambda: self.lib.kci_sparse_search(
            self._searcher, idx.ctypes.data_as(c_u32p), val.ctypes.data_as(c_f32p), len(idx),
            int(k), int(min(budget, 2 ** 32 - 1)), int(rerank if rerank is not None else 3 * k),
            ids.ctypes.data_as(c_u64p), sc.ctypes.data_as(c_f32p)))
        if n < 0:
            raise RuntimeError("kci_sparse_search failed")
        return [(int(ids[i]), float(sc[i])) for i in range(n)]

    def build_blocks(self, list_len=0, block_size=0, summary_mass=0.0, threads=0, cluster_terms=0, centre_terms=0, head_mass=0.0, global_cut=False):
        """The structure search_blocks needs (call build() first). 0 = the default of each setting; see
        KciSparseBlockParams in kci_sparse.h. global_cut is experimental (very slow builds on large data): leave it
        False. Returns the seconds it took."""
        p = _BlockParams(C.sizeof(_BlockParams), int(list_len), int(block_size), float(summary_mass), int(threads), int(cluster_terms),
                         int(centre_terms), float(head_mass), 1 if global_cut else 0)
        secs = C.c_double(0)
        if self.lib.kci_sparse_build_blocks(self._s, C.byref(p), C.byref(secs)) != 0:
            raise RuntimeError("kci_sparse_build_blocks failed (not built, a bad setting, or out of memory)")
        return secs.value

    def build_neighbours(self, knn=20, query_terms=0, heap_factor=0.0, threads=0):
        """For every row, its knn most similar rows (call build_blocks() first). One block search per row, so minutes on
        millions of rows. Returns the seconds it took."""
        secs = C.c_double(0)
        if self.lib.kci_sparse_build_knn(self._s, int(knn), int(query_terms), float(heap_factor), int(threads), C.byref(secs)) != 0:
            raise RuntimeError("kci_sparse_build_knn failed (no blocks, a bad setting, or out of memory)")
        return secs.value

    def drop_lists(self):
        """Free the memory search() needs once the blocks are built; search() is then refused until the next build()."""
        if self.lib.kci_sparse_drop_lists(self._s) != 0:
            raise RuntimeError("kci_sparse_drop_lists failed (call build_blocks() first)")

    def drop_rows(self):
        """For a finished index: also free the full rows (and the lists). search_blocks() goes on working, with scores
        from the 16-bit copy of the rows (within about 1e-4 of the exact ones); add(), insert(), build(), build_blocks(),
        build_neighbours() and search() are refused from then on. delete(), save() and load() work."""
        if self.lib.kci_sparse_drop_rows(self._s) != 0:
            raise RuntimeError("kci_sparse_drop_rows failed (needs build_blocks(), no inserted rows waiting, dimension below 65536)")

    @property
    def has_rows(self):
        """False after drop_rows() (also for an index loaded from a file saved in that state)."""
        return bool(self.lib.kci_sparse_has_rows(self._s))

    def search_blocks(self, indices, values, k=10, query_terms=8, heap_factor=1.0, neighbours=0, filter=None):
        """(see the header for this function)"""
        if not self.lib.kci_sparse_is_built(self._s):
            raise RuntimeError("call build() first (rows were added since the last build)")
        if not self.lib.kci_sparse_has_blocks(self._s):
            raise RuntimeError("call build_blocks() first")
        if k <= 0 or neighbours < 0 or not (0 < heap_factor <= 4):
            raise ValueError("k must be positive, neighbours not negative, heap_factor in (0, 4]")
        if self._searcher is None:
            self._searcher = self.lib.kci_sparse_searcher_new(self._s)
            if not self._searcher:
                raise MemoryError("kci_sparse_searcher_new failed")
        idx, val = _pairs(indices, values, self.dim)
        ids = np.empty(k, dtype=np.uint64)
        sc = np.empty(k, dtype=np.float32)
        n = self._with_filter(filter, lambda: self.lib.kci_sparse_search_blocks(
            self._searcher, idx.ctypes.data_as(c_u32p), val.ctypes.data_as(c_f32p), len(idx),
            int(k), int(query_terms), float(heap_factor), int(neighbours),
            ids.ctypes.data_as(c_u64p), sc.ctypes.data_as(c_f32p)))
        if n < 0:
            raise RuntimeError("kci_sparse_search_blocks failed")
        return [(int(ids[i]), float(sc[i])) for i in range(n)]


def seal(data, key):
    """Encrypt bytes with a passphrase (kci_seal: PBKDF2-HMAC-SHA256, AES-256-CTR, AES-256-CMAC). -> bytes, 40 longer."""
    data, key = bytes(data), (key.encode("utf-8") if isinstance(key, str) else bytes(key))
    if not key: raise ValueError("seal: empty key")
    lib = _lib(); out = C.create_string_buffer(len(data) + 40)
    lib.kci_seal.restype = C.c_int64; lib.kci_seal.argtypes = [C.c_char_p, C.c_uint64, C.c_char_p, C.c_int, C.c_char_p]
    n = lib.kci_seal(data, len(data), key, len(key), out)
    if n != len(data) + 40: raise RuntimeError("kci_seal failed")
    return out.raw


def unseal(blob, key):
    """The bytes back from seal(). Wrong key, a changed or missing byte -> RuntimeError (nothing is returned)."""
    blob, key = bytes(blob), (key.encode("utf-8") if isinstance(key, str) else bytes(key))
    if not key: raise ValueError("unseal: empty key")
    lib = _lib(); out = C.create_string_buffer(max(1, len(blob) - 40))
    lib.kci_unseal.restype = C.c_int64; lib.kci_unseal.argtypes = [C.c_char_p, C.c_uint64, C.c_char_p, C.c_int, C.c_char_p]
    n = lib.kci_unseal(blob, len(blob), key, len(key), out)
    if n < 0: raise RuntimeError("unseal failed (wrong key, or the data was changed or cut off)")
    return out.raw[:n]


def hybrid_search_sparse(vec, sparse, query_vec, q_indices, q_values, k=10, depth=None, w_vec=1.0, w_sparse=1.0,
                         budget=0, rerank=None, filter=None, method="rrf", **search_kw):
    """Dense search (kci_vec.KciVec) + sparse search (KciSparse) merged by reciprocal rank (RRF). method="mix" merges
    by score instead (hybrid.mix); weights from hybrid.choose_weight. depth: candidates taken from each side
    (default 4k). Taking many more than that lets rows that only one side cares about rise; tune w_vec / w_sparse
    first. filter: a set / list of allowed ids or a callable id -> bool, applied to both sides. search_kw goes to
    vec.search. Returns [(id, fused_score, vec_rank, sparse_rank)], best first; a rank is None when that side did
    not return the id."""
    depth = depth or 4 * k
    vhits = vec.search(query_vec, k=depth, filter=filter, **search_kw)
    shits = sparse.search(q_indices, q_values, k=depth, budget=budget, rerank=rerank, filter=filter)
    if method not in ("rrf", "mix"):
        raise ValueError("method must be 'rrf' or 'mix'")
    fuse = mix if method == "mix" else rrf
    return [(id, s, r[0], r[1]) for id, s, r in fuse([vhits, shits], k=k, weights=[w_vec, w_sparse])]
