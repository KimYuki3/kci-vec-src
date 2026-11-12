/*
 * kci_vec.h — KCI-Vec, embeddable approximate nearest-neighbour search (public API)
 *
 * Encrypted, embeddable, dimension-agnostic int8 HNSW.
 *   - int8 vectors: 1/4 the memory of float32, NEON SDOT distance.
 *   - HNSW graph (Malkov & Yashunin): O(log N) approximate search.
 *   - Multithreaded build AND batch search (per-node spinlocks on build;
 *     lock-free graph reads on search).
 *   - Cosine similarity on unit-normalized vectors (int8 dot, fixed scale 127).
 *   - Optional encryption of the on-disk snapshot: AES-256-CTR with an
 *     AES-256-CMAC tag over the whole file (damage / tampering is refused).
 *
 * One embeddable library, no server. No domain assumptions: the dimension is a
 * runtime parameter and ids are opaque u64 keys chosen by the caller.
 *
 * Thread-safety: build and search take an explicit thread count and manage
 * their own workers. A KciVec is otherwise single-writer; do not call _add or
 * _build concurrently with searches on the same handle.
 */
#ifndef KCI_VEC_H
#define KCI_VEC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KciVec KciVec;

/* Create an empty index. `dim` is the vector dimension (must be a multiple of
 * 4). M / efc are HNSW build parameters (M=16, efc=200 are good defaults;
 * pass 0 to use them). Returns NULL on bad args / OOM. */
KciVec *kci_vec_new(int dim, int M, int efc);
/* L2 (Euclidean) distance: available in the evaluation wheel (kci_vec_new_l2 / kci_vec_is_l2); its source is not part
 * of this release yet. */
void    kci_vec_free(KciVec *v);

/* Reserve capacity for `n` vectors up front (avoids realloc churn). */
int kci_vec_reserve(KciVec *v, uint32_t n);

/* Append one vector. `id` is the caller's opaque key returned by searches.
 * _add takes float32 (length dim) and L2-normalizes + quantizes to int8.
 * _add_i8 takes a pre-quantized unit-norm int8 vector (length dim).
 * Returns the new vector count, or -1 on error. Adding invalidates any
 * previously built graph (call kci_vec_build again). */
int64_t kci_vec_add(KciVec *v, uint64_t id, const float *vec);
int64_t kci_vec_add_i8(KciVec *v, uint64_t id, const int8_t *vec);

/* The graph and traversal stay int8 (1/4 RAM); in production the float store is memory-mapped from disk so RAM
 * stays int8 — only a handful of float rows are read per query. */
int kci_vec_keep_float(KciVec *v, int on);

/* Move the retained float store to `path` and mmap it read-only, so the resident set stays int8 (codes + graph,
 * ~1/4 of float32) and only the float rows touched by re-rank fault in. Call once after kci_vec_build (with
 * kci_vec_keep_float on). The float store becomes read-only afterwards (no insert/compact). Returns 0 on success,
 * -1 otherwise. */
int kci_vec_float_mmap(KciVec *v, const char *path);

/* Pass 0 to leave a value unchanged. Stored vectors are unaffected. */
int kci_vec_set_params(KciVec *v, int M, int efc);

/* Build (or rebuild) the HNSW graph over all added vectors.
 * threads <= 0 selects a sensible default (cpu cores). Returns 0 on success,
 * and writes the wall-clock build seconds to *out_secs if non-NULL. */
int kci_vec_build(KciVec *v, int threads, double *out_secs);

/* Approximate top-k search via HNSW. Fills out_ids[0..ret) and (if non-NULL) out_scores[] with cosine similarity,
 * best first. Returns the number of results (<= k), or -1 if no graph is built. */
int kci_vec_search(KciVec *v, const float *query, int k, int ef,
                   uint64_t *out_ids, float *out_scores);

/* Reusable per-thread search context. Create one per worker thread; the
 * visited-set and heaps are allocated once and reused across queries. Multiple
 * searchers may run concurrently on one (already-built, read-only) KciVec. */
typedef struct KciVecSearcher KciVecSearcher;
KciVecSearcher *kci_vec_searcher_new(KciVec *v);
void            kci_vec_searcher_free(KciVecSearcher *s);
int             kci_vec_search_r(KciVecSearcher *s, const float *query,
                                 int k, int ef, uint64_t *out_ids, float *out_scores);

/* Two-stage search: int8 HNSW shortlist of `rerank_n` candidates → full- precision float re-rank → top-k. Requires
 * kci_vec_keep_float. Returns -2 if no float store is present. */
int             kci_vec_search_rerank(KciVecSearcher *s, const float *query,
                                      int k, int ef, int rerank_n,
                                      uint64_t *out_ids, float *out_scores);

/* Exact brute-force top-k (ground truth / small-N fallback), multithreaded.
 * Does not require a built graph. threads <= 0 = default. */
int kci_vec_search_exact(KciVec *v, const float *query, int k, int threads,
                         uint64_t *out_ids, float *out_scores);

/* ── Incremental update (single-writer: do NOT call concurrently with searches;
 * pause serving, batch updates, resume). All require a built graph; insert also
 * requires kci_vec_keep_float. Bulk loads should still use add() + build(). */
/* Insert one vector into the live graph. Returns the new count, -1 on error,
 * -2 if floats were not retained (kci_vec_keep_float). */
int64_t kci_vec_insert(KciVec *v, uint64_t id, const float *vec);
/* Tombstone the vector with caller id (excluded from results, kept as a routing
 * hop). Returns 0, or -1 if id not found. Update = delete then insert. */
int kci_vec_delete(KciVec *v, uint64_t id);
/* Forget id for good: every row of it (also tombstoned earlier versions) is zeroed and removed, its list entries
 * cleared, and the centroid of every list it was in computed again from the remaining live members (their codes
 * encoded again): nothing of the vector stays in RAM or in a later save, except its share in the global 4-bit
 * codebooks (trained on a sample of up to 16,384 residuals). -2 (rebuild instead) for a graph, mmap'd floats,
 * low-rank lists, or an index with neither floats nor a float source. Rows removed (0 = absent), -1 on error.
 * Recreate searchers afterwards. */
int kci_vec_forget(KciVec *v, uint64_t id);
/* Rebuild over only the live vectors, reclaiming tombstoned slots. Returns the new
 * live count. Requires retained floats; a no-op (returns count) if nothing dead. */
uint32_t kci_vec_compact(KciVec *v);

/* ── Filtered search (metadata / ACL at the engine layer) ──────────────────
 * Predicate over the caller's id: return non-zero to keep a vector eligible. The
 * graph is traversed through non-matching nodes (connectivity preserved) but only
 * matching nodes enter the result set, so selective filters still find their
 * nearest matches — unlike post-filtering a plain top-k. Bounded by a visit budget
 * (~64*ef) so a very selective filter degrades to best-effort rather than scanning
 * the whole graph; for tiny matching sets prefer brute force + your own filter.
 * rerank_n>0 adds the float re-rank (needs kci_vec_keep_float). filt==NULL behaves
 * like kci_vec_search_rerank. */
typedef int (*KciVecFilter)(uint64_t id, void *ctx);
/* Float source. A list index (no graph) can drop its resident floats and fetch a candidate's float from the caller
 * during the re-rank instead: fn(id, out, ctx) fills out[dim] (any length, it is normalized) and returns 0, or
 * nonzero if the id is gone (it then ranks last). An application that keeps every vector in its own store (for
 * example an encrypted key-value store) saves dim*4 bytes per vector of RAM, and no plaintext copy of the vectors
 * stays resident. Costs rerank_n calls per query. After it: kci_vec_insert works from the given vector; exact
 * search ranks by the int8 rows; kci_vec_compact does nothing (rebuild from your store instead). ctx_free(ctx), if
 * given, runs when the index is freed or the source replaced. Returns 0, -1 (no lists, a graph, or mmap'd floats). */
typedef int (*KciVecFloatFn)(uint64_t id, float *out, void *ctx);
int kci_vec_set_float_source(KciVec *v, KciVecFloatFn fn, void *ctx, void (*ctx_free)(void *));
int kci_vec_search_filtered(KciVecSearcher *s, const float *query, int k, int ef,
                            int rerank_n, KciVecFilter filt, void *ctx,
                            uint64_t *out_ids, float *out_scores);

/* Needs kci_vec_keep_float(v, 1). Build it after the vectors are added — and after kci_vec_build if you also want
 * the graph, because that build renumbers the vectors and drops this index. nlist = number of lists (0:
 * 2*sqrt(count); 2,000 suits ~1 M vectors). Inserts (kci_vec_insert) and deletes work on it; kci_vec_compact
 * rebuilds it; kci_vec_save / _load keep it. Returns 0, or -1 (no floats, out of memory). */
int kci_vec_build_ivf(KciVec *v, int nlist, int threads);
/* Same, with a second list per vector (SOAR, as in ScaNN; each vector is also coded in the list of a second
 * centroid chosen so that its error is unlike the first one's (weight soar_lambda; 1.5 is ScaNN's value; 0 = one
 * list only, which is what kci_vec_build_ivf does). Saved with the index. */
int kci_vec_build_ivf_ex(KciVec *v, int nlist, int threads, float soar_lambda);
/* All build options . code_dims > 0: the 4-bit codes describe the residual projected onto its top code_dims
 * principal directions (PCA of a sample of residuals) instead of all dims, so the scan reads code_dims/2 bytes per
 * vector instead of dim/2; the re-rank (float, or int8 with kci_vec_searcher_set_rerank_int8) restores the
 * precision. Even, at least 8, below dim; 0 = all dims. Zero the struct for the defaults. */
/* lowrank > 0 low-rank scoring instead of the 4-bit tables (the LoRANN idea). Each list keeps the top-lowrank
 * principal directions of its own residuals (int8) and every vector lowrank int8 coefficients on them; a query is
 * projected onto a probed list's directions once, then each vector costs a lowrank-term int8 dot — lowrank bytes
 * per vector instead of dim/2. Multiples of 16, at most 256, below dim; 64 suits 384-768 dims. Use a larger
 * rerank_n (400 at k=100) and the int8 or float re-rank. Overrides code_dims. */
typedef struct { int nlist; int threads; float soar; int code_dims; int lowrank; } KciVecIvfParams;
int kci_vec_build_ivf_p(KciVec *v, const KciVecIvfParams *params);
/* Its weight (dims-1) t^2 / (1-t^2) is off below 1: at 0.2 that is below 26 dims. Kept with the index (save /
 * load, compact). Returns 0, or -1 on bad args. */
int kci_vec_set_ivf_codes(KciVec *v, int dims_per_code, float avq_threshold);
/* How the next kci_vec_build_ivf* trains the centroids: k-means iterations (0 = 10) over a sample of at most
 * `sample` vectors (0 = 250,000; UINT32_MAX = all). The codes and the search are unchanged. Returns 0, or -1
 * (iterations outside 0..1000). */
int kci_vec_set_ivf_training(KciVec *v, int iterations, uint32_t sample);
/* kci_vec_search_ivf re-ranks the shortlist with the int8 rows instead of the floats 1/4 of the bytes to read. Off
 * by default. Scores stay cosine. */
int kci_vec_searcher_set_rerank_int8(KciVecSearcher *s, int on);
/* Probe gap for kci_vec_search_ivf of the best nprobe lists, scan only those whose centroid score (cosine) is
 * within `gap` of the nearest one's, but at least `min_probe`. nprobe stays the upper bound. The best gap depends
 * on the data: try 0.15-0.35. Returns 0, or -1 (no searcher, gap NaN). */
int kci_vec_searcher_set_probe_gap(KciVecSearcher *s, float gap, int min_probe);
/* Exploration radius for the graph search , the stop rule NGT-qg uses. eps > 0: the result set is kept at `ef`
 * entries (pass a small ef, k..3k) and the search goes on while the next candidate's cosine distance is within
 * (1+eps) x the ef-th best; eps <= 0 (the default) is plain HNSW (stop when the next candidate is worse than the
 * ef-th best). Sweep per data set. Applies to kci_vec_search_r / _rerank / _filtered. Returns 0, or -1 (no
 * searcher, eps NaN). */
int kci_vec_searcher_set_eps(KciVecSearcher *s, float eps);
/* Quantized graph layer 0 of the graph laid out the way NGT-qg / qsgngt search it. Each node keeps its neighbours'
 * 4-bit codes beside it, so one hop reads one contiguous block and scores every neighbour with SIMD table lookups
 * (the lists' kernel) instead of reading M0 int8 rows at random. The upper layers and the exact re-score of the
 * result set still use the rows. The anisotropic threshold of kci_vec_set_ivf_codes applies if set. Not saved;
 * dropped by kci_vec_insert, kci_vec_compact and kci_vec_build (build it again after). Returns 0, or -1 (no graph,
 * bad args, out of memory). */
int kci_vec_build_qg(KciVec *v, int dims_per_code, int threads);
/* Bytes the quantized graph takes (0 = none). */
int64_t kci_vec_qg_bytes(const KciVec *v);
/* Search over the quantized graph (on = 1): kci_vec_search_r re-scores the result set with the int8 rows,
 * kci_vec_search_rerank with the floats. Use a small ef (k..3k) with kci_vec_searcher_set_eps, since the ef-th best is
 * an estimate now. kci_vec_search_filtered and the plain kci_vec_search do not use it yet. Returns 0, -1 (no
 * searcher), -2 (the index has no quantized graph: kci_vec_build_qg first). */
int kci_vec_searcher_set_qg(KciVecSearcher *s, int on);
/* Search the lists: score the centroids, scan the best `nprobe` lists with the 4-bit tables, keep the best
 * `rerank_n` and re-score them with the float vectors. Scores are cosine. Returns the hit count, or -1 if the
 * index has no lists. */
int kci_vec_search_ivf(KciVecSearcher *s, const float *query, int k, int nprobe, int rerank_n,
                       uint64_t *out_ids, float *out_scores);
/* The same with a filter over the caller's ids (metadata / ACL; only vectors with filt(id, ctx) != 0 enter the
 * shortlist and the answer. The predicate is asked only for candidates that beat the shortlist's worst (a few
 * hundred per query), so a Python callback is fine. A selective filter needs a larger nprobe / rerank_n to find k
 * matches; for very small matching sets use kci_vec_search_exact + your own filter. */
int kci_vec_search_ivf_filtered(KciVecSearcher *s, const float *query, int k, int nprobe, int rerank_n,
                                KciVecFilter filt, void *ctx, uint64_t *out_ids, float *out_scores);

/* On-disk snapshot. If key != NULL (keylen bytes), the serialized index is encrypted at rest (format KCIVEC05:
 * AES-256-CTR, key from PBKDF2-HMAC-SHA256 with a per-save salt, and an AES-256-CMAC tag over the whole file);
 * pass the same key to _load. With key=NULL the snapshot is plaintext (KCIVEC04). The graph is saved too, so _load
 * needs no rebuild. Older KCIVEC04 files encrypted with ChaCha20 (no tag) still load. */
int     kci_vec_save(KciVec *v, const char *path, const uint8_t *key, int keylen);
KciVec *kci_vec_load(const char *path, const uint8_t *key, int keylen);
/* kci_vec_save with options. KCI_VEC_SAVE_NO_FLOAT leaves the retained floats out of the file (for a caller that
 * stores the vectors elsewhere, e.g. an application's own store): a smaller file, less to decrypt at load. */
#define KCI_VEC_SAVE_NO_FLOAT 1
/* Such a file loads only with kci_vec_load_ex(..., KCI_VEC_LOAD_RAW_KEY), and a passphrase file only without it.
 * Use it only with a key of full strength (32 random bytes). */
#define KCI_VEC_SAVE_RAW_KEY 2
#define KCI_VEC_LOAD_RAW_KEY 2
int     kci_vec_save_ex(KciVec *v, const char *path, const uint8_t *key, int keylen, int flags);
KciVec *kci_vec_load_ex(const char *path, const uint8_t *key, int keylen, int flags);
/* Set the retained float of the vector with this id (after a KCI_VEC_SAVE_NO_FLOAT load). 0, or -1 (unknown id, no
 * float store). */
int     kci_vec_set_float(KciVec *v, uint64_t id, const float *vec);

/* Plain, not encrypted: keep the file where the tables are. On an attached index the searches work
 * (kci_vec_search_ivf*, kci_vec_search_exact, kci_vec_get_float); anything that writes (add, insert, delete,
 * forget, compact, build*, keep_float, set_float, float_mmap, protect, reserve) returns -1 (compact: the count).
 * The graph is not in the image, so kci_vec_search / _search_r return -1. kci_vec_save works (an encrypted copy of
 * an attached index). kci_vec_free unmaps the file. kci_vec_attach_mem does the same over memory the caller mapped
 * (a shared segment); it must outlive the index. image_save: 0 / -1 (write error, floats mmap'd). attach: NULL (no
 * file, not an image, truncated or inconsistent, an index kind this release does not have). The file is for the
 * machine that wrote it (byte order, alignment). */
int     kci_vec_image_save(const KciVec *v, const char *path);
KciVec *kci_vec_attach(const char *path);
KciVec *kci_vec_attach_mem(const void *image, size_t len);
/* kci_vec_thaw: a writable copy of an attached image (rows, floats, tombstones, lists, same centroids and
 * codebooks), independent of the mapping. Inserts and deletes work on it, and kci_vec_image_save writes the next
 * image: a refresh that applies only the changes (assign + encode each new vector against the existing centroids;
 * no k-means, no pass over the old rows). NULL: not attached, out of memory. The centroids are not retrained:
 * after many refreshes that shift the data, build again. kci_vec_squeeze: drop the tombstoned rows of a lists-only
 * index without retraining: the rows move down, every list is laid out again with its live entries, the id hash is
 * rebuilt. Searchers must be recreated. Returns the rows removed (0 = none dead), -1 on error, -2 for a graph
 * index or one without lists (use kci_vec_compact). */
KciVec *kci_vec_thaw(const KciVec *attached);
int64_t kci_vec_squeeze(KciVec *v);

/* Buffers grown later stay protected. Call after build or load. Returns the bits that held for every buffer: 1 =
 * out of core dumps, 2 = locked in RAM (0 = neither, e.g. on Windows without a raised working set), -1 error. */
int     kci_vec_protect(KciVec *v);

/* How much of the index is plaintext in RAM, in bytes, by kind: out[0] vector codes (int8 rows, 4-bit / low-rank
 * list codes) out[1] floats kept in RAM (0 if mmap'd or absent) out[2] derived from the vectors (centroids,
 * projection, codebooks, per-list bases) out[3] graph (neighbour lists, levels, tombstones) out[4] caller ids (and
 * the id lookup table) out[5] total Allocated sizes (capacity, not only live rows): what is in memory. Searchers'
 * scratch (the last query) is not counted. 0 / -1. */
int     kci_vec_plain_bytes(const KciVec *v, uint64_t out[6]);

/* Introspection. */
/* Reconstruct a float (unit-norm) view of stored vector i into out[dim]:
 * the exact float if keep_float was on, else the int8 row dequantized. */
int      kci_vec_get_float(const KciVec *v, uint32_t i, float *out);

uint32_t kci_vec_count(const KciVec *v);
int      kci_vec_dim(const KciVec *v);
int      kci_vec_is_built(const KciVec *v);

#ifdef __cplusplus
}
#endif

#endif /* KCI_VEC_H */
