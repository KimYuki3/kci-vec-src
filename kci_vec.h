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
/* An index by L2 (Euclidean) distance instead of cosine. Vectors keep their length (not normalized); every search
 * returns the nearest by |q - x| and out_scores holds the squared distance |q - x|^2, smallest first. Inside it is
 * the cosine index of dim+2 dims (each vector gets two more coordinates that turn the L2 order into a cosine order
 * exactly), so the graph, the lists, the re-rank, save and load all work as for cosine. max_norm: the largest |x|
 * that will ever be added. That first build also fixes the range of lengths: afterwards, adding or inserting a
 * vector much longer or shorter than those (outside twice their spread of |x|^2) returns -1; build a new index for
 * it. A max_norm far above the real lengths wastes precision. Not for L2: kci_vec_add_i8 and
 * kci_vec_set_float_source (both return -1). kci_vec_dim returns dim, and kci_vec_get_float the vector as given.
 * Returns NULL on bad args (dim <= 0, max_norm < 0 or not finite) / OOM. */
KciVec *kci_vec_new_l2(int dim, int M, int efc, float max_norm);
int     kci_vec_is_l2(const KciVec *v);   /* 1 = made by kci_vec_new_l2 (or loaded from such a file) */
/* An index by inner product (maximum inner product search) instead of cosine. Vectors keep their length; every
 * search returns the vectors with the largest q.x and out_scores holds q.x, largest first. Inside it is the cosine
 * index of dim+1 dims: each vector is scaled to length <= 1 and gets one more coordinate that brings its length to
 * 1, the query gets a 0 there (the published reduction of Bachrach et al. 2014). It changes no ranking (every
 * score moves by the same q.mean, which is added back) and lets the int8 rows and the codes describe what differs
 * between the vectors instead of what they share. Measure both with your own queries. 0 / -1 (not an inner-product
 * index, or already built). max_norm: the largest |x| that will ever be added. Afterwards a longer vector
 * (centred: one farther from the mean than the farthest at that build; with max_norm: any vector longer than
 * max_norm) returns -1 from add / insert. Not for inner product: kci_vec_add_i8 and kci_vec_set_float_source (both
 * return -1). kci_vec_dim returns dim, kci_vec_get_float the vector as given. Not in a KCI_VEC_NO_L2 build.
 * Returns NULL on bad args (dim <= 0, max_norm < 0 or not finite) / OOM. */
KciVec *kci_vec_new_ip(int dim, int M, int efc, float max_norm);
int     kci_vec_set_ip_centre(KciVec *v, int on);
int     kci_vec_metric(const KciVec *v);  /* 0 = cosine, 1 = L2, 2 = inner product; -1 = NULL */
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
/* The graph is of the same kind (same file format, same searches, rows can be added and deleted after); efc is
 * used for the upper layers only. Below 20,000 rows it is the same as kci_vec_build. Returns as kci_vec_build. */
int kci_vec_build_fast(KciVec *v, int threads, double *out_secs);

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
/* The same with the neighbours' codes over the top code_dims principal directions only (even, 8..dim-1; 0 = all
 * dimensions, as kci_vec_build_qg). A smaller block per node: less memory and a cheaper hop, coarser estimates.
 * Returns 0, or -1 (as kci_vec_build_qg, or a bad code_dims). */
int kci_vec_build_qg_ex(KciVec *v, int dims_per_code, int code_dims, int threads);
/* As kci_vec_build_qg_ex, with the neighbours' codes over the first gate_dims of the code_dims directions only
 * (even, 8..code_dims; 0 = code_dims, the same as kci_vec_build_qg_ex). The short rows of
 * kci_vec_searcher_set_qg_exact(s, 2) keep all code_dims coordinates, so the block per node shrinks while the
 * ranking inside the search stays as fine. Use it for the memory, with set_qg_exact(s, 2). Returns 0, or -1 (as
 * kci_vec_build_qg_ex, or a bad gate_dims). */
int kci_vec_build_qg_gate(KciVec *v, int dims_per_code, int code_dims, int gate_dims, int threads);
/* A quantized graph with residual codes: the memory-first form. Each neighbour is kept as its difference from the
 * node on the first gate_dims of the code_dims principal directions (2 bits a dimension; gate_dims 0 = code_dims),
 * with one scale per neighbour, so a neighbour takes gate_dims / 4 + 4 bytes. The search always scores with the
 * short rows (as kci_vec_searcher_set_qg_exact(s, 2)); use it with kci_vec_search_rerank. code_dims as
 * kci_vec_build_qg_ex (required, > 0). Saved and loaded like the other forms; a library older than this one
 * refuses such a file. Returns 0, or -1 (as kci_vec_build_qg_ex, code_dims 0, or no memory for the short rows). */
int kci_vec_build_qg_res(KciVec *v, int code_dims, int gate_dims, int threads);
/* Bytes the quantized graph takes (0 = none). */
int64_t kci_vec_qg_bytes(const KciVec *v);
/* Search over the quantized graph (on = 1): kci_vec_search_r re-scores the result set with the int8 rows,
 * kci_vec_search_rerank with the floats. Use a small ef (k..3k) with kci_vec_searcher_set_eps, since the ef-th
 * best is an estimate now. kci_vec_search_filtered uses it too rows that fail the filter are walked through and
 * left out of the result, as on the plain graph); the plain kci_vec_search does not. Returns 0, -1 (no searcher),
 * -2 (the index has no quantized graph: kci_vec_build_qg first). on = -1: automatic, and the default of a new
 * searcher: the quantized graph is used whenever the index has one (built, loaded from a file or attached from an
 * image), with the short rows if it has them (as kci_vec_searcher_set_qg_exact(s, 2); that switch takes -1 for
 * automatic too). */
int kci_vec_searcher_set_qg(KciVecSearcher *s, int on);
/* Worth it where the estimates are coarse (high dimensions, 2 dims a code); off by default. Returns 0, or -1 (no
 * searcher). on = 2: with a graph built by kci_vec_build_qg_ex (code_dims > 0), score them with short int8 rows
 * instead (the code_dims coordinates, code_dims bytes a row; the graph keeps them, count x code_dims bytes).
 * Without code_dims, 2 is the same as 1. */
int kci_vec_searcher_set_qg_exact(KciVecSearcher *s, int on);
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
 * needs no rebuild. Older KCIVEC04 files encrypted with ChaCha20 (no tag) still load. The quantized graph
 * (kci_vec_build_qg*, with its short rows) and the halves (kci_vec_half_floats) are saved too, inside the
 * encrypted and authenticated part: the loaded index answers kci_vec_search_rerank through the quantized graph
 * without building it again (kci_vec_searcher_set_qg as usual). */
int     kci_vec_save(KciVec *v, const char *path, const uint8_t *key, int keylen);
KciVec *kci_vec_load(const char *path, const uint8_t *key, int keylen);
/* kci_vec_save with options. KCI_VEC_SAVE_NO_FLOAT leaves the retained floats out of the file (for a caller that
 * stores the vectors elsewhere, e.g. an application's own store): a smaller file, less to decrypt at load. */
#define KCI_VEC_SAVE_NO_FLOAT 1
/* Such a file loads only with kci_vec_load_ex(..., KCI_VEC_LOAD_RAW_KEY), and a passphrase file only without it.
 * Use it only with a key of full strength (32 random bytes). */
#define KCI_VEC_SAVE_RAW_KEY 2
#define KCI_VEC_LOAD_RAW_KEY 2
/* KCI_VEC_LOAD_AUTH_ONLY: kci_vec_load_ex accepts only a file that is encrypted with the authentication tag
 * (AES-256-CTR + CMAC: saved with a key by this library since . A plaintext file and an old ChaCha20 file (no tag)
 * return NULL, whatever key is given. For indexes that arrive from outside: a file without a tag can be altered
 * without the load noticing. Combine with KCI_VEC_LOAD_RAW_KEY by |. */
#define KCI_VEC_LOAD_AUTH_ONLY 4
int     kci_vec_save_ex(KciVec *v, const char *path, const uint8_t *key, int keylen, int flags);
KciVec *kci_vec_load_ex(const char *path, const uint8_t *key, int keylen, int flags);
/* Set the retained float of the vector with this id (after a KCI_VEC_SAVE_NO_FLOAT load). 0, or -1 (unknown id, no
 * float store). */
int     kci_vec_set_float(KciVec *v, uint64_t id, const float *vec);

/* Lists learned from a query sample, for data whose queries come from another distribution than the vectors (text
 * queries over images, the queries and keys of an attention head) and whose answers concentrate on few vectors.
 * kci_vec_learn: `queries` = nq sample queries (dim floats each, the caller's dim), `answers` = their true or
 * approximate nearest ids, per_query each (an id not in the index is skipped). Needs a built index (graph or
 * lists). 0 / -1. The learned lists name rows: kci_vec_build and kci_vec_compact drop them (learn again), inserts
 * keep them (new vectors are just not in them), they are not saved. kci_vec_search_learned: the `nclusters`
 * (1..64) nearest clusters, the first `per_list` vectors of each, then ef = 0 : those are scored exactly on the
 * kept floats and the best k returned. Fast; a query unlike the sample gets a poor answer, there is no fallback;
 * ef > 0 : they are added as starting points of the graph search (ef as in kci_vec_search_rerank, re-rank of the
 * ef best), so the graph completes what the lists miss. Needs the graph (kci_vec_build). ef < 0 : needs
 * kci_vec_learn_pack. The lists' own int8 copies are scanned in order and only the -ef best (150 - 500) are scored
 * exactly. The fastest form when a query reads thousands of list entries; like ef = 0 it has no fallback for a
 * query unlike the sample. kci_vec_learn_pack (after kci_vec_learn): gives every list an int8 copy of its vectors
 * (dim bytes per list entry), centred on the mean of all vectors, which changes no ranking and is what lets one
 * byte per dimension tell them apart. 0 / -1. Dropped with the lists. Needs kci_vec_keep_float. Returns the count,
 * -1 on error, -2 if nothing is learned (or it was dropped; ef < 0: not packed). Same scores as the other searches
 * (cosine, |q-x|^2 for L2, q.x for inner product). */
int kci_vec_learn(KciVec *v, const float *queries, uint32_t nq, const uint64_t *answers, int per_query,
                  int nclusters, int list_cap, int threads, float *out_top5);
int kci_vec_has_learned(const KciVec *v);   /* 1 = kci_vec_search_learned can be used */
int kci_vec_learn_pack(KciVec *v);
/* For ef > 0 on packed lists: start the graph from the `best` rows of the int8 scan of the list prefixes instead
 * of from all of them (0 = all, the default). Fewer, better starting points. 0 / -1. */
int kci_vec_searcher_set_learned_seeds(KciVecSearcher *s, int best);
/* Choose the lists in two stages: kci_vec_learn also groups its clusters into sqrt(nclusters) pages; with pages >
 * 0 a search compares the query with the pages, then only with the clusters of the nearest `pages` of them (0 =
 * compare with every cluster, the default). Cheaper when few lists are read; with too few pages a list is missed.
 * No effect below 64 clusters. 0 / -1. */
int kci_vec_searcher_set_learned_pages(KciVecSearcher *s, int pages);
/* Also read the first `rows` of the hot rows: kci_vec_learn keeps, besides the lists, the rows most often an
 * answer over the whole sample (at most a quarter of the index, most frequent first; packed with the lists). For
 * the last answers that the nearest lists miss, on data whose answers are concentrated. 0 = off, the default. 0 /
 * -1. */
int kci_vec_searcher_set_learned_hot(KciVecSearcher *s, int rows);
int kci_vec_search_learned(KciVecSearcher *s, const float *query, int k, int nclusters, int per_list, int ef,
                           uint64_t *out_ids, float *out_scores);
/* For data whose answers are not concentrated (kci_vec_learn's concentration is low): the sample queries
 * themselves are the index. kci_vec_vote_set keeps the answers of the nq sample queries (ids of this index,
 * per_query each) and an int8 copy of every vector centred on the mean (dim bytes per vector). 0 / -1. Needs
 * kci_vec_keep_float and a built index; dropped by kci_vec_build / kci_vec_compact (set again), not saved.
 * kci_vec_search_voted: `voters` are numbers (0 .. nq-1) of the sample queries nearest to the query, found by the
 * caller (for instance with a second KciVec holding the sample queries). The union of their answers is scanned on
 * the int8 copies and the `rerank` best (at least k, at most 4096) are scored exactly. Returns the count, -1 on
 * error, -2 if nothing is set. A query unlike the sample gets a poor answer; there is no fallback. Related work:
 * MLANN. */
int kci_vec_vote_set(KciVec *v, const uint64_t *answers, uint32_t nq, int per_query);
int kci_vec_search_voted(KciVecSearcher *s, const float *query, const uint32_t *voters, int nvoters, int k, int rerank,
                         uint64_t *out_ids, float *out_scores);
/* Optional short codes for the scan of the union, for high dimensions. kci_vec_vote_short (after
 * kci_vec_vote_set): `rows` holds one short vector (sdim floats, at most 4096) for each of the n = count vectors,
 * in the order of `ids`; the caller chooses the map (short query . short row should approximate the score, up to a
 * constant per query). Kept as int8 centred on the mean, sdim bytes per vector. 0 / -1.
 * kci_vec_search_voted_short: the union is ranked by qshort . short row, the `keep` best (at least `rerank`) go
 * through the full int8 rows, and the `rerank` best of those are scored exactly. -2 without short codes. */
int kci_vec_vote_short(KciVec *v, const uint64_t *ids, const float *rows, uint32_t n, int sdim);
int kci_vec_search_voted_short(KciVecSearcher *s, const float *query, const float *qshort, int keep, const uint32_t *voters, int nvoters,
                               int k, int rerank, uint64_t *out_ids, float *out_scores);

/* Plain, not encrypted: keep the file where the tables are. On an attached index the searches work
 * (kci_vec_search_ivf*, kci_vec_search_exact, kci_vec_get_float); anything that writes (add, insert, delete,
 * forget, compact, build*, keep_float, set_float, float_mmap, protect, reserve) returns -1 (compact: the count).
 * An index with a graph (kci_vec_build) takes it into the image, with the quantized graph (kci_vec_build_qg*, its
 * short rows) and the halves (kci_vec_half_floats) if they are there; the attached index then answers
 * kci_vec_search / _search_r / _search_rerank (with kci_vec_searcher_set_qg) as the index it was saved from. An
 * image written by this version is read by an earlier one as an image without a graph. The file is for the machine
 * that wrote it (byte order, alignment). */
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

/* One way: the floats are wiped and freed. Returns 0, or -1 (not built, no resident floats, floats mapped from a
 * file, halves made already). */
int     kci_vec_half_floats(KciVec *v);
/* Search-only: for a built index with a quantized graph (kci_vec_build_qg*) and an exact pass (the halves of
 * kci_vec_half_floats, or the retained floats), give up what a quantized-graph search does not read: the int8 rows
 * (dim bytes a row) and the plain graph's layer-0 links. One way. Afterwards only kci_vec_search_rerank and
 * kci_vec_search_filtered (with rerank_n > 0) work, over the quantized graph (a searcher set to the plain graph
 * gets -1); kci_vec_search, _search_r, _search_exact, insert, delete, save, image_save, protect and every build
 * return -1. Without short rows (no code_dims) the exact pass always covers the whole result set (rerank_n is
 * taken as ef). To get the rest back, build the index again from the vectors. Returns 0, or -1 (not built, no
 * quantized graph, no halves or floats, lists / learned lists / votes present, an attached image). */
int     kci_vec_search_only(KciVec *v);

/* How much of the index is plaintext in RAM, in bytes, by kind: out[0] vector codes (int8 rows, 4-bit / low-rank
 * list codes) out[1] floats kept in RAM (0 if mmap'd or absent; or the halves) out[2] derived from the vectors
 * (centroids, projection, codebooks, per-list bases) out[3] graph (neighbour lists, levels, tombstones) out[4]
 * caller ids (and the id lookup table) out[5] total Allocated sizes (capacity, not only live rows): what is in
 * memory. Searchers' scratch (the last query) is not counted. 0 / -1. */
int     kci_vec_plain_bytes(const KciVec *v, uint64_t out[6]);

/* Introspection. */
/* Reconstruct a float (unit-norm) view of stored vector i into out[dim]:
 * the exact float if keep_float was on, else the int8 row dequantized. */
int      kci_vec_get_float(const KciVec *v, uint32_t i, float *out);

uint32_t kci_vec_count(const KciVec *v);
/* How many rows are exact copies of another row? Counted on the int8 rows of a built index (graph or lists),
 * deleted rows left out. out[0] = live rows, out[1] = groups of 2 or more identical rows, out[2] = rows in those
 * groups, out[3] = the largest group, out[4] = groups larger than the layer-0 neighbour list (2 M; such a group
 * cannot even link all of its own members), out[5] = distinct rows (what is left when every group is kept once),
 * out[6], out[7] = 0 (reserved). Returns 0, or -1 (NULL, not built, no int8 rows any more: after
 * kci_vec_search_only; out of memory). Why look: texts that repeat (boilerplate, empty fields) give identical
 * embeddings, and a graph search spends its steps inside such groups. When out[2] is more than a few percent of
 * out[0], put identical rows in once (keep the list of their ids yourself) or search with a larger ef. */
int      kci_vec_dup_stats(const KciVec *v, uint64_t out[8]);
/* A filter sees every id. kci_vec_delete of an id that shares its vector removes the id only. kci_vec_count counts
 * ids; kci_vec_dup_stats: out[6] = ids kept beside rows, out[7] = rows that have such ids (out[0..5] then count
 * the rows, which hold no duplicates any more). Default 0: nothing changes for an index that does not ask. Needs
 * the floats in memory while building (as the build does); an index with a float source or mmap'd floats is left
 * as it is. kci_vec_save writes such an index with the ids, under a magic of its own (KCIVTW04 .. 09 for KCIVEC04
 * .. An index loaded from such a file has dedup on. NOT YET: kci_vec_image_save refuses (-1) an index that holds
 * such ids (the image has no place for them yet). Returns 0, or -1 (NULL, an attached image). */
int      kci_vec_set_dedup(KciVec *v, int on);
int      kci_vec_searcher_set_unique(KciVecSearcher *s, int on);
int      kci_vec_dim(const KciVec *v);
int      kci_vec_is_built(const KciVec *v);

#ifdef __cplusplus
}
#endif

#endif /* KCI_VEC_H */
