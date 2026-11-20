/*
 * kci_sparse.h — sparse-vector index (public API), the companion of kci_vec
 *
 * A sparse vector is a short list of (dimension, weight) pairs out of a very
 * large vocabulary: learned sparse embeddings (SPLADE and the like), BM25 /
 * TF-IDF term weights. Similarity is the inner product.
 *   - Exact top-k, and an approximate search with a work budget.
 *   - Reusable per-thread searchers; many may search one built index at once.
 *   - kci_hybrid_rrf: merge the ranking of this index with the ranking of a
 *     dense kci_vec index (or any other ranking) into one.
 *
 * New file. kci_vec.c stays a one-file dense index; this file does
 * not include it and needs nothing from it. Ids are opaque u64 keys chosen by
 * the caller — use the same id for the same document on both sides.
 *
 * Thread-safety: single-writer. Do not call _add or _build concurrently with
 * searches on the same handle.
 */
#ifndef KCI_SPARSE_H
#define KCI_SPARSE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KciSparse KciSparse;

/* Create an empty index. `dim` is the vocabulary size: every dimension number
 * must be < dim (SPLADE with the BERT vocabulary: 30522). Returns NULL on bad
 * args / OOM. */
KciSparse *kci_sparse_new(uint32_t dim);
void       kci_sparse_free(KciSparse *s);

/* Append one row. idx[0..nnz) are the dimension numbers, val[0..nnz) their
 * weights; any order. A dimension given twice is summed, zero weights are
 * dropped, negative weights are allowed. Returns the new count, or -1 on bad
 * args (a dimension >= dim, a NaN / infinite weight) / OOM — the row is then
 * not added. Rows added after kci_sparse_build need another build. */
int64_t kci_sparse_add(KciSparse *s, uint64_t id, const uint32_t *idx, const float *val, int nnz);

/* Build the search structure over everything added so far. threads <= 0 = cpu
 * cores. Returns 0 on success, -1 on OOM; writes wall-clock seconds to
 * *out_secs if non-NULL. */
int kci_sparse_build(KciSparse *s, int threads, double *out_secs);

/* Add one row to an index that is already built, without rebuilding. Same arguments and return value as
 * kci_sparse_add; -1 also if the index is not built. Not to be called while searches run on the same handle. */
int64_t kci_sparse_insert(KciSparse *s, uint64_t id, const uint32_t *idx, const float *val, int nnz);
uint32_t kci_sparse_pending(const KciSparse *s);   /* rows inserted since the last build */

/* Remove every row that was added under this id. Searches stop returning it at
 * once — no rebuild is needed and existing searchers stay valid; its memory is
 * given back by the next kci_sparse_build. Returns how many rows were removed
 * (0 = no such id), or -1 on bad args / OOM. Like _add, not to be called while
 * searches run on the same handle. */
int kci_sparse_delete(KciSparse *s, uint64_t id);

/* Reusable per-thread search context (one per worker thread). */
typedef struct KciSparseSearcher KciSparseSearcher;
KciSparseSearcher *kci_sparse_searcher_new(KciSparse *s);
void               kci_sparse_searcher_free(KciSparseSearcher *S);

/* Restrict every later search of this searcher to the rows whose id passes
 * `fn` (non-zero = allowed; same shape as KciVecFilter, so one function can
 * serve both sides of a hybrid search). fn == NULL removes the restriction.
 * The function is called only for rows that would otherwise enter the result,
 * from the thread that runs the search. With a filter that lets few rows
 * through, the approximate searches find fewer of them: raise budget /
 * query_terms, or use the exact search. */
typedef int (*KciSparseFilter)(uint64_t id, void *ctx);
void kci_sparse_searcher_set_filter(KciSparseSearcher *S, KciSparseFilter fn, void *ctx);

/* Exact top-k by inner product. Fills out_ids[0..ret) and (if non-NULL)
 * out_scores[], best first; equal scores come in the order the rows were
 * added. Only rows that share a dimension with the query are returned, so ret
 * may be < k. Returns -1 on bad args / not built / a searcher older than the
 * last build. */
int kci_sparse_search_exact(KciSparseSearcher *S, const uint32_t *qidx, const float *qval, int qnnz,
                            int k, uint64_t *out_ids, float *out_scores);

/* Approximate top-k. The best `rerank_n` rows found within the budget are then scored exactly and the top k of
 * those returned (rerank_n < k is raised to k; a few times k is a good start). The scores returned are exact inner
 * products. Same return values as above. */
int kci_sparse_search(KciSparseSearcher *S, const uint32_t *qidx, const float *qval, int qnnz,
                      int k, uint32_t budget, int rerank_n, uint64_t *out_ids, float *out_scores);

/* ── Block search: for large indexes ───────────────────────────── An extra structure on top of a built index. The
 * front of every dimension's list (its `list_len` largest weights) is cut into groups of similar rows, and each
 * group keeps a short summary that bounds the score any of its rows can reach. A query looks only at its
 * `query_terms` heaviest dimensions, skips the groups whose bound cannot enter the current top k, and scores the
 * rows of the others exactly. Zero / NULL = the default for that field. struct_size: set it to
 * sizeof(KciSparseBlockParams) (or leave it 0 after a memset: 0 means "the fields up to global_cut", the layout of
 * this first published version). Fields added later come after global_cut and are read only when struct_size
 * covers them, so a program compiled against this header keeps working with a newer library. Returns 0, -1 on bad
 * args / not built / OOM (the index itself stays usable). A later kci_sparse_build drops the structure; build it
 * again. */
typedef struct {
    unsigned struct_size; /* sizeof(KciSparseBlockParams) as the caller compiled it; 0 = up to global_cut */
    int   list_len;       /* rows kept per dimension, largest weights first (default 3000) */
    int   block_size;     /* target rows per group (default 6) */
    float summary_mass;   /* share of a summary's weight that is kept, 0 < x <= 1 (default 0.4; 1 = a true bound). */
    int   threads;        /* <= 0 = cpu cores */
    int   cluster_terms;
    int   centre_terms;   /* the same for the row a group is formed around: 0 = as cluster_terms, < 0 = all of its dimensions */
    float head_mass;      /* candidates are scored from the heaviest pairs of a row only, as many as carry this share of the sum of its
 * |weights| (default 0.8); the best 2k by that score are then scored whole. Smaller = less memory read per query,
 * slightly less exact. >= 1 = whole rows, which is also the default when summary_mass is 1 */
    int   global_cut;     /* EXPERIMENTAL, leave 0. 1 = one weight level for all dimensions instead of list_len rows for
                             each, set so that the lists average list_len rows (common dimensions keep more). The build
                             then handles the long lists on one thread and can take very long on millions of rows
                             (2.68 million rows: not finished after 19 minutes); it may change or go away */
} KciSparseBlockParams;
int kci_sparse_build_blocks(KciSparse *s, const KciSparseBlockParams *params, double *out_secs);
int kci_sparse_has_blocks(const KciSparse *s);
/* Element counts of the parts: out[0] pairs in the rows, [1] pairs in the lists (0 once dropped), [2] groups,
 * [3] rows in groups, [4] summary runs, [5] summary entries, [6] neighbour slots, [7] pairs in the half-size rows. */
void kci_sparse_sizes(const KciSparse *s, uint64_t out[8]);

/* Give back the memory of the per-dimension lists once the blocks are built (about as much as the rows
 * themselves). Returns 0, -1 if there are no blocks. */
int kci_sparse_drop_lists(KciSparse *s);

/* Give back the full rows as well, for an index that is finished: built, blocks built (and the neighbour lists, if
 * wanted), no rows waiting from kci_sparse_insert, dimension below 65536. The per-dimension lists go too. Keep the
 * source data if that may be needed. Returns 0, or -1 when the index is not in the state above (nothing is freed
 * then). */
int kci_sparse_drop_rows(KciSparse *s);
int kci_sparse_has_rows(const KciSparse *s);   /* 0 after kci_sparse_drop_rows (also for a loaded index saved in that state) */

/* Neighbour lists for every row, the `knn` rows most similar to it, found with the block search itself
 * (query_terms / heap_factor as below; <= 0 = 5 and 0.8). Needs kci_sparse_build_blocks. Costs one block search
 * per row. Returns 0, -1 on bad args / no blocks / OOM. A later kci_sparse_build drops the lists. */
int kci_sparse_build_knn(KciSparse *s, int knn, int query_terms, float heap_factor, int threads, double *out_secs);
int kci_sparse_knn(const KciSparse *s);     /* neighbours kept per row, 0 = not built */

/* Scores returned are exact inner products. Returns the count, -1 on bad args, -2 if kci_sparse_build_blocks has
 * not been called. */
int kci_sparse_search_blocks(KciSparseSearcher *S, const uint32_t *qidx, const float *qval, int qnnz,
                             int k, int query_terms, float heap_factor, int knn,
                             uint64_t *out_ids, float *out_scores);

/* With head_mass < 1, how many rows kci_sparse_search_blocks scores whole at the end (the best ones by the score
 * of their heavy part). 0 = the default, 2 * k. More = closer to scoring every candidate whole. */
void kci_sparse_searcher_set_pool(KciSparseSearcher *S, int rows);

/* Work done by the last kci_sparse_search_blocks on this searcher:
 * out[0] = summaries looked at, out[1] = groups opened, out[2] = rows scored. */
void kci_sparse_searcher_work(const KciSparseSearcher *S, uint64_t out[3]);

uint32_t kci_sparse_count(const KciSparse *s);      /* rows added and not removed */
uint32_t kci_sparse_dim(const KciSparse *s);
int      kci_sparse_is_built(const KciSparse *s);   /* 1 = built and no row added since */
int64_t  kci_sparse_bytes(const KciSparse *s);      /* memory held, in bytes */

/* Save the index to a file and read it back: the rows and whatever has been built from them (the lists, the
 * groups, the neighbour lists), so nothing has to be built again after a load. Rows removed and rows inserted
 * since the build are kept as they are.
 *   key == NULL (keylen 0): the file is written as is.
 *   key != NULL: the file is encrypted with AES-256-CTR and carries an AES-256-CMAC tag; the keys come from
 *     `key` (any bytes, keylen > 0) through PBKDF2-HMAC-SHA256 with a random salt, as for kci_vec_save.
 * save writes <path>.tmp and renames it, so a failed save leaves the old file. Returns 0, or -1.
 * load returns NULL when the file cannot be read, is not this kind of file, does not match its own header,
 * was written with another key, fails the tag (damaged or altered), or holds an offset or row number that
 * points outside its arrays. A file saved with a key opens only with that key; a file saved without one is
 * refused when a key is given (the caller expected an encrypted file). Little-endian machines only. */
int        kci_sparse_save(const KciSparse *s, const char *path, const uint8_t *key, int keylen);
KciSparse *kci_sparse_load(const char *path, const uint8_t *key, int keylen);

/* Merge several rankings into one by reciprocal rank fusion:
 *     score(id) = sum over lists of  weight / (c + rank of id in that list),  rank from 1.
 * Only the order inside each list matters, so a dense (cosine) list and a
 * sparse (inner product) list can be merged without scaling their scores.
 * lists[i] holds lens[i] ids, best first; an id repeated inside one list counts
 * once, at its best rank. weights == NULL means 1 for every list; c <= 0 means
 * 60 (the usual value). Fills out_ids[0..ret) and (if non-NULL) out_scores[],
 * best first; equal scores come smaller id first. Returns ret (<= k), or -1 on
 * bad args / OOM.
 *
 * Typical hybrid search: take 3-5 times k from kci_vec_search_r and from
 * kci_sparse_search, then merge the two lists down to k here. */
int kci_hybrid_rrf(const uint64_t *const *lists, const int *lens, const float *weights, int nlists,
                   float c, int k, uint64_t *out_ids, float *out_scores);

/* Merge several rankings into one by their scores score(id) = sum over lists of weight * s, s = that list's score
 * of id stretched so that the list's first row is 1 and its last row is 0; 0 for a list that does not hold id.
 * Unlike kci_hybrid_rrf this uses how far apart the scores are, not only the order: a row far ahead in one list
 * stays ahead. The lists must be best first; the direction of the scores does not matter (cosine, inner product,
 * or a distance where smaller is better), nor their unit. scores[i][j] belongs to lists[i][j]. An id repeated
 * inside one list counts once, at its first place. weights == NULL means 1 for every list. Fills out_ids[0..ret)
 * and (if non-NULL) out_scores[], best first; equal scores come smaller id first. Returns ret (<= k), or -1 on bad
 * args (a score that is not finite included) / OOM. */
int kci_hybrid_mix(const uint64_t *const *lists, const float *const *scores, const int *lens, const float *weights,
                   int nlists, int k, uint64_t *out_ids, float *out_scores);

/* Encrypt / decrypt a block of bytes with a passphrase — for what lies beside an index, e.g. the texts the vectors
 * were made from. Returns n + 40, or -1 (bad args, no random source). Two seals of the same bytes differ (the
 * salt). kci_unseal: in[0..n) -> out[0..n - 40); `out` must hold n - 40 bytes. Returns n - 40, or -1 when the key
 * is wrong, a byte was changed, the block is cut off or is not a sealed block; `out` is then untouched. The whole
 * block is held in memory; in and out must not overlap. */
int64_t kci_seal(const uint8_t *in, uint64_t n, const uint8_t *key, int keylen, uint8_t *out);
int64_t kci_unseal(const uint8_t *in, uint64_t n, const uint8_t *key, int keylen, uint8_t *out);

#ifdef __cplusplus
}
#endif
#endif /* KCI_SPARSE_H */
