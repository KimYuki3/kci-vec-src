/*
 Sparse index: the exact search must equal a plain all-pairs computation, the
 * budgeted search must return exact scores and reach the exact answer as the
 * budget grows, bad input must be refused, and rank fusion must follow the
 * formula. Run under ASan/UBSan.
 */
#include "kci_sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 2000
#define N   5000
#define NQ  200
#define K   30

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float    rndf(void){ return (float)(rnd() % 100000) / 100000.0f; }

/* A row: a few dimensions, low numbers far more often (like words), weights in (0, 3]. */
static int mkrow(uint32_t *idx, float *val, int maxn, int neg){
    int n = 5 + (int)(rnd() % (uint32_t)(maxn - 5));
    for (int i = 0; i < n; i++){
        float u = rndf(); idx[i] = (uint32_t)(u * u * u * (DIM - 1));
        val[i] = 0.05f + 3.0f * rndf();
        if (neg && rnd() % 5 == 0) val[i] = -val[i];
    }
    return n;   /* may repeat a dimension: the index must sum them */
}

static float *dense;   /* [N*DIM] the same rows, spread out, for the all-pairs answer */
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

typedef struct { float s; uint32_t d; } Ref;
static int cmpref(const void *a, const void *b){
    const Ref *x = a, *y = b;
    if (x->s != y->s) return x->s > y->s ? -1 : 1;
    return x->d < y->d ? -1 : x->d > y->d;
}

static void run(int neg){
    uint32_t idx[64]; float val[64];
    KciSparse *s = kci_sparse_new(DIM);
    memset(dense, 0, sizeof(float) * N * DIM);
    for (uint32_t d = 0; d < N; d++){
        int n = mkrow(idx, val, 60, neg);
        for (int i = 0; i < n; i++) dense[(size_t)d * DIM + idx[i]] += val[i];
        int64_t r = kci_sparse_add(s, 1000 + d, idx, val, n);
        CHECK(r == (int64_t)d + 1, "add returned %lld", (long long)r);
    }
    CHECK(kci_sparse_searcher_new(s) == NULL, "searcher before build must be NULL");
    double secs; CHECK(kci_sparse_build(s, 4, &secs) == 0, "build");
    CHECK(kci_sparse_is_built(s) == 1 && kci_sparse_count(s) == N && kci_sparse_dim(s) == DIM, "getters");
    KciSparseSearcher *S = kci_sparse_searcher_new(s);

    Ref *ref = malloc(sizeof(Ref) * N);
    float *qd = malloc(sizeof(float) * DIM);
    uint64_t ids[100]; float sc[100];
    int exact_bad = 0, score_bad = 0; long hit_small = 0, hit_big = 0, tot = 0; int full_bad = 0;
    for (int qi = 0; qi < NQ; qi++){
        int qn = mkrow(idx, val, 40, neg);
        memset(qd, 0, sizeof(float) * DIM);
        for (int i = 0; i < qn; i++) qd[idx[i]] += val[i];
        int nref = 0;
        for (uint32_t d = 0; d < N; d++){
            double sum = 0; int share = 0; const float *row = dense + (size_t)d * DIM;
            for (int t = 0; t < DIM; t++) if (qd[t] != 0 && row[t] != 0){ sum += (double)qd[t] * row[t]; share = 1; }
            if (share){ ref[nref].s = (float)sum; ref[nref].d = d; nref++; }
        }
        qsort(ref, (size_t)nref, sizeof(Ref), cmpref);
        int want = nref < K ? nref : K;

        int n = kci_sparse_search_exact(S, idx, val, qn, K, ids, sc);
        if (n != want) exact_bad++;
        else for (int i = 0; i < n; i++){
            /* float sums in a different order: allow the last bits, and a swap between near-ties */
            if (fabsf(sc[i] - ref[i].s) > 1e-3f * (1.0f + fabsf(ref[i].s))) { exact_bad++; break; }
            if (ids[i] != 1000 + ref[i].d && fabsf(sc[i] - ref[i].s) > 1e-4f) { exact_bad++; break; }
        }

        for (int pass = 0; pass < 3; pass++){
            uint32_t budget = pass == 0 ? 300 : pass == 1 ? 3000 : 1000000000u;
            n = kci_sparse_search(S, idx, val, qn, K, budget, 3 * K, ids, sc);
            for (int i = 0; i < n; i++){
                uint32_t d = (uint32_t)(ids[i] - 1000); double sum = 0;
                for (int t = 0; t < DIM; t++) sum += (double)qd[t] * dense[(size_t)d * DIM + t];
                if (fabsf(sc[i] - (float)sum) > 1e-3f * (1.0f + fabsf((float)sum))) score_bad++;
                if (i > 0 && sc[i] > sc[i-1]) score_bad++;
            }
            int hit = 0;
            for (int i = 0; i < n; i++) for (int j = 0; j < want; j++) if (ids[i] == 1000 + ref[j].d){ hit++; break; }
            if (pass == 0) hit_small += hit; else if (pass == 1) hit_big += hit;
            else if (n != want || hit < want - 1) full_bad++;
        }
        tot += want;
    }
    printf("%s: exact vs all-pairs mismatches %d ; wrong scores %d ; recall@%d budget 300 = %.3f, 3000 = %.3f ; huge budget mismatches %d\n",
           neg ? "signed  " : "positive", exact_bad, score_bad, K, (double)hit_small / tot, (double)hit_big / tot, full_bad);
    CHECK(exact_bad == 0, "exact search differs from all-pairs");
    CHECK(score_bad == 0, "budgeted search returned a wrong or unordered score");
    CHECK(full_bad == 0, "a budget larger than the index must equal the exact search");
    CHECK(hit_big >= hit_small, "recall must not fall as the budget grows");
    CHECK((double)hit_big / tot > 0.80, "recall at budget 3000 too low");

    /* a row added after the build: the old searcher is refused until the next build */
    idx[0] = 3; val[0] = 1.0f;
    CHECK(kci_sparse_add(s, 999999, idx, val, 1) == N + 1, "add after build");
    CHECK(kci_sparse_is_built(s) == 0, "is_built after add");
    CHECK(kci_sparse_search_exact(S, idx, val, 1, K, ids, sc) == -1, "search on a stale index must be -1");
    CHECK(kci_sparse_build(s, 0, NULL) == 0, "rebuild");
    CHECK(kci_sparse_search_exact(S, idx, val, 1, K, ids, sc) == -1, "old searcher after rebuild must be -1");
    kci_sparse_searcher_free(S);
    S = kci_sparse_searcher_new(s);
    idx[0] = DIM - 1; val[0] = 1.0f;   /* a rare dimension, so the new row's list is short */
    CHECK(kci_sparse_add(s, 888888, idx, val, 1) == N + 2, "second add after build");
    CHECK(kci_sparse_build(s, 0, NULL) == 0, "second rebuild");
    kci_sparse_searcher_free(S);
    S = kci_sparse_searcher_new(s);
    int n = kci_sparse_search_exact(S, idx, val, 1, 100, ids, sc);
    int found = 0; for (int i = 0; i < n; i++) if (ids[i] == 888888) found = 1;
    CHECK(n > 0 && (found || n == 100), "row added after the build not found after the rebuild (n=%d)", n);

    free(ref); free(qd);
    kci_sparse_searcher_free(S); kci_sparse_free(s);
}

static void bad_input(void){
    uint32_t idx[4] = {1, 2, 2, 7}; float val[4] = {1.0f, 2.0f, 3.0f, 0.0f}; uint64_t ids[8]; float sc[8];
    CHECK(kci_sparse_new(0) == NULL, "dim 0");
    KciSparse *s = kci_sparse_new(8);
    uint32_t big[1] = {8}; float one[1] = {1.0f}, nanv[1] = {NAN}, infv[1] = {INFINITY};
    CHECK(kci_sparse_add(s, 1, big, one, 1) == -1, "dimension >= dim must be refused");
    CHECK(kci_sparse_add(s, 1, idx, nanv, 1) == -1, "NaN must be refused");
    CHECK(kci_sparse_add(s, 1, idx, infv, 1) == -1, "inf must be refused");
    CHECK(kci_sparse_add(s, 1, NULL, NULL, 2) == -1, "NULL arrays");
    CHECK(kci_sparse_add(s, 1, idx, val, -1) == -1, "negative nnz");
    CHECK(kci_sparse_count(s) == 0, "refused rows must not be added");
    CHECK(kci_sparse_add(s, 10, idx, val, 4) == 1, "add with a repeat and a zero");   /* {1:1, 2:5} */
    CHECK(kci_sparse_add(s, 11, NULL, NULL, 0) == 2, "empty row");
    float v2[2] = {2.0f, -2.0f}; uint32_t i2[2] = {5, 5};
    CHECK(kci_sparse_add(s, 12, i2, v2, 2) == 3, "row that sums to nothing");
    CHECK(kci_sparse_build(s, 1, NULL) == 0, "build");
    KciSparseSearcher *S = kci_sparse_searcher_new(s);
    uint32_t qi[1] = {2}; float qv[1] = {2.0f};
    int n = kci_sparse_search_exact(S, qi, qv, 1, 8, ids, sc);
    CHECK(n == 1 && ids[0] == 10 && fabsf(sc[0] - 10.0f) < 1e-6f, "repeat must be summed: n=%d score=%g", n, n > 0 ? sc[0] : 0);
    uint32_t q5[1] = {5};
    CHECK(kci_sparse_search_exact(S, q5, qv, 1, 8, ids, sc) == 0, "nothing shares dimension 5");
    CHECK(kci_sparse_search_exact(S, big, one, 1, 8, ids, sc) == -1, "query dimension >= dim");
    CHECK(kci_sparse_search_exact(S, qi, nanv, 1, 8, ids, sc) == -1, "query NaN");
    CHECK(kci_sparse_search_exact(S, qi, qv, 1, 0, ids, sc) == -1, "k 0");
    CHECK(kci_sparse_search_exact(S, qi, qv, 1, 8, NULL, sc) == -1, "NULL out_ids");
    CHECK(kci_sparse_search_exact(S, NULL, NULL, 0, 8, ids, sc) == 0, "empty query");
    CHECK(kci_sparse_search_exact(S, qi, qv, 1, 8, ids, NULL) == 1, "NULL out_scores is allowed");
    CHECK(kci_sparse_search_exact(S, qi, qv, 1, 8, ids, sc) == 1 && ids[0] == 10, "searcher still clean after refused queries");
    kci_sparse_searcher_free(S); kci_sparse_free(s);
    kci_sparse_free(NULL); kci_sparse_searcher_free(NULL);

    /* an index with no rows */
    s = kci_sparse_new(8);
    CHECK(kci_sparse_build(s, 2, NULL) == 0, "build of an empty index");
    S = kci_sparse_searcher_new(s);
    CHECK(S && kci_sparse_search_exact(S, qi, qv, 1, 8, ids, sc) == 0, "search of an empty index");
    kci_sparse_searcher_free(S); kci_sparse_free(s);
}

/* Removing rows: gone from the results at once, the rest unchanged, gone for good after a rebuild. */
static void removal(void){
    enum { M = 3000 };
    KciSparse *s = kci_sparse_new(50);
    uint32_t idx[3]; float val[3]; uint64_t ids[M]; float sc[M];
    CHECK(kci_sparse_delete(s, 5) == 0, "delete on an empty index");
    for (uint32_t d = 0; d < M; d++){
        idx[0] = d % 50; idx[1] = (d * 7 + 1) % 50; idx[2] = 49 - d % 50;
        val[0] = 1.0f + (float)(d % 13); val[1] = 0.5f; val[2] = 0.25f;
        kci_sparse_add(s, d, idx, val, 3);
    }
    kci_sparse_add(s, 77, idx, val, 3);                     /* a second row under id 77 */
    CHECK(kci_sparse_build(s, 2, NULL) == 0, "build");
    KciSparseSearcher *S = kci_sparse_searcher_new(s);
    uint32_t qi[2] = {3, 10}; float qv[2] = {1.0f, 2.0f};
    uint64_t before[M]; float bsc[M];
    int nb = kci_sparse_search_exact(S, qi, qv, 2, M, before, bsc);
    CHECK(nb > 100, "enough rows share the query's dimensions (n=%d)", nb);

    /* remove every third result, and id 77 (two rows) */
    int gone = 0;
    for (int i = 0; i < nb; i += 3) if (before[i] != 77){ CHECK(kci_sparse_delete(s, before[i]) == 1, "delete id %llu", (unsigned long long)before[i]); gone++; }
    CHECK(kci_sparse_delete(s, 77) == 2, "id 77 has two rows");
    CHECK(kci_sparse_delete(s, 77) == 0, "deleting twice removes nothing");
    CHECK(kci_sparse_delete(s, 123456789) == 0, "unknown id");
    CHECK(kci_sparse_count(s) == (uint32_t)(M + 1 - gone - 2), "count after delete");
    CHECK(kci_sparse_is_built(s) == 1, "delete must not need a rebuild");

    for (int pass = 0; pass < 2; pass++){                   /* pass 0: marked only; pass 1: after a rebuild */
        int n = kci_sparse_search_exact(S, qi, qv, 2, M, ids, sc);
        int j = 0, ok = 1;
        for (int i = 0; i < nb; i++){
            if (i % 3 == 0 || before[i] == 77) continue;
            if (j >= n || ids[j] != before[i] || sc[j] != bsc[i]) { ok = 0; break; }
            j++;
        }
        CHECK(ok && j == n, "pass %d: the rows left must come back in the same order with the same scores (n=%d, matched %d)", pass, n, j);
        n = kci_sparse_search(S, qi, qv, 2, 30, 200, 60, ids, sc);
        for (int i = 0; i < n; i++) for (int t = 0; t < nb; t += 3) if (ids[i] == before[t]) { CHECK(0, "pass %d: a removed row came back from the budgeted search", pass); break; }
        if (pass == 0){
            int64_t r = kci_sparse_add(s, 5000000, qi, qv, 2);   /* add after delete, then rebuild */
            CHECK(r == (int64_t)kci_sparse_count(s), "add returns the live count");
            CHECK(kci_sparse_delete(s, 5000000) == 1 && kci_sparse_add(s, 5000001, qi, qv, 2) > 0, "delete then add before the rebuild");
            int64_t bytes0 = kci_sparse_bytes(s);
            CHECK(kci_sparse_build(s, 2, NULL) == 0, "rebuild");
            CHECK(kci_sparse_bytes(s) <= bytes0, "rebuild must not grow the index");
            kci_sparse_searcher_free(S); S = kci_sparse_searcher_new(s);
            n = kci_sparse_search_exact(S, qi, qv, 2, M, ids, sc);
            int seen = 0; for (int i = 0; i < n; i++){ if (ids[i] == 5000001) seen++; if (ids[i] == 5000000) seen = -99; }
            CHECK(seen == 1, "after the rebuild: the row added is found once, the row removed is not (seen=%d)", seen);
            CHECK(kci_sparse_delete(s, 5000001) == 1, "delete after the rebuild");
        }
    }
    kci_sparse_searcher_free(S); kci_sparse_free(s);
}

/* Block search. With whole lists, whole summaries, every query dimension and heap_factor 1 nothing
 * can be skipped wrongly, so it must equal the exact search; with the defaults it is approximate but
 * every score is still the exact one. */
static void blocks(int neg){
    enum { BN = 4000, BQ = 100 };
    uint32_t idx[64]; float val[64]; uint64_t ids[K], eids[K]; float sc[K], esc[K];
    KciSparse *s = kci_sparse_new(DIM);
    for (uint32_t d = 0; d < BN; d++){ int n = mkrow(idx, val, 60, neg); kci_sparse_add(s, 7000 + d, idx, val, n); }
    CHECK(kci_sparse_build_blocks(s, NULL, NULL) == -1, "build_blocks before build must be -1");
    kci_sparse_build(s, 2, NULL);
    KciSparseSearcher *S = kci_sparse_searcher_new(s);
    idx[0] = 1; val[0] = 1.0f;
    CHECK(kci_sparse_search_blocks(S, idx, val, 1, K, 0, 1.0f, 0, ids, sc) == -2, "search_blocks without blocks must be -2");
    CHECK(kci_sparse_build_knn(s, 10, 0, 0, 2, NULL) == -1 && kci_sparse_knn(s) == 0, "build_knn without blocks must be -1");
    CHECK(kci_sparse_has_blocks(s) == 0, "has_blocks before");
    KciSparseBlockParams bad; memset(&bad, 0, sizeof bad); bad.summary_mass = 1.5f; bad.threads = 1;
    CHECK(kci_sparse_build_blocks(s, &bad, NULL) == -1 && kci_sparse_has_blocks(s) == 0, "summary_mass > 1 must be refused");

    KciSparseBlockParams full; memset(&full, 0, sizeof full); full.struct_size = sizeof full; full.list_len = 1000000; full.block_size = 8; full.summary_mass = 1.0f; full.threads = 2;
    CHECK(kci_sparse_build_blocks(s, &full, NULL) == 0 && kci_sparse_has_blocks(s) == 1, "build_blocks (whole lists, whole summaries)");
    uint64_t save = rs; int differ = 0; uint64_t rows_scored = 0, w[3];
    for (int qi = 0; qi < BQ; qi++){
        int qn = mkrow(idx, val, 40, neg);
        int ne = kci_sparse_search_exact(S, idx, val, qn, K, eids, esc);
        int n = kci_sparse_search_blocks(S, idx, val, qn, K, 0, 1.0f, 0, ids, sc);
        kci_sparse_searcher_work(S, w); rows_scored += w[2];
        if (n != ne) { differ++; continue; }
        for (int i = 0; i < n; i++)
            if (fabsf(sc[i] - esc[i]) > 1e-3f * (1.0f + fabsf(esc[i])) || (ids[i] != eids[i] && fabsf(sc[i] - esc[i]) > 1e-4f)) { differ++; break; }
    }
    CHECK(differ == 0, "%s: block search with nothing cut differs from exact in %d queries", neg ? "signed" : "positive", differ);
    if (!neg) CHECK(rows_scored < (uint64_t)BQ * BN, "summaries should let some rows go unscored (scored %llu of %d)", (unsigned long long)rows_scored, BQ * BN);

    CHECK(kci_sparse_build_blocks(s, NULL, NULL) == 0, "build_blocks with the defaults (replaces the first)");
    CHECK(kci_sparse_build_knn(s, 0, 0, 0, 2, NULL) == -1, "knn 0 must be refused");
    CHECK(kci_sparse_build_knn(s, 12, 0, 0, 2, NULL) == 0 && kci_sparse_knn(s) == 12, "build_knn");
    long hits[2] = {0, 0}, tot = 0; int wrong = 0;
    for (int pass = 0; pass < 2; pass++){                   /* pass 0: groups only; pass 1: plus 12 neighbours of each result */
        rs = save;
        for (int qi = 0; qi < BQ; qi++){
            int qn = mkrow(idx, val, 40, neg);
            int ne = kci_sparse_search_exact(S, idx, val, qn, K, eids, esc);
            int n = kci_sparse_search_blocks(S, idx, val, qn, K, 3, 0.8f, pass ? 12 : 0, ids, sc);
            if (n < 0 || n > K) { wrong++; continue; }
            for (int i = 0; i < n; i++){
                if (i > 0 && sc[i] > sc[i-1]) wrong++;
                for (int x = 0; x < i; x++) if (ids[x] == ids[i]) wrong++;
                for (int e = 0; e < ne; e++) if (ids[i] == eids[e]) { hits[pass]++; if (fabsf(sc[i] - esc[e]) > 1e-3f * (1.0f + fabsf(esc[e]))) wrong++; break; }
            }
            if (pass == 0) tot += ne;
        }
    }
    printf("%s: block search, defaults, 3 query dimensions, heap_factor 0.8: recall@%d = %.3f, with 12 neighbours = %.3f\n",
           neg ? "signed  " : "positive", K, (double)hits[0] / tot, (double)hits[1] / tot);
    CHECK(wrong == 0, "block search returned a wrong, repeated or unordered result");
    CHECK((double)hits[0] / tot > 0.3, "block search recall far too low");
    CHECK(hits[1] >= hits[0], "neighbours must not lower the recall");
    CHECK(kci_sparse_search_blocks(S, idx, val, 1, K, 0, 1.0f, -1, ids, sc) == -1, "negative knn must be refused");
    CHECK(kci_sparse_search_blocks(S, idx, val, 1, K, 0, 1.0f, 1000, ids, sc) >= 0, "knn larger than built is capped");

    CHECK(kci_sparse_search_blocks(S, idx, val, 1, K, 0, 0.0f, 0, ids, sc) == -1, "heap_factor 0 must be refused");
    CHECK(kci_sparse_search_blocks(S, idx, val, 1, K, 0, 4.5f, 0, ids, sc) == -1, "heap_factor > 4 must be refused");
    int qn = mkrow(idx, val, 40, neg);
    int n = kci_sparse_search_blocks(S, idx, val, qn, K, 0, 1.0f, 12, ids, sc);
    CHECK(n > 0 && kci_sparse_delete(s, ids[0]) == 1, "delete the best row");
    uint64_t gone = ids[0];
    n = kci_sparse_search_blocks(S, idx, val, qn, K, 0, 1.0f, 12, ids, sc);
    for (int i = 0; i < n; i++) if (ids[i] == gone) { CHECK(0, "a removed row came back from the block search"); break; }
    n = kci_sparse_search_blocks(S, idx, val, qn, K, 2, 0.7f, 12, ids, sc);
    for (int i = 0; i < n; i++) if (ids[i] == gone) { CHECK(0, "a removed row came back through the neighbour lists"); break; }
    /* without the lists: the block search is unchanged, the list searches are refused, a rebuild brings them back */
    n = kci_sparse_search_blocks(S, idx, val, qn, K, 4, 0.9f, 12, ids, sc);
    int64_t bytes0 = kci_sparse_bytes(s);
    CHECK(kci_sparse_drop_lists(s) == 0 && kci_sparse_bytes(s) < bytes0, "drop_lists frees memory");
    int n2 = kci_sparse_search_blocks(S, idx, val, qn, K, 4, 0.9f, 12, eids, esc);
    int same = n2 == n; for (int i = 0; i < n && same; i++) same = eids[i] == ids[i] && esc[i] == sc[i];
    CHECK(same, "block search after drop_lists must give the same result");
    CHECK(kci_sparse_search_exact(S, idx, val, qn, K, eids, esc) == -1 && kci_sparse_search(S, idx, val, qn, K, 100, 60, eids, esc) == -1,
          "list searches after drop_lists must be -1");
    CHECK(kci_sparse_build_blocks(s, NULL, NULL) == -1 && kci_sparse_has_blocks(s) == 1, "build_blocks after drop_lists must be -1 and keep the blocks");
    CHECK(kci_sparse_build(s, 2, NULL) == 0 && kci_sparse_has_blocks(s) == 0 && kci_sparse_knn(s) == 0, "a rebuild drops the blocks and the neighbour lists");
    kci_sparse_searcher_free(S); S = kci_sparse_searcher_new(s);
    CHECK(kci_sparse_drop_lists(s) == -1, "drop_lists without blocks must be -1");
    CHECK(kci_sparse_search_exact(S, idx, val, qn, K, eids, esc) >= 0, "the rebuild brings the lists back");
    kci_sparse_searcher_free(S); kci_sparse_free(s);
}

/* Filter: every result passes it, the exact search returns exactly the allowed part of the unfiltered
 * ranking, and removing the filter restores the old result. */
static int mod3(uint64_t id, void *ctx){ (*(long*)ctx)++; return id % 3 == 0; }
static int only_id(uint64_t id, void *ctx){ return id == *(uint64_t*)ctx; }
static void filtered(void){
    enum { FN = 3000 };
    uint32_t idx[64]; float val[64]; static uint64_t all[FN], ids[FN]; static float asc[FN], sc[FN];
    KciSparse *s = kci_sparse_new(DIM);
    for (uint32_t d = 0; d < FN; d++){ int n = mkrow(idx, val, 60, 0); kci_sparse_add(s, d, idx, val, n); }
    kci_sparse_build(s, 2, NULL);
    kci_sparse_build_blocks(s, NULL, NULL); kci_sparse_build_knn(s, 10, 0, 0, 2, NULL);
    KciSparseSearcher *S = kci_sparse_searcher_new(s);
    int bad = 0, impure = 0; long calls = 0, rows_seen = 0;
    for (int qi = 0; qi < 50; qi++){
        int qn = mkrow(idx, val, 40, 0);
        kci_sparse_searcher_set_filter(S, NULL, NULL);
        int na = kci_sparse_search_exact(S, idx, val, qn, FN, all, asc);
        kci_sparse_searcher_set_filter(S, mod3, &calls);
        int n = kci_sparse_search_exact(S, idx, val, qn, K, ids, sc);
        int j = 0;
        for (int i = 0; i < na && j < n; i++) if (all[i] % 3 == 0){ if (ids[j] != all[i] || sc[j] != asc[i]) bad++; j++; }
        int allowed = 0; for (int i = 0; i < na; i++) allowed += all[i] % 3 == 0;
        if (j != n || n != (allowed < K ? allowed : K)) bad++;
        rows_seen += na;
        n = kci_sparse_search(S, idx, val, qn, K, 500, 60, ids, sc);
        for (int i = 0; i < n; i++) if (ids[i] % 3) impure++;
        n = kci_sparse_search_blocks(S, idx, val, qn, K, 5, 0.9f, 10, ids, sc);
        for (int i = 0; i < n; i++) if (ids[i] % 3) impure++;
        if (n == 0) bad++;
    }
    CHECK(bad == 0, "filtered exact search differs from the allowed part of the unfiltered ranking (%d)", bad);
    CHECK(impure == 0, "%d results did not pass the filter", impure);
    CHECK(calls < rows_seen, "the filter should be asked only for rows that would enter (%ld calls, %ld rows)", calls, rows_seen);
    /* one allowed id: it comes back alone, with its score */
    int qn = mkrow(idx, val, 40, 0);
    kci_sparse_searcher_set_filter(S, NULL, NULL);
    int na = kci_sparse_search_exact(S, idx, val, qn, FN, all, asc);
    uint64_t target = all[na / 2];
    kci_sparse_searcher_set_filter(S, only_id, &target);
    int n = kci_sparse_search_exact(S, idx, val, qn, K, ids, sc);
    CHECK(na > 2 && n == 1 && ids[0] == target && sc[0] == asc[na / 2], "single allowed id: n=%d", n);
    kci_sparse_searcher_set_filter(S, NULL, NULL);
    n = kci_sparse_search_exact(S, idx, val, qn, K, ids, sc);
    int same = n == (na < K ? na : K); for (int i = 0; i < n && same; i++) same = ids[i] == all[i];
    CHECK(same, "removing the filter must restore the unfiltered result");
    kci_sparse_searcher_set_filter(NULL, mod3, NULL);
    kci_sparse_searcher_free(S); kci_sparse_free(s);
}

/* Rows inserted after the build: every search must give what an index built with all the rows gives. */
static void inserted(void){
    enum { IN0 = 2000, INS = 120, IQ = 80 };
    static uint32_t ridx[IN0 + INS][64]; static float rval[IN0 + INS][64]; static int rn[IN0 + INS];
    uint32_t idx[64]; float val[64]; uint64_t ia[K], ib[K]; float sa[K], sb[K];
    KciSparse *a = kci_sparse_new(DIM), *b = kci_sparse_new(DIM);
    for (int d = 0; d < IN0 + INS; d++){ rn[d] = mkrow(ridx[d], rval[d], 60, 0); kci_sparse_add(b, 100 + (uint64_t)d, ridx[d], rval[d], rn[d]); }
    for (int d = 0; d < IN0; d++) kci_sparse_add(a, 100 + (uint64_t)d, ridx[d], rval[d], rn[d]);
    CHECK(kci_sparse_insert(a, 1, ridx[0], rval[0], rn[0]) == -1, "insert before build must be -1");
    kci_sparse_build(a, 2, NULL); kci_sparse_build(b, 2, NULL);
    kci_sparse_build_blocks(a, NULL, NULL); kci_sparse_build_knn(a, 10, 0, 0, 2, NULL);
    KciSparseSearcher *A = kci_sparse_searcher_new(a), *B = kci_sparse_searcher_new(b);
    for (int d = IN0; d < IN0 + INS; d++)
        CHECK(kci_sparse_insert(a, 100 + (uint64_t)d, ridx[d], rval[d], rn[d]) == d + 1, "insert returns the count");
    CHECK(kci_sparse_is_built(a) == 1 && kci_sparse_pending(a) == INS && kci_sparse_count(a) == IN0 + INS, "built, pending, count after inserts");
    uint32_t bigd[1] = { DIM }; float one[1] = { 1.0f };
    CHECK(kci_sparse_insert(a, 5, bigd, one, 1) == -1 && kci_sparse_pending(a) == INS, "a bad row is refused by insert too");

    int differ = 0, wrong = 0, self_miss = 0;
    for (int qi = 0; qi < IQ; qi++){
        int qn = mkrow(idx, val, 40, 0);
        int nb = kci_sparse_search_exact(B, idx, val, qn, K, ib, sb);
        int na = kci_sparse_search_exact(A, idx, val, qn, K, ia, sa);
        if (na != nb) differ++;
        else for (int i = 0; i < na; i++) if (fabsf(sa[i] - sb[i]) > 1e-3f * (1.0f + fabsf(sb[i])) || (ia[i] != ib[i] && fabsf(sa[i] - sb[i]) > 1e-4f)) { differ++; break; }
        /* approximate searches: every id returned once, each score the exact one (looked up in a big exact list of b) */
        for (int mode = 0; mode < 2; mode++){
            static uint64_t fb[IN0 + INS]; static float fs[IN0 + INS];
            int nf = kci_sparse_search_exact(B, idx, val, qn, IN0 + INS, fb, fs);
            int n = mode ? kci_sparse_search_blocks(A, idx, val, qn, K, 5, 0.9f, 10, ia, sa) : kci_sparse_search(A, idx, val, qn, K, 2000, 60, ia, sa);
            for (int i = 0; i < n; i++){
                int found = 0;
                for (int x = 0; x < i; x++) if (ia[x] == ia[i]) wrong++;
                for (int e = 0; e < nf; e++) if (fb[e] == ia[i]) { found = 1; if (fabsf(sa[i] - fs[e]) > 1e-3f * (1.0f + fabsf(fs[e]))) wrong++; break; }
                if (!found) wrong++;
            }
        }
    }
    CHECK(differ == 0, "exact search over built + inserted rows differs from an index built with all rows (%d queries)", differ);
    CHECK(wrong == 0, "approximate search over built + inserted rows: %d wrong, repeated or unknown results", wrong);
    /* asked for by the content of an inserted row: whenever the true best row (from b) is an inserted one, the
     * approximate searches must return it first too — inserted rows are always scored, never skipped */
    int cases = 0;
    for (int d = IN0; d < IN0 + INS; d++){
        if (kci_sparse_search_exact(B, ridx[d], rval[d], rn[d], 1, ib, sb) != 1 || ib[0] < 100 + (uint64_t)IN0) continue;
        cases++;
        if (kci_sparse_search(A, ridx[d], rval[d], rn[d], 1, 500, 30, ia, sa) != 1 || ia[0] != ib[0]) self_miss++;
        if (kci_sparse_search_blocks(A, ridx[d], rval[d], rn[d], 1, 3, 0.8f, 10, ia, sa) != 1 || ia[0] != ib[0]) self_miss++;
    }
    CHECK(cases > 10, "too few queries whose best row is an inserted one (%d)", cases);
    CHECK(self_miss == 0, "an inserted row that is the true best was not returned first (%d of %d cases)", self_miss, cases);
    /* delete and filter apply to inserted rows too */
    int d0 = IN0 + 3; uint64_t id0 = 100 + (uint64_t)d0;
    CHECK(kci_sparse_delete(a, id0) == 1, "delete an inserted row");
    int n = kci_sparse_search_exact(A, ridx[d0], rval[d0], rn[d0], K, ia, sa);
    for (int i = 0; i < n; i++) if (ia[i] == id0) { CHECK(0, "a deleted inserted row came back"); break; }
    long calls = 0; kci_sparse_searcher_set_filter(A, mod3, &calls);
    n = kci_sparse_search_blocks(A, ridx[IN0 + 10], rval[IN0 + 10], rn[IN0 + 10], K, 5, 0.9f, 10, ia, sa);
    for (int i = 0; i < n; i++) if (ia[i] % 3) { CHECK(0, "the filter did not apply to an inserted row"); break; }
    kci_sparse_searcher_set_filter(A, NULL, NULL);
    /* a rebuild takes the inserted rows into the lists */
    CHECK(kci_sparse_build(a, 2, NULL) == 0 && kci_sparse_pending(a) == 0 && kci_sparse_count(a) == IN0 + INS - 1, "rebuild after inserts");
    kci_sparse_searcher_free(A); A = kci_sparse_searcher_new(a);
    n = kci_sparse_search_exact(A, ridx[IN0 + 10], rval[IN0 + 10], rn[IN0 + 10], 1, ia, sa);
    CHECK(n == 1 && ia[0] == 100 + (uint64_t)(IN0 + 10), "an inserted row is still found after the rebuild");
    kci_sparse_searcher_free(A); kci_sparse_searcher_free(B); kci_sparse_free(a); kci_sparse_free(b);
}

static void wide_vocab(void){
    enum { WD = 100000, WN = 4000, WQ = 60, WK = 20 };
    KciSparse *s = kci_sparse_new(WD);
    uint64_t r = 0x9E3779B97F4A7C15ULL; uint32_t ix[32]; float vv[32];
    #define WRND() (r ^= r << 13, r ^= r >> 7, r ^= r << 17, r)
    #define WROW(n_) do { n_ = 0; for (int j_ = 0; j_ < 28; j_++){ uint64_t a_ = WRND() % 317, b_ = WRND() % 317; uint32_t d_ = (uint32_t)(a_ * b_ % WD); \
        if (j_ % 4 == 0){ d_ = (uint32_t)(WRND() % WD); } int dup_ = 0; for (int z_ = 0; z_ < n_; z_++) dup_ |= ix[z_] == d_; \
        if (!dup_){ ix[n_] = d_; vv[n_] = 0.05f + (float)(WRND() % 1000) / 400.0f; n_++; } } } while (0)
    int n; int big = 0;
    for (uint64_t id = 0; id < WN; id++){ WROW(n); for (int j = 0; j < n; j++) big |= ix[j] >= 65536; kci_sparse_add(s, id + 1, ix, vv, n); }
    CHECK(big && kci_sparse_build(s, 2, NULL) == 0, "wide vocabulary: rows with dimensions above 65,535, built");
    KciSparseSearcher *S = kci_sparse_searcher_new(s);
    KciSparseBlockParams full; memset(&full, 0, sizeof full); full.struct_size = sizeof full; full.list_len = 1000000; full.block_size = 8; full.summary_mass = 1.0f; full.threads = 2;
    CHECK(kci_sparse_build_blocks(s, &full, NULL) == 0, "wide vocabulary: build_blocks");
    uint64_t z[8]; kci_sparse_sizes(s, z);
    CHECK(z[7] == 0 && kci_sparse_drop_rows(s) == -1 && kci_sparse_has_rows(s) == 1, "wide vocabulary: no half-size rows, drop_rows refused");
    uint64_t ids[WK], eids[WK]; float sc[WK], esc[WK]; uint64_t keep = r; int differ = 0;
    for (int q = 0; q < WQ; q++){
        WROW(n);
        int ne = kci_sparse_search_exact(S, ix, vv, n, WK, eids, esc), nb = kci_sparse_search_blocks(S, ix, vv, n, WK, 0, 1.0f, 0, ids, sc);
        if (nb != ne){ differ++; continue; }
        for (int i = 0; i < nb; i++) if (fabsf(sc[i] - esc[i]) > 1e-3f * (1.0f + fabsf(esc[i])) || (ids[i] != eids[i] && fabsf(sc[i] - esc[i]) > 1e-4f)){ differ++; break; }
    }
    CHECK(differ == 0, "wide vocabulary: block search with nothing cut differs from exact in %d of %d queries", differ, WQ);
    CHECK(kci_sparse_build_blocks(s, NULL, NULL) == 0 && kci_sparse_build_knn(s, 12, 0, 0, 2, NULL) == 0, "wide vocabulary: default blocks and neighbour lists");
    long hit = 0, tot = 0; int wrong = 0; uint64_t first[WQ][WK]; int firstn[WQ];
    r = keep;
    for (int q = 0; q < WQ; q++){
        WROW(n);
        int ne = kci_sparse_search_exact(S, ix, vv, n, WK, eids, esc), nb = kci_sparse_search_blocks(S, ix, vv, n, WK, 8, 1.0f, 12, ids, sc);
        firstn[q] = nb; for (int i = 0; i < nb; i++) first[q][i] = ids[i];
        tot += ne;
        for (int i = 0; i < nb; i++){
            int found = 0; for (int j = 0; j < ne; j++) if (eids[j] == ids[i]){ found = 1; if (fabsf(sc[i] - esc[j]) > 1e-3f * (1.0f + fabsf(esc[j]))) wrong++; }
            hit += found;
        }
    }
    CHECK(wrong == 0 && hit * 10 >= tot * 6, "wide vocabulary: defaults give exact scores and recall@%d %.3f (>= 0.6)", WK, tot ? (double)hit / tot : 0.0);
    kci_sparse_searcher_free(S);
    CHECK(kci_sparse_save(s, "test_sparse_wide.kcs", (const uint8_t*)"wide key", 8) == 0, "wide vocabulary: save");
    KciSparse *b = kci_sparse_load("test_sparse_wide.kcs", (const uint8_t*)"wide key", 8);
    CHECK(b != NULL && kci_sparse_load("test_sparse_wide.kcs", (const uint8_t*)"wide kez", 8) == NULL, "wide vocabulary: load, wrong key refused");
    if (b){
        KciSparseSearcher *B = kci_sparse_searcher_new(b); int same = 1;
        r = keep;
        for (int q = 0; q < WQ; q++){
            WROW(n);
            int nb = kci_sparse_search_blocks(B, ix, vv, n, WK, 8, 1.0f, 12, ids, sc);
            same &= nb == firstn[q]; for (int i = 0; same && i < nb; i++) same &= ids[i] == first[q][i];
        }
        CHECK(same, "wide vocabulary: the loaded index answers the same");
        kci_sparse_searcher_free(B); kci_sparse_free(b);
    }
    remove("test_sparse_wide.kcs");
    kci_sparse_free(s);
    #undef WRND
    #undef WROW
}

static void params_size(void){
    KciSparse *a = kci_sparse_new(DIM), *b = kci_sparse_new(DIM), *c = kci_sparse_new(DIM);
    uint64_t rs = 99; uint32_t ix[12]; float vv[12];
    for (uint64_t id = 0; id < 3000; id++){
        int n = 0;
        for (int j = 0; j < 12; j++){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; uint32_t d = (uint32_t)(rs % DIM); int dup = 0;
            for (int z = 0; z < n; z++) dup |= ix[z] == d;
            if (!dup){ ix[n] = d; vv[n] = 0.1f + (float)(rs >> 40 & 1023) / 512.0f; n++; } }
        kci_sparse_add(a, id, ix, vv, n); kci_sparse_add(b, id, ix, vv, n); kci_sparse_add(c, id, ix, vv, n);
    }
    kci_sparse_build(a, 1, NULL); kci_sparse_build(b, 1, NULL); kci_sparse_build(c, 1, NULL);
    KciSparseBlockParams full; memset(&full, 0, sizeof full); full.struct_size = sizeof full; full.list_len = 200; full.block_size = 5; full.threads = 1;
    struct { unsigned struct_size; int list_len, block_size; float summary_mass; int threads; } shortp = { 0, 200, 5, 0, 1 };
    shortp.struct_size = sizeof shortp;
    struct { KciSparseBlockParams p; int later[6]; } longp; memset(&longp, 0x7F, sizeof longp); longp.p = full; longp.p.struct_size = sizeof longp;
    int ra = kci_sparse_build_blocks(a, &full, NULL), rb = kci_sparse_build_blocks(b, (const KciSparseBlockParams*)(const void*)&shortp, NULL),
        rc = kci_sparse_build_blocks(c, &longp.p, NULL);
    uint64_t za[8], zb[8], zc[8]; kci_sparse_sizes(a, za); kci_sparse_sizes(b, zb); kci_sparse_sizes(c, zc);
    CHECK(ra == 0 && rb == 0 && rc == 0 && memcmp(za, zb, sizeof za) == 0 && memcmp(za, zc, sizeof za) == 0 && za[2] > 0,
          "a shorter and a longer params struct build the same index as the full one (%d %d %d, groups %llu %llu %llu)", ra, rb, rc,
          (unsigned long long)za[2], (unsigned long long)zb[2], (unsigned long long)zc[2]);
    KciSparseBlockParams tiny; memset(&tiny, 0, sizeof tiny); tiny.struct_size = 4;
    CHECK(kci_sparse_build_blocks(a, &tiny, NULL) == -1, "a struct_size too small for any setting is refused");
    kci_sparse_free(a); kci_sparse_free(b); kci_sparse_free(c);
}

static void seal(void){
    enum { SN = 64 };       /* small: every unseal runs PBKDF2 (100,000 rounds) */
    uint8_t in[SN], box[SN + 40], box2[SN + 40], out[SN], key[] = "seal key 2026";
    for (int i = 0; i < SN; i++) in[i] = (uint8_t)(i * 37 + (i >> 3));
    int64_t r = kci_seal(in, SN, key, 13, box);
    CHECK(r == SN + 40 && memcmp(box, "KCISEAL1", 8) == 0, "seal returns n + 40");
    int plain = 0; for (int i = 0; i + 16 <= SN; i++) if (memcmp(box + 24 + i, in + i, 16) == 0) plain++;
    CHECK(plain == 0, "the sealed block does not hold the plain bytes");
    memset(out, 0, SN);
    CHECK(kci_unseal(box, SN + 40, key, 13, out) == SN && memcmp(out, in, SN) == 0, "unseal gives the bytes back");
    CHECK(kci_seal(in, SN, key, 13, box2) == SN + 40 && memcmp(box, box2, SN + 40) != 0, "two seals of the same bytes differ");
    memset(out, 0xEE, SN);
    CHECK(kci_unseal(box, SN + 40, (const uint8_t*)"seal key 2027", 13, out) == -1 && out[0] == 0xEE && out[SN-1] == 0xEE, "a wrong key is refused, out untouched");
    int refused = 0;
    for (int i = 0; i < SN + 40; i++){ box[i] ^= 0x01; refused += kci_unseal(box, SN + 40, key, 13, out) == -1; box[i] ^= 0x01; }
    CHECK(refused == SN + 40, "every one-byte change is refused: %d of %d", refused, SN + 40);
    int cut = 0; for (int n = 0; n < SN + 40; n += 7) cut += kci_unseal(box, (uint64_t)n, key, 13, out) == -1;
    CHECK(cut == (SN + 40 + 6) / 7, "a cut-off block is refused");
    uint8_t e[40];
    CHECK(kci_seal(NULL, 0, key, 13, e) == 40 && kci_unseal(e, 40, key, 13, NULL) == 0, "an empty block seals and opens");
    CHECK(kci_seal(in, SN, NULL, 0, box2) == -1 && kci_seal(in, SN, key, 0, box2) == -1 && kci_seal(in, SN, key, 13, NULL) == -1 &&
          kci_unseal(NULL, SN + 40, key, 13, out) == -1 && kci_unseal(box, SN + 40, NULL, 0, out) == -1, "bad arguments are refused");
}

static void fusion(void){
    /* dense says 1 2 3 4, sparse says 3 5 1. c = 60:
     *   1: 1/61 + 1/63   3: 1/63 + 1/61   2: 1/62   5: 1/62   4: 1/64
     * 1 and 3 tie (smaller id first), then 2 and 5 tie, then 4. */
    uint64_t a[4] = {1, 2, 3, 4}, b[3] = {3, 5, 1}, out[8]; float sc[8];
    const uint64_t *lists[2] = {a, b}; int lens[2] = {4, 3};
    int n = kci_hybrid_rrf(lists, lens, NULL, 2, 0, 8, out, sc);
    CHECK(n == 5 && out[0] == 1 && out[1] == 3 && out[2] == 2 && out[3] == 5 && out[4] == 4,
          "rrf order: n=%d %llu %llu %llu %llu %llu", n, (unsigned long long)out[0], (unsigned long long)out[1],
          (unsigned long long)out[2], (unsigned long long)out[3], (unsigned long long)out[4]);
    CHECK(fabsf(sc[0] - (1.0f/61 + 1.0f/63)) < 1e-7f && fabsf(sc[4] - 1.0f/64) < 1e-7f, "rrf scores");
    float w[2] = {1.0f, 3.0f};   /* trust the sparse side more: 3 first, then 1, then 5 */
    n = kci_hybrid_rrf(lists, lens, w, 2, 60.0f, 3, out, sc);
    CHECK(n == 3 && out[0] == 3 && out[1] == 1 && out[2] == 5, "weighted rrf order");
    uint64_t rep[3] = {9, 9, 8}; const uint64_t *l1[1] = {rep}; int n1[1] = {3};
    n = kci_hybrid_rrf(l1, n1, NULL, 1, 60.0f, 8, out, sc);
    CHECK(n == 2 && out[0] == 9 && fabsf(sc[0] - 1.0f/61) < 1e-7f && out[1] == 8, "a repeat inside one list counts once");
    int zero[2] = {0, 0};
    CHECK(kci_hybrid_rrf(lists, zero, NULL, 2, 60.0f, 8, out, sc) == 0, "empty lists");
    int neg[2] = {4, -1};
    CHECK(kci_hybrid_rrf(lists, neg, NULL, 2, 60.0f, 8, out, sc) == -1, "negative length");
    CHECK(kci_hybrid_rrf(NULL, lens, NULL, 2, 60.0f, 8, out, sc) == -1, "NULL lists");
    CHECK(kci_hybrid_rrf(lists, lens, NULL, 2, 60.0f, 0, out, sc) == -1, "k 0");
    {
        float sa[4] = {0.9f, 0.8f, 0.5f, 0.1f}, sb[3] = {30.0f, 12.0f, 10.0f};
        const float *scs[2] = {sa, sb};
        n = kci_hybrid_mix(lists, scs, lens, NULL, 2, 8, out, sc);
        CHECK(n == 5 && out[0] == 3 && out[1] == 1 && out[2] == 2 && out[3] == 5 && out[4] == 4, "mix order: n=%d %llu %llu %llu",
              n, (unsigned long long)out[0], (unsigned long long)out[1], (unsigned long long)out[2]);
        CHECK(fabsf(sc[0] - 1.5f) < 1e-6f && fabsf(sc[1] - 1.0f) < 1e-6f && fabsf(sc[2] - 0.875f) < 1e-6f && fabsf(sc[3] - 0.1f) < 1e-6f && sc[4] == 0, "mix scores");
        float wm[2] = {1.0f, 0.2f};      /* the sparse side counts little: 1: 1.0   2: 0.875   3: 0.7 */
        n = kci_hybrid_mix(lists, scs, lens, wm, 2, 3, out, sc);
        CHECK(n == 3 && out[0] == 1 && out[1] == 2 && out[2] == 3 && fabsf(sc[2] - 0.7f) < 1e-6f, "weighted mix order");
        float da[4] = {0.1f, 0.2f, 0.5f, 0.9f};      /* a distance, smaller is better, still best first: the same stretch */
        const float *scd[2] = {da, sb};
        n = kci_hybrid_mix(lists, scd, lens, NULL, 2, 8, out, sc);
        CHECK(n == 5 && out[0] == 3 && out[1] == 1 && fabsf(sc[1] - 1.0f) < 1e-6f && fabsf(sc[2] - 0.875f) < 1e-6f, "mix with a distance on one side");
        float flat[3] = {2.0f, 2.0f, 2.0f}; const float *sf[1] = {flat};
        n = kci_hybrid_mix(l1, sf, n1, NULL, 1, 8, out, sc);
        CHECK(n == 2 && out[0] == 8 && out[1] == 9 && sc[0] == 1.0f && sc[1] == 1.0f, "mix: equal scores all count 1, a repeat once");
        CHECK(kci_hybrid_mix(lists, scs, zero, NULL, 2, 8, out, sc) == 0, "mix: empty lists");
        CHECK(kci_hybrid_mix(lists, scs, neg, NULL, 2, 8, out, sc) == -1, "mix: negative length");
        CHECK(kci_hybrid_mix(lists, NULL, lens, NULL, 2, 8, out, sc) == -1, "mix: NULL scores");
        CHECK(kci_hybrid_mix(lists, scs, lens, NULL, 2, 0, out, sc) == -1, "mix: k 0");
        float bad[3] = {30.0f, NAN, 10.0f}; const float *sn[2] = {sa, bad};
        CHECK(kci_hybrid_mix(lists, sn, lens, NULL, 2, 8, out, sc) == -1, "mix: a score that is not a number is refused");
    }
}

#define SV_PATH "test_sparse_tmp.kcs"
static long sv_size(const char *p){ FILE *f = fopen(p, "rb"); if (!f) return -1; fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f); return n; }
static int sv_poke(const char *p, long at, int xorv, int setv){   /* change one byte; returns the old one */
    FILE *f = fopen(p, "r+b"); if (!f) return -1;
    fseek(f, at, SEEK_SET); int c = fgetc(f);
    fseek(f, at, SEEK_SET); fputc(setv >= 0 ? setv : (c ^ xorv), f); fclose(f);
    return c;
}
/* every kind of search on a and b gives the same ids and scores */
static int sv_same(KciSparse *a, KciSparse *b, int blocks_too, int lists_too){
    uint32_t idx[64]; float val[64]; uint64_t ia[K], ib[K]; float sa[K], sb[K]; int differ = 0;
    KciSparseSearcher *A = kci_sparse_searcher_new(a), *B = kci_sparse_searcher_new(b);
    if (!A || !B){ kci_sparse_searcher_free(A); kci_sparse_searcher_free(B); return -1; }
    uint64_t keep = rs; rs = 1234567891234567ull;             /* the same queries at every call */
    for (int qi = 0; qi < 60; qi++){
        int qn = mkrow(idx, val, 40, 0);
        for (int mode = 0; mode < 3; mode++){
            if ((mode < 2 && !lists_too) || (mode == 2 && !blocks_too)) continue;
            int na = mode == 0 ? kci_sparse_search_exact(A, idx, val, qn, K, ia, sa) : mode == 1 ? kci_sparse_search(A, idx, val, qn, K, 3000, 60, ia, sa)
                               : kci_sparse_search_blocks(A, idx, val, qn, K, 5, 0.9f, 10, ia, sa);
            int nb = mode == 0 ? kci_sparse_search_exact(B, idx, val, qn, K, ib, sb) : mode == 1 ? kci_sparse_search(B, idx, val, qn, K, 3000, 60, ib, sb)
                               : kci_sparse_search_blocks(B, idx, val, qn, K, 5, 0.9f, 10, ib, sb);
            if (na != nb || na < 0) { differ++; continue; }
            for (int i = 0; i < na; i++) if (ia[i] != ib[i] || sa[i] != sb[i]) { differ++; break; }
        }
    }
    rs = keep;
    kci_sparse_searcher_free(A); kci_sparse_searcher_free(B);
    return differ;
}
static void sv_roundtrip(KciSparse *a, const char *what, int blocks_too, int lists_too){
    static const uint8_t key[] = "a pass phrase"; static const uint8_t other[] = "another phrase";
    for (int enc = 0; enc < 2; enc++){
        CHECK(kci_sparse_save(a, SV_PATH, enc ? key : NULL, enc ? 13 : 0) == 0, "%s: save (%s)", what, enc ? "key" : "plain");
        KciSparse *b = kci_sparse_load(SV_PATH, enc ? key : NULL, enc ? 13 : 0);
        CHECK(b != NULL, "%s: load (%s)", what, enc ? "key" : "plain");
        if (!b) continue;
        CHECK(kci_sparse_count(a) == kci_sparse_count(b) && kci_sparse_dim(a) == kci_sparse_dim(b) && kci_sparse_is_built(a) == kci_sparse_is_built(b)
              && kci_sparse_pending(a) == kci_sparse_pending(b) && kci_sparse_has_blocks(a) == kci_sparse_has_blocks(b) && kci_sparse_knn(a) == kci_sparse_knn(b),
              "%s: counts and state differ after a load", what);
        if (kci_sparse_is_built(a)){
            int d = sv_same(a, b, blocks_too, lists_too);
            CHECK(d == 0, "%s (%s): %d searches differ after a load", what, enc ? "key" : "plain", d);
        }
        kci_sparse_free(b);
        if (enc){
            CHECK(kci_sparse_load(SV_PATH, other, 14) == NULL, "%s: the wrong key must be refused", what);
            CHECK(kci_sparse_load(SV_PATH, key, 12) == NULL, "%s: a shorter key must be refused", what);
            CHECK(kci_sparse_load(SV_PATH, NULL, 0) == NULL, "%s: an encrypted file without a key must be refused", what);
        } else CHECK(kci_sparse_load(SV_PATH, key, 13) == NULL, "%s: a plain file must be refused when a key is given", what);
    }
}
static void saved(void){
    enum { SN = 1500, SI = 40 };
    static uint32_t ridx[SN + SI][64]; static float rval[SN + SI][64]; static int rn[SN + SI];
    static const uint8_t key[] = "a pass phrase";
    KciSparse *a = kci_sparse_new(DIM);
    for (int d = 0; d < SN + SI; d++) rn[d] = mkrow(ridx[d], rval[d], 60, d % 7 == 0);
    for (int d = 0; d < SN; d++) kci_sparse_add(a, 1000 + (uint64_t)d, ridx[d], rval[d], rn[d]);
    CHECK(kci_sparse_save(NULL, SV_PATH, NULL, 0) == -1 && kci_sparse_save(a, NULL, NULL, 0) == -1 && kci_sparse_save(a, SV_PATH, key, 0) == -1
          && kci_sparse_save(a, SV_PATH, NULL, 5) == -1, "save must refuse bad arguments");
    CHECK(kci_sparse_load("no_such_file.kcs", NULL, 0) == NULL && kci_sparse_load(NULL, NULL, 0) == NULL, "load must refuse a missing file");
    sv_roundtrip(a, "not built", 0, 0);
    { KciSparse *b = NULL; CHECK(kci_sparse_save(a, SV_PATH, NULL, 0) == 0 && (b = kci_sparse_load(SV_PATH, NULL, 0)) != NULL && kci_sparse_build(b, 2, NULL) == 0
            && kci_sparse_build(a, 2, NULL) == 0 && sv_same(a, b, 0, 1) == 0, "a file saved before the build builds to the same index"); kci_sparse_free(b); }
    sv_roundtrip(a, "lists only", 0, 1);
    kci_sparse_build_blocks(a, NULL, NULL);
    sv_roundtrip(a, "lists + groups", 1, 1);
    for (int d = 0; d < 30; d++) kci_sparse_delete(a, 1000 + (uint64_t)(d * 37));
    for (int d = SN; d < SN + SI; d++) kci_sparse_insert(a, 1000 + (uint64_t)d, ridx[d], rval[d], rn[d]);
    CHECK(kci_sparse_build_knn(a, 10, 0, 0, 2, NULL) == 0, "neighbour lists with inserted rows waiting");
    sv_roundtrip(a, "everything, with removed and inserted rows", 1, 1);

    /* the loaded index goes on working: insert, delete, rebuild */
    CHECK(kci_sparse_save(a, SV_PATH, key, 13) == 0, "save");
    KciSparse *b = kci_sparse_load(SV_PATH, key, 13);
    CHECK(b != NULL, "load");
    if (b){
        uint64_t id[1]; float sc[1];
        CHECK(kci_sparse_insert(b, 777777, ridx[5], rval[5], rn[5]) == kci_sparse_insert(a, 777777, ridx[5], rval[5], rn[5]), "insert after a load");
        CHECK(kci_sparse_delete(b, 1000 + 11) == 1 && kci_sparse_delete(a, 1000 + 11) == 1, "delete after a load");
        CHECK(sv_same(a, b, 1, 1) == 0, "searches differ after insert + delete on a loaded index");
        CHECK(kci_sparse_build(b, 2, NULL) == 0 && kci_sparse_build(a, 2, NULL) == 0 && sv_same(a, b, 0, 1) == 0, "rebuild of a loaded index");
        KciSparseSearcher *B = kci_sparse_searcher_new(b);
        CHECK(kci_sparse_search_exact(B, ridx[SN + 3], rval[SN + 3], rn[SN + 3], 1, id, sc) == 1, "search after the rebuild");
        kci_sparse_searcher_free(B); kci_sparse_free(b);
    }
    kci_sparse_build_blocks(a, NULL, NULL); kci_sparse_build_knn(a, 10, 0, 0, 2, NULL); kci_sparse_drop_lists(a);
    sv_roundtrip(a, "lists dropped", 1, 0);

    /* an altered or cut encrypted file is refused wherever the change is */
    CHECK(kci_sparse_save(a, SV_PATH, key, 13) == 0, "save");
    long sz = sv_size(SV_PATH); int accepted = 0;
    for (int i = 0; i < 60; i++){
        long at = i < 12 ? i * 9 : i == 59 ? sz - 1 : (long)(rnd() % (uint32_t)sz);
        int old = sv_poke(SV_PATH, at, 1 << (i % 8), -1);
        KciSparse *x = kci_sparse_load(SV_PATH, key, 13);
        if (x){ accepted++; kci_sparse_free(x); }
        sv_poke(SV_PATH, at, 0, old);
    }
    CHECK(accepted == 0, "an encrypted file with one bit changed was accepted (%d of 60 places)", accepted);
    { KciSparse *x = kci_sparse_load(SV_PATH, key, 13); CHECK(x != NULL, "the file restored loads again"); kci_sparse_free(x); }
    { FILE *f = fopen(SV_PATH, "ab"); fputc(0, f); fclose(f);
      CHECK(kci_sparse_load(SV_PATH, key, 13) == NULL, "a file one byte longer must be refused"); }
    { FILE *f = fopen(SV_PATH, "rb"), *g = fopen(SV_PATH ".cut", "wb"); long half = sz / 2;
      for (long i = 0; i < half; i++) fputc(fgetc(f), g);
      fclose(f); fclose(g);
      CHECK(kci_sparse_load(SV_PATH ".cut", key, 13) == NULL, "a file cut in half must be refused"); remove(SV_PATH ".cut"); }

    /* a damaged file without a key: refused, or still safe to search */
    kci_sparse_build(a, 2, NULL); kci_sparse_build_blocks(a, NULL, NULL); kci_sparse_build_knn(a, 10, 0, 0, 2, NULL);
    for (int d = SN; d < SN + 5; d++) kci_sparse_insert(a, 5000 + (uint64_t)d, ridx[d], rval[d], rn[d]);
    CHECK(kci_sparse_save(a, SV_PATH, NULL, 0) == 0, "save");
    sz = sv_size(SV_PATH); int loaded = 0, refused = 0;
    for (int i = 0; i < 400; i++){
        long at = i < 120 ? i : (long)(rnd() % (uint32_t)sz);
        int old = sv_poke(SV_PATH, at, 0, i % 3 == 0 ? 0xff : i % 3 == 1 ? 0x00 : (int)(rnd() & 0xff));
        KciSparse *x = kci_sparse_load(SV_PATH, NULL, 0);
        if (!x) refused++;
        else {
            loaded++;
            KciSparseSearcher *X = kci_sparse_searcher_new(x); uint64_t id[K]; float sc[K];
            for (int q = 0; X && q < 25; q++){
                int d = (q * 61 + i) % SN;
                kci_sparse_search_exact(X, ridx[d], rval[d], rn[d], K, id, sc);
                kci_sparse_search(X, ridx[d], rval[d], rn[d], K, 2000, 60, id, sc);
                kci_sparse_search_blocks(X, ridx[d], rval[d], rn[d], K, 5, 0.9f, 10, id, sc);
            }
            kci_sparse_searcher_free(X); kci_sparse_free(x);
        }
        sv_poke(SV_PATH, at, 0, old);
    }
    CHECK(refused > 50, "a damaged plain file: only %d of 400 changes were refused (%d loaded)", refused, loaded);
    printf("saved: a damaged file without a key: %d of 400 one-byte changes refused, %d loaded and searched\n", refused, loaded);
    remove(SV_PATH);
    kci_sparse_free(a);
}

static void rows_dropped(void){
    enum { SN = 1500, NQD = 60 };
    static uint32_t ridx[SN + 2][64]; static float rval[SN + 2][64]; static int rn[SN + 2];
    static uint64_t ia[NQD][K]; static float sa[NQD][K]; static int na[NQD];
    static const uint8_t key[] = "a pass phrase";
    uint64_t ib[K]; float sb[K];
    KciSparse *a = kci_sparse_new(DIM);
    for (int d = 0; d < SN + 2; d++) rn[d] = mkrow(ridx[d], rval[d], 60, d % 5 == 0);
    for (int d = 0; d < SN; d++) kci_sparse_add(a, 1000 + (uint64_t)d, ridx[d], rval[d], rn[d]);
    CHECK(kci_sparse_drop_rows(NULL) == -1 && kci_sparse_drop_rows(a) == -1, "drop_rows before the build must be -1");
    kci_sparse_build(a, 2, NULL);
    CHECK(kci_sparse_drop_rows(a) == -1 && kci_sparse_has_rows(a) == 1, "drop_rows without blocks must be -1");
    kci_sparse_build_blocks(a, NULL, NULL); kci_sparse_build_knn(a, 10, 0, 0, 2, NULL);
    kci_sparse_insert(a, 999, ridx[SN], rval[SN], rn[SN]);
    CHECK(kci_sparse_drop_rows(a) == -1 && kci_sparse_has_rows(a) == 1, "drop_rows with inserted rows waiting must be -1");
    kci_sparse_build(a, 2, NULL); kci_sparse_build_blocks(a, NULL, NULL); kci_sparse_build_knn(a, 10, 0, 0, 2, NULL);

    KciSparseSearcher *A = kci_sparse_searcher_new(a);
    for (int q = 0; q < NQD; q++){ int d = (q * 23) % SN; na[q] = kci_sparse_search_blocks(A, ridx[d], rval[d], rn[d], K, 5, 0.9f, 10, ia[q], sa[q]); }
    int64_t before = kci_sparse_bytes(a);
    CHECK(kci_sparse_drop_rows(a) == 0 && kci_sparse_has_rows(a) == 0, "drop_rows on a finished index");
    CHECK(kci_sparse_bytes(a) < before - (int64_t)SN * 8, "drop_rows frees the rows (%lld -> %lld bytes)", (long long)before, (long long)kci_sparse_bytes(a));
    CHECK(kci_sparse_drop_rows(a) == -1, "drop_rows twice must be -1");
    int moved = 0, off = 0;
    for (int q = 0; q < NQD; q++){
        int d = (q * 23) % SN;
        int nb = kci_sparse_search_blocks(A, ridx[d], rval[d], rn[d], K, 5, 0.9f, 10, ib, sb);
        if (nb != na[q]){ off++; continue; }
        for (int i = 0; i < nb; i++){
            float tol = 2e-3f * (fabsf(sa[q][i]) + 1.0f);
            if (fabsf(sa[q][i] - sb[i]) > tol) off++;        /* the score at each rank stays, whichever row holds it */
            if (ia[q][i] != ib[i]) moved++;
        }
    }
    CHECK(off == 0, "after drop_rows %d scores moved by more than the 16-bit weights allow", off);
    CHECK(moved < NQD * K / 20, "after drop_rows %d of %d places hold another row", moved, NQD * K);
    uint64_t one[1]; float sc1[1];
    CHECK(kci_sparse_add(a, 5, ridx[0], rval[0], rn[0]) == -1 && kci_sparse_insert(a, 5, ridx[0], rval[0], rn[0]) == -1
          && kci_sparse_build(a, 2, NULL) == -1 && kci_sparse_build_blocks(a, NULL, NULL) == -1 && kci_sparse_build_knn(a, 10, 0, 0, 2, NULL) == -1
          && kci_sparse_search_exact(A, ridx[0], rval[0], rn[0], 1, one, sc1) == -1 && kci_sparse_search(A, ridx[0], rval[0], rn[0], 1, 100, 10, one, sc1) == -1,
          "after drop_rows adding, building and the list searches must be -1");
    CHECK(kci_sparse_has_blocks(a) == 1 && kci_sparse_knn(a) == 10 && kci_sparse_is_built(a) == 1, "the refused calls must leave the index as it was");
    CHECK(kci_sparse_search_blocks(A, ridx[7], rval[7], rn[7], 1, 0, 1.0f, 10, one, sc1) == 1 && one[0] == 1007, "a row finds itself after drop_rows");
    CHECK(kci_sparse_delete(a, 1007) == 1 && kci_sparse_search_blocks(A, ridx[7], rval[7], rn[7], 1, 0, 1.0f, 10, one, sc1) == 1 && one[0] != 1007,
          "delete after drop_rows");
    kci_sparse_searcher_free(A);

    sv_roundtrip(a, "rows dropped", 1, 0);
    CHECK(kci_sparse_save(a, SV_PATH, key, 13) == 0, "save");
    { KciSparse *x = kci_sparse_load(SV_PATH, key, 13);
      CHECK(x != NULL && kci_sparse_has_rows(x) == 0 && kci_sparse_add(x, 5, ridx[0], rval[0], rn[0]) == -1 && kci_sparse_build(x, 2, NULL) == -1
            && kci_sparse_delete(x, 1008) == 1, "a loaded index without rows stays without rows");
      kci_sparse_free(x); }
    /* a damaged file without a key: refused, or still safe to search (the half-size rows are read as they are) */
    CHECK(kci_sparse_save(a, SV_PATH, NULL, 0) == 0, "save");
    long sz = sv_size(SV_PATH); int loaded = 0, refused = 0;
    for (int i = 0; i < 400; i++){
        long at = i < 120 ? i : i < 200 ? sz - 1 - (long)(rnd() % (uint32_t)(sz / 3)) : (long)(rnd() % (uint32_t)sz);
        int old = sv_poke(SV_PATH, at, 0, i % 3 == 0 ? 0xff : i % 3 == 1 ? 0x00 : (int)(rnd() & 0xff));
        KciSparse *x = kci_sparse_load(SV_PATH, NULL, 0);
        if (!x) refused++;
        else {
            loaded++;
            KciSparseSearcher *X = kci_sparse_searcher_new(x);
            for (int q = 0; X && q < 25; q++){ int d = (q * 61 + i) % SN; kci_sparse_search_blocks(X, ridx[d], rval[d], rn[d], K, 5, 0.9f, 10, ib, sb); }
            kci_sparse_searcher_free(X); kci_sparse_free(x);
        }
        sv_poke(SV_PATH, at, 0, old);
    }
    { KciSparse *w = kci_sparse_new(DIM); KciSparseBlockParams wp; memset(&wp, 0, sizeof wp); wp.head_mass = 1.0f; wp.threads = 2;
      for (int d = 0; d < SN; d++) kci_sparse_add(w, 1000 + (uint64_t)d, ridx[d], rval[d], rn[d]);
      kci_sparse_build(w, 2, NULL);
      CHECK(kci_sparse_build_blocks(w, &wp, NULL) == 0 && kci_sparse_build_knn(w, 10, 0, 0, 2, NULL) == 0 && kci_sparse_drop_rows(w) == 0, "whole rows: build and drop");
      sv_roundtrip(w, "rows dropped, whole rows", 1, 0);
      wp.head_mass = -0.5f;
      CHECK(kci_sparse_build_blocks(w, &wp, NULL) == -1, "a negative head_mass must be refused");
      kci_sparse_free(w); }
    CHECK(refused > 50, "a damaged plain file without rows: only %d of 400 changes were refused (%d loaded)", refused, loaded);
    printf("rows dropped: a damaged file without a key: %d of 400 one-byte changes refused, %d loaded and searched\n", refused, loaded);
    remove(SV_PATH);
    kci_sparse_free(a);
}

#define FX_DIM 600
#define FX_N   200
#define FX_KEY "fixture key 2026"
static int fx_row(int d, uint32_t *idx, float *val){
    uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t)(d + 1) + 12345;
    int n = 0; uint32_t k = 0;
    for (;;){
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        k += 1 + (uint32_t)(x >> 33) % 60;
        if (k >= FX_DIM || n >= 40) break;
        float v = 0.05f + (float)((x >> 12) % 1950) / 1000.0f;
        if (d % 6 == 0 && ((x >> 5) & 3) == 0) v = -v;
        idx[n] = k; val[n] = v; n++;
    }
    return n;
}
static int fx_same(KciSparse *a, KciSparse *b, int lists_too){
    uint32_t idx[64]; float val[64]; uint64_t ia[K], ib[K]; float sa[K], sb[K]; int differ = 0;
    KciSparseSearcher *A = kci_sparse_searcher_new(a), *B = kci_sparse_searcher_new(b);
    if (!A || !B){ kci_sparse_searcher_free(A); kci_sparse_searcher_free(B); return -1; }
    for (int qi = 0; qi < 80; qi++){
        int qn = fx_row(5000 + qi, idx, val);
        for (int mode = lists_too ? 0 : 2; mode < 4; mode++){
            int na = mode == 0 ? kci_sparse_search_exact(A, idx, val, qn, K, ia, sa) : mode == 1 ? kci_sparse_search(A, idx, val, qn, K, 300, 60, ia, sa)
                               : kci_sparse_search_blocks(A, idx, val, qn, K, mode == 2 ? 4 : 0, mode == 2 ? 0.9f : 1.0f, mode == 2 ? 6 : 0, ia, sa);
            int nb = mode == 0 ? kci_sparse_search_exact(B, idx, val, qn, K, ib, sb) : mode == 1 ? kci_sparse_search(B, idx, val, qn, K, 300, 60, ib, sb)
                               : kci_sparse_search_blocks(B, idx, val, qn, K, mode == 2 ? 4 : 0, mode == 2 ? 0.9f : 1.0f, mode == 2 ? 6 : 0, ib, sb);
            if (na != nb || na <= 0){ differ++; continue; }
            for (int i = 0; i < na; i++) if (ia[i] != ib[i] || sa[i] != sb[i]){ differ++; break; }
        }
    }
    kci_sparse_searcher_free(A); kci_sparse_searcher_free(B);
    return differ;
}
static void old_files(void){
    uint32_t idx[64]; float val[64];
    KciSparse *a = kci_sparse_new(FX_DIM);
    for (int d = 0; d < FX_N; d++){ int n = fx_row(d, idx, val); kci_sparse_add(a, 100 + (uint64_t)d, idx, val, n); }
    KciSparseBlockParams bp; memset(&bp, 0, sizeof bp); bp.list_len = 40; bp.block_size = 5; bp.threads = 1;
    bp.head_mass = 1.0f;                                     /* whole rows, as the version that wrote the files scored them */
    kci_sparse_build(a, 1, NULL); kci_sparse_build_blocks(a, &bp, NULL); kci_sparse_build_knn(a, 6, 0, 0, 1, NULL);
    for (int d = 0; d < 5; d++) kci_sparse_delete(a, 100 + (uint64_t)(d * 37));
    for (int d = FX_N; d < FX_N + 3; d++){ int n = fx_row(d, idx, val); kci_sparse_insert(a, 100 + (uint64_t)d, idx, val, n); }
    for (int enc = 0; enc < 2; enc++){
        const char *path = enc ? "tests/data/sparse_v1_key.kcs" : "tests/data/sparse_v1_plain.kcs";
        KciSparse *b = kci_sparse_load(path, enc ? (const uint8_t *)FX_KEY : NULL, enc ? (int)strlen(FX_KEY) : 0);
        CHECK(b != NULL, "%s (written by the first version of the format) must load — run from the repository root", path);
        if (!b) continue;
        CHECK(kci_sparse_count(b) == kci_sparse_count(a) && kci_sparse_pending(b) == 3 && kci_sparse_knn(b) == 6 && kci_sparse_has_blocks(b) == 1
              && kci_sparse_has_rows(b) == 1, "%s: counts and state", path);
        int d = fx_same(a, b, 1);
        CHECK(d == 0, "%s: %d searches differ from the index built now", path, d);
        /* and it can be saved again (in today's format) and read back to the same */
        CHECK(kci_sparse_save(b, SV_PATH, NULL, 0) == 0, "%s: save again", path);
        KciSparse *c = kci_sparse_load(SV_PATH, NULL, 0);
        CHECK(c != NULL && fx_same(a, c, 1) == 0, "%s: saved again and loaded, searches differ", path);
        kci_sparse_free(c);
        CHECK(kci_sparse_build(b, 1, NULL) == 0 && kci_sparse_build_blocks(b, &bp, NULL) == 0, "%s: rebuild after the load", path);
        kci_sparse_free(b);
        if (enc) CHECK(kci_sparse_load(path, (const uint8_t *)"wrong key", 9) == NULL, "%s: the wrong key must be refused", path);
    }
    /* a damaged file in the old layout goes through the conversion: refused, or still safe to search */
    { FILE *f = fopen("tests/data/sparse_v1_plain.kcs", "rb"), *g = fopen(SV_PATH, "wb"); int c;
      if (f && g) while ((c = fgetc(f)) != EOF) fputc(c, g);
      if (f) fclose(f);
      if (g) fclose(g); }
    long sz = sv_size(SV_PATH); int loaded = 0, refused = 0; uint64_t ib[K]; float sb[K];
    for (int i = 0; sz > 0 && i < 500; i++){
        long at = i < 120 ? i : i < 400 ? sz - 1 - (long)(rnd() % (uint32_t)(sz * 3 / 4)) : (long)(rnd() % (uint32_t)sz);
        int old = sv_poke(SV_PATH, at, 0, i % 3 == 0 ? 0xff : i % 3 == 1 ? 0x00 : (int)(rnd() & 0xff));
        KciSparse *x = kci_sparse_load(SV_PATH, NULL, 0);
        if (!x) refused++;
        else {
            loaded++;
            KciSparseSearcher *X = kci_sparse_searcher_new(x);
            for (int q = 0; X && q < 25; q++){
                int qn = fx_row(7000 + q + i, idx, val);
                kci_sparse_search_exact(X, idx, val, qn, K, ib, sb);
                kci_sparse_search_blocks(X, idx, val, qn, K, 4, 0.9f, 6, ib, sb);
                kci_sparse_search_blocks(X, idx, val, qn, K, 0, 1.0f, 0, ib, sb);
            }
            kci_sparse_searcher_free(X); kci_sparse_free(x);
        }
        sv_poke(SV_PATH, at, 0, old);
    }
    CHECK(sz > 0 && refused > 50, "a damaged file in the old layout: only %d of 500 changes were refused (%d loaded)", refused, loaded);
    printf("old files: a damaged file in the old layout: %d of 500 one-byte changes refused, %d loaded and searched\n", refused, loaded);
    remove(SV_PATH);
    kci_sparse_free(a);
}

static void pool_small(void){
    uint32_t idx[64]; float val[64]; uint64_t ids[K], pids[K]; float sc[K], psc[K];
    KciSparseBlockParams bp; memset(&bp, 0, sizeof bp); bp.threads = 2; bp.head_mass = 0.5f;
    KciSparse *s = kci_sparse_new(DIM);
    for (uint32_t d = 0; d < 300; d++){ int n = mkrow(idx, val, 60, 0); kci_sparse_add(s, 100 + (uint64_t)d, idx, val, n); }
    kci_sparse_build(s, 2, NULL);
    CHECK(kci_sparse_build_blocks(s, &bp, NULL) == 0, "pool: build_blocks");
    KciSparseSearcher *S = kci_sparse_searcher_new(s);
    CHECK(kci_sparse_drop_rows(s) == 0, "pool: drop_rows");
    int cut = 0, partial = 0;
    for (int qi = 0; qi < 20; qi++){
        int qn = mkrow(idx, val, 40, 0);
        kci_sparse_searcher_set_pool(S, 0);
        int n = kci_sparse_search_blocks(S, idx, val, qn, K, 0, 1.0f, 0, ids, sc);
        kci_sparse_searcher_set_pool(S, 3);
        int np = kci_sparse_search_blocks(S, idx, val, qn, K, 0, 1.0f, 0, pids, psc);
        if (n <= 0 || np != n){ cut++; continue; }
        for (int i = 0; i < np; i++)
            for (int j = 0; j < n; j++) if (pids[i] == ids[j] && psc[i] != sc[j]){ partial++; break; }
    }
    CHECK(cut == 0, "a pool set below k: fewer rows returned in %d of 20 queries", cut);
    CHECK(partial == 0, "a pool set below k: %d rows came back with another score than with the default pool", partial);
    kci_sparse_searcher_set_pool(S, 0);
    kci_sparse_searcher_free(S); kci_sparse_free(s);

    enum { TN = 8, TQ = 20 };                                  /* fewer rows than k */
    static uint64_t eid[TQ][K]; static float esc[TQ][K]; int ne[TQ];
    s = kci_sparse_new(DIM);
    for (uint32_t d = 0; d < TN; d++){ int n = mkrow(idx, val, 60, 0); kci_sparse_add(s, 100 + (uint64_t)d, idx, val, n); }
    kci_sparse_build(s, 2, NULL);
    CHECK(kci_sparse_build_blocks(s, &bp, NULL) == 0 && kci_sparse_build_knn(s, 4, 0, 0, 2, NULL) == 0, "pool: a tiny index");
    S = kci_sparse_searcher_new(s);
    uint64_t save = rs;
    for (int qi = 0; qi < TQ; qi++){ int qn = mkrow(idx, val, 40, 0); ne[qi] = kci_sparse_search_exact(S, idx, val, qn, K, eid[qi], esc[qi]); }
    CHECK(kci_sparse_drop_rows(s) == 0, "pool: drop_rows on the tiny index");
    rs = save; int wrong = 0;
    for (int qi = 0; qi < TQ; qi++){
        int qn = mkrow(idx, val, 40, 0);
        /* without the neighbour lists: they also bring in rows that share no dimension with the query (score 0),
         * which the exact search does not return */
        int n = kci_sparse_search_blocks(S, idx, val, qn, K, 0, 4.0f, 0, ids, sc);
        if (n != ne[qi]){ wrong++; continue; }
        for (int i = 0; i < n; i++){
            int e = 0; while (e < ne[qi] && eid[qi][e] != ids[i]) e++;
            if (e == ne[qi] || fabsf(sc[i] - esc[qi][e]) > 1e-3f * (1.0f + fabsf(esc[qi][e]))){ wrong++; break; }
        }
    }
    CHECK(wrong == 0, "rows dropped, fewer rows than k: %d of %d queries differ from the exact scores", wrong, TQ);
    kci_sparse_searcher_free(S); kci_sparse_free(s);
}

static void head_saved(void){
    static const float hm[] = { 0.7345f, 0.999f, 0.001f, 0.5f, 1.0f };
    uint32_t idx[64]; float val[64];
    for (int v = 0; v < 5; v++){
        for (int drop = 0; drop < 2; drop++){
            KciSparse *s = kci_sparse_new(DIM); uint64_t keep = rs; rs = 424242 + (uint64_t)v;
            for (uint32_t d = 0; d < 800; d++){ int n = mkrow(idx, val, 60, 0); kci_sparse_add(s, 100 + (uint64_t)d, idx, val, n); }
            rs = keep;
            KciSparseBlockParams bp; memset(&bp, 0, sizeof bp); bp.threads = 2; bp.head_mass = hm[v];
            kci_sparse_build(s, 2, NULL);
            CHECK(kci_sparse_build_blocks(s, &bp, NULL) == 0 && kci_sparse_build_knn(s, 8, 0, 0, 2, NULL) == 0, "head_mass %.4f: build", hm[v]);
            if (drop) CHECK(kci_sparse_drop_rows(s) == 0, "head_mass %.4f: drop_rows", hm[v]);
            CHECK(kci_sparse_save(s, SV_PATH, NULL, 0) == 0, "head_mass %.4f%s: save", hm[v], drop ? ", no rows" : "");
            KciSparse *b = kci_sparse_load(SV_PATH, NULL, 0);
            CHECK(b != NULL, "head_mass %.4f%s: the saved file must load", hm[v], drop ? ", no rows" : "");
            if (b){ int d = sv_same(s, b, 1, 0); CHECK(d == 0, "head_mass %.4f%s: %d searches differ after a load", hm[v], drop ? ", no rows" : "", d); }
            kci_sparse_free(b); kci_sparse_free(s);
        }
    }
    remove(SV_PATH);
}

int main(void){
    dense = malloc(sizeof(float) * N * DIM);
    if (!dense) return 2;
    run(0);
    run(1);
    bad_input();
    removal();
    blocks(0);
    blocks(1);
    filtered();
    inserted();
    saved();
    rows_dropped();
    pool_small();
    head_saved();
    old_files();
    fusion();
    seal();
    params_size();
    wide_vocab();
    free(dense);
    printf(fails ? "test_sparse: %d FAILED\n" : "test_sparse: all OK\n", fails);
    return fails ? 1 : 0;
}
