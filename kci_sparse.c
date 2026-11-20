/*
 * kci_sparse.c — sparse-vector index (inner product) + rank fusion with a dense index.
 * See kci_sparse.h for the API. No dependencies, does not include kci_vec (kci_crypt.h is shared with it).
 */
#include "kci_sparse.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#ifdef _WIN32
#include "kci_win_compat.h"
#else
#include <unistd.h>
#endif
#include <pthread.h>
#include "kci_crypt.h"
#if defined(__AVX2__) && !defined(KCI_SPARSE_NO_SIMD)
#include <immintrin.h>
#define SP_AVX2 1
#endif

typedef struct { uint32_t k; float v; } SpPair;      /* (dimension, weight) in a row; (row, weight) in a list */
#define SP_HF_MAX 4.0f
#ifndef SP_PF_MAX
#define SP_PF_MAX 1024        /* bytes of a half-size row asked for ahead (a row of 227 pairs is 940) */
#endif
#define SP_CROW_PAD 32        /* entries after the last row, so that a 32-byte load at the end of a row stays inside */
typedef struct { uint32_t ep; float v; } SpSlot;     /* per-row running score; valid when ep == the query's epoch */
typedef struct { float s; uint32_t d; } SpCand;

struct KciSparse {
    uint32_t dim, n, ncap;
    uint64_t *ids;          /* [ncap] caller's id of each row */
    uint8_t  *dead;         /* [ncap] 1 = removed by kci_sparse_delete, still in f / post until the next build */
    uint32_t  ndead;
    uint32_t *map;          /* id -> row + 1, open addressing, made by the first delete (NULL = not made) */
    uint32_t  mapcap;       /* a power of two, >= 2 * n */
    uint64_t *row;          /* [ncap+1] offset of each row in f */
    SpPair   *f;            /* rows, dimensions ascending inside a row */
    uint64_t  fnnz, fcap;
    uint64_t *tptr;         /* [dim+1] offset of each list in post (NULL until built) */
    SpPair   *post;         /* lists, largest |weight| first */
    int       built;
    uint32_t  nbuilt;       /* rows [0, nbuilt) are in the lists; rows [nbuilt, n) came by kci_sparse_insert since
                               the last build and are scored one by one in every search */
    uint32_t  gen;          /* bumped by every build: a searcher from an older build is refused */
    SpPair   *tmp; int tmpcap;
    /* blocks (kci_sparse_build_blocks); bptr == NULL = not built */
    uint64_t *bptr;         /* [dim+1] first group of each dimension */
    uint64_t *bdoc;         /* [nblk+1] offset of each group's rows in brow */
    uint32_t *brow;         /* rows of every group */
    uint64_t *sdp;          /* [dim+1] first run of list t (in sdim / slen) */
    uint16_t *sdim16; uint32_t *sdim32;   /* dimension of each run (16-bit when dim <= 65536), ascending inside a list */
    uint8_t  *slen;         /* pairs in each run, 1..255 */
    uint64_t *sck;          /* [runs / 16 + 1] first pair of run 16 * i; made from slen, not saved */
    uint64_t *sep;          /* [dim+1] first pair of list t */
    uint8_t  *ent;          /* the pairs: ent[2 * i] = step, ent[2 * i + 1] = byte */
    float    *bscale;       /* [nblk] */
    uint64_t  nblk;
    float head_mass;        /* share of a row's weight in the first part of its half-size copy; 1 = all of it */
    int norows;             /* kci_sparse_drop_rows: f is gone, the half-size rows are the only copy */
    uint32_t *boff;
    uint32_t pf_bytes;      /* bytes asked for ahead when a row's own count is not at hand (a neighbour): the average first part + a line */
    uint8_t  *bhl;          /* beside boff: how many 64-byte lines the first part of that row touches (255 = or more) */
    uint16_t *crow; uint64_t *coff; uint32_t ncrow;   /* coff[ncrow + 1]; rows [0, ncrow) have a half-size copy */
    uint32_t  nknn_rows;    /* rows that have a neighbour list (n when the lists were made) */
    uint32_t *nbr;          /* [nknn_rows * nknn] neighbour rows of each row, most similar first, UINT32_MAX = none */
    int       nknn;
};

struct KciSparseSearcher {
    KciSparse *s;
    uint32_t   gen, ep, nslot;
    SpSlot    *slot;        /* [n] */
    uint32_t  *touched;     /* [n] rows met by this query */
    uint64_t  *vis; uint32_t nvis;
    float     *qd;          /* [dim] the query spread out, zero outside a call */
    uint16_t  *q16; float *qv16; int q16cap; uint32_t m16;   /* the query, dimensions ascending, for sp_crow_dot */
    SpPair    *q;   uint64_t *cut;  int qcap;
    SpCand    *heap; int hcap;
    uint32_t  *seed; int seedcap;
    float     *bacc; SpCand *bcand; uint32_t bcap;      /* bounds of the groups of one list */
    SpCand *pool; int pn, pcap, pall, pool_want;
    KciSparseFilter filt; void *fctx;
    uint64_t   work[3];
};

static double sp_now(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int sp_cmp_key(const void *a, const void *b){
    uint32_t x = ((const SpPair*)a)->k, y = ((const SpPair*)b)->k;
    return x < y ? -1 : x > y;
}
/* largest |weight| first; equal weights keep row order so a build is repeatable */
static int sp_cmp_post(const void *a, const void *b){
    const SpPair *x = a, *y = b;
    float fx = fabsf(x->v), fy = fabsf(y->v);
    if (fx != fy) return fx > fy ? -1 : 1;
    return x->k < y->k ? -1 : x->k > y->k;
}

/* Sort pairs by dimension, sum repeats, drop zeros. Returns the new length, -1 if a sum overflowed. */
static int sp_canon(SpPair *p, int n){
    if (n > 1) qsort(p, (size_t)n, sizeof *p, sp_cmp_key);
    int m = 0;
    for (int i = 0; i < n; ){
        uint32_t k = p[i].k; float v = 0;
        for (; i < n && p[i].k == k; i++) v += p[i].v;
        if (!isfinite(v)) return -1;
        if (v != 0){ p[m].k = k; p[m].v = v; m++; }
    }
    return m;
}

/* Check and copy caller's pairs into *buf (grown as needed), canonical. Returns length or -1. */
static int sp_take(SpPair **buf, int *cap, uint32_t dim, const uint32_t *idx, const float *val, int nnz){
    if (nnz < 0 || (nnz > 0 && (!idx || !val))) return -1;
    for (int i = 0; i < nnz; i++) if (idx[i] >= dim || !isfinite(val[i])) return -1;
    if (nnz > *cap){
        int nc = nnz < 64 ? 64 : nnz;
        SpPair *nb = realloc(*buf, (size_t)nc * sizeof **buf);
        if (!nb) return -1;
        *buf = nb; *cap = nc;
    }
    for (int i = 0; i < nnz; i++){ (*buf)[i].k = idx[i]; (*buf)[i].v = val[i]; }
    return sp_canon(*buf, nnz);
}

KciSparse *kci_sparse_new(uint32_t dim){
    if (dim == 0) return NULL;
    KciSparse *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->dim = dim; s->head_mass = 1.0f;
    s->row = calloc(1, sizeof *s->row);   /* row[0] = 0 */
    if (!s->row){ free(s); return NULL; }
    return s;
}

static void sp_blocks_free(KciSparse *s){
    free(s->bptr); free(s->bdoc); free(s->brow); free(s->bscale);
    free(s->sdp); free(s->sdim16); free(s->sdim32); free(s->slen); free(s->sck); free(s->sep); free(s->ent);
    s->bptr = NULL; s->bdoc = NULL; s->brow = NULL; s->bscale = NULL;
    s->sdp = NULL; s->sdim16 = NULL; s->sdim32 = NULL; s->slen = NULL; s->sck = NULL; s->sep = NULL; s->ent = NULL;
    s->nblk = 0;
    free(s->nbr); s->nbr = NULL; s->nknn = 0; s->nknn_rows = 0;   /* made with the blocks, dropped with them */
    free(s->crow); free(s->coff); free(s->boff); free(s->bhl); s->crow = NULL; s->coff = NULL; s->boff = NULL; s->bhl = NULL;
}

void kci_sparse_free(KciSparse *s){
    if (!s) return;
    sp_blocks_free(s);
    free(s->dead); free(s->map);
    free(s->ids); free(s->row); free(s->f); free(s->tptr); free(s->post); free(s->tmp);
    free(s);
}

static inline uint32_t sp_hash(uint64_t x){
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 27; x *= 0x94d049bb133111ebull; x ^= x >> 31;
    return (uint32_t)x;
}
static void sp_map_put(KciSparse *s, uint32_t row){
    uint32_t mask = s->mapcap - 1, h = sp_hash(s->ids[row]) & mask;
    while (s->map[h]) h = (h + 1) & mask;
    s->map[h] = row + 1;
}
static int sp_map_make(KciSparse *s){
    uint32_t cap = 1024;
    while ((uint64_t)cap < (uint64_t)s->n * 2 && cap < 0x80000000u) cap <<= 1;
    if ((uint64_t)cap < (uint64_t)s->n + 1) return -1;       /* must keep one empty slot to stop a probe */
    uint32_t *m = calloc(cap, sizeof *m);
    if (!m) return -1;
    free(s->map); s->map = m; s->mapcap = cap;
    for (uint32_t d = 0; d < s->n; d++) sp_map_put(s, d);
    return 0;
}

int kci_sparse_delete(KciSparse *s, uint64_t id){
    if (!s) return -1;
    if (s->n == 0) return 0;
    if (!s->map && sp_map_make(s) != 0) return -1;
    int removed = 0;
    uint32_t mask = s->mapcap - 1;
    for (uint32_t h = sp_hash(id) & mask; s->map[h]; h = (h + 1) & mask){
        uint32_t d = s->map[h] - 1;
        if (s->ids[d] == id && !s->dead[d]){ s->dead[d] = 1; s->ndead++; removed++; }
    }
    return removed;
}

/* Drop the removed rows from ids / row / f for good. Called by build. */
static void sp_compact(KciSparse *s){
    uint32_t w = 0; uint64_t fw = 0;
    for (uint32_t d = 0; d < s->n; d++){
        if (s->dead[d]) continue;
        uint64_t a = s->row[d], len = s->row[d + 1] - a;
        if (fw != a && len) memmove(s->f + fw, s->f + a, (size_t)len * sizeof *s->f);
        s->ids[w] = s->ids[d]; s->row[w] = fw; fw += len; w++;
    }
    s->row[w] = fw; s->n = w; s->fnnz = fw; s->ndead = 0;
    if (w) memset(s->dead, 0, w);
    free(s->map); s->map = NULL; s->mapcap = 0;              /* row numbers changed */
}

static int64_t sp_append(KciSparse *s, uint64_t id, const uint32_t *idx, const float *val, int nnz, int keep_built){
    if (s && s->norows) return -1;                           /* the rows were given back: nothing can be added or built */
    if (!s || s->n == UINT32_MAX) return -1;
    int m = sp_take(&s->tmp, &s->tmpcap, s->dim, idx, val, nnz);
    if (m < 0) return -1;
    if (s->n == s->ncap){
        uint64_t nc = s->ncap ? (uint64_t)s->ncap * 2 : 1024;
        if (nc > UINT32_MAX) nc = UINT32_MAX;
        uint64_t *ni = realloc(s->ids, nc * sizeof *ni);
        if (!ni) return -1;
        s->ids = ni;
        uint8_t *nd = realloc(s->dead, nc);
        if (!nd) return -1;
        s->dead = nd;
        uint64_t *nr = realloc(s->row, (nc + 1) * sizeof *nr);
        if (!nr) return -1;
        s->row = nr; s->ncap = (uint32_t)nc;
    }
    if (s->fnnz + (uint64_t)m > s->fcap){
        uint64_t nc = s->fcap ? s->fcap * 2 : 65536;
        while (nc < s->fnnz + (uint64_t)m) nc *= 2;
        SpPair *nf = realloc(s->f, nc * sizeof *nf);
        if (!nf) return -1;
        s->f = nf; s->fcap = nc;
    }
    if (m) memcpy(s->f + s->fnnz, s->tmp, (size_t)m * sizeof *s->f);
    s->fnnz += (uint64_t)m;
    s->ids[s->n] = id; s->dead[s->n] = 0;
    s->row[++s->n] = s->fnnz;
    if (s->map){
        if ((uint64_t)s->n * 2 > s->mapcap){ free(s->map); s->map = NULL; s->mapcap = 0; }   /* remade by the next delete */
        else sp_map_put(s, s->n - 1);
    }
    if (!keep_built) s->built = 0;
    return (int64_t)(s->n - s->ndead);
}

int64_t kci_sparse_add(KciSparse *s, uint64_t id, const uint32_t *idx, const float *val, int nnz){
    return sp_append(s, id, idx, val, nnz, 0);
}

int64_t kci_sparse_insert(KciSparse *s, uint64_t id, const uint32_t *idx, const float *val, int nnz){
    if (!s || !s->built) return -1;
    return sp_append(s, id, idx, val, nnz, 1);
}
uint32_t kci_sparse_pending(const KciSparse *s){ return s && s->built ? s->n - s->nbuilt : 0; }

typedef struct { KciSparse *s; uint32_t next; } SpSortJob;
static void *sp_sort_worker(void *arg){
    SpSortJob *j = arg; KciSparse *s = j->s;
    for (;;){
        uint32_t t0 = __sync_fetch_and_add(&j->next, 256u);
        if (t0 >= s->dim) break;
        uint32_t t1 = s->dim - t0 > 256u ? t0 + 256u : s->dim;
        for (uint32_t t = t0; t < t1; t++){
            uint64_t len = s->tptr[t + 1] - s->tptr[t];
            if (len > 1) qsort(s->post + s->tptr[t], (size_t)len, sizeof *s->post, sp_cmp_post);
        }
    }
    return NULL;
}

int kci_sparse_build(KciSparse *s, int threads, double *out_secs){
    if (!s) return -1;
    if (s->norows) return -1;
    double t0 = sp_now();
    if (threads <= 0){ long c = sysconf(_SC_NPROCESSORS_ONLN); threads = c > 0 ? (int)c : 1; }
    if (threads > 64) threads = 64;
    free(s->tptr); free(s->post); s->tptr = NULL; s->post = NULL; s->built = 0;
    sp_blocks_free(s);
    if (s->ndead) sp_compact(s);

    if (s->fcap > s->fnnz && s->fnnz){
        SpPair *nf = realloc(s->f, (size_t)s->fnnz * sizeof *nf);
        if (nf){ s->f = nf; s->fcap = s->fnnz; }
    }
    uint64_t *tptr = calloc((size_t)s->dim + 1, sizeof *tptr);
    uint64_t *cur  = malloc((size_t)s->dim * sizeof *cur);
    SpPair   *post = malloc((size_t)(s->fnnz ? s->fnnz : 1) * sizeof *post);
    if (!tptr || !cur || !post){ free(tptr); free(cur); free(post); return -1; }
    for (uint64_t i = 0; i < s->fnnz; i++) tptr[s->f[i].k + 1]++;
    for (uint32_t t = 0; t < s->dim; t++){ tptr[t + 1] += tptr[t]; cur[t] = tptr[t]; }
    for (uint32_t d = 0; d < s->n; d++)
        for (uint64_t i = s->row[d]; i < s->row[d + 1]; i++){
            SpPair *p = &post[cur[s->f[i].k]++];
            p->k = d; p->v = s->f[i].v;
        }
    free(cur);
    s->tptr = tptr; s->post = post;

    SpSortJob job = { s, 0 };
    pthread_t th[64]; int started = 0;
    for (int i = 1; i < threads; i++){
        if (pthread_create(&th[started], NULL, sp_sort_worker, &job) != 0) break;
        started++;
    }
    sp_sort_worker(&job);
    for (int i = 0; i < started; i++) pthread_join(th[i], NULL);

    s->gen++; s->built = 1; s->nbuilt = s->n;
    if (out_secs) *out_secs = sp_now() - t0;
    return 0;
}

KciSparseSearcher *kci_sparse_searcher_new(KciSparse *s){
    if (!s || !s->built) return NULL;
    KciSparseSearcher *S = calloc(1, sizeof *S);
    if (!S) return NULL;
    S->s = s; S->gen = s->gen;
    size_t n = s->n ? s->n : 1;
    S->nslot = (uint32_t)n;
    S->slot    = calloc(n, sizeof *S->slot);
    S->touched = malloc(n * sizeof *S->touched);
    S->vis     = calloc(n / 64 + 1, sizeof *S->vis);
    S->qd      = calloc(s->dim, sizeof *S->qd);
    if (!S->slot || !S->touched || !S->qd || !S->vis){ kci_sparse_searcher_free(S); return NULL; }
    return S;
}

void kci_sparse_searcher_free(KciSparseSearcher *S){
    if (!S) return;
    free(S->slot); free(S->touched); free(S->vis); free(S->qd); free(S->q16); free(S->qv16); free(S->pool); free(S->q); free(S->cut); free(S->heap); free(S->seed); free(S->bacc); free(S->bcand);
    free(S);
}

static inline int sp_before(SpCand a, SpCand b){ return a.s > b.s || (a.s == b.s && a.d < b.d); }
static int sp_cmp_cand(const void *a, const void *b){
    SpCand x = *(const SpCand*)a, y = *(const SpCand*)b;
    return sp_before(x, y) ? -1 : sp_before(y, x);
}
/* heap[0] is the candidate that ranks last among the kept ones */
static inline void sp_heap_down(SpCand *h, int n, int i){
    for (;;){
        int l = 2 * i + 1, r = l + 1, w = i;
        if (l < n && sp_before(h[w], h[l])) w = l;
        if (r < n && sp_before(h[w], h[r])) w = r;
        if (w == i) return;
        SpCand t = h[i]; h[i] = h[w]; h[w] = t; i = w;
    }
}
/* the other way round: heap[0] is the candidate that ranks first */
static inline void sp_best_down(SpCand *h, uint32_t n, uint32_t i){
    for (;;){
        uint32_t l = 2 * i + 1, r = l + 1, w = i;
        if (l < n && sp_before(h[l], h[w])) w = l;
        if (r < n && sp_before(h[r], h[w])) w = r;
        if (w == i) return;
        SpCand t = h[i]; h[i] = h[w]; h[w] = t; i = w;
    }
}
static inline void sp_heap_offer(SpCand *h, int *n, int cap, SpCand c){
    if (*n < cap){
        int i = (*n)++; h[i] = c;
        while (i > 0){
            int p = (i - 1) / 2;
            if (!sp_before(h[p], h[i])) break;
            SpCand t = h[i]; h[i] = h[p]; h[p] = t; i = p;
        }
    } else if (sp_before(c, h[0])){ h[0] = c; sp_heap_down(h, cap, 0); }
}

void kci_sparse_searcher_set_filter(KciSparseSearcher *S, KciSparseFilter fn, void *ctx){
    if (!S) return;
    S->filt = fn; S->fctx = fn ? ctx : NULL;
}

/* Offer a row that passed everything but the filter: the filter (a call into the caller's code) is asked
 * only if the row would enter. */
static inline void sp_offer_filtered(const KciSparseSearcher *S, SpCand *h, int *hn, int cap, SpCand c){
    if (S->filt){
        if (*hn == cap && !sp_before(c, h[0])) return;
        if (!S->filt(S->s->ids[c.d], S->fctx)) return;
    }
    sp_heap_offer(h, hn, cap, c);
}

/* The rows inserted since the last build: score each exactly (S->qd must hold the query) and offer it. */
static void sp_offer_pending(const KciSparseSearcher *S, SpCand *h, int *hn, int cap){
    const KciSparse *s = S->s; const float *qd = S->qd;
    for (uint32_t d = s->nbuilt; d < s->n; d++){
        if (s->dead[d]) continue;
        float sum = 0; int share = 0;
        for (uint64_t y = s->row[d]; y < s->row[d + 1]; y++){
            float qv = qd[s->f[y].k];
            if (qv != 0){ sum += qv * s->f[y].v; share = 1; }
        }
        if (!share) continue;                                 /* like the lists: only rows that share a dimension */
        SpCand c = { sum, d };
        sp_offer_filtered(S, h, hn, cap, c);
    }
}

/* How many pairs at the front of a list (largest |weight| first) have |weight| >= x. */
static inline uint64_t sp_front(const SpPair *p, uint64_t len, float x){
    uint64_t lo = 0, hi = len;
    while (lo < hi){
        uint64_t mid = lo + (hi - lo) / 2;
        if (fabsf(p[mid].v) >= x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* Choose how far to walk each list so that about `budget` pairs are read in total, and they are the
 * pairs with the largest |weight_q * weight_row| over all the query's lists: find the level by
 * bisection, each list answers "how many of mine are above it" by a binary search. Returns 1 if any
 * list was cut short. */
static int sp_cuts(const KciSparse *s, const SpPair *q, uint64_t *cut, int m, uint32_t budget){
    uint64_t total = 0; float top = 0;
    for (int i = 0; i < m; i++){
        uint64_t len = s->tptr[q[i].k + 1] - s->tptr[q[i].k];
        cut[i] = len; total += len;
        if (len){ float c = fabsf(q[i].v) * fabsf(s->post[s->tptr[q[i].k]].v); if (c > top) top = c; }
    }
    if (budget == 0 || total <= budget) return 0;
    float lo = 0, hi = top;                 /* above(lo) > budget; hi is the answer so far */
    for (int it = 0; it < 20; it++){
        float mid = 0.5f * (lo + hi);
        if (!(mid > lo && mid < hi)) break;
        uint64_t above = 0;
        for (int i = 0; i < m && above <= budget; i++)
            above += sp_front(s->post + s->tptr[q[i].k], s->tptr[q[i].k + 1] - s->tptr[q[i].k], mid / fabsf(q[i].v));
        if (above > budget) lo = mid; else hi = mid;
    }
    for (int i = 0; i < m; i++)
        cut[i] = sp_front(s->post + s->tptr[q[i].k], s->tptr[q[i].k + 1] - s->tptr[q[i].k], hi / fabsf(q[i].v));
    return 1;
}

int kci_sparse_search(KciSparseSearcher *S, const uint32_t *qidx, const float *qval, int qnnz,
                      int k, uint32_t budget, int rerank_n, uint64_t *out_ids, float *out_scores){
    if (!S || k <= 0 || !out_ids) return -1;
    KciSparse *s = S->s;
    if (!s->built || S->gen != s->gen || !s->post) return -1;
    int m = sp_take(&S->q, &S->qcap, s->dim, qidx, qval, qnnz);
    if (m < 0) return -1;
    if (m == 0 || s->n == 0) return 0;
    uint64_t *nc = realloc(S->cut, (size_t)S->qcap * sizeof *nc);
    if (!nc) return -1;
    S->cut = nc;
    const SpPair *q = S->q;

    int partial = sp_cuts(s, q, S->cut, m, budget);
    int want = partial ? (rerank_n > k ? rerank_n : k) : k;
    if ((uint32_t)want > s->n) want = (int)s->n;
    if (want > S->hcap){
        SpCand *nh = realloc(S->heap, (size_t)want * sizeof *nh);
        if (!nh) return -1;
        S->heap = nh; S->hcap = want;
    }

    if (++S->ep == 0){ memset(S->slot, 0, (size_t)S->nslot * sizeof *S->slot); S->ep = 1; }
    uint32_t ep = S->ep, nt = 0; SpSlot *slot = S->slot; uint32_t *touched = S->touched;
    for (int i = 0; i < m; i++){
        const SpPair *p = s->post + s->tptr[q[i].k]; float w = q[i].v; uint64_t c = S->cut[i];
        for (uint64_t j = 0; j < c; j++){
            SpSlot *a = &slot[p[j].k];
            if (a->ep != ep){ a->ep = ep; a->v = w * p[j].v; touched[nt++] = p[j].k; }
            else a->v += w * p[j].v;
        }
    }

    SpCand *h = S->heap; int hn = 0;
    const uint8_t *dead = s->ndead ? s->dead : NULL;
    for (uint32_t i = 0; i < nt; i++){
        if (dead && dead[touched[i]]) continue;
        SpCand c = { slot[touched[i]].v, touched[i] };
        sp_offer_filtered(S, h, &hn, want, c);
    }
    if (partial || s->n > s->nbuilt){
        for (int i = 0; i < m; i++) S->qd[q[i].k] = q[i].v;
        if (partial)                             /* the running score missed the tail of each list: score exactly */
            for (int i = 0; i < hn; i++){
                float sum = 0;
                for (uint64_t j = s->row[h[i].d]; j < s->row[h[i].d + 1]; j++) sum += S->qd[s->f[j].k] * s->f[j].v;
                h[i].s = sum;
            }
        if (s->n > s->nbuilt) sp_offer_pending(S, h, &hn, want);
        for (int i = 0; i < m; i++) S->qd[q[i].k] = 0;
    }
    qsort(h, (size_t)hn, sizeof *h, sp_cmp_cand);
    if (hn > k) hn = k;
    for (int i = 0; i < hn; i++){ out_ids[i] = s->ids[h[i].d]; if (out_scores) out_scores[i] = h[i].s; }
    return hn;
}

int kci_sparse_search_exact(KciSparseSearcher *S, const uint32_t *qidx, const float *qval, int qnnz,
                            int k, uint64_t *out_ids, float *out_scores){
    return kci_sparse_search(S, qidx, qval, qnnz, k, 0, 0, out_ids, out_scores);
}

uint32_t kci_sparse_count(const KciSparse *s){ return s ? s->n - s->ndead : 0; }
uint32_t kci_sparse_dim(const KciSparse *s){ return s ? s->dim : 0; }
int      kci_sparse_is_built(const KciSparse *s){ return s ? s->built : 0; }
int64_t  kci_sparse_bytes(const KciSparse *s){
    if (!s) return 0;
    uint64_t b = sizeof *s + (uint64_t)s->ncap * (sizeof *s->ids + 1) + ((uint64_t)s->ncap + 1) * sizeof *s->row
               + (uint64_t)s->mapcap * sizeof *s->map
               + s->fcap * sizeof *s->f + (uint64_t)s->tmpcap * sizeof *s->tmp;
    if (s->post) b += ((uint64_t)s->dim + 1) * sizeof *s->tptr + (s->fnnz ? s->fnnz : 1) * sizeof *s->post;
    if (s->bptr) b += ((uint64_t)s->dim + 1) * (sizeof *s->bptr + sizeof *s->sdp + sizeof *s->sep) + (s->nblk + 1) * sizeof *s->bdoc
                    + s->bdoc[s->nblk] * sizeof *s->brow + s->nblk * sizeof *s->bscale
                    + s->sdp[s->dim] * ((s->sdim16 ? 2 : 4) + 1) + (s->sdp[s->dim] / 16 + 1) * sizeof *s->sck + s->sep[s->dim] * 2;
    if (s->nbr) b += (uint64_t)s->nknn_rows * (uint64_t)s->nknn * sizeof *s->nbr;
    if (s->crow) b += (s->coff[s->ncrow] + SP_CROW_PAD) * sizeof *s->crow + ((uint64_t)s->ncrow + 1) * sizeof *s->coff;
    if (s->boff) b += s->bdoc[s->nblk] * (sizeof *s->boff + 1);
    return (int64_t)b;
}

typedef struct {
    uint32_t nblk; uint32_t *doff; uint32_t *rows; float *scale;
    uint32_t *soff;                                       /* per group, into the scratch summary; only during the build of the list */
    uint32_t nd, nent; uint32_t *tdim; uint8_t *tlen, *ent;   /* runs and pairs of this list, in the final layout */
} SpTermBlocks;
typedef struct { uint32_t c; float v; } SpCEnt;         /* (centre, its weight) in the centres' list of one dimension */
typedef struct {
    KciSparse *s; SpTermBlocks *tb; int list_len, block_size, cterms, centre_terms; float mass;
    uint32_t next, end; int err;                          /* lists [next, end) are this round's */
    float thr;                                            /* > 0: a list is its pairs with |weight| >= thr (global_cut) */
} SpBlockJob;

/* The T-th largest |weight| of a row (0 if the row has no more than T pairs): its heaviest T dimensions
 * are the ones with |weight| >= this. */
static float sp_row_cut(const KciSparse *s, uint32_t d, int T){
    if (s->row[d + 1] - s->row[d] <= (uint64_t)T) return 0;
    float top[64]; int nt = 0;                                /* largest first */
    for (uint64_t x = s->row[d]; x < s->row[d + 1]; x++){
        float a = fabsf(s->f[x].v);
        if (nt < T){ int p = nt++; while (p > 0 && top[p - 1] < a){ top[p] = top[p - 1]; p--; } top[p] = a; }
        else if (a > top[T - 1]){ int p = T - 1; while (p > 0 && top[p - 1] < a){ top[p] = top[p - 1]; p--; } top[p] = a; }
    }
    return top[T - 1];
}

/* How many pairs of list t go into the groups. */
static uint64_t sp_list_take(const KciSparse *s, uint32_t t, int list_len, int block_size, float thr){
    uint64_t full = s->tptr[t + 1] - s->tptr[t];
    if (!(thr > 0)) return full > (uint64_t)list_len ? (uint64_t)list_len : full;
    uint64_t L = sp_front(s->post + s->tptr[t], full, thr), cap = (uint64_t)65535 * (uint64_t)block_size;
    uint64_t least = full > (uint64_t)block_size ? (uint64_t)block_size : full;   /* a rare dimension keeps one group */
    if (L < least) L = least;
    return L > cap ? cap : L;
}

typedef struct {
    int32_t *head; uint32_t *cend; float *smax; uint32_t *tk; uint32_t ntk, tkcap;      /* [dim] scratch, clean between uses */
    SpCEnt *ce; size_t cecap; float *acc; uint32_t *assign, *cnt, *order, *mark, *tc; size_t lcap, bcap;
    SpPair *sum; size_t sumcap; float *scale;
    uint64_t *keys; size_t keycap;
} SpBlockScratch;

static int sp_cmp_u64(const void *a, const void *b){
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return x < y ? -1 : x > y;
}

/* Write the runs and pairs of one list from its sorted keys (dimension << 24 | group << 8 | byte). With
 * tdim == NULL only counts. */
static void sp_enc_runs(const uint64_t *keys, size_t nkeys, uint32_t *tdim, uint8_t *tlen, uint8_t *ent, uint32_t *out_nd, uint32_t *out_ne){
    uint32_t nd = 0, ne = 0; size_t x = 0;
    while (x < nkeys){
        uint32_t k = (uint32_t)(keys[x] >> 24), g = 0, len = 0;
        for (; x < nkeys && (uint32_t)(keys[x] >> 24) == k; x++){
            uint32_t grp = (uint32_t)(keys[x] >> 8) & 0xffffu, q = (uint32_t)keys[x] & 0xffu;
            for (;;){
                if (len == 255){ if (tdim){ tdim[nd] = k; tlen[nd] = 255; } nd++; len = 0; g = 0; }
                uint32_t step = grp - g;
                if (step > 255){ if (ent){ ent[2 * (size_t)ne] = 255; ent[2 * (size_t)ne + 1] = 0; } ne++; len++; g += 255; continue; }
                if (ent){ ent[2 * (size_t)ne] = (uint8_t)step; ent[2 * (size_t)ne + 1] = (uint8_t)q; }
                ne++; len++; g = grp; break;
            }
        }
        if (tdim){ tdim[nd] = k; tlen[nd] = (uint8_t)len; }
        nd++;
    }
    *out_nd = nd; *out_ne = ne;
}

/* The offset of every 16th run, from the run lengths. */
static int sp_make_sck(KciSparse *s){
    uint64_t nr = s->sdp[s->dim], e = 0;
    free(s->sck);
    s->sck = malloc((size_t)(nr / 16 + 1) * sizeof *s->sck);
    if (!s->sck) return -1;
    for (uint64_t r = 0; r < nr; r++){ if ((r & 15) == 0) s->sck[r >> 4] = e; e += s->slen[r]; }
    if ((nr & 15) == 0) s->sck[nr >> 4] = e;
    return 0;
}

static int sp_block_term(const SpBlockJob *j, SpBlockScratch *w, uint32_t t){
    const KciSparse *s = j->s; SpTermBlocks *out = &j->tb[t];
    uint32_t L = (uint32_t)sp_list_take(s, t, j->list_len, j->block_size, j->thr);
    if (L == 0) return 0;
    const SpPair *pl = s->post + s->tptr[t];
    uint32_t nb = (L + (uint32_t)j->block_size - 1) / (uint32_t)j->block_size;
    if (nb > 65535) nb = 65535;                               /* a group's number inside its list is kept in 16 bits */

    if (L > w->lcap){
        uint32_t *na = realloc(w->assign, (size_t)L * sizeof *na); if (!na) return -1; w->assign = na;
        uint32_t *no = realloc(w->order, (size_t)L * sizeof *no);  if (!no) return -1; w->order = no;
        w->lcap = L;
    }
    if (nb + 1 > w->bcap){
        float *na = realloc(w->acc, (size_t)(nb + 1) * sizeof *na);    if (!na) return -1; w->acc = na;
        uint32_t *nc = realloc(w->cnt, (size_t)(nb + 2) * sizeof *nc); if (!nc) return -1; w->cnt = nc;
        float *ns = realloc(w->scale, (size_t)(nb + 1) * sizeof *ns);  if (!ns) return -1; w->scale = ns;
        uint32_t *nm = realloc(w->mark, (size_t)(nb + 1) * sizeof *nm); if (!nm) return -1; w->mark = nm;
        uint32_t *nt = realloc(w->tc, (size_t)(nb + 1) * sizeof *nt);   if (!nt) return -1; w->tc = nt;
        w->bcap = nb + 1;
    }

    /* 1. assign every row to a centre */
    if (nb == 1) for (uint32_t i = 0; i < L; i++) w->assign[i] = 0;
    else {
        size_t ne = 0;
        for (uint32_t c = 0; c < nb; c++){
            uint32_t d = pl[(uint64_t)c * L / nb].k;
            float cthr = j->centre_terms < 0 ? 0 : sp_row_cut(s, d, j->centre_terms);
            w->acc[c] = cthr;                                 /* kept for the second pass; acc is not in use yet */
            for (uint64_t x = s->row[d]; x < s->row[d + 1]; x++){
                uint32_t k = s->f[x].k;
                if (k == t || fabsf(s->f[x].v) < cthr) continue;
                if (w->cend[k]++ == 0) w->tk[w->ntk++] = k;
                ne++;
            }
        }
        if (ne > 0x7fffffffu) return -1;
        if (ne > w->cecap){
            SpCEnt *n2 = realloc(w->ce, ne * sizeof *n2); if (!n2) return -1;
            w->ce = n2; w->cecap = ne;
        }
        { uint32_t pos = 0;
          for (uint32_t x = 0; x < w->ntk; x++){ uint32_t k = w->tk[x], n = w->cend[k]; w->head[k] = (int32_t)pos; w->cend[k] = pos; pos += n; } }
        for (uint32_t c = 0; c < nb; c++){
            uint32_t d = pl[(uint64_t)c * L / nb].k;
            float cthr = w->acc[c];
            for (uint64_t x = s->row[d]; x < s->row[d + 1]; x++){
                uint32_t k = s->f[x].k;
                if (k == t || fabsf(s->f[x].v) < cthr) continue;
                SpCEnt *e = &w->ce[w->cend[k]++]; e->c = c; e->v = s->f[x].v;
            }
        }
        memset(w->mark, 0, (size_t)nb * sizeof *w->mark);
        for (uint32_t i = 0; i < L; i++){
            uint32_t d = pl[i].k, ntc = 0;
            float thr = sp_row_cut(s, d, j->cterms);
            for (uint64_t x = s->row[d]; x < s->row[d + 1]; x++){
                float v = s->f[x].v;
                if (fabsf(v) < thr) continue;
                uint32_t k = s->f[x].k;
                if (w->head[k] < 0) continue;
                for (const SpCEnt *e = w->ce + w->head[k], *ee = w->ce + w->cend[k]; e < ee; e++){
                    uint32_t c = e->c;
                    if (w->mark[c] != i + 1){ w->mark[c] = i + 1; w->acc[c] = 0; w->tc[ntc++] = c; }
                    w->acc[c] += v * e->v;
                }
            }
            uint32_t pos = (uint32_t)((uint64_t)i * nb / L), best = pos;                 /* nothing shared: by position */
            float bv = w->mark[pos] == i + 1 ? w->acc[pos] : 0;
            if (bv < 0){                                      /* a centre it shares nothing with (sum 0) can win: all of them, in order */
                for (uint32_t c = 0; c < nb; c++){ float a = w->mark[c] == i + 1 ? w->acc[c] : 0; if (a > bv){ bv = a; best = c; } }
            } else {
                float mx = bv; uint32_t cm = pos;             /* the largest sum; among equals the lowest centre, as a scan in order gives */
                for (uint32_t x = 0; x < ntc; x++){
                    uint32_t c = w->tc[x]; float a = w->acc[c];
                    if (a > mx || (a == mx && a > bv && c < cm)){ mx = a; cm = c; }
                }
                best = cm;
            }
            w->assign[i] = best;
        }
        for (uint32_t x = 0; x < w->ntk; x++){ w->head[w->tk[x]] = -1; w->cend[w->tk[x]] = 0; }
        w->ntk = 0;
    }

    /* 2. group the rows by centre, dropping centres that got none */
    memset(w->cnt, 0, (size_t)(nb + 2) * sizeof *w->cnt);
    for (uint32_t i = 0; i < L; i++) w->cnt[w->assign[i] + 1]++;
    uint32_t used = 0;
    for (uint32_t c = 0; c < nb; c++) if (w->cnt[c + 1]) used++;
    out->doff = malloc((size_t)(used + 1) * sizeof *out->doff);
    out->soff = malloc((size_t)(used + 1) * sizeof *out->soff);
    out->rows = malloc((size_t)L * sizeof *out->rows);
    if (!out->doff || !out->soff || !out->rows) return -1;
    for (uint32_t c = 0; c < nb; c++) w->cnt[c + 1] += w->cnt[c];          /* cnt[c] = start of centre c */
    for (uint32_t i = 0; i < L; i++) w->order[i] = w->cnt[w->assign[i]]++; /* now cnt[c] = end of centre c */
    for (uint32_t i = 0; i < L; i++) out->rows[w->order[i]] = pl[i].k;

    /* 3. one summary per group */
    size_t nsum = 0; uint32_t b = 0, start = 0;
    for (uint32_t c = 0; c < nb; c++){
        uint32_t end = w->cnt[c];
        if (end == start) continue;
        out->doff[b] = start; out->soff[b] = (uint32_t)nsum;
        for (uint32_t i = start; i < end; i++){
            uint32_t d = out->rows[i];
            for (uint64_t x = s->row[d]; x < s->row[d + 1]; x++){
                uint32_t k = s->f[x].k; float a = fabsf(s->f[x].v);
                if (w->smax[k] == 0) w->tk[w->ntk++] = k;
                if (a > w->smax[k]) w->smax[k] = a;
            }
        }
        if (nsum + w->ntk > w->sumcap){
            size_t ncap = (nsum + w->ntk) * 2;
            SpPair *n2 = realloc(w->sum, ncap * sizeof *n2); if (!n2) return -1;
            w->sum = n2; w->sumcap = ncap;
        }
        SpPair *sm = w->sum + nsum; double total = 0; float top = 0;
        for (uint32_t x = 0; x < w->ntk; x++){
            sm[x].k = w->tk[x]; sm[x].v = w->smax[w->tk[x]]; total += sm[x].v; w->smax[w->tk[x]] = 0;
            if (sm[x].v > top) top = sm[x].v;
        }
        uint32_t n = w->ntk; w->ntk = 0;
        if (j->mass < 1.0f && n > 1){
            /* keep the heaviest part only. No sort (a group's summary has thousands of entries and there are
             * about as many groups as rows / block_size): 256 equal steps of weight, add them up from the
             * top until the share is reached, keep every entry in those steps. */
            double step[256]; memset(step, 0, sizeof step);
            float sc = 255.0f / top;
            for (uint32_t x = 0; x < n; x++) step[(int)(sm[x].v * sc)] += sm[x].v;
            double run = 0, want = total * (double)j->mass; int cutb = 255;
            for (; cutb > 0; cutb--){ run += step[cutb]; if (run >= want) break; }
            uint32_t keep = 0;
            for (uint32_t x = 0; x < n; x++) if ((int)(sm[x].v * sc) >= cutb) sm[keep++] = sm[x];
            n = keep;
        }
        nsum += n;
        if (nsum > 0xffffffffu) return -1;
        w->scale[b] = top / 255.0f;
        start = end; b++;
    }
    out->doff[b] = L; out->soff[b] = (uint32_t)nsum; out->nblk = b;
    out->scale = malloc((size_t)(b ? b : 1) * sizeof *out->scale);
    if (!out->scale) return -1;
    memcpy(out->scale, w->scale, (size_t)b * sizeof *out->scale);

    /* 4. turn the summaries round: sort (dimension, group, byte) and cut into one run per dimension */
    if (nsum > w->keycap){
        uint64_t *nk = realloc(w->keys, nsum * 2 * sizeof *nk); if (!nk) return -1;
        w->keys = nk; w->keycap = nsum * 2;
    }
    for (uint32_t g = 0; g < b; g++){
        float inv = out->scale[g] > 0 ? 1.0f / out->scale[g] : 0;
        for (uint32_t x = out->soff[g]; x < out->soff[g + 1]; x++){
            int qv = (int)ceilf(w->sum[x].v * inv);           /* up, so that byte * scale >= the weight */
            while (qv < 255 && (float)qv * out->scale[g] < w->sum[x].v) qv++;
            if (qv > 255) qv = 255;
            if (qv < 1) qv = 1;
            w->keys[x] = ((uint64_t)w->sum[x].k << 24) | ((uint64_t)g << 8) | (uint64_t)qv;
        }
    }
    free(out->soff); out->soff = NULL;
    if (nsum > 1) qsort(w->keys, nsum, sizeof *w->keys, sp_cmp_u64);
    uint32_t nd = 0, ne = 0;
    sp_enc_runs(w->keys, nsum, NULL, NULL, NULL, &nd, &ne);
    out->tdim = malloc((size_t)(nd ? nd : 1) * sizeof *out->tdim);
    out->tlen = malloc((size_t)(nd ? nd : 1));
    out->ent  = malloc((size_t)(ne ? ne : 1) * 2);
    if (!out->tdim || !out->tlen || !out->ent) return -1;
    sp_enc_runs(w->keys, nsum, out->tdim, out->tlen, out->ent, &nd, &ne);
    out->nd = nd; out->nent = ne;
    return 0;
}

static void *sp_block_worker(void *arg){
    SpBlockJob *j = arg; uint32_t dim = j->s->dim;
    SpBlockScratch w; memset(&w, 0, sizeof w);
    w.head = malloc((size_t)dim * sizeof *w.head);
    w.cend = calloc(dim, sizeof *w.cend);
    w.smax = calloc(dim, sizeof *w.smax);
    w.tk   = malloc((size_t)dim * sizeof *w.tk);
    int err = (!w.head || !w.cend || !w.smax || !w.tk);
    if (!err) for (uint32_t k = 0; k < dim; k++) w.head[k] = -1;
    while (!err && !j->err){
        uint32_t t0 = __sync_fetch_and_add(&j->next, 4u);
        if (t0 >= j->end) break;
        uint32_t t1 = j->end - t0 > 4u ? t0 + 4u : j->end;
        for (uint32_t t = t0; t < t1 && !err; t++) err = sp_block_term(j, &w, t) != 0;
    }
    if (err) __sync_fetch_and_or(&j->err, 1);
    free(w.head); free(w.cend); free(w.smax); free(w.tk); free(w.ce); free(w.acc); free(w.assign); free(w.cnt); free(w.order); free(w.mark); free(w.tc); free(w.sum); free(w.scale); free(w.keys);
    return NULL;
}

/* The half-size rows of every row there is now (dim <= 65536). Made with the groups, and again at load. */
static int sp_cmp_f32_desc(const void *a, const void *b){
    float x = *(const float*)a, y = *(const float*)b;
    return x > y ? -1 : x < y;
}
/* The smallest |weight| of the first part of a row: its heaviest pairs up to `mass` of the sum. 0 = all. */
static float sp_head_cut(const SpPair *fp, uint32_t len, float mass, float *tmp){
    if (!(mass < 1.0f) || len < 2) return 0;
    double tot = 0, run = 0;
    for (uint32_t y = 0; y < len; y++){ tmp[y] = fabsf(fp[y].v); tot += tmp[y]; }
    qsort(tmp, len, sizeof *tmp, sp_cmp_f32_desc);
    double want = tot * (double)mass;
    for (uint32_t i = 0; i < len; i++){ run += tmp[i]; if (run >= want) return tmp[i]; }
    return 0;
}
typedef struct { KciSparse *s; float *cut; uint32_t next, maxlen; int pass, err; } SpCrowJob;
static void *sp_crow_worker(void *arg){
    SpCrowJob *j = arg; KciSparse *s = j->s;
    float *tmp = j->pass == 0 ? malloc((size_t)j->maxlen * sizeof *tmp) : NULL;
    if (j->pass == 0 && !tmp){ __sync_fetch_and_or(&j->err, 1); return NULL; }
    for (;;){
        uint32_t d0 = __sync_fetch_and_add(&j->next, 2048u);
        if (d0 >= s->n) break;
        uint32_t d1 = s->n - d0 > 2048u ? d0 + 2048u : s->n;
        for (uint32_t dd = d0; dd < d1; dd++){
            uint32_t len = (uint32_t)(s->row[dd + 1] - s->row[dd]), nh = 0; const SpPair *fp = s->f + s->row[dd];
            if (j->pass == 0){
                float ct = sp_head_cut(fp, len, s->head_mass, tmp);
                for (uint32_t y = 0; y < len; y++) if (fabsf(fp[y].v) >= ct) nh++;
                uint64_t nd = ((uint64_t)nh + 15) / 16;
                j->cut[dd] = ct; s->coff[dd] = 3 + nd + 16 * nd + nh + 2 * (uint64_t)(len - nh);   /* its size; turned into offsets after */
                continue;
            }
            float top = 0, ct = j->cut[dd];
            for (uint32_t y = 0; y < len; y++){ float a = fabsf(fp[y].v); if (a > top) top = a; if (a >= ct) nh++; }
            uint32_t nd = (nh + 15) / 16, nt = len - nh;
            float sc = top / 32767.0f, inv = sc > 0 ? 1.0f / sc : 0;
            uint16_t *h = s->crow + s->coff[dd], *dir = h + 3, *ck = dir + nd; int16_t *cv = (int16_t*)(ck + 16 * nd);
            uint16_t *tk = (uint16_t*)(cv + nh); int16_t *tv = (int16_t*)(tk + nt);
            h[0] = (uint16_t)nh; memcpy(h + 1, &sc, sizeof sc);
            uint32_t ih = 0, it = 0;
            for (uint32_t y = 0; y < len; y++){
                float v = fp[y].v * inv; int iv = (int)(v < 0 ? v - 0.5f : v + 0.5f);
                int16_t q = (int16_t)(iv > 32767 ? 32767 : iv < -32767 ? -32767 : iv);
                if (fabsf(fp[y].v) >= ct){ ck[ih] = (uint16_t)fp[y].k; cv[ih] = q; ih++; }
                else                     { tk[it] = (uint16_t)fp[y].k; tv[it] = q; it++; }
            }
            for (uint32_t y = nh; y < 16 * nd; y++) ck[y] = 0xFFFF;
            for (uint32_t c = 0; c < nd; c++) dir[c] = ck[16 * c];
            uint64_t used = 3 + (uint64_t)nd + 16 * (uint64_t)nd + nh + 2 * (uint64_t)nt;
            if (s->coff[dd + 1] - s->coff[dd] > used) h[used] = 0;        /* the entry between two rows, where there is one */
        }
    }
    free(tmp);
    return NULL;
}
static int sp_crow_run(SpCrowJob *j, int threads, int pass){
    j->pass = pass; j->next = 0;
    pthread_t th[64]; int started = 0;
    for (int i = 1; i < threads; i++){
        if (pthread_create(&th[started], NULL, sp_crow_worker, j) != 0) break;
        started++;
    }
    sp_crow_worker(j);
    for (int i = 0; i < started; i++) pthread_join(th[i], NULL);
    return j->err ? -1 : 0;
}
static int sp_make_crow(KciSparse *s, int threads){
    free(s->crow); free(s->coff); s->crow = NULL; s->ncrow = 0;
    if (threads <= 0){ long c = sysconf(_SC_NPROCESSORS_ONLN); threads = c > 0 ? (int)c : 1; }
    if (threads > 64) threads = 64;
    SpCrowJob job; memset(&job, 0, sizeof job); job.s = s; job.maxlen = 1;
    for (uint32_t dd = 0; dd < s->n; dd++){ uint64_t l = s->row[dd + 1] - s->row[dd]; if (l > job.maxlen) job.maxlen = (uint32_t)l; }
    job.cut = malloc((size_t)(s->n ? s->n : 1) * sizeof *job.cut);
    s->coff = malloc(((size_t)s->n + 1) * sizeof *s->coff);
    if (!s->coff || !job.cut || sp_crow_run(&job, threads, 0) != 0){ free(job.cut); free(s->coff); s->coff = NULL; return -1; }
    uint64_t tot = 0;
    for (uint32_t dd = 0; dd < s->n; dd++){
        uint64_t sz = s->coff[dd];
        s->coff[dd] = tot; tot += sz;
        tot += tot & 1;                                      /* every row starts on an even entry (boff) */
    }
    s->coff[s->n] = tot;
    s->crow = malloc(((size_t)tot + SP_CROW_PAD) * sizeof *s->crow);
    if (!s->crow || sp_crow_run(&job, threads, 1) != 0){ free(job.cut); free(s->crow); free(s->coff); s->crow = NULL; s->coff = NULL; return -1; }
    free(job.cut);
    memset(s->crow + tot, 0, SP_CROW_PAD * sizeof *s->crow);
    s->ncrow = s->n;
    return 0;
}

/* boff from coff (see KciSparse.boff). Not having it is not an error. */
static void sp_make_boff(KciSparse *s){
    free(s->boff); free(s->bhl); s->boff = NULL; s->bhl = NULL; s->pf_bytes = SP_PF_MAX;
    if (!s->crow || !s->bptr) return;
    { uint64_t sum = 0;
      for (uint32_t d = 0; d < s->ncrow; d++){ uint64_t nh = s->crow[s->coff[d]], nd = (nh + 15) / 16; sum += (3 + nd + 16 * nd + nh) * 2; }
      uint64_t avg = s->ncrow ? sum / s->ncrow : 0;
      s->pf_bytes = (uint32_t)(avg + 64 > SP_PF_MAX ? SP_PF_MAX : avg + 64); }
    if (s->coff[s->ncrow] / 2 > 0xffffffffu) return;
    for (uint32_t d = 0; d < s->ncrow; d++) if (s->coff[d] & 1) return;
    uint64_t nr = s->bdoc[s->nblk];
    uint32_t *bo = malloc((size_t)(nr ? nr : 1) * sizeof *bo);
    uint8_t *hl = malloc((size_t)(nr ? nr : 1));
    if (!bo || !hl){ free(bo); free(hl); return; }
    for (uint64_t i = 0; i < nr; i++){
        if (s->brow[i] >= s->ncrow){ free(bo); free(hl); return; }
        uint64_t o = s->coff[s->brow[i]]; bo[i] = (uint32_t)(o / 2);
        /* the lines from the row's first byte to the last byte of its first part; the row sits anywhere in a line */
        uint64_t nh = s->crow[o], nd = (nh + 15) / 16, bytes = (3 + nd + 16 * nd + nh) * 2;
        uint64_t first = ((uintptr_t)(s->crow + o)) & 63, lines = (first + bytes + 63) / 64;
        hl[i] = (uint8_t)(lines > 255 ? 255 : lines);
    }
    s->boff = bo; s->bhl = hl;
}

int kci_sparse_has_blocks(const KciSparse *s){ return s && s->bptr != NULL; }

void kci_sparse_sizes(const KciSparse *s, uint64_t out[8]){
    memset(out, 0, 8 * sizeof *out);
    if (!s) return;
    out[0] = s->fnnz; out[1] = s->post ? s->fnnz : 0;
    if (s->bptr){ out[2] = s->nblk; out[3] = s->bdoc[s->nblk]; out[4] = s->sdp[s->dim]; out[5] = s->sep[s->dim]; }
    out[6] = s->nbr ? (uint64_t)s->nknn_rows * (uint64_t)s->nknn : 0; out[7] = s->crow ? s->fnnz : 0;
}

int kci_sparse_drop_rows(KciSparse *s){
    if (!s || !s->built || !s->bptr || !s->crow || s->norows || s->n != s->nbuilt || s->ncrow != s->n) return -1;
    free(s->post); s->post = NULL;
    free(s->f); s->f = NULL; s->fcap = 0;
    free(s->tmp); s->tmp = NULL; s->tmpcap = 0;
    s->norows = 1;
    return 0;
}
int kci_sparse_has_rows(const KciSparse *s){ return s && !s->norows; }

int kci_sparse_drop_lists(KciSparse *s){
    if (!s || !s->built || !s->bptr) return -1;
    free(s->post); s->post = NULL;                           /* tptr stays: it is small and says how long each list was */
    return 0;
}

int kci_sparse_build_blocks(KciSparse *s, const KciSparseBlockParams *params, double *out_secs){
    KciSparseBlockParams local, *p = NULL;
    if (params){
        size_t have = params->struct_size ? params->struct_size : sizeof local;   /* when fields are added: 0 must keep meaning "up to global_cut" */
        if (have < sizeof(unsigned) + sizeof(int)) return -1;
        memset(&local, 0, sizeof local);
        memcpy(&local, params, have < sizeof local ? have : sizeof local);
        p = &local;
    }
    if (!s || !s->built || !s->post) return -1;
    double t0 = sp_now();
    int list_len = p && p->list_len > 0 ? p->list_len : 3000;
    int block_size = p && p->block_size > 0 ? p->block_size : 6;
    float mass = p && p->summary_mass != 0 ? p->summary_mass : 0.4f;
    int threads = p ? p->threads : 0;
    int cterms = p && p->cluster_terms > 0 ? p->cluster_terms : 16;
    if (cterms > 64) cterms = 64;
    int centre_terms = p && p->centre_terms != 0 ? p->centre_terms : cterms;
    if (centre_terms > 64) centre_terms = 64;
    if (!(mass > 0 && mass <= 1.0f) || (p && (p->list_len < 0 || p->block_size < 0 || p->cluster_terms < 0 || p->head_mass < 0))) return -1;
    if (threads <= 0){ long c = sysconf(_SC_NPROCESSORS_ONLN); threads = c > 0 ? (int)c : 1; }
    if (threads > 64) threads = 64;
    sp_blocks_free(s);

    SpBlockJob job; memset(&job, 0, sizeof job);
    job.s = s; job.list_len = list_len; job.block_size = block_size; job.mass = mass; job.cterms = cterms; job.centre_terms = centre_terms;
    job.tb = calloc(s->dim, sizeof *job.tb);
    if (!job.tb) return -1;

    if (p && p->global_cut){
        uint64_t nonempty = 0; float hi = 0, lo = 0;
        for (uint32_t t = 0; t < s->dim; t++)
            if (s->tptr[t + 1] > s->tptr[t]){ nonempty++; float a = fabsf(s->post[s->tptr[t]].v); if (a > hi) hi = a; }
        uint64_t want = nonempty * (uint64_t)list_len;
        if (want >= s->tptr[s->dim] || !(hi > 0)) job.thr = 0;                 /* everything fits: plain lists */
        else {
            for (int it = 0; it < 40; it++){
                float mid = lo + (hi - lo) * 0.5f; uint64_t c = 0;
                for (uint32_t t = 0; t < s->dim; t++) c += sp_front(s->post + s->tptr[t], s->tptr[t + 1] - s->tptr[t], mid);
                if (c > want) lo = mid; else hi = mid;
            }
            job.thr = hi;
        }
    }
    uint64_t nrow = 0, bmax = 0;
    for (uint32_t t = 0; t < s->dim; t++){
        uint64_t Lt = sp_list_take(s, t, list_len, block_size, job.thr);
        uint64_t nb = (Lt + (uint64_t)block_size - 1) / (uint64_t)block_size;
        nrow += Lt; bmax += nb > 65535 ? 65535 : nb;
    }
    uint64_t ecap = nrow * 8 + 1024, dcap = nrow / 2 + 1024;   /* first guesses, grown below */
    int rc = 0;
    s->bptr = malloc(((size_t)s->dim + 1) * sizeof *s->bptr);
    s->sdp  = malloc(((size_t)s->dim + 1) * sizeof *s->sdp);
    s->sep  = malloc(((size_t)s->dim + 1) * sizeof *s->sep);
    s->bdoc = malloc((size_t)(bmax + 1) * sizeof *s->bdoc);
    s->brow = malloc((size_t)(nrow ? nrow : 1) * sizeof *s->brow);
    s->bscale = malloc((size_t)(bmax ? bmax : 1) * sizeof *s->bscale);
    s->slen = malloc((size_t)dcap);
    s->ent  = malloc((size_t)ecap * 2);
    if (s->dim <= 65536) s->sdim16 = malloc((size_t)dcap * sizeof *s->sdim16);
    else                 s->sdim32 = malloc((size_t)dcap * sizeof *s->sdim32);
    if (!s->bptr || !s->sdp || !s->sep || !s->bdoc || !s->brow || !s->bscale || !s->slen || !s->ent
        || (!s->sdim16 && !s->sdim32)) rc = -1;

    uint64_t b = 0, r = 0, e = 0, d = 0;
    uint64_t target = nrow / 64 + 1;                          /* rows a round: at most about 65 rounds */
    for (uint32_t t0 = 0; t0 < s->dim && rc == 0; ){
        uint32_t t1 = t0; uint64_t acc = 0;
        while (t1 < s->dim && acc < target){ acc += sp_list_take(s, t1, list_len, block_size, job.thr); t1++; }
        job.next = t0; job.end = t1;
        pthread_t th[64]; int started = 0;
        for (int i = 1; i < threads; i++){
            if (pthread_create(&th[started], NULL, sp_block_worker, &job) != 0) break;
            started++;
        }
        sp_block_worker(&job);
        for (int i = 0; i < started; i++) pthread_join(th[i], NULL);
        if (job.err) rc = -1;

        for (uint32_t t = t0; t < t1; t++){
            SpTermBlocks *tb = &job.tb[t];
            if (rc == 0){ s->bptr[t] = b; s->sdp[t] = d; s->sep[t] = e; }
            if (rc == 0 && tb->nblk){
                if (e + tb->nent > ecap){
                    uint64_t nc = (e + tb->nent) + (e + tb->nent) / 2;
                    uint8_t *n1 = realloc(s->ent, (size_t)nc * 2);
                    if (!n1) rc = -1; else { s->ent = n1; ecap = nc; }
                }
                if (rc == 0 && d + tb->nd > dcap){
                    uint64_t nc = (d + tb->nd) + (d + tb->nd) / 2;
                    uint8_t *n1 = realloc(s->slen, (size_t)nc); if (n1) s->slen = n1;
                    int ok = n1 != NULL;
                    if (s->sdim16){ uint16_t *n2 = realloc(s->sdim16, (size_t)nc * sizeof *n2); if (n2) s->sdim16 = n2; else ok = 0; }
                    else          { uint32_t *n2 = realloc(s->sdim32, (size_t)nc * sizeof *n2); if (n2) s->sdim32 = n2; else ok = 0; }
                    if (!ok) rc = -1; else dcap = nc;
                }
            }
            if (rc == 0 && tb->nblk){
                for (uint32_t i = 0; i < tb->nblk; i++){ s->bdoc[b] = r + tb->doff[i]; s->bscale[b] = tb->scale[i]; b++; }
                memcpy(s->brow + r, tb->rows, (size_t)tb->doff[tb->nblk] * sizeof *s->brow);
                r += tb->doff[tb->nblk];
                memcpy(s->ent + 2 * e, tb->ent, (size_t)tb->nent * 2);
                memcpy(s->slen + d, tb->tlen, tb->nd);
                if (s->sdim16) for (uint32_t x = 0; x < tb->nd; x++) s->sdim16[d + x] = (uint16_t)tb->tdim[x];
                else memcpy(s->sdim32 + d, tb->tdim, (size_t)tb->nd * sizeof *s->sdim32);
                e += tb->nent; d += tb->nd;
            }
            free(tb->doff); free(tb->rows); free(tb->soff); free(tb->scale); free(tb->tdim); free(tb->tlen); free(tb->ent);
            memset(tb, 0, sizeof *tb);
        }
        t0 = t1;
    }
    free(job.tb);
    if (rc == 0){
        s->bptr[s->dim] = b; s->sdp[s->dim] = d; s->sep[s->dim] = e; s->bdoc[b] = r; s->nblk = b;
        /* give the unused ends back; a failed shrink keeps the larger piece, which is still right */
        void *q;
        if ((q = realloc(s->bdoc, (size_t)(b + 1) * sizeof *s->bdoc)) != NULL) s->bdoc = q;
        if ((q = realloc(s->bscale, (size_t)(b ? b : 1) * sizeof *s->bscale)) != NULL) s->bscale = q;
        if ((q = realloc(s->ent, (size_t)(e ? e : 1) * 2)) != NULL) s->ent = q;
        if ((q = realloc(s->slen, (size_t)(d ? d : 1))) != NULL) s->slen = q;
        if (s->sdim16){ if ((q = realloc(s->sdim16, (size_t)(d ? d : 1) * sizeof *s->sdim16)) != NULL) s->sdim16 = q; }
        else          { if ((q = realloc(s->sdim32, (size_t)(d ? d : 1) * sizeof *s->sdim32)) != NULL) s->sdim32 = q; }
    } else sp_blocks_free(s);
    s->head_mass = p && p->head_mass != 0 ? p->head_mass : mass >= 1.0f ? 1.0f : 0.8f;
    if (!(p && p->head_mass != 0) && s->head_mass < 1.0f)
        for (uint64_t y = 0; y < s->fnnz; y++) if (s->f[y].v < 0){ s->head_mass = 1.0f; break; }
    if (s->head_mass > 0 && s->head_mass < 1.0f){
        long hk = lrintf(s->head_mass * 200.0f);
        s->head_mass = hk < 1 ? 1.0f / 200.0f : hk >= 200 ? 1.0f : (float)hk / 200.0f;
    } else s->head_mass = 1.0f;
    if (rc == 0 && sp_make_sck(s) != 0){ sp_blocks_free(s); rc = -1; }
    if (rc == 0 && s->dim < 65536 && sp_make_crow(s, threads) != 0){ sp_blocks_free(s); rc = -1; }
    if (rc == 0) sp_make_boff(s);
    if (out_secs) *out_secs = sp_now() - t0;
    return rc;
}

static int sp_cmp_u16(const void *a, const void *b){ return (int)*(const uint16_t*)a - (int)*(const uint16_t*)b; }
static int sp_cmp_absw_desc(const void *a, const void *b){
    float x = fabsf(((const SpPair*)a)->v), y = fabsf(((const SpPair*)b)->v);
    if (x != y) return x > y ? -1 : 1;
    return ((const SpPair*)a)->k < ((const SpPair*)b)->k ? -1 : ((const SpPair*)a)->k > ((const SpPair*)b)->k;
}

static inline float sp_crow_dot(const uint16_t *h, const uint16_t *q16, const float *qv, uint32_t m, const float *qd){
    uint32_t n = h[0], nd = (n + 15) >> 4; float sc; memcpy(&sc, h + 1, sizeof sc);
    const uint16_t *dir = h + 3, *a = dir + nd; const int16_t *w = (const int16_t*)(a + 16 * nd);
#ifdef SP_AVX2
    if ((uint64_t)m * 6 > n){
        __m256 vacc = _mm256_setzero_ps(); uint32_t y = 0; float sum = 0;
        for (; y + 8 <= n; y += 8){
            __m256i vk = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(a + y)));
            __m256i vw = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(w + y)));
            vacc = _mm256_add_ps(vacc, _mm256_mul_ps(_mm256_i32gather_ps(qd, vk, 4), _mm256_cvtepi32_ps(vw)));
        }
        __m128 hs = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
        hs = _mm_add_ps(hs, _mm_movehl_ps(hs, hs)); hs = _mm_add_ss(hs, _mm_shuffle_ps(hs, hs, 1));
        sum = _mm_cvtss_f32(hs);
        for (; y < n; y++) sum += qd[a[y]] * (float)w[y];
        return sum * sc;
    }
    float acc[4] = { 0, 0, 0, 0 };
    if (nd <= 16){
        const __m256i iota = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m256i valid = _mm256_cmpgt_epi16(_mm256_set1_epi16((short)nd), iota);
        const __m256i vd = _mm256_loadu_si256((const __m256i*)(const void*)dir);     /* past nd: masked by `valid` */
        for (uint32_t i = 0; i < m; i++){
            __m256i vq = _mm256_set1_epi16((short)q16[i]);
            uint32_t le = (uint32_t)_mm256_movemask_epi8(_mm256_and_si256(_mm256_cmpeq_epi16(_mm256_max_epu16(vd, vq), vq), valid));
            uint32_t c = (uint32_t)__builtin_popcount(le) >> 1;
            c -= (c != 0);
            uint32_t mk = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_loadu_si256((const __m256i*)(const void*)(a + 16 * c)), vq));
            uint32_t pos = 16 * c + ((uint32_t)__builtin_ctz(mk | 0x80000000u) >> 1);
            int wi = (int)w[pos] & -(int)(mk != 0);           /* no match: pos is still inside the row's piece, the weight is dropped */
            acc[i & 3] += qv[i] * (float)wi;
        }
    } else {
        const __m256i iota = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        for (uint32_t i = 0; i < m; i++){
            __m256i vq = _mm256_set1_epi16((short)q16[i]); uint32_t c = 0;
            for (uint32_t r = 0; r < nd; r += 16){
                uint32_t left = nd - r < 16 ? nd - r : 16;
                __m256i valid = _mm256_cmpgt_epi16(_mm256_set1_epi16((short)left), iota);
                __m256i vd = _mm256_loadu_si256((const __m256i*)(const void*)(dir + r));
                c += (uint32_t)__builtin_popcount((uint32_t)_mm256_movemask_epi8(_mm256_and_si256(_mm256_cmpeq_epi16(_mm256_max_epu16(vd, vq), vq), valid))) >> 1;
            }
            c -= (c != 0);
            uint32_t mk = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_loadu_si256((const __m256i*)(const void*)(a + 16 * c)), vq));
            uint32_t pos = 16 * c + ((uint32_t)__builtin_ctz(mk | 0x80000000u) >> 1);
            int wi = (int)w[pos] & -(int)(mk != 0);
            acc[i & 3] += qv[i] * (float)wi;
        }
    }
    return ((acc[0] + acc[1]) + (acc[2] + acc[3])) * sc;
#else
    (void)q16; (void)qv; (void)m;
    float sum = 0;
    for (uint32_t y = 0; y < n; y++) sum += qd[a[y]] * (float)w[y];
    return sum * sc;
#endif
}

/* What the rest of a half-size row (its tail, see KciSparse.crow) adds to the score. ext = the row's extent in
 * entries. The tail is short and has no list of pieces: its pairs are gone through one by one. */
static inline float sp_crow_tail(const uint16_t *h, uint64_t ext, const float *qd){
    uint32_t n = h[0], nd = (n + 15) >> 4; uint64_t used = 3 + (uint64_t)nd + 16 * (uint64_t)nd + n;
    if (ext <= used) return 0;
    uint32_t nt = (uint32_t)((ext - used) / 2); float sc; memcpy(&sc, h + 1, sizeof sc);
    const uint16_t *tk = h + used; const int16_t *tv = (const int16_t*)(tk + nt);
    float sum = 0; uint32_t y = 0;
#ifdef SP_AVX2
    __m256 vacc = _mm256_setzero_ps();
    for (; y + 8 <= nt; y += 8){
        __m256i vk = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(tk + y)));
        __m256i vw = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(tv + y)));
        vacc = _mm256_add_ps(vacc, _mm256_mul_ps(_mm256_i32gather_ps(qd, vk, 4), _mm256_cvtepi32_ps(vw)));
    }
    __m128 hs = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
    hs = _mm_add_ps(hs, _mm_movehl_ps(hs, hs)); hs = _mm_add_ss(hs, _mm_shuffle_ps(hs, hs, 1));
    sum = _mm_cvtss_f32(hs);
#endif
    for (; y < nt; y++) sum += qd[tk[y]] * (float)tv[y];
    return sum * sc;
}

#ifndef SP_AHEAD
#define SP_AHEAD 5            /* groups taken from the heap ahead of the one being scored (at least 1) */
#endif
#define SP_AHEAD_ON (SP_AHEAD >= 4)
/* Ask for what scoring a list of rows will read, in two steps because the second needs what the first
 * fetched: (1) each row's "met already" slot and its offset, (2) the row itself, if not met yet. */
#define SP_SEEN(vis, d) (((vis)[(d) >> 6] >> ((d) & 63)) & 1)
/* offs: for the rows of a group, where each one's half-size row starts (KciSparse.boff), or NULL = look it up */
static inline void sp_pf_meta(const KciSparseSearcher *S, const uint32_t *rows, const uint32_t *offs, uint32_t n){
    const KciSparse *s = S->s; const uint64_t *vis = S->vis;
    for (uint32_t x = 0; x < n; x++){
        uint32_t d = rows[x];
        if (d == UINT32_MAX) break;
        __builtin_prefetch(&vis[d >> 6]);
        if (!offs) __builtin_prefetch(s->coff ? (const void*)&s->coff[d] : (const void*)&s->row[d]);
    }
}
static inline void sp_pf_data(const KciSparseSearcher *S, const uint32_t *rows, const uint32_t *offs, const uint8_t *hl, uint32_t n){
    const KciSparse *s = S->s; const uint64_t *vis = S->vis;
    for (uint32_t x = 0; x < n; x++){
        uint32_t d = rows[x];
        if (d == UINT32_MAX) break;
        if (SP_SEEN(vis, d)) continue;
        if (s->crow){
            const uint16_t *h = offs ? s->crow + 2 * (uint64_t)offs[x] : s->crow + s->coff[d];
            const char *p = (const char*)h;
            __builtin_prefetch(p);
            if (offs){ uint32_t ln = hl[x] > 16 ? 16 : hl[x]; for (uint32_t o = 1; o < ln; o++) __builtin_prefetch(p + 64 * o); }
            else {
                size_t bytes = (size_t)(s->coff[d + 1] - s->coff[d]) * sizeof *s->crow;
                if (bytes > s->pf_bytes) bytes = s->pf_bytes;
                for (size_t o = 64; o < bytes; o += 64) __builtin_prefetch(p + o);
            }
        } else {
            const char *p = (const char*)(s->f + s->row[d]);
            __builtin_prefetch(p); __builtin_prefetch(p + 64); __builtin_prefetch(p + 128);
        }
    }
}

static inline void sp_score_rows(KciSparseSearcher *S, const uint32_t *rows, const uint32_t *offs, const uint8_t *hl, uint32_t n, const uint8_t *dead,
                                 SpCand *h, int *hn, int want, uint64_t *w_rows, int ahead){
    const KciSparse *s = S->s; uint64_t *vis = S->vis; const float *qd = S->qd;
    for (uint32_t x = 0; x < n; x++) if (rows[x] == UINT32_MAX){ n = x; break; }
    if (!ahead){ sp_pf_meta(S, rows, offs, n); sp_pf_data(S, rows, offs, hl, n); }
    const uint16_t *crow = s->crow, *q16 = S->q16; const float *qv16 = S->qv16; uint32_t m16 = S->m16;
    for (uint32_t x = 0; x < n; x++){
        uint32_t d = rows[x];
        if (SP_SEEN(vis, d)) continue;
        vis[d >> 6] |= (uint64_t)1 << (d & 63); S->touched[S->nvis++] = d;
        if (dead && dead[d]) continue;
        float sum = 0;
        if (crow) sum = sp_crow_dot(offs ? crow + 2 * (uint64_t)offs[x] : crow + s->coff[d], q16, qv16, m16, qd);
        else
        for (uint64_t y = s->row[d]; y < s->row[d + 1]; y++) sum += qd[s->f[y].k] * s->f[y].v;
        (*w_rows)++;
        SpCand c = { sum, d };
        if (!S->pcap) sp_offer_filtered(S, h, hn, want, c);
        else {                                                /* the heap gives the level to skip by, the pool the answer */
            if (S->filt){
                if (S->pn == S->pcap && !sp_before(c, S->pool[0])) continue;      /* the pool holds the heap's rows and more */
                if (!S->filt(s->ids[d], S->fctx)) continue;
            }
            sp_heap_offer(h, hn, want, c);
            sp_heap_offer(S->pool, &S->pn, S->pcap, c);
        }
    }
}

/* The block search on the query already in S->q[0..m) (canonical). Leaves the best rows in S->heap, best
 * first, and returns how many (<= k), or -1 on OOM. */
static int sp_blocks_core(KciSparseSearcher *S, int m, int k, int query_terms, float heap_factor, int knn){
    KciSparse *s = S->s;
    S->work[0] = S->work[1] = S->work[2] = 0;
    if (m == 0 || s->n == 0) return 0;
    int want = (uint32_t)k > s->n ? (int)s->n : k;
    if (want > S->hcap){
        SpCand *nh = realloc(S->heap, (size_t)want * sizeof *nh);
        if (!nh) return -1;
        S->heap = nh; S->hcap = want;
    }
    S->pn = 0; S->pcap = 0;
    if (s->crow && s->head_mass < 1.0f){
        int pc = S->pool_want > 0 ? S->pool_want : want > 0x3fffffff ? want : 2 * want;
        if (pc < want) pc = want;
        if ((uint32_t)pc > s->n) pc = (int)s->n;
        if (pc > 0){
            if (pc > S->pall){ SpCand *np = realloc(S->pool, (size_t)pc * sizeof *np); if (!np) return -1; S->pool = np; S->pall = pc; }
            S->pcap = pc;
        }
    }
    SpPair *q = S->q;
    if (m > 1) qsort(q, (size_t)m, sizeof *q, sp_cmp_absw_desc);
    int nt = query_terms > 0 && query_terms < m ? query_terms : m;
    float *qd = S->qd;
    for (int i = 0; i < m; i++) qd[q[i].k] = q[i].v;
    uint32_t m16 = 0;
    if (s->crow){                                             /* the query once more, dimensions ascending */
        if (m > S->q16cap){
            uint16_t *n16 = realloc(S->q16, (size_t)m * sizeof *n16); if (n16) S->q16 = n16;
            float *nv = realloc(S->qv16, (size_t)m * sizeof *nv);     if (nv) S->qv16 = nv;
            if (!n16 || !nv){ for (int i = 0; i < m; i++) qd[q[i].k] = 0; return -1; }
            S->q16cap = m;
        }
        for (int i = 0; i < m; i++) S->q16[i] = (uint16_t)q[i].k;
        if (m > 1) qsort(S->q16, (size_t)m, sizeof *S->q16, sp_cmp_u16);
        for (int i = 0; i < m; i++) S->qv16[i] = qd[S->q16[i]];
        m16 = (uint32_t)m;
    }
    S->m16 = m16;

    S->nvis = 0; const uint32_t *boff = s->boff;
    const uint8_t *dead = s->ndead ? s->dead : NULL;
    SpCand *h = S->heap; int hn = 0, rc = 0;
    uint64_t w_sum = 0, w_open = 0, w_rows = 0;
    for (int i = 0; i < nt; i++){
        uint32_t t = q[i].k; uint64_t b0 = s->bptr[t]; uint32_t nb = (uint32_t)(s->bptr[t + 1] - b0), b = 0;
        /* until k rows are in hand there is nothing to compare a bound with */
        for (; b < nb && hn < want; b++){
            w_open++;
            sp_score_rows(S, s->brow + s->bdoc[b0 + b], boff ? boff + s->bdoc[b0 + b] : NULL, boff ? s->bhl + s->bdoc[b0 + b] : NULL, (uint32_t)(s->bdoc[b0 + b + 1] - s->bdoc[b0 + b]), dead, h, &hn, want, &w_rows, 0);
        }
        if (b == nb) continue;
        if (nb > S->bcap){
            float *na = realloc(S->bacc, (size_t)nb * sizeof *na);
            if (!na){ rc = -1; break; }
            S->bacc = na;
            SpCand *nc = realloc(S->bcand, (size_t)nb * sizeof *nc);
            if (!nc){ rc = -1; break; }
            S->bcand = nc; S->bcap = nb;
        }
        /* the bounds of every group of this list: only the runs of the query's own dimensions are read */
        float *acc = S->bacc; memset(acc, 0, (size_t)nb * sizeof *acc);
        uint64_t r0 = s->sdp[t], nd = s->sdp[t + 1] - r0;
        const uint16_t *sd16 = s->sdim16 ? s->sdim16 + r0 : NULL;
        const uint32_t *sd32 = s->sdim32 ? s->sdim32 + r0 : NULL;
        const uint8_t *slen = s->slen, *ent = s->ent; const uint64_t *sck = s->sck;
        for (int x = 0; x < m; x++){
            uint32_t key = q[x].k; uint64_t lo = 0, hi = nd;
            if (sd16){ while (lo < hi){ uint64_t mid = lo + (hi - lo) / 2; if (sd16[mid] < key) lo = mid + 1; else hi = mid; } }
            else     { while (lo < hi){ uint64_t mid = lo + (hi - lo) / 2; if (sd32[mid] < key) lo = mid + 1; else hi = mid; } }
            float aq = fabsf(q[x].v);
            for (; lo < nd && (sd16 ? sd16[lo] : sd32[lo]) == key; lo++){       /* one run, or several when it has more than 255 pairs */
                uint64_t r = r0 + lo, e = sck[r >> 4];
                for (uint64_t y = r & ~(uint64_t)15; y < r; y++) e += slen[y];
                const uint8_t *p = ent + 2 * e; uint32_t len = slen[r], g = 0;
                for (uint32_t i = 0; i < len; i++){ g += p[2 * i]; acc[g] += aq * (float)p[2 * i + 1]; }
            }
        }
        w_sum += nb - b;
        /* the groups that can still enter, best bound first: the k-th best score only rises, so a group that
         * fails now fails later too, and once one fails in this order the rest do */
        SpCand *cand = S->bcand; uint32_t ncand = 0; float thr = h[0].s;
        for (uint32_t x = b; x < nb; x++){
            float est = acc[x] * s->bscale[b0 + x];
            if (!(est * heap_factor < thr)){ cand[ncand].s = est; cand[ncand].d = x; ncand++; }
        }
        for (uint32_t x = ncand / 2; x-- > 0; ) sp_best_down(cand, ncand, x);
        SpCand pipe[SP_AHEAD]; int np = 0;
        for (;;){
            while (np < SP_AHEAD && ncand){
                SpCand top = cand[0];
                if (top.s * heap_factor < h[0].s){ ncand = 0; break; }
                cand[0] = cand[--ncand]; sp_best_down(cand, ncand, 0);
                __builtin_prefetch(&s->bdoc[b0 + top.d]); __builtin_prefetch(&s->bdoc[b0 + top.d + 1]);
                pipe[np++] = top;
            }
            if (np == 0) break;
            if (np > 3){ uint64_t g = b0 + pipe[3].d; const char *r = (const char*)(s->brow + s->bdoc[g]);
                         __builtin_prefetch(r); __builtin_prefetch(r + 64);
                         if (boff){ r = (const char*)(boff + s->bdoc[g]); __builtin_prefetch(r); __builtin_prefetch(r + 64); __builtin_prefetch(s->bhl + s->bdoc[g]); } }
            if (np > 2){ uint64_t g = b0 + pipe[2].d; sp_pf_meta(S, s->brow + s->bdoc[g], boff ? boff + s->bdoc[g] : NULL, (uint32_t)(s->bdoc[g + 1] - s->bdoc[g])); }
            if (np > 1){ uint64_t g = b0 + pipe[1].d; sp_pf_data(S, s->brow + s->bdoc[g], boff ? boff + s->bdoc[g] : NULL, boff ? s->bhl + s->bdoc[g] : NULL, (uint32_t)(s->bdoc[g + 1] - s->bdoc[g])); }
            SpCand top = pipe[0];
            for (int x = 1; x < np; x++) pipe[x - 1] = pipe[x];
            np--;
            if (top.s * heap_factor < h[0].s){ np = 0; ncand = 0; break; }   /* the ones behind it rank lower still */
            uint64_t g = b0 + top.d;
            w_open++;
            sp_score_rows(S, s->brow + s->bdoc[g], boff ? boff + s->bdoc[g] : NULL, boff ? s->bhl + s->bdoc[g] : NULL, (uint32_t)(s->bdoc[g + 1] - s->bdoc[g]), dead, h, &hn, want, &w_rows, SP_AHEAD_ON);
        }
    }
    if (rc == 0 && knn > 0 && s->nbr && hn > 0){
        if (knn > s->nknn) knn = s->nknn;
        if (hn > S->seedcap){
            uint32_t *ns = realloc(S->seed, (size_t)want * sizeof *ns);
            if (!ns) rc = -1;
            else { S->seed = ns; S->seedcap = want; }
        }
        if (rc == 0){
            int nseed = hn;
            for (int i = 0; i < nseed; i++){
                S->seed[i] = h[i].d;
                if (h[i].d < s->nknn_rows){ const char *r = (const char*)(s->nbr + (size_t)h[i].d * (size_t)s->nknn);
                                            __builtin_prefetch(r); __builtin_prefetch(r + 64); }
            }
            /* the same steps ahead as for the groups: seed i is scored, i + 1 has its rows asked for, i + 2 its slots */
            #define SP_NBR(i) (s->nbr + (size_t)S->seed[i] * (size_t)s->nknn)
            for (int i = 0; i < nseed; i++){
                if (i + 2 < nseed && S->seed[i + 2] < s->nknn_rows) sp_pf_meta(S, SP_NBR(i + 2), NULL, (uint32_t)knn);
                if (i + 1 < nseed && S->seed[i + 1] < s->nknn_rows) sp_pf_data(S, SP_NBR(i + 1), NULL, NULL, (uint32_t)knn);
                if (S->seed[i] < s->nknn_rows)
                    sp_score_rows(S, SP_NBR(i), NULL, NULL, (uint32_t)knn, dead, h, &hn, want, &w_rows, SP_AHEAD_ON && i >= 2);
            }
            #undef SP_NBR
        }
    }
    if (rc == 0 && S->pcap && S->pn > 0){                    /* the pool with its tails: the best `want` are the answer */
        SpCand *pl = S->pool; int pn = S->pn;
        for (int i = 0; i < pn; i++){ const char *p = (const char*)(s->crow + s->coff[pl[i].d]); __builtin_prefetch(p + 448); __builtin_prefetch(p + 512); }
        for (int i = 0; i < pn; i++){
            uint32_t d = pl[i].d;
            pl[i].s += sp_crow_tail(s->crow + s->coff[d], s->coff[d + 1] - s->coff[d], qd);
        }
        qsort(pl, (size_t)pn, sizeof *pl, sp_cmp_cand);
        hn = pn < want ? pn : want;
        memcpy(h, pl, (size_t)hn * sizeof *h);
        for (int i = hn / 2; i-- > 0; ) sp_heap_down(h, hn, i);              /* a heap again, for the rows still to be offered */
    }
    if (rc == 0 && s->n > s->nbuilt) sp_offer_pending(S, h, &hn, want);
    if (rc == 0 && s->crow && !s->norows)                    /* the rows kept were scored from the half-size copy: once more, exactly */
        for (int i = 0; i < hn; i++){
            float sum = 0; uint32_t d = h[i].d;
            for (uint64_t y = s->row[d]; y < s->row[d + 1]; y++) sum += qd[s->f[y].k] * s->f[y].v;
            h[i].s = sum;
        }
    for (int i = 0; i < m; i++) qd[q[i].k] = 0;
    for (uint32_t i = 0; i < S->nvis; i++) S->vis[S->touched[i] >> 6] = 0;   /* a whole word: all its set bits are this query's */
    S->nvis = 0;
    if (rc != 0) return -1;
    S->work[0] = w_sum; S->work[1] = w_open; S->work[2] = w_rows;
    qsort(h, (size_t)hn, sizeof *h, sp_cmp_cand);
    return hn;
}

int kci_sparse_search_blocks(KciSparseSearcher *S, const uint32_t *qidx, const float *qval, int qnnz,
                             int k, int query_terms, float heap_factor, int knn,
                             uint64_t *out_ids, float *out_scores){
    if (!S || k <= 0 || knn < 0 || !out_ids || !(heap_factor > 0 && heap_factor <= SP_HF_MAX)) return -1;
    KciSparse *s = S->s;
    if (!s->built || S->gen != s->gen) return -1;
    if (!s->bptr) return -2;
    int m = sp_take(&S->q, &S->qcap, s->dim, qidx, qval, qnnz);
    if (m < 0) return -1;
    int hn = sp_blocks_core(S, m, k, query_terms, heap_factor, knn);
    if (hn < 0) return -1;
    const SpCand *h = S->heap;
    for (int i = 0; i < hn; i++){ out_ids[i] = s->ids[h[i].d]; if (out_scores) out_scores[i] = h[i].s; }
    return hn;
}

/* ── neighbour lists ── */
typedef struct { KciSparse *s; uint32_t *nbr; int knn, terms; float hf; uint32_t next; int err; } SpKnnJob;

static void *sp_knn_worker(void *arg){
    SpKnnJob *j = arg; KciSparse *s = j->s;
    KciSparseSearcher *S = kci_sparse_searcher_new(s);
    int err = !S;
    while (!err && !j->err){
        uint32_t d0 = __sync_fetch_and_add(&j->next, 256u);
        if (d0 >= s->n) break;
        uint32_t d1 = s->n - d0 > 256u ? d0 + 256u : s->n;
        for (uint32_t d = d0; d < d1; d++){
            uint32_t *out = j->nbr + (size_t)d * (size_t)j->knn;
            for (int x = 0; x < j->knn; x++) out[x] = UINT32_MAX;
            int m = (int)(s->row[d + 1] - s->row[d]);
            if (m == 0 || s->dead[d]) continue;
            if (m > S->qcap){
                SpPair *nq = realloc(S->q, (size_t)m * sizeof *nq);
                if (!nq){ err = 1; break; }
                S->q = nq; S->qcap = m;
            }
            memcpy(S->q, s->f + s->row[d], (size_t)m * sizeof *S->q);     /* a stored row is already canonical */
            int hn = sp_blocks_core(S, m, j->knn + 1, j->terms, j->hf, 0);
            if (hn < 0){ err = 1; break; }
            int w = 0;
            for (int i = 0; i < hn && w < j->knn; i++)
                if (S->heap[i].d != d && S->heap[i].d < s->nbuilt) out[w++] = S->heap[i].d;
        }
    }
    if (err) __sync_fetch_and_or(&j->err, 1);
    kci_sparse_searcher_free(S);
    return NULL;
}

int kci_sparse_knn(const KciSparse *s){ return s && s->nbr ? s->nknn : 0; }

int kci_sparse_build_knn(KciSparse *s, int knn, int query_terms, float heap_factor, int threads, double *out_secs){
    if (!s || !s->built || !s->bptr || s->norows || knn <= 0 || knn > 1024) return -1;
    if (query_terms <= 0) query_terms = 5;
    if (heap_factor <= 0) heap_factor = 0.8f;                /* and the search using the lists 0.9375 -> 0.9340 at the same speed */
    if (!(heap_factor <= SP_HF_MAX)) return -1;
    double t0 = sp_now();
    if (threads <= 0){ long c = sysconf(_SC_NPROCESSORS_ONLN); threads = c > 0 ? (int)c : 1; }
    if (threads > 64) threads = 64;
    free(s->nbr); s->nbr = NULL; s->nknn = 0; s->nknn_rows = 0;
    SpKnnJob job; memset(&job, 0, sizeof job);
    job.s = s; job.knn = knn; job.terms = query_terms; job.hf = heap_factor;
    job.nbr = malloc((size_t)(s->n ? s->n : 1) * (size_t)knn * sizeof *job.nbr);
    if (!job.nbr) return -1;
    pthread_t th[64]; int started = 0;
    for (int i = 1; i < threads; i++){
        if (pthread_create(&th[started], NULL, sp_knn_worker, &job) != 0) break;
        started++;
    }
    sp_knn_worker(&job);
    for (int i = 0; i < started; i++) pthread_join(th[i], NULL);
    if (job.err){ free(job.nbr); return -1; }
    s->nbr = job.nbr; s->nknn = knn; s->nknn_rows = s->n;
    if (out_secs) *out_secs = sp_now() - t0;
    return 0;
}

void kci_sparse_searcher_set_pool(KciSparseSearcher *S, int rows){ if (S) S->pool_want = rows > 0 ? rows : 0; }

void kci_sparse_searcher_work(const KciSparseSearcher *S, uint64_t out[3]){
    if (!out) return;
    out[0] = S ? S->work[0] : 0; out[1] = S ? S->work[1] : 0; out[2] = S ? S->work[2] : 0;
}

/* ── rank fusion ── */
typedef struct { uint64_t id; int list, rank; } SpRank;
typedef struct { uint64_t id; float s; } SpFused;
static int sp_cmp_rank(const void *a, const void *b){
    const SpRank *x = a, *y = b;
    if (x->id != y->id) return x->id < y->id ? -1 : 1;
    if (x->list != y->list) return x->list < y->list ? -1 : 1;
    return x->rank < y->rank ? -1 : x->rank > y->rank;
}
static int sp_cmp_fused(const void *a, const void *b){
    const SpFused *x = a, *y = b;
    if (x->s != y->s) return x->s > y->s ? -1 : 1;
    return x->id < y->id ? -1 : x->id > y->id;
}

int kci_hybrid_rrf(const uint64_t *const *lists, const int *lens, const float *weights, int nlists,
                   float c, int k, uint64_t *out_ids, float *out_scores){
    if (!lists || !lens || nlists <= 0 || k <= 0 || !out_ids || !isfinite(c)) return -1;
    if (c <= 0) c = 60.0f;
    size_t total = 0;
    for (int i = 0; i < nlists; i++){
        if (lens[i] < 0 || (lens[i] > 0 && !lists[i])) return -1;
        if (weights && !isfinite(weights[i])) return -1;
        total += (size_t)lens[i];
    }
    if (total == 0) return 0;
    SpRank  *r = malloc(total * sizeof *r);
    SpFused *f = malloc(total * sizeof *f);
    if (!r || !f){ free(r); free(f); return -1; }
    size_t n = 0;
    for (int i = 0; i < nlists; i++)
        for (int j = 0; j < lens[i]; j++){ r[n].id = lists[i][j]; r[n].list = i; r[n].rank = j + 1; n++; }
    qsort(r, n, sizeof *r, sp_cmp_rank);
    size_t nf = 0;
    for (size_t i = 0; i < n; i++){
        if (i > 0 && r[i].id == r[i-1].id && r[i].list == r[i-1].list) continue;   /* repeat inside one list */
        float add = (weights ? weights[r[i].list] : 1.0f) / (c + (float)r[i].rank);
        if (nf > 0 && f[nf-1].id == r[i].id) f[nf-1].s += add;
        else { f[nf].id = r[i].id; f[nf].s = add; nf++; }
    }
    qsort(f, nf, sizeof *f, sp_cmp_fused);
    int ret = nf < (size_t)k ? (int)nf : k;
    for (int i = 0; i < ret; i++){ out_ids[i] = f[i].id; if (out_scores) out_scores[i] = f[i].s; }
    free(r); free(f);
    return ret;
}

typedef struct { uint64_t id; int list, rank; float v; } SpMix;
static int sp_cmp_mix(const void *a, const void *b){
    const SpMix *x = a, *y = b;
    if (x->id != y->id) return x->id < y->id ? -1 : 1;
    if (x->list != y->list) return x->list < y->list ? -1 : 1;
    return x->rank < y->rank ? -1 : x->rank > y->rank;
}
int kci_hybrid_mix(const uint64_t *const *lists, const float *const *scores, const int *lens, const float *weights,
                   int nlists, int k, uint64_t *out_ids, float *out_scores){
    if (!lists || !scores || !lens || nlists <= 0 || k <= 0 || !out_ids) return -1;
    size_t total = 0;
    for (int i = 0; i < nlists; i++){
        if (lens[i] < 0 || (lens[i] > 0 && (!lists[i] || !scores[i]))) return -1;
        if (weights && !isfinite(weights[i])) return -1;
        for (int j = 0; j < lens[i]; j++) if (!isfinite(scores[i][j])) return -1;
        total += (size_t)lens[i];
    }
    if (total == 0) return 0;
    SpMix   *r = malloc(total * sizeof *r);
    SpFused *f = malloc(total * sizeof *f);
    if (!r || !f){ free(r); free(f); return -1; }
    size_t n = 0;
    for (int i = 0; i < nlists; i++){
        if (lens[i] == 0) continue;
        float hi = scores[i][0], lo = scores[i][lens[i] - 1], span = hi - lo;   /* best first: works for "smaller is better" too */
        for (int j = 0; j < lens[i]; j++){
            float v = span != 0 ? (scores[i][j] - lo) / span : 1.0f;
            r[n].id = lists[i][j]; r[n].list = i; r[n].rank = j; r[n].v = v < 0 ? 0 : v > 1 ? 1 : v; n++;
        }
    }
    qsort(r, n, sizeof *r, sp_cmp_mix);
    size_t nf = 0;
    for (size_t i = 0; i < n; i++){
        if (i > 0 && r[i].id == r[i-1].id && r[i].list == r[i-1].list) continue;   /* repeat inside one list */
        float add = (weights ? weights[r[i].list] : 1.0f) * r[i].v;
        if (nf > 0 && f[nf-1].id == r[i].id) f[nf-1].s += add;
        else { f[nf].id = r[i].id; f[nf].s = add; nf++; }
    }
    qsort(f, nf, sizeof *f, sp_cmp_fused);
    int ret = nf < (size_t)k ? (int)nf : k;
    for (int i = 0; i < ret; i++){ out_ids[i] = f[i].id; if (out_scores) out_scores[i] = f[i].s; }
    free(r); free(f);
    return ret;
}

#define SP_MAGIC     "KCISPAR1"
#define SP_KEYCHK    "KCISPARSE-KEYCHK"
#define SP_LABEL_ENC "KCISPAR1 encryption key"
#define SP_LABEL_MAC "KCISPAR1 mac key"
#define SP_KC_NONCE  0xFFFFFFFEull          /* the key check's CTR nonce; the pieces count up from 0 */
#define SP_PIECE     (64u * 1024 * 1024)
#define SP_F_ENC 1u
#define SP_F_BUILT 2u
#define SP_F_LISTS 4u
#define SP_F_BLOCKS 8u
#define SP_F_KNN 16u
#define SP_F_NOROWS 32u
#define SP_F_SUM2 64u
#define SP_F_ALL 127u
#define SP_F_HEAD(f) (((f) >> 8) & 0xffu)
#ifdef _WIN32
#define sp_fseek _fseeki64
#define sp_ftell _ftelli64
static int sp_replace_file(const char *from, const char *to){
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
}
#else
#define sp_fseek fseeko
#define sp_ftell ftello
static int sp_replace_file(const char *from, const char *to){ return rename(from, to); }
#endif

typedef struct { FILE *f; int enc; KvAes aes; KvCmac mac; uint64_t piece; uint8_t *tmp; } SpIO;

static int sp_io_keys(SpIO *io, const uint8_t *key, int keylen, const uint8_t salt[16]){
    uint8_t mk[32], ek[32], ak[32];
    kdf_derive(key, keylen, salt, 16, mk);
    kv_subkey(mk, SP_LABEL_ENC, ek); kv_subkey(mk, SP_LABEL_MAC, ak);
    kv_aes_init(&io->aes, ek); kv_cmac_init(&io->mac, ak); io->enc = 1;
    memset(mk, 0, 32); memset(ek, 0, 32); memset(ak, 0, 32);
    io->tmp = malloc(SP_PIECE);
    return io->tmp ? 0 : -1;
}
static int sp_wraw(SpIO *io, const void *p, size_t n){       /* written as is; part of the tag */
    if (io->enc) kv_cmac_update(&io->mac, p, n);
    return fwrite(p, 1, n, io->f) == n ? 0 : -1;
}
static int sp_wr(SpIO *io, const void *p, uint64_t n){
    const uint8_t *b = p;
    while (n){
        size_t c = n < SP_PIECE ? (size_t)n : SP_PIECE;
        if (!io->enc){ if (fwrite(b, 1, c, io->f) != c) return -1; }
        else {
            if (io->piece >= SP_KC_NONCE) return -1;
            memcpy(io->tmp, b, c); kv_ctr(&io->aes, io->piece++, io->tmp, c); kv_cmac_update(&io->mac, io->tmp, c);
            if (fwrite(io->tmp, 1, c, io->f) != c) return -1;
        }
        b += c; n -= c;
    }
    return 0;
}
static int sp_rd(SpIO *io, void *p, uint64_t n){
    uint8_t *b = p;
    while (n){
        size_t c = n < SP_PIECE ? (size_t)n : SP_PIECE;
        if (fread(b, 1, c, io->f) != c) return -1;
        if (io->enc){ if (io->piece >= SP_KC_NONCE) return -1; kv_ctr(&io->aes, io->piece++, b, c); }
        b += c; n -= c;
    }
    return 0;
}

/* how many bytes the arrays of a file with this header take; 0 = the header cannot be right */
static uint64_t sp_payload(uint32_t flags, const uint32_t h32[8], const uint64_t h64[6], uint64_t limit){
    uint64_t dim = h32[1], n = h32[2], nknn = h32[4], nkr = h32[5];
    for (int i = 0; i < 6; i++) if (h64[i] > limit) return 0;
    if (n > limit) return 0;
    uint64_t sz = n * 8 + n + (n + 1) * 8, ctot = (uint64_t)h32[6] | ((uint64_t)h32[7] << 32);
    if (flags & SP_F_NOROWS){
        if ((flags & (SP_F_BUILT | SP_F_BLOCKS | SP_F_LISTS)) != (SP_F_BUILT | SP_F_BLOCKS) || dim >= 65536 || ctot > limit || h32[3] != h32[2]) return 0;
        sz += (n + 1) * 8 + ctot * 2;
    } else {
        if (ctot) return 0;
        sz += h64[0] * 8;
    }
    if (flags & SP_F_BUILT){ if (dim > limit) return 0; sz += (dim + 1) * 8; }
    if (flags & SP_F_LISTS) sz += h64[5] * 8;
    if (flags & SP_F_BLOCKS)
        sz += (dim + 1) * 8 * 3 + (h64[1] + 1) * 8 + h64[2] * 4 + h64[1] * 4 + h64[3] * (dim <= 65536 ? 2 : 4)
            + ((flags & SP_F_SUM2) ? h64[3] + h64[4] * 2 : h64[3] * 4 + h64[4] * 3);
    if (flags & SP_F_KNN){ if (nknn == 0 || nknn > 1024 || nkr > n) return 0; sz += nkr * nknn * 4; }
    return sz;
}

int kci_sparse_save(const KciSparse *s, const char *path, const uint8_t *key, int keylen){
    if (!s || !path || (key && keylen <= 0) || (!key && keylen != 0)) return -1;
    int built = s->built, lists = built && s->post, blocks = built && s->bptr, knn = blocks && s->nbr;
    SpIO io; memset(&io, 0, sizeof io);
    uint8_t salt[16];
    if (key){
        if (kv_randbytes(salt, 16) != 0) return -1;            /* no random salt, no encrypted file */
        if (sp_io_keys(&io, key, keylen, salt) != 0) return -1;
    }
    char tmp[1100];
    if (snprintf(tmp, sizeof tmp, "%s.tmp", path) >= (int)sizeof tmp){ free(io.tmp); return -1; }
    io.f = fopen(tmp, "wb");
    if (!io.f){ free(io.tmp); return -1; }
    uint32_t dim = s->dim;
    uint64_t ctot = s->norows ? s->coff[s->n] : 0;
    uint32_t h32[8] = { (io.enc ? SP_F_ENC : 0) | (built ? SP_F_BUILT : 0) | (lists ? SP_F_LISTS : 0) | (blocks ? SP_F_BLOCKS | SP_F_SUM2 : 0) | (knn ? SP_F_KNN : 0)
                        | (s->norows ? SP_F_NOROWS : 0)
                        | (blocks && s->head_mass < 1.0f ? (uint32_t)lrintf(s->head_mass * 200.0f) << 8 : 0),
                        dim, s->n, built ? s->nbuilt : 0, knn ? (uint32_t)s->nknn : 0, knn ? s->nknn_rows : 0, (uint32_t)ctot, (uint32_t)(ctot >> 32) };
    uint64_t h64[6] = { s->fnnz, blocks ? s->nblk : 0, blocks ? s->bdoc[s->nblk] : 0, blocks ? s->sdp[dim] : 0,
                        blocks ? s->sep[dim] : 0, built ? s->tptr[dim] : 0 };
    int ok = sp_wraw(&io, SP_MAGIC, 8) == 0 && sp_wraw(&io, h32, sizeof h32) == 0 && sp_wraw(&io, h64, sizeof h64) == 0;
    if (io.enc) ok = ok && sp_wraw(&io, salt, 16) == 0;
    uint8_t kc[16]; memcpy(kc, SP_KEYCHK, 16);
    if (io.enc) kv_ctr(&io.aes, SP_KC_NONCE, kc, 16);
    ok = ok && sp_wraw(&io, kc, 16) == 0;
    ok = ok && sp_wr(&io, s->ids, (uint64_t)s->n * 8) == 0 && sp_wr(&io, s->dead, s->n) == 0
            && sp_wr(&io, s->row, ((uint64_t)s->n + 1) * 8) == 0 && (s->norows || sp_wr(&io, s->f, s->fnnz * sizeof *s->f) == 0);
    if (built) ok = ok && sp_wr(&io, s->tptr, ((uint64_t)dim + 1) * 8) == 0;
    if (lists) ok = ok && sp_wr(&io, s->post, h64[5] * sizeof *s->post) == 0;
    if (blocks){
        ok = ok && sp_wr(&io, s->bptr, ((uint64_t)dim + 1) * 8) == 0 && sp_wr(&io, s->sdp, ((uint64_t)dim + 1) * 8) == 0
                && sp_wr(&io, s->sep, ((uint64_t)dim + 1) * 8) == 0 && sp_wr(&io, s->bdoc, (h64[1] + 1) * 8) == 0
                && sp_wr(&io, s->brow, h64[2] * 4) == 0 && sp_wr(&io, s->bscale, h64[1] * 4) == 0;
        if (s->sdim16) ok = ok && sp_wr(&io, s->sdim16, h64[3] * 2) == 0;
        else           ok = ok && sp_wr(&io, s->sdim32, h64[3] * 4) == 0;
        ok = ok && sp_wr(&io, s->slen, h64[3]) == 0 && sp_wr(&io, s->ent, h64[4] * 2) == 0;
    }
    if (knn) ok = ok && sp_wr(&io, s->nbr, (uint64_t)s->nknn_rows * (uint64_t)s->nknn * 4) == 0;
    if (s->norows) ok = ok && sp_wr(&io, s->coff, ((uint64_t)s->n + 1) * 8) == 0 && sp_wr(&io, s->crow, ctot * 2) == 0;
    if (ok && io.enc){ uint8_t tag[16]; kv_cmac_final(&io.mac, tag); ok = fwrite(tag, 1, 16, io.f) == 16; }
    if (fclose(io.f) != 0) ok = 0;
    free(io.tmp); memset(&io, 0, sizeof io);
    if (!ok){ remove(tmp); return -1; }
    if (sp_replace_file(tmp, path) != 0){ remove(tmp); return -1; }
    return 0;
}

static int sp_mono(const uint64_t *p, uint64_t n, uint64_t last){
    if (p[0] != 0) return -1;
    for (uint64_t i = 0; i < n; i++) if (p[i] > p[i + 1]) return -1;
    return p[n] == last ? 0 : -1;
}

/* Everything a search follows without looking: offsets in order and inside their arrays, row numbers inside
 * the rows. A file that passes can still be wrong (it then gives wrong answers) but cannot make a search read
 * or write outside the index. */
static int sp_check(KciSparse *s, uint32_t flags, const uint64_t h64[6]){
    uint32_t dim = s->dim, n = s->n, nb_rows = s->nbuilt;
    if (sp_mono(s->row, n, s->fnnz) != 0) return -1;
    uint32_t nd = 0;
    if (s->norows && (s->coff[0] != 0 || nb_rows != n)) return -1;
    for (uint32_t d = 0; d < n; d++){
        if (s->dead[d] > 1) return -1;
        nd += s->dead[d];
        if (s->row[d + 1] - s->row[d] > 0x7fffffffu) return -1;
        if (!s->norows){
            for (uint64_t x = s->row[d]; x < s->row[d + 1]; x++){
                if (s->f[x].k >= dim || !isfinite(s->f[x].v)) return -1;
                if (x > s->row[d] && s->f[x - 1].k >= s->f[x].k) return -1;
            }
        } else {                                              /* the half-size row: sp_crow_dot trusts every field of it */
            uint64_t len = s->row[d + 1] - s->row[d], o = s->coff[d];
            if (len > 65535 || s->coff[d + 1] < o + 3 || s->coff[d + 1] > s->coff[n]) return -1;
            const uint16_t *h = s->crow + o; float sc; memcpy(&sc, h + 1, sizeof sc);
            uint64_t nh = h[0], cnd = (nh + 15) / 16, nt = len - nh;          /* first part, then the tail */
            uint64_t ext = s->coff[d + 1] - o, used = 3 + cnd + 16 * cnd + nh + 2 * nt;
            if (nh > len || (ext != used && ext != used + 1) || !isfinite(sc) || sc < 0) return -1;   /* + 1: rows start on even entries */
            if (nt > 0 && !(s->head_mass < 1.0f)) return -1;  /* a tail in a file that says whole rows: it would never be added */
            const uint16_t *dir = h + 3, *ck = dir + cnd, *tk = ck + 16 * cnd + nh;
            for (uint64_t y = 0; y < 16 * cnd; y++){
                if (y < nh ? (ck[y] >= dim || (y > 0 && ck[y - 1] >= ck[y])) : ck[y] != 0xFFFF) return -1;
            }
            for (uint64_t c = 0; c < cnd; c++) if (dir[c] != ck[16 * c]) return -1;
            for (uint64_t y = 0; y < nt; y++) if (tk[y] >= dim || (y > 0 && tk[y - 1] >= tk[y])) return -1;
        }
    }
    s->ndead = nd;
    if (!(flags & SP_F_BUILT)) return (flags & (SP_F_LISTS | SP_F_BLOCKS | SP_F_KNN)) || nb_rows != 0 ? -1 : 0;
    if (nb_rows > n || sp_mono(s->tptr, dim, h64[5]) != 0 || h64[5] != s->row[nb_rows]) return -1;
    if (flags & SP_F_LISTS)
        for (uint32_t t = 0; t < dim; t++)
            for (uint64_t x = s->tptr[t]; x < s->tptr[t + 1]; x++){
                if (s->post[x].k >= nb_rows || !isfinite(s->post[x].v)) return -1;
                if (x > s->tptr[t] && fabsf(s->post[x - 1].v) < fabsf(s->post[x].v)) return -1;
            }
    if (flags & SP_F_BLOCKS){
        if (sp_mono(s->bptr, dim, s->nblk) != 0 || sp_mono(s->bdoc, s->nblk, h64[2]) != 0
            || sp_mono(s->sdp, dim, h64[3]) != 0 || sp_mono(s->sep, dim, h64[4]) != 0) return -1;
        for (uint64_t g = 0; g < s->nblk; g++)
            if (s->bdoc[g + 1] - s->bdoc[g] > 0xffffffffu || !isfinite(s->bscale[g]) || s->bscale[g] < 0) return -1;
        for (uint64_t x = 0; x < h64[2]; x++) if (s->brow[x] >= nb_rows) return -1;
        for (uint32_t t = 0; t < dim; t++){
            uint64_t nb = s->bptr[t + 1] - s->bptr[t], e = s->sep[t];
            if (nb > 65535) return -1;
            for (uint64_t x = s->sdp[t]; x < s->sdp[t + 1]; x++){
                uint32_t k = s->sdim16 ? s->sdim16[x] : s->sdim32[x], len = s->slen[x], g = 0;
                if (k >= dim || len == 0 || e + len > s->sep[t + 1]) return -1;
                if (x > s->sdp[t] && (s->sdim16 ? s->sdim16[x - 1] : s->sdim32[x - 1]) > k) return -1;
                for (uint64_t y = e; y < e + len; y++){ g += s->ent[2 * y]; if (g >= nb) return -1; }   /* every group a search adds into */
                e += len;
            }
            if (e != s->sep[t + 1]) return -1;
        }
    } else if (!(flags & SP_F_LISTS)) return -1;             /* built, but neither way to search it */
    if (flags & SP_F_KNN){
        if (!(flags & SP_F_BLOCKS)) return -1;
        uint64_t tot = (uint64_t)s->nknn_rows * (uint64_t)s->nknn;
        for (uint64_t x = 0; x < tot; x++) if (s->nbr[x] != UINT32_MAX && s->nbr[x] >= nb_rows) return -1;
    }
    return 0;
}

static int sp_sum_from_v1(KciSparse *s, const uint32_t *soff, const uint16_t *eblk, const uint8_t *eq, uint64_t tot[2]){
    uint32_t dim = s->dim; int rc = -1;
    if (sp_mono(s->bptr, dim, s->nblk) != 0 || sp_mono(s->sdp, dim, tot[0]) != 0 || sp_mono(s->sep, dim, tot[1]) != 0) return -1;
    uint64_t maxe = 0;
    for (uint32_t t = 0; t < dim; t++){ uint64_t et = s->sep[t + 1] - s->sep[t]; if (et > 0xffffffffu) return -1; if (et > maxe) maxe = et; }
    uint64_t *keys = malloc((size_t)(maxe ? maxe : 1) * sizeof *keys);
    uint32_t *tdim = NULL; uint8_t *tlen = NULL, *slen = NULL, *ent = NULL; uint16_t *d16 = NULL; uint32_t *d32 = NULL;
    uint64_t *nsdp = malloc(((size_t)dim + 1) * sizeof *nsdp), *nsep = malloc(((size_t)dim + 1) * sizeof *nsep);
    uint64_t nd_all = 0, ne_all = 0, tcap = 0;
    if (!keys || !nsdp || !nsep) goto done;
    for (int pass = 0; pass < 2; pass++){
        uint64_t d = 0, e = 0;
        for (uint32_t t = 0; t < dim; t++){
            uint64_t nb = s->bptr[t + 1] - s->bptr[t], et = s->sep[t + 1] - s->sep[t], n = 0;
            if (nb > 65535) goto done;
            for (uint64_t x = s->sdp[t]; x < s->sdp[t + 1]; x++){
                uint32_t k = s->sdim16 ? s->sdim16[x] : s->sdim32[x];
                uint64_t e0 = soff[x], e1 = x + 1 < s->sdp[t + 1] ? soff[x + 1] : et;
                if (k >= dim || e0 > e1 || e1 > et) goto done;
                if (x > s->sdp[t] && (s->sdim16 ? s->sdim16[x - 1] : s->sdim32[x - 1]) >= k) goto done;
                if (x == s->sdp[t] && e0 != 0) goto done;
                for (uint64_t y = e0; y < e1; y++){
                    uint32_t g = eblk[s->sep[t] + y];
                    if (g >= nb || (y > e0 && eblk[s->sep[t] + y - 1] >= g)) goto done;
                    keys[n++] = ((uint64_t)k << 24) | ((uint64_t)g << 8) | (uint64_t)eq[s->sep[t] + y];
                }
            }
            uint32_t nd = 0, ne = 0;
            if (pass == 0){
                sp_enc_runs(keys, (size_t)n, NULL, NULL, NULL, &nd, &ne);
                if (nd > tcap) tcap = nd;
            } else {
                sp_enc_runs(keys, (size_t)n, tdim, tlen, ent + 2 * e, &nd, &ne);
                memcpy(slen + d, tlen, nd);
                if (d16) for (uint32_t x = 0; x < nd; x++) d16[d + x] = (uint16_t)tdim[x];
                else memcpy(d32 + d, tdim, (size_t)nd * sizeof *d32);
                nsdp[t] = d; nsep[t] = e;
            }
            d += nd; e += ne;
        }
        if (pass == 0){
            nd_all = d; ne_all = e;
            tdim = malloc((size_t)(tcap ? tcap : 1) * sizeof *tdim); tlen = malloc((size_t)(tcap ? tcap : 1));
            slen = malloc((size_t)(nd_all ? nd_all : 1)); ent = malloc((size_t)(ne_all ? ne_all : 1) * 2);
            if (s->sdim16) d16 = malloc((size_t)(nd_all ? nd_all : 1) * sizeof *d16); else d32 = malloc((size_t)(nd_all ? nd_all : 1) * sizeof *d32);
            if (!tdim || !tlen || !slen || !ent || (!d16 && !d32)) goto done;
        } else { nsdp[dim] = d; nsep[dim] = e; }
    }
    if (s->sdim16){ free(s->sdim16); s->sdim16 = d16; d16 = NULL; } else { free(s->sdim32); s->sdim32 = d32; d32 = NULL; }
    free(s->sdp); s->sdp = nsdp; nsdp = NULL; free(s->sep); s->sep = nsep; nsep = NULL;
    s->slen = slen; slen = NULL; s->ent = ent; ent = NULL;
    tot[0] = nd_all; tot[1] = ne_all; rc = 0;
done:
    free(keys); free(tdim); free(tlen); free(slen); free(ent); free(d16); free(d32); free(nsdp); free(nsep);
    return rc;
}

KciSparse *kci_sparse_load(const char *path, const uint8_t *key, int keylen){
    if (!path || (key && keylen <= 0) || (!key && keylen != 0)) return NULL;
    SpIO io; memset(&io, 0, sizeof io);
    KciSparse *s = NULL; int ok = 0;
    io.f = fopen(path, "rb");
    if (!io.f) return NULL;
    uint8_t magic[8], salt[16], kc[16]; uint32_t h32[8]; uint64_t h64[6];
    do {
        if (fread(magic, 1, 8, io.f) != 8 || memcmp(magic, SP_MAGIC, 8) != 0) break;
        if (fread(h32, sizeof h32, 1, io.f) != 1 || fread(h64, sizeof h64, 1, io.f) != 1) break;
        uint32_t flags = h32[0], dim = h32[1];
        if ((flags & ~(SP_F_ALL | 0xff00u)) || SP_F_HEAD(flags) >= 200 || dim == 0 || ((flags & SP_F_SUM2) && !(flags & SP_F_BLOCKS))) break;
        /* a key says the caller expects an encrypted, authenticated file: a plain one in its place is refused */
        if (((flags & SP_F_ENC) != 0) != (key != NULL)) break;
        if (sp_fseek(io.f, 0, SEEK_END) != 0) break;
        int64_t fsize = (int64_t)sp_ftell(io.f);
        uint64_t head = 8 + sizeof h32 + sizeof h64 + (key ? 16 : 0) + 16, tail = key ? 16 : 0;
        if (fsize < 0 || (uint64_t)fsize < head + tail) break;
        uint64_t pay = sp_payload(flags, h32, h64, (uint64_t)fsize);
        if (pay == 0 || pay != (uint64_t)fsize - head - tail) break;
        if (sp_fseek(io.f, (int64_t)(8 + sizeof h32 + sizeof h64), SEEK_SET) != 0) break;
        if (key){
            if (fread(salt, 1, 16, io.f) != 16 || sp_io_keys(&io, key, keylen, salt) != 0) break;
        }
        if (fread(kc, 1, 16, io.f) != 16) break;
        if (key) kv_ctr(&io.aes, SP_KC_NONCE, kc, 16);
        if (memcmp(kc, SP_KEYCHK, 16) != 0) break;            /* wrong key (or not this kind of file) */
        if (key){
            uint64_t left = (uint64_t)fsize - 16; uint8_t tag[16], want[16], diff = 0;
            if (sp_fseek(io.f, 0, SEEK_SET) != 0) break;
            int bad = 0;
            while (left && !bad){
                size_t c = left < SP_PIECE ? (size_t)left : SP_PIECE;
                if (fread(io.tmp, 1, c, io.f) != c) bad = 1; else kv_cmac_update(&io.mac, io.tmp, c);
                left -= c;
            }
            if (bad || fread(want, 1, 16, io.f) != 16) break;
            kv_cmac_final(&io.mac, tag);
            for (int i = 0; i < 16; i++) diff |= (uint8_t)(tag[i] ^ want[i]);
            if (diff) break;
            if (sp_fseek(io.f, (int64_t)head, SEEK_SET) != 0) break;
        }
        s = calloc(1, sizeof *s);
        if (!s) break;
        uint32_t n = h32[2];
        s->dim = dim; s->n = n; s->ncap = n; s->fnnz = h64[0]; s->fcap = h64[0]; s->gen = 1;
        s->built = (flags & SP_F_BUILT) != 0; s->nbuilt = h32[3];
        s->ids = malloc(((size_t)n ? (size_t)n : 1) * 8); s->dead = malloc((size_t)n ? (size_t)n : 1);
        s->norows = (flags & SP_F_NOROWS) != 0;
        s->head_mass = SP_F_HEAD(flags) ? (float)SP_F_HEAD(flags) / 200.0f : 1.0f;
        if (s->norows) s->fcap = 0;
        s->row = malloc(((size_t)n + 1) * 8);
        if (!s->norows) s->f = malloc((size_t)(h64[0] ? h64[0] : 1) * sizeof *s->f);
        if (!s->ids || !s->dead || !s->row || (!s->norows && !s->f)) break;
        if (sp_rd(&io, s->ids, (uint64_t)n * 8) != 0 || sp_rd(&io, s->dead, n) != 0
            || sp_rd(&io, s->row, ((uint64_t)n + 1) * 8) != 0 || (!s->norows && sp_rd(&io, s->f, h64[0] * sizeof *s->f) != 0)) break;
        if (flags & SP_F_BUILT){
            s->tptr = malloc(((size_t)dim + 1) * 8);
            if (!s->tptr || sp_rd(&io, s->tptr, ((uint64_t)dim + 1) * 8) != 0) break;
        }
        if (flags & SP_F_LISTS){
            s->post = malloc((size_t)(h64[5] ? h64[5] : 1) * sizeof *s->post);
            if (!s->post || sp_rd(&io, s->post, h64[5] * sizeof *s->post) != 0) break;
        }
        if (flags & SP_F_BLOCKS){
            size_t nblk = (size_t)h64[1], nr = (size_t)h64[2], nd = (size_t)h64[3], ne = (size_t)h64[4];
            s->nblk = h64[1];
            s->bptr = malloc(((size_t)dim + 1) * 8); s->sdp = malloc(((size_t)dim + 1) * 8); s->sep = malloc(((size_t)dim + 1) * 8);
            s->bdoc = malloc((nblk + 1) * 8); s->brow = malloc((nr ? nr : 1) * 4); s->bscale = malloc((nblk ? nblk : 1) * 4);
            if (dim <= 65536) s->sdim16 = malloc((nd ? nd : 1) * 2); else s->sdim32 = malloc((nd ? nd : 1) * 4);
            if (!s->bptr || !s->sdp || !s->sep || !s->bdoc || !s->brow || !s->bscale || (!s->sdim16 && !s->sdim32)) break;
            if (sp_rd(&io, s->bptr, ((uint64_t)dim + 1) * 8) != 0 || sp_rd(&io, s->sdp, ((uint64_t)dim + 1) * 8) != 0
                || sp_rd(&io, s->sep, ((uint64_t)dim + 1) * 8) != 0 || sp_rd(&io, s->bdoc, (h64[1] + 1) * 8) != 0
                || sp_rd(&io, s->brow, h64[2] * 4) != 0 || sp_rd(&io, s->bscale, h64[1] * 4) != 0) break;
            if (s->sdim16 ? sp_rd(&io, s->sdim16, h64[3] * 2) != 0 : sp_rd(&io, s->sdim32, h64[3] * 4) != 0) break;
            if (flags & SP_F_SUM2){
                s->slen = malloc(nd ? nd : 1); s->ent = malloc((ne ? ne : 1) * 2);
                if (!s->slen || !s->ent || sp_rd(&io, s->slen, h64[3]) != 0 || sp_rd(&io, s->ent, h64[4] * 2) != 0) break;
            } else {
                uint32_t *soff = malloc((nd ? nd : 1) * 4); uint16_t *eblk = malloc((ne ? ne : 1) * 2); uint8_t *eq = malloc(ne ? ne : 1);
                int bad = !soff || !eblk || !eq || sp_rd(&io, soff, h64[3] * 4) != 0 || sp_rd(&io, eblk, h64[4] * 2) != 0 || sp_rd(&io, eq, h64[4]) != 0;
                if (!bad) bad = sp_sum_from_v1(s, soff, eblk, eq, &h64[3]) != 0;
                free(soff); free(eblk); free(eq);
                if (bad) break;
            }
        }
        if (flags & SP_F_KNN){
            s->nknn = (int)h32[4]; s->nknn_rows = h32[5];
            uint64_t tot = (uint64_t)h32[4] * (uint64_t)h32[5];
            s->nbr = malloc((size_t)(tot ? tot : 1) * 4);
            if (!s->nbr || sp_rd(&io, s->nbr, tot * 4) != 0) break;
        }
        if (s->norows){
            uint64_t ctot = (uint64_t)h32[6] | ((uint64_t)h32[7] << 32);
            s->coff = malloc(((size_t)n + 1) * 8); s->crow = malloc(((size_t)ctot + SP_CROW_PAD) * sizeof *s->crow);
            if (!s->coff || !s->crow || sp_rd(&io, s->coff, ((uint64_t)n + 1) * 8) != 0 || sp_rd(&io, s->crow, ctot * 2) != 0) break;
            memset(s->crow + ctot, 0, SP_CROW_PAD * sizeof *s->crow);
            if (s->coff[n] != ctot) break;
            s->ncrow = n;
        }
        if (sp_check(s, flags, h64) != 0) break;
        if ((flags & SP_F_BLOCKS) && sp_make_sck(s) != 0) break;
        if ((flags & SP_F_BLOCKS) && dim < 65536 && !s->norows && sp_make_crow(s, 0) != 0) break;
        if (flags & SP_F_BLOCKS) sp_make_boff(s);
        ok = 1;
    } while (0);
    fclose(io.f); free(io.tmp); memset(&io, 0, sizeof io);
    if (!ok){ kci_sparse_free(s); return NULL; }
    return s;
}

#define SP_SEAL_MAGIC "KCISEAL1"
#define SP_SEAL_ENC   "KCISEAL1 encryption key"
#define SP_SEAL_MAC   "KCISEAL1 mac key"
#define SP_SEAL_OVER  40u
static void sp_seal_keys(const uint8_t *key, int keylen, const uint8_t salt[16], KvAes *aes, KvCmac *mac){
    uint8_t mk[32], ek[32], ak[32];
    kdf_derive(key, keylen, salt, 16, mk);
    kv_subkey(mk, SP_SEAL_ENC, ek); kv_subkey(mk, SP_SEAL_MAC, ak);
    kv_aes_init(aes, ek); kv_cmac_init(mac, ak);
    memset(mk, 0, 32); memset(ek, 0, 32); memset(ak, 0, 32);
}
static void sp_seal_ctr(const KvAes *aes, uint8_t *buf, uint64_t n){
    uint64_t piece = 0;
    while (n){ size_t c = n < SP_PIECE ? (size_t)n : SP_PIECE; kv_ctr(aes, piece++, buf, c); buf += c; n -= c; }
}
int64_t kci_seal(const uint8_t *in, uint64_t n, const uint8_t *key, int keylen, uint8_t *out){
    if ((!in && n) || !key || keylen <= 0 || !out || n > (uint64_t)INT64_MAX - SP_SEAL_OVER) return -1;
    uint8_t salt[16];
    if (kv_randbytes(salt, 16) != 0) return -1;                /* no random salt, nothing written */
    KvAes aes; KvCmac mac;
    sp_seal_keys(key, keylen, salt, &aes, &mac);
    memcpy(out, SP_SEAL_MAGIC, 8); memcpy(out + 8, salt, 16);
    if (n) memcpy(out + 24, in, (size_t)n);
    sp_seal_ctr(&aes, out + 24, n);
    kv_cmac_update(&mac, out, (size_t)(24 + n));
    kv_cmac_final(&mac, out + 24 + n);
    memset(&aes, 0, sizeof aes); memset(&mac, 0, sizeof mac);
    return (int64_t)(n + SP_SEAL_OVER);
}
int64_t kci_unseal(const uint8_t *in, uint64_t n, const uint8_t *key, int keylen, uint8_t *out){
    if (!in || !key || keylen <= 0 || n < SP_SEAL_OVER || n > (uint64_t)INT64_MAX) return -1;
    if (memcmp(in, SP_SEAL_MAGIC, 8) != 0) return -1;
    uint64_t m = n - SP_SEAL_OVER;
    if (m && !out) return -1;
    KvAes aes; KvCmac mac; uint8_t tag[16], diff = 0;
    sp_seal_keys(key, keylen, in + 8, &aes, &mac);
    kv_cmac_update(&mac, in, (size_t)(24 + m));
    kv_cmac_final(&mac, tag);
    for (int i = 0; i < 16; i++) diff |= (uint8_t)(tag[i] ^ in[24 + m + i]);
    if (!diff && m){ memcpy(out, in + 24, (size_t)m); sp_seal_ctr(&aes, out, m); }
    memset(&aes, 0, sizeof aes); memset(&mac, 0, sizeof mac);
    return diff ? -1 : (int64_t)m;
}

