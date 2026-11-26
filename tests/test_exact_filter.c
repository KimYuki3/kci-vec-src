/* kci_vec_search_exact_filtered, kci_vec_filter_id_bits, and a searcher kept across inserts.
 *   gcc -O2 -march=native -I. tests/test_exact_filter.c kci_vec.c -lpthread -lm -o test_exact_filter */
#include "kci_vec.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if(!(c)){ fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while(0)

enum { N = 6000, D = 48, K = 10, NQ = 60 };
static float X[N][D], Q[NQ][D];
static unsigned char dead[N];
static unsigned rs = 12345;
static float rnd(void){ rs = rs*1664525u + 1013904223u; return (float)(rs>>8)/(float)(1u<<24) - 0.5f; }

typedef struct { int mod, rem; } Mod;
static int f_mod(uint64_t id, void *c){ const Mod *m = c; return (int)(id % (unsigned)m->mod) == m->rem; }
static int f_none(uint64_t id, void *c){ (void)id; (void)c; return 0; }

/* the true top-k among live ids that pass, by the metric (cosine: larger first; l2: smaller first) */
static int truth(int l2, const float *q, KciVecFilter f, void *ctx, int n, int k, uint64_t *out){
    static double sc[N + 4000]; static int idx[N + 4000]; int m = 0;
    double qn = 0; for(int d=0;d<D;d++) qn += (double)q[d]*q[d];
    for(int i=0;i<n;i++){ if(dead[i] || (f && !f((uint64_t)i, ctx))) continue;
        double dot=0, xn=0, dd=0; for(int d=0;d<D;d++){ dot += (double)q[d]*X[i][d]; xn += (double)X[i][d]*X[i][d]; double e=(double)q[d]-X[i][d]; dd += e*e; }
        sc[m] = l2 ? -dd : dot/sqrt(xn*qn); idx[m++] = i; }
    int r = m < k ? m : k;
    for(int a=0;a<r;a++){ int b=a; for(int j=a+1;j<m;j++) if(sc[j]>sc[b]) b=j;
        double t=sc[a]; sc[a]=sc[b]; sc[b]=t; int u=idx[a]; idx[a]=idx[b]; idx[b]=u; out[a]=(uint64_t)idx[a]; }
    return r;
}
static int same_set(const uint64_t *a, int na, const uint64_t *b, int nb){
    if(na != nb) return 0;
    for(int i=0;i<na;i++){ int ok=0; for(int j=0;j<nb;j++) if(a[i]==b[j]){ ok=1; break; } if(!ok) return 0; }
    return 1;
}

static void run(int l2){
    const char *name = l2 ? "l2" : "cosine";
    memset(dead, 0, sizeof dead);
    KciVec *v = l2 ? kci_vec_new_l2(D, 16, 200, 0.0f) : kci_vec_new(D, 16, 200);
    CHECK(v != NULL, "%s: new", name); if(!v) return;
    kci_vec_keep_float(v, 1);
    for(int i=0;i<N;i++) kci_vec_add(v, (uint64_t)i, X[i]);
    CHECK(kci_vec_build(v, 4, NULL) == 0, "%s: build", name);
    for(int i=0;i<N;i+=7){ kci_vec_delete(v, (uint64_t)i); dead[i]=1; }          /* deleted ids must not come back */

    Mod few = { 50, 3 }, quarter = { 4, 1 }, all = { 1, 0 };
    struct { KciVecFilter f; void *c; const char *what; } F[] = { { f_mod, &few, "1 in 50" }, { f_mod, &quarter, "1 in 4" }, { f_mod, &all, "all" } };
    uint64_t got[64], want[64]; float sc[64];
    for(int fi=0; fi<3; fi++) for(int th=1; th<=4; th+=3){
        int bad=0, badsc=0;
        for(int q=0;q<NQ;q++){
            int n = kci_vec_search_exact_filtered(v, Q[q], K, th, F[fi].f, F[fi].c, got, sc);
            int m = truth(l2, Q[q], F[fi].f, F[fi].c, N, K, want);
            if(!same_set(got, n, want, m)) bad++;
            for(int i=0;i<n;i++){ if(dead[got[i]] || !F[fi].f(got[i], F[fi].c)) bad++; }
            for(int i=1;i<n;i++) if(l2 ? sc[i] < sc[i-1] : sc[i] > sc[i-1]) badsc++;
        }
        CHECK(bad == 0 && badsc == 0, "%s, filter %s, %d thread(s): %d queries differ from the truth, %d score orders wrong", name, F[fi].what, th, bad, badsc);
    }
    /* fewer ids pass than k: all of them, no more; nothing passes: 0; no filter: the unfiltered function */
    Mod rare = { 2000, 12 };                                        /* ids 12, 2012, 4012: three ids, none deleted (not multiples of 7) */
    int n = kci_vec_search_exact_filtered(v, Q[0], K, 1, f_mod, &rare, got, sc);
    int m = truth(l2, Q[0], f_mod, &rare, N, K, want);
    CHECK(n == 3 && same_set(got, n, want, m), "%s: 3 ids pass, got %d", name, n);
    CHECK(kci_vec_search_exact_filtered(v, Q[0], K, 1, f_none, NULL, got, sc) == 0, "%s: nothing passes", name);
    uint64_t g2[64]; int n1 = kci_vec_search_exact_filtered(v, Q[1], K, 1, NULL, NULL, got, NULL), n2 = kci_vec_search_exact(v, Q[1], K, 1, g2, NULL);
    CHECK(n1 == n2 && memcmp(got, g2, sizeof(uint64_t)*(size_t)n1) == 0, "%s: filt NULL = kci_vec_search_exact", name);

    /* the ready-made bitmap filter: with brute force and with the graph's filtered search */
    static uint64_t bits[(N + 4000)/64 + 1]; memset(bits, 0, sizeof bits);
    for(int i=0;i<N;i++) if(i % 5 == 2) bits[i>>6] |= 1ULL << (i & 63);
    KciVecIdBits B = { bits, N };
    Mod fifth = { 5, 2 };
    CHECK(kci_vec_filter_id_bits(2, &B) == 1 && kci_vec_filter_id_bits(3, &B) == 0 && kci_vec_filter_id_bits((uint64_t)N + 2, &B) == 0
          && kci_vec_filter_id_bits(~0ULL, &B) == 0, "%s: bitmap filter on single ids", name);
    int bad = 0;
    for(int q=0;q<NQ;q++){ n = kci_vec_search_exact_filtered(v, Q[q], K, 4, kci_vec_filter_id_bits, &B, got, NULL);
        m = truth(l2, Q[q], f_mod, &fifth, N, K, want); if(!same_set(got, n, want, m)) bad++; }
    CHECK(bad == 0, "%s: bitmap filter with brute force: %d queries differ", name, bad);

    /* a searcher made now, kept while the index grows by two thirds: the graph searches must stay inside their marks
     * (ASan) and find the new ids */
    KciVecSearcher *s = kci_vec_searcher_new(v);
    CHECK(s != NULL, "%s: searcher", name);
    static float Y[4000][D];
    for(int i=0;i<4000;i++){ for(int d=0;d<D;d++) Y[i][d] = rnd(); if(kci_vec_insert(v, (uint64_t)(N + i), Y[i]) < 0){ CHECK(0, "%s: insert %d", name, i); break; } }
    int found = 0, allowed_only = 1;
    for(int i=0;i<200;i++){
        n = kci_vec_search_rerank(s, Y[i*20], K, 100, 100, got, NULL); for(int j=0;j<n;j++) if(got[j] == (uint64_t)(N + i*20)) found++;
        n = kci_vec_search_r(s, Y[i*20], K, 100, got, NULL); (void)n;
        n = kci_vec_search_filtered(s, Y[i*20], K, 100, 100, kci_vec_filter_id_bits, &B, got, NULL);
        for(int j=0;j<n;j++) if(got[j] % 5 != 2 || got[j] >= (uint64_t)N) allowed_only = 0;
    }
    CHECK(found >= 196, "%s: the old searcher finds %d of 200 inserted vectors", name, found);
    CHECK(allowed_only, "%s: the old searcher's filtered search returned an id outside the bitmap", name);
    /* brute force sees the inserted rows too */
    n = kci_vec_search_exact_filtered(v, Y[5], 1, 1, f_mod, &all, got, NULL);
    CHECK(n == 1 && got[0] == (uint64_t)(N + 5), "%s: brute force finds an inserted vector", name);
    kci_vec_searcher_free(s); kci_vec_free(v);
}

static void dedup(void){
    /* identical vectors kept once (kci_vec_set_dedup): a row is scored when any id on it passes; only passing ids come out */
    KciVec *v = kci_vec_new(D, 16, 200); kci_vec_keep_float(v, 1);
    if(kci_vec_set_dedup(v, 1) != 0){ CHECK(0, "dedup: set_dedup"); kci_vec_free(v); return; }
    for(int i=0;i<2000;i++) kci_vec_add(v, (uint64_t)i, X[i % 400]);          /* every vector five times: ids i, i+400, ... */
    CHECK(kci_vec_build(v, 2, NULL) == 0, "dedup: build");
    Mod odd = { 2, 1 }; uint64_t got[64]; float sc[64]; int bad = 0;
    for(int q=0;q<40;q++){
        int n = kci_vec_search_exact_filtered(v, X[q], 5, 1, f_mod, &odd, got, sc);
        /* ids of vector q: q, q+400, q+800, q+1200, q+1600, all of q's parity: for an odd q all five pass and score ~1, for an even q none of its own */
        int self = 0; for(int j=0;j<n;j++){ if(got[j] % 2 != 1) bad++; if(got[j] % 400 == (uint64_t)q && sc[j] > 0.999f) self++; }
        if(self != (q % 2 ? 5 : 0) || n != 5) bad++;
    }
    CHECK(bad == 0, "dedup: %d problems", bad);
    kci_vec_free(v);
}

int main(void){
    for(int i=0;i<N;i++) for(int d=0;d<D;d++) X[i][d] = rnd();
    for(int q=0;q<NQ;q++) for(int d=0;d<D;d++) Q[q][d] = rnd();
    run(0); run(1); dedup();
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails != 0;
}
