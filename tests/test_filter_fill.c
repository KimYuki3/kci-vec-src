/* A filter that passes fewer than k ids returns every one of them.
 *   gcc -O2 -march=native -I. tests/test_filter_fill.c kci_vec.c -lpthread -lm -o test_filter_fill && ./test_filter_fill */
#include "kci_vec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if(!(c)){ fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while(0)

enum { N = 60000, D = 32, K = 10, NQ = 40 };
static float *X, Q[NQ][D];
static unsigned rs = 2468;
static float rnd(void){ rs = rs*1664525u + 1013904223u; return (float)(rs>>8)/(float)(1u<<24) - 0.5f; }

static int same_set(const uint64_t *a, int na, const uint64_t *b, int nb){
    if(na != nb) return 0;
    for(int i=0;i<na;i++){ int ok=0; for(int j=0;j<nb;j++) if(a[i]==b[j]){ ok=1; break; } if(!ok) return 0; }
    return 1;
}

static void run(int l2, int lists){
    char name[32]; snprintf(name, sizeof name, "%s/%s", l2 ? "l2" : "cosine", lists ? "lists" : "graph");
    KciVec *v = l2 ? kci_vec_new_l2(D, 16, 100, 0.0f) : kci_vec_new(D, 16, 100);
    CHECK(v != NULL, "%s: new", name); if(!v) return;
    kci_vec_keep_float(v, 1);
    for(int i=0;i<N;i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i*D);
    CHECK(kci_vec_build(v, 8, NULL) == 0, "%s: build", name);
    if(lists) CHECK(kci_vec_build_ivf(v, 256, 8) == 0, "%s: build_ivf", name);
    KciVecSearcher *s = kci_vec_searcher_new(v);
    CHECK(s != NULL, "%s: searcher", name); if(!s){ kci_vec_free(v); return; }

    static uint64_t bits[N/64 + 1];
    uint64_t got[64], off[64], want[64]; float sc[64], scw[64];
    /* 3 ids pass (fewer than k): all 3 must come back, with the scores brute force gives */
    int short_off = 0, bad = 0, badsc = 0;
    for(int q=0;q<NQ;q++){
        memset(bits, 0, sizeof bits);
        for(int j=0;j<3;j++){ unsigned id = (unsigned)((q*7919u + j*20011u + 13u) % N); bits[id>>6] |= 1ULL << (id & 63); }
        KciVecIdBits B = { bits, N };
        int m = kci_vec_search_exact_filtered(v, Q[q], K, 1, kci_vec_filter_id_bits, &B, want, scw);
        kci_vec_searcher_set_filter_fill(s, 0);
        int n0 = lists ? kci_vec_search_ivf_filtered(s, Q[q], K, 8, 100, kci_vec_filter_id_bits, &B, off, NULL)
                       : kci_vec_search_filtered(s, Q[q], K, 50, 100, kci_vec_filter_id_bits, &B, off, NULL);
        if(n0 < m) short_off++;
        kci_vec_searcher_set_filter_fill(s, 1);
        int n = lists ? kci_vec_search_ivf_filtered(s, Q[q], K, 8, 100, kci_vec_filter_id_bits, &B, got, sc)
                      : kci_vec_search_filtered(s, Q[q], K, 50, 100, kci_vec_filter_id_bits, &B, got, sc);
        if(m != 3 || !same_set(got, n, want, m)) bad++;
        else for(int i=0;i<n;i++) if(got[i] != want[i] || sc[i] != scw[i]) badsc++;
    }
    CHECK(bad == 0 && badsc == 0, "%s: 3 ids pass: %d of %d queries did not return all of them, %d orders / scores differ from brute force", name, bad, NQ, badsc);
    /* the test must be able to fail: without the fill the index's own search leaves ids out */
    CHECK(short_off > 0, "%s: with the fill off every query already returned all 3 (the test shows nothing)", name);
    printf("%s: with the fill off %d of %d queries returned fewer than the 3 that pass\n", name, short_off, NQ);

    /* half of the ids pass (k answers found): the answer is the index's own, the same with the fill on or off */
    memset(bits, 0, sizeof bits);
    for(int i=0;i<N;i+=2) bits[i>>6] |= 1ULL << (i & 63);
    KciVecIdBits H = { bits, N }; int diff = 0;
    for(int q=0;q<NQ;q++){
        kci_vec_searcher_set_filter_fill(s, 0);
        int n0 = lists ? kci_vec_search_ivf_filtered(s, Q[q], K, 8, 100, kci_vec_filter_id_bits, &H, off, NULL)
                       : kci_vec_search_filtered(s, Q[q], K, 50, 100, kci_vec_filter_id_bits, &H, off, NULL);
        kci_vec_searcher_set_filter_fill(s, 1);
        int n = lists ? kci_vec_search_ivf_filtered(s, Q[q], K, 8, 100, kci_vec_filter_id_bits, &H, got, NULL)
                      : kci_vec_search_filtered(s, Q[q], K, 50, 100, kci_vec_filter_id_bits, &H, got, NULL);
        if(n != K || n0 != K || memcmp(got, off, sizeof(uint64_t)*K) != 0) diff++;
    }
    CHECK(diff == 0, "%s: half pass: %d queries changed with the fill on", name, diff);
    /* nothing passes: 0 */
    memset(bits, 0, sizeof bits);
    int n = lists ? kci_vec_search_ivf_filtered(s, Q[0], K, 8, 100, kci_vec_filter_id_bits, &H, got, NULL)
                  : kci_vec_search_filtered(s, Q[0], K, 50, 100, kci_vec_filter_id_bits, &H, got, NULL);
    CHECK(n == 0, "%s: nothing passes, got %d", name, n);
    kci_vec_searcher_free(s); kci_vec_free(v);
}

int main(void){
    X = malloc(sizeof(float)*(size_t)N*D); if(!X) return 2;
    for(size_t i=0;i<(size_t)N*D;i++) X[i] = rnd();
    for(int q=0;q<NQ;q++) for(int d=0;d<D;d++) Q[q][d] = rnd();
    run(0, 0); run(1, 0); run(0, 1); run(1, 1);
    free(X);
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails != 0;
}
