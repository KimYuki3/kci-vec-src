/* Groups of nearly equal vectors: every row must be findable.
 *   gcc -O2 -march=native -I. tests/test_near_dup.c kci_vec.c -lpthread -lm -o test_near_dup && ./test_near_dup */
#include "kci_vec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if(!(c)){ fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while(0)

enum { D = 48, NC = 200, PER = 100, N = NC*PER, K = 10, NQ = 100 };
static float C[NC][D], Q[NQ][D], *X;
static unsigned rs = 97531;
static float rnd(void){ rs = rs*1664525u + 1013904223u; return (float)(rs>>8)/(float)(1u<<24) - 0.5f; }

static void run(int l2, int threads){
    char name[32]; snprintf(name, sizeof name, "%s/%d threads", l2 ? "l2" : "cosine", threads);
    KciVec *v = l2 ? kci_vec_new_l2(D, 16, 200, 0.0f) : kci_vec_new(D, 16, 200);
    kci_vec_keep_float(v, 1);
    for(int i=0;i<N;i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i*D);
    CHECK(kci_vec_build(v, threads, NULL) == 0, "%s: build", name);
    KciVecSearcher *s = kci_vec_searcher_new(v); uint64_t a[K], t[K]; long hit = 0, self = 0;
    for(int q=0;q<NQ;q++){
        int m = kci_vec_search_exact(v, Q[q], K, 4, t, NULL), n = kci_vec_search_rerank(s, Q[q], K, 200, 200, a, NULL);
        for(int i=0;i<m;i++) for(int j=0;j<n;j++) if(t[i]==a[j]){ hit++; break; }
    }
    /* a search for a stored row finds that row: it has a link leading to it */
    for(int i=0;i<N;i+=7){ int n = kci_vec_search_rerank(s, X + (size_t)i*D, 1, 200, 200, a, NULL); self += (n==1 && a[0]==(uint64_t)i); }
    double r = (double)hit/(NQ*K), sf = (double)self/((N+6)/7);
    printf("%s: recall@%d %.3f at ef 200, a stored row found by a search for itself %.4f\n", name, K, r, sf);
    CHECK(r >= 0.95, "%s: recall %.3f", name, r);
    CHECK(sf >= 0.99, "%s: %.4f of the stored rows found by a search for themselves", name, sf);
    kci_vec_searcher_free(s); kci_vec_free(v);
}

int main(void){
    X = malloc(sizeof(float)*(size_t)N*D); if(!X) return 2;
    for(int c=0;c<NC;c++) for(int d=0;d<D;d++) C[c][d] = rnd();
    for(int i=0;i<N;i++) for(int d=0;d<D;d++) X[(size_t)i*D+d] = C[i%NC][d] + 0.02f*rnd();
    for(int q=0;q<NQ;q++) for(int d=0;d<D;d++) Q[q][d] = C[q*37%NC][d] + 0.02f*rnd();
    run(0, 1); run(0, 8); run(1, 1); run(1, 8);
    free(X);
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails != 0;
}
