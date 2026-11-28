/* Groups of nearly equal vectors: every row must be findable.
 *   gcc -O2 -march=native -I. tests/test_near_dup.c kci_vec.c -lpthread -lm -o test_near_dup && ./test_near_dup */
#include "kci_vec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static void reach(int gn, int gs, int dim, int M, float noise, int metric, int threads){
    int n = gn*gs; float *Y = malloc(sizeof(float)*(size_t)n*dim), *c = malloc(sizeof(float)*dim);
    const char *name = metric==0 ? "l2" : metric==1 ? "cosine" : "ip";
    for(int g=0;g<gn;g++){ for(int d=0;d<dim;d++) c[d] = rnd();
        for(int i=0;i<gs;i++) for(int d=0;d<dim;d++) Y[((size_t)g*gs+i)*dim+d] = c[d] + noise*rnd(); }
    KciVec *v = metric==0 ? kci_vec_new_l2(dim, M, 100, 0.0f) : metric==1 ? kci_vec_new(dim, M, 100) : kci_vec_new_ip(dim, M, 100, 0.0f);
    kci_vec_keep_float(v, 1);
    for(int i=0;i<n;i++) kci_vec_add(v, (uint64_t)i, Y + (size_t)i*dim);
    CHECK(kci_vec_build(v, threads, NULL) == 0, "reach %s: build", name);
    CHECK(kci_vec_unreached(v) == 0, "reach %s, %d x %d, M %d, %d threads: %u rows unreached", name, gn, gs, M, threads, kci_vec_unreached(v));
    if(metric != 2 && gn > 1){                 /* a search as wide as two groups for a stored row */
        KciVecSearcher *s = kci_vec_searcher_new(v); int found = 0, nq = 300; uint64_t ids[10];
        for(int q=0;q<nq;q++){ uint64_t r = (uint64_t)q*2654435761u % (uint64_t)n;
            int m = kci_vec_search_rerank(s, Y + (size_t)r*dim, 10, 2*gs, 2*gs, ids, NULL);
            for(int i=0;i<m;i++) if(ids[i]==r){ found++; break; } }
        CHECK(found >= nq*9/10, "reach %s, %d x %d: %d of %d rows found by a search for themselves", name, gn, gs, found, nq);
        printf("reach %s, %d x %d, M %d, %d threads: unreached %u, %d of %d rows found by a search for themselves\n", name, gn, gs, M, threads, kci_vec_unreached(v), found, nq);
        kci_vec_searcher_free(s);
    } else printf("reach %s, %d x %d, M %d, %d threads: unreached %u\n", name, gn, gs, M, threads, kci_vec_unreached(v));
    kci_vec_free(v); free(Y); free(c);
}

static float gauss(void){ float a = rnd() + 0.5f + 1e-7f, b = rnd() + 0.5f; return sqrtf(-2*logf(a))*cosf(6.2831853f*b); }
static void reach_short(void){
    enum { SN = 30000, SD = 64 };
    float *Y = malloc(sizeof(float)*(size_t)SN*SD);
    for(int i=0;i<SN;i++){ float len = expf(gauss()); for(int d=0;d<SD;d++) Y[(size_t)i*SD+d] = len*gauss(); }
    KciVec *v = kci_vec_new_ip(SD, 16, 200, 0.0f);
    kci_vec_keep_float(v, 1);
    for(int i=0;i<SN;i++) kci_vec_add(v, (uint64_t)i, Y + (size_t)i*SD);
    CHECK(kci_vec_build(v, 8, NULL) == 0, "short rows: build");
    printf("ip with rows of very different lengths: unreached %u of %d\n", kci_vec_unreached(v), SN);
    CHECK(kci_vec_unreached(v) == 0, "short rows: %u rows unreached", kci_vec_unreached(v));
    kci_vec_free(v); free(Y);
}

static void order(int gn, int gs, int dim, float noise, int metric){
    int n = gn*gs, nq = 100; float *Y = malloc(sizeof(float)*(size_t)n*dim), *c = malloc(sizeof(float)*dim); double *dd = malloc(sizeof(double)*n);
    const char *name = metric==0 ? "l2" : "cosine";
    for(int g=0;g<gn;g++){ for(int d=0;d<dim;d++) c[d] = rnd();
        for(int i=0;i<gs;i++) for(int d=0;d<dim;d++) Y[((size_t)g*gs+i)*dim+d] = c[d] + noise*rnd(); }
    KciVec *v = metric==0 ? kci_vec_new_l2(dim, 16, 100, 0.0f) : kci_vec_new(dim, 16, 100);
    kci_vec_keep_float(v, 1);
    for(int i=0;i<n;i++) kci_vec_add(v, (uint64_t)i, Y + (size_t)i*dim);
    CHECK(kci_vec_build(v, 8, NULL) == 0, "order %s: build", name);
    KciVecSearcher *s = kci_vec_searcher_new(v); long he = 0, hg = 0; uint64_t a[10], b[10], top[10];
    for(int q=0;q<nq;q++){ size_t r = (size_t)((uint64_t)q*2654435761u % (uint64_t)n); const float *qv = Y + r*dim;
        double qn = 0; for(int d=0;d<dim;d++) qn += (double)qv[d]*qv[d];
        for(int i=0;i<n;i++){ const float *x = Y + (size_t)i*dim; double e = 0, xn = 0, dp = 0;
            for(int d=0;d<dim;d++){ double t = (double)qv[d]-x[d]; e += t*t; xn += (double)x[d]*x[d]; dp += (double)qv[d]*x[d]; }
            dd[i] = metric==0 ? e : 1.0 - dp/sqrt(qn*xn); }
        for(int k=0;k<10;k++){ int bi = -1; for(int i=0;i<n;i++) if(dd[i] >= 0 && (bi < 0 || dd[i] < dd[bi])) bi = i;
            top[k] = (uint64_t)bi; dd[bi] = -1; }
        int m = kci_vec_search_exact(v, qv, 10, 4, b, NULL), g = kci_vec_search_rerank(s, qv, 10, 4*gs, 4*gs, a, NULL);
        for(int j=0;j<10;j++){ for(int i=0;i<m;i++) if(b[i]==top[j]){ he++; break; } for(int i=0;i<g;i++) if(a[i]==top[j]){ hg++; break; } } }
    double re = (double)he/(nq*10), rg = (double)hg/(nq*10);
    printf("order %s, %d x %d, dim %d: of the 10 nearest in double, brute force %.3f, graph at ef %d %.3f\n", name, gn, gs, dim, re, 4*gs, rg);
    CHECK(re >= 0.90, "order %s: brute force %.3f of the 10 nearest in double", name, re);
    CHECK(rg >= 0.88, "order %s: graph %.3f of the 10 nearest in double", name, rg);
    kci_vec_searcher_free(s); kci_vec_free(v); free(Y); free(c); free(dd);
}

int main(void){
    X = malloc(sizeof(float)*(size_t)N*D); if(!X) return 2;
    for(int c=0;c<NC;c++) for(int d=0;d<D;d++) C[c][d] = rnd();
    for(int i=0;i<N;i++) for(int d=0;d<D;d++) X[(size_t)i*D+d] = C[i%NC][d] + 0.02f*rnd();
    for(int q=0;q<NQ;q++) for(int d=0;d<D;d++) Q[q][d] = C[q*37%NC][d] + 0.02f*rnd();
    run(0, 1); run(0, 8); run(1, 1); run(1, 8);
    reach_short();
    order(30, 300, 64, 0.001f, 0); order(30, 300, 64, 0.001f, 1); order(10, 300, 384, 0.001f, 1);
    for(int m=0;m<3;m++){ reach(60, 400, 64, 16, 0.002f, m, 8); reach(1, 30000, 128, 4, 2.0f, m, m ? 8 : 1); }
    free(X);
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails != 0;
}
