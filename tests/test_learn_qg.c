/*
 test_learn_qg.c — learned starting points over the quantized graph .
 *
 *   gcc -O2 -march=native -I. tests/test_learn_qg.c kci_vec.c -lpthread -lm -o test_learn_qg   (from the repo root)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "kci_vec.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail = 1; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } else { printf("  ok: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint64_t g_rs = 88172645463325252ull;
static double urand(void){ g_rs ^= g_rs<<13; g_rs ^= g_rs>>7; g_rs ^= g_rs<<17; return (double)(g_rs>>11)/9007199254740992.0; }
static double grand(void){ double a=urand()+1e-12, b=urand(); return sqrt(-2*log(a))*cos(6.283185307179586*b); }

enum { N=40000, D=48, NS=4000, NQ=200, K=10, ND=12 };
static float DIR[ND][D], X[N][D], S[NS][D], Q[NQ][D];
static void mkq(float *q){ int a=(int)(urand()*ND); if(a>=ND) a=ND-1; for(int d=0;d<D;d++) q[d]=DIR[a][d]+0.45f*(float)grand(); }
static double dotd(const float *a, const float *b){ double s=0; for(int d=0;d<D;d++) s+=(double)a[d]*b[d]; return s; }
static void exact(int ip, const float *q, uint64_t *out){
    static double sc[N]; static double xn[N]; static int have;
    if(!have){ for(int i=0;i<N;i++) xn[i]=sqrt(dotd(X[i],X[i])); have=1; }
    for(int i=0;i<N;i++) sc[i] = ip ? dotd(q,X[i]) : dotd(q,X[i])/xn[i];
    for(int j=0;j<K;j++){ int b=0; for(int i=1;i<N;i++) if(sc[i]>sc[b]) b=i; out[j]=(uint64_t)b; sc[b]=-1e300; }
}
static double recall(const uint64_t *got, int n, const uint64_t *gt){ int h=0; for(int j=0;j<n;j++) for(int g=0;g<K;g++) if(got[j]==gt[g]){ h++; break; } return (double)h/K; }

static void run(int ip){
    printf("%s index\n", ip ? "inner-product" : "cosine");
    static uint64_t A[NS*K], GT[NQ*K], before[NQ*K]; static float bsc[NQ*K];
    for(int i=0;i<NS;i++) exact(ip, S[i], A+(size_t)i*K);
    for(int i=0;i<NQ;i++) exact(ip, Q[i], GT+(size_t)i*K);
    KciVec *v = ip ? kci_vec_new_ip(D, 12, 100, 0.0f) : kci_vec_new(D, 12, 100);
    kci_vec_keep_float(v, 1);
    for(int i=0;i<N;i++) kci_vec_add(v, (uint64_t)i, X[i]);
    double secs=0; CHECK(kci_vec_build(v, 1, &secs)==0, "build (1 thread: the same graph every run)");
    CHECK(kci_vec_learn(v, &S[0][0], NS, A, K, 32, 0, 1, NULL)==0, "learn (32 clusters)");
    KciVecSearcher *s=kci_vec_searcher_new(v); uint64_t ids[K]; float sc[K];
    const int LISTS=2, PER=24, EF=40;
    for(int i=0;i<NQ;i++){ int c=kci_vec_search_learned(s, Q[i], K, LISTS, PER, EF, before+(size_t)i*K, bsc+(size_t)i*K); if(c!=K){ CHECK(0,"plain graph: %d answers", c); break; } }
    kci_vec_searcher_free(s);

    CHECK(kci_vec_build_qg(v, 1, 1)==0, "build the quantized graph (1 dim a code)");
    s=kci_vec_searcher_new(v); kci_vec_searcher_set_qg(s, 0); int same=0;
    for(int i=0;i<NQ;i++){ int c=kci_vec_search_learned(s, Q[i], K, LISTS, PER, EF, ids, sc);
        same += c==K && memcmp(ids, before+(size_t)i*K, sizeof ids)==0 && memcmp(sc, bsc+(size_t)i*K, sizeof sc)==0; }
    CHECK(same==NQ, "set_qg(s, 0): the same ids and scores as before the quantized graph (%d of %d queries)", same, NQ);
    kci_vec_searcher_free(s);
    /* 2. the default searcher: over the quantized graph, with the starting points */
    s=kci_vec_searcher_new(v); double rl=0, rq=0, rl_small=0, rq_small=0; int okn=1, okord=1, okscore=1, samep=0;
    for(int i=0;i<NQ;i++){
        int c=kci_vec_search_learned(s, Q[i], K, LISTS, PER, EF, ids, sc); if(c!=K) okn=0;
        samep += c==K && memcmp(ids, before+(size_t)i*K, sizeof ids)==0;
        for(int j=1;j<c;j++) if(sc[j]>sc[j-1]+1e-6f) okord=0;
        for(int j=0;j<c;j++){ double e = ip ? dotd(Q[i], X[ids[j]]) : dotd(Q[i], X[ids[j]])/sqrt(dotd(X[ids[j]],X[ids[j]])*dotd(Q[i],Q[i])); if(fabs(e-sc[j])>2e-3*(1+fabs(e))) okscore=0; }
        rl+=recall(ids, c, GT+(size_t)i*K);
        c=kci_vec_search_rerank(s, Q[i], K, EF, EF, ids, sc); rq+=recall(ids, c, GT+(size_t)i*K);
        c=kci_vec_search_learned(s, Q[i], K, 4, 64, K, ids, sc); rl_small+=recall(ids, c, GT+(size_t)i*K);
        c=kci_vec_search_rerank(s, Q[i], K, K, K, ids, sc); rq_small+=recall(ids, c, GT+(size_t)i*K); }
    rl/=NQ; rq/=NQ; rl_small/=NQ; rq_small/=NQ;
    CHECK(okn, "quantized graph + starting points: k answers every time");
    CHECK(samep<NQ, "the walk is over the quantized graph: %d of %d queries answer exactly as the plain graph did (before 2026-11-22: all)", samep, NQ);
    CHECK(okord, "best first");
    CHECK(okscore, "exact scores");
    CHECK(rl>=rq-0.01, "recall@10 at ef %d: %.4f with the starting points, %.4f without (not below it)", EF, rl, rq);
    CHECK(rl_small>rq_small+0.005, "at ef %d the starting points (4 lists x 64) are really used: %.4f against %.4f", K, rl_small, rq_small);
    kci_vec_searcher_free(s); kci_vec_free(v);
}
int main(void){
    for(int a=0;a<ND;a++){ double n=0; for(int d=0;d<D;d++){ DIR[a][d]=(float)grand(); n+=(double)DIR[a][d]*DIR[a][d]; } n=sqrt(n); for(int d=0;d<D;d++) DIR[a][d]=(float)(DIR[a][d]/n*1.5); }
    for(int i=0;i<N;i++){ double len=0.5+1.5*urand(), n=0; for(int d=0;d<D;d++){ X[i][d]=(float)grand(); n+=(double)X[i][d]*X[i][d]; } n=sqrt(n); for(int d=0;d<D;d++) X[i][d]=(float)(X[i][d]/n*len); }
    for(int i=0;i<NS;i++) mkq(S[i]);
    for(int i=0;i<NQ;i++) mkq(Q[i]);
    run(0); run(1);
    printf(g_fail ? "test_learn_qg: FAILED\n" : "test_learn_qg: PASSED\n");
    return g_fail;
}
