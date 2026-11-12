/*

 */
#include "kci_vec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 20000
#define D 64
#define NQ 300
#define K 10

static uint64_t sm_state = 0x9E3779B97F4A7C15ULL;
static uint64_t sm_next(void){ uint64_t z=(sm_state+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static float sm_gauss(void){ double u=(sm_next()>>11)*(1.0/9007199254740992.0)+1e-12, w=(sm_next()>>11)*(1.0/9007199254740992.0); return (float)(sqrt(-2*log(u))*cos(6.283185307179586*w)); }
static void unit(float *x){ double n=0; for(int d=0;d<D;d++) n+=x[d]*(double)x[d]; n=sqrt(n); for(int d=0;d<D;d++) x[d]=(float)(x[d]/n); }

static int fails=0;
#define CHECK(c, ...) do{ if(!(c)){ fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } }while(0)

static float *base, *qry; static uint64_t gt[NQ][K];
static void brute(void){
    for(int q=0;q<NQ;q++){ float best[K]; uint64_t bid[K]; int n=0;
        for(int i=0;i<N;i++){ float s=0; for(int d=0;d<D;d++) s+=qry[q*D+d]*base[(size_t)i*D+d];
            int j; if(n<K) j=n++; else { if(s<=best[K-1]) continue; j=K-1; }
            best[j]=s; bid[j]=(uint64_t)i;
            while(j>0 && best[j]>best[j-1]){ float t=best[j]; best[j]=best[j-1]; best[j-1]=t; uint64_t u=bid[j]; bid[j]=bid[j-1]; bid[j-1]=u; j--; } }
        for(int j=0;j<K;j++) gt[q][j]=bid[j]; }
}
static double recall(KciVecSearcher *s, int mode, int ef, float eps, int qg, int *bad){
    kci_vec_searcher_set_eps(s, eps); CHECK(kci_vec_searcher_set_qg(s, qg)==0, "set_qg %d", qg);
    uint64_t ids[K]; float sc[K]; long hit=0; if(bad) *bad=0;
    for(int q=0;q<NQ;q++){
        int n = mode ? kci_vec_search_rerank(s, qry+q*D, K, ef, ef, ids, sc) : kci_vec_search_r(s, qry+q*D, K, ef, ids, sc);
        if(n!=K && bad) (*bad)++;
        for(int a=0;a<n;a++){
            if(!(sc[a]==sc[a]) || fabsf(sc[a])>1.0001f) { if(bad) (*bad)++; }
            if(a && sc[a]>sc[a-1]+1e-6f) { if(bad) (*bad)++; }
            for(int b=0;b<a;b++) if(ids[b]==ids[a]) { if(bad) (*bad)++; }
            if(mode){ float c=0; for(int d=0;d<D;d++) c+=qry[q*D+d]*base[ids[a]*D+d]; if(fabsf(c-sc[a])>1e-4f && bad) (*bad)++; }
            for(int b=0;b<K;b++) if(gt[q][b]==ids[a]){ hit++; break; } }
    }
    return (double)hit/((double)NQ*K);
}

static void env_half(int off){
#ifdef _WIN32
    _putenv_s("KCI_VEC_QG_NO_HALF", off ? "1" : "");
#else
    if(off) setenv("KCI_VEC_QG_NO_HALF","1",1); else unsetenv("KCI_VEC_QG_NO_HALF");
#endif
}
static void half_check(int dim, int M, int dp){
    const int n=6000, nq=100, k=10; float *b=malloc((size_t)n*dim*sizeof(float)), *q=malloc((size_t)nq*dim*sizeof(float));
    for(int i=0;i<n+nq;i++){ float *x = i<n ? b+(size_t)i*dim : q+(size_t)(i-n)*dim; double s=0;
        for(int d=0;d<dim;d++){ x[d]=sm_gauss(); s+=x[d]*(double)x[d]; } s=sqrt(s); for(int d=0;d<dim;d++) x[d]=(float)(x[d]/s); }
    KciVec *v=kci_vec_new(dim,M,100); kci_vec_keep_float(v,1); kci_vec_reserve(v,n);
    for(int i=0;i<n;i++) kci_vec_add(v,(uint64_t)i,b+(size_t)i*dim);
    double secs; CHECK(kci_vec_build(v,4,&secs)==0, "half_check build");
    uint64_t *ids[2]; float *sc[2]; int64_t by[2]; long diff=0;
    for(int lay=0;lay<2;lay++){
        env_half(lay==0); CHECK(kci_vec_build_qg(v,dp,0)==0, "half_check build_qg"); by[lay]=kci_vec_qg_bytes(v);
        KciVecSearcher *s=kci_vec_searcher_new(v); CHECK(kci_vec_searcher_set_qg(s,1)==0, "half_check set_qg");
        ids[lay]=calloc((size_t)nq*3*k,sizeof(uint64_t)); sc[lay]=calloc((size_t)nq*3*k,sizeof(float));
        for(int t=0;t<3;t++){ kci_vec_searcher_set_eps(s, t==2 ? 0.1f : 0.0f);
            for(int j=0;j<nq;j++){ uint64_t *oi=ids[lay]+((size_t)t*nq+j)*k; float *os=sc[lay]+((size_t)t*nq+j)*k;
                int got = t==1 ? kci_vec_search_r(s, q+(size_t)j*dim, k, 40, oi, os) : kci_vec_search_rerank(s, q+(size_t)j*dim, k, 40, 40, oi, os);
                CHECK(got==k, "half_check got %d", got); } }
        kci_vec_searcher_free(s);
    }
    env_half(0);
    for(size_t i=0;i<(size_t)nq*3*k;i++) if(ids[0][i]!=ids[1][i] || sc[0][i]!=sc[1][i]) diff++;
    printf("half block  dim %d M %d dp %d: %lld -> %lld bytes, %ld of %d results differ\n", dim, M, dp, (long long)by[0], (long long)by[1], diff, nq*3*k);
    CHECK(diff==0, "dim %d M %d dp %d: the 16-slot block changed %ld results", dim, M, dp, diff);
    CHECK(by[1]<by[0], "dim %d M %d dp %d: the 16-slot block is not smaller (%lld vs %lld)", dim, M, dp, (long long)by[1], (long long)by[0]);
    for(int lay=0;lay<2;lay++){ free(ids[lay]); free(sc[lay]); }
    kci_vec_free(v); free(b); free(q);
}

int main(void){
    half_check(63, 8, 1); half_check(64, 8, 2); half_check(63, 24, 1); half_check(64, 4, 1);
    base=malloc((size_t)N*D*sizeof(float)); qry=malloc((size_t)NQ*D*sizeof(float));
    for(int i=0;i<N;i++){ for(int d=0;d<D;d++) base[(size_t)i*D+d]=sm_gauss(); unit(base+(size_t)i*D); }
    for(int q=0;q<NQ;q++){ for(int d=0;d<D;d++) qry[q*D+d]=0.7f*sm_gauss()+base[(size_t)(q*37)*D+d]; unit(qry+q*D); }   /* near some base vector */
    brute();
    KciVec *v=kci_vec_new(D,16,100); kci_vec_keep_float(v,1); kci_vec_reserve(v,N+10);
    for(int i=0;i<N;i++) kci_vec_add(v,(uint64_t)i,base+(size_t)i*D);
    CHECK(kci_vec_build_qg(v,1,0)==-1, "build_qg before the graph must fail");
    double secs; CHECK(kci_vec_build(v,4,&secs)==0, "build");
    KciVecSearcher *s=kci_vec_searcher_new(v);
    CHECK(kci_vec_searcher_set_qg(s,1)==-2, "set_qg without a quantized graph must be -2");
    CHECK(kci_vec_qg_bytes(v)==0, "no bytes yet");
    CHECK(kci_vec_build_qg(v,3,0)==-1, "dims_per_code 3 refused");
    int bad;
    double r_plain  = recall(s, 0, 100, 0.0f, 0, &bad); CHECK(bad==0, "plain bad %d", bad);
    double r_plainR = recall(s, 1, 100, 0.0f, 0, &bad); CHECK(bad==0, "plain rerank bad %d", bad);
    double r_plainE = recall(s, 1, 20, 0.2f, 0, &bad);
    printf("plain       ef100 int8 %.4f  rerank %.4f  | ef20 eps0.2 rerank %.4f\n", r_plain, r_plainR, r_plainE);
    for(int dp=1; dp<=2; dp++){
        CHECK(kci_vec_build_qg(v,dp,0)==0, "build_qg dp %d", dp);
        int64_t by=kci_vec_qg_bytes(v); CHECK(by>(int64_t)N*64, "bytes %lld", (long long)by);
        double q_i8  = recall(s, 0, 100, 0.0f, 1, &bad); CHECK(bad==0, "qg int8 bad %d", bad);
        double q_rr  = recall(s, 1, 100, 0.0f, 1, &bad); CHECK(bad==0, "qg rerank bad %d", bad);
        double q_e   = recall(s, 1, 20, 0.2f, 1, &bad);  CHECK(bad==0, "qg eps bad %d", bad);
        double q_e10 = recall(s, 0, 10, 0.3f, 1, &bad);  CHECK(bad==0, "qg eps int8 bad %d", bad);
        printf("qg dp %d     ef100 int8 %.4f  rerank %.4f  | ef20 eps0.2 rerank %.4f | ef10 eps0.3 int8 %.4f  (%.1f MB)\n", dp, q_i8, q_rr, q_e, q_e10, by/1048576.0);
        CHECK(q_rr >= r_plainR-(dp==1 ? 0.03 : 0.06), "dp %d: rerank recall %.4f far below plain %.4f", dp, q_rr, r_plainR);   /* random 64-d data: the pair codes lose ~0.04 */
        CHECK(q_i8 >= r_plain-(dp==1 ? 0.05 : 0.08), "dp %d: int8 recall %.4f far below plain %.4f", dp, q_i8, r_plain);
        CHECK(q_e >= (dp==1 ? 0.8 : 0.6), "dp %d: eps path recall %.4f", dp, q_e);   /* pair codes are coarser: the small W of the eps path loses more */
        CHECK(q_e10 > 0.5, "dp %d: eps int8 path recall %.4f", dp, q_e10);
        /* plain searches must be untouched while a quantized graph exists and the switch is off */
        double r2 = recall(s, 1, 100, 0.0f, 0, &bad); CHECK(r2==r_plainR && bad==0, "plain changed with qg present: %.4f vs %.4f", r2, r_plainR);
    }
    /* delete keeps the quantized graph; the deleted id never comes back */
    { uint64_t victim=gt[0][0]; CHECK(kci_vec_delete(v,victim)==0, "delete");
      CHECK(kci_vec_qg_bytes(v)>0, "delete must keep the quantized graph");
      kci_vec_searcher_set_qg(s,1); kci_vec_searcher_set_eps(s,0.2f);
      uint64_t ids[K]; float sc[K]; int n=kci_vec_search_rerank(s, qry, K, 20, 20, ids, sc);
      for(int a=0;a<n;a++) CHECK(ids[a]!=victim, "deleted id returned");
      n=kci_vec_search_r(s, qry, K, 20, ids, sc);
      for(int a=0;a<n;a++) CHECK(ids[a]!=victim, "deleted id returned (int8)"); }
    /* insert drops it; a rebuild brings it back */
    { float x[D]; for(int d=0;d<D;d++) x[d]=sm_gauss(); unit(x);
      CHECK(kci_vec_insert(v, 777777, x)>0, "insert");
      CHECK(kci_vec_qg_bytes(v)==0, "insert must drop the quantized graph");
      CHECK(kci_vec_searcher_set_qg(s,1)==-2, "set_qg after the drop must be -2");
      kci_vec_searcher_set_qg(s,0);
      uint64_t ids[K]; float sc[K]; int n=kci_vec_search_rerank(s, x, K, 50, 50, ids, sc); CHECK(n==K && ids[0]==777777, "inserted vector found by the plain search");
      CHECK(kci_vec_build_qg(v,1,0)==0, "rebuild after insert");
      CHECK(kci_vec_searcher_set_qg(s,1)==0, "set_qg after rebuild");
      kci_vec_searcher_set_eps(s,0.2f);
      n=kci_vec_search_rerank(s, x, K, 20, 20, ids, sc); CHECK(n==K && ids[0]==777777, "inserted vector found over the quantized graph (%d, %llu)", n, (unsigned long long)(n?ids[0]:0)); }
    kci_vec_searcher_free(s); kci_vec_free(v); free(base); free(qry);
    printf(fails? "FAILED %d\n" : "ALL PASS\n", fails);
    return fails?1:0;
}
