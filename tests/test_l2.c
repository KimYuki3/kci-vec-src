/*
 test_l2.c — L2 (Euclidean) indexes: kci_vec_new_l2 .
 *
 *   gcc -O2 -march=native -mprefer-vector-width=256 -I. tests/test_l2.c -lpthread -lm -o test_l2   (from the repo root)
 */
#include "kci_vec.c"
#include <stdio.h>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail = 1; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t g_rs = 88172645463325252ull;
static double urand(void){ g_rs ^= g_rs<<13; g_rs ^= g_rs>>7; g_rs ^= g_rs<<17; return (double)(g_rs>>11)/9007199254740992.0; }
static double grand(void){ double a=urand()+1e-12, b=urand(); return sqrt(-2*log(a))*cos(6.283185307179586*b); }

/* random direction, length lo..hi (log-uniform), shifted so the data is not centred at 0 */
static double g_lo=0.5, g_hi=2.0;
static void mkvec(float *x, int D){
    double n=0; for(int d=0;d<D;d++){ x[d]=(float)grand(); n+=(double)x[d]*x[d]; }
    double len=g_lo*pow(g_hi/g_lo, urand()); n=sqrt(n);
    for(int d=0;d<D;d++) x[d]=(float)(x[d]/n*len + (d%3==0 ? 0.3 : 0.0));
}
static double l2sq(const float *a, const float *b, int D){ double s=0; for(int d=0;d<D;d++){ double t=(double)a[d]-b[d]; s+=t*t; } return s; }
/* exact top-K by L2 over the live ids (dead[i] = 1 skips) */
static void topk(const float *X, int N, const unsigned char *dead, const float *q, int D, int K, int *ids, double *ds){
    for(int j=0;j<K;j++){ ids[j]=-1; ds[j]=1e300; }
    for(int i=0;i<N;i++){ if(dead && dead[i]) continue; double d=l2sq(X+(size_t)i*D,q,D);
        if(d<ds[K-1]){ int j=K-1; while(j>0 && ds[j-1]>d){ ds[j]=ds[j-1]; ids[j]=ids[j-1]; j--; } ds[j]=d; ids[j]=i; } }
}
static int hits(const uint64_t *got, int m, const int *want, int K){
    int h=0; for(int i=0;i<m;i++) for(int j=0;j<K;j++) if((int)got[i]==want[j]){ h++; break; } return h;
}
static int file_magic(const char *p, char out[9]){ FILE *f=fopen(p,"rb"); if(!f) return -1; size_t r=fread(out,1,8,f); fclose(f); out[8]=0; return r==8?0:-1; }

static void run_dim(int D, double lo, double hi){
    enum { N=6000, NQ=60, K=10 };
    g_lo=lo; g_hi=hi; int rec = hi/lo <= 4.01;
    printf("D=%d, lengths %.2f..%.2f\n", D, lo, hi);
    float *X=malloc(sizeof(float)*N*D), *Q=malloc(sizeof(float)*NQ*D);
    for(int i=0;i<N;i++) mkvec(X+(size_t)i*D, D);
    for(int i=0;i<NQ;i++) mkvec(Q+(size_t)i*D, D);
    KciVec *v=kci_vec_new_l2(D, 16, 200, 0);
    CHECK(v && kci_vec_is_l2(v) && kci_vec_dim(v)==D, "new_l2 / is_l2 / dim");
    kci_vec_reserve(v, N); kci_vec_keep_float(v, 1);
    for(int i=0;i<N;i++) CHECK(kci_vec_add(v, (uint64_t)i, X+(size_t)i*D)==i+1, "add %d", i);
    { int8_t c[128]={0}; CHECK(kci_vec_add_i8(v, 99999, c)==-1, "add_i8 refused for L2"); }
    int want[K]; double wd[K]; uint64_t ids[K]; float sc[K];

    int ex_hit=0; double maxrel=0;
    for(int q=0;q<NQ;q++){ topk(X,N,NULL,Q+(size_t)q*D,D,K,want,wd);
        int m=kci_vec_search_exact(v, Q+(size_t)q*D, K, 2, ids, sc); ex_hit+=hits(ids,m,want,K);
        for(int i=0;i<m;i++){ double rel=fabs(sc[i]-wd[i])/(wd[i]+1e-3); if(rel>maxrel) maxrel=rel; } }
    printf("  exact: recall %.4f, max rel. error of |q-x|^2 %.2e\n", (double)ex_hit/(NQ*K), maxrel);
    CHECK(ex_hit>=(rec ? NQ*K : NQ*K*99/100), "exact search: L2 order (%d / %d)", ex_hit, NQ*K);
    CHECK(maxrel<(rec ? 1e-4 : 1e-2), "exact search: squared distances (max rel. error %.2e)", maxrel);
    { float back[128]; CHECK(kci_vec_get_float(v, 17, back)==0, "get_float");
      double e=0, n=0; for(int d=0;d<D;d++){ e+=fabs(back[d]-X[17*D+d]); n+=fabs(X[17*D+d]); }
      CHECK(e/n<1e-5, "get_float gives the vector back (%.2e)", e/n); }

    /* lists: every list probed, large re-rank → exact */
    CHECK(kci_vec_build_ivf(v, 32, 2)==0, "build_ivf");
    KciVecSearcher *s=kci_vec_searcher_new(v);
    int iv_hit=0, iv_np_hit=0;
    for(int q=0;q<NQ;q++){ topk(X,N,NULL,Q+(size_t)q*D,D,K,want,wd);
        int m=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 32, N, ids, sc); iv_hit+=hits(ids,m,want,K);
        for(int i=1;i<m;i++) CHECK(sc[i]>=sc[i-1], "ivf scores ascending");
        m=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 8, 200, ids, sc); iv_np_hit+=hits(ids,m,want,K); }
    /* the same data and queries in a cosine index, unit length (what the lists find on structureless random data
     * at nprobe 8 / R 200, the yardstick for the L2 index) */
    int cos_hit=0;
    { KciVec *c=kci_vec_new(D,16,200); kci_vec_reserve(c,N); kci_vec_keep_float(c,1);
      for(int i=0;i<N;i++) kci_vec_add(c,(uint64_t)i,X+(size_t)i*D);
      kci_vec_build_ivf(c,32,2); KciVecSearcher *cs=kci_vec_searcher_new(c);
      for(int q=0;q<NQ;q++){ const float *y=Q+(size_t)q*D; int w[K]; double ws[K]; double ny=0;
          for(int d=0;d<D;d++) ny+=(double)y[d]*y[d];
          for(int j=0;j<K;j++){ w[j]=-1; ws[j]=-9; }
          for(int i=0;i<N;i++){ const float *x=X+(size_t)i*D; double dt=0,nx=0; for(int d=0;d<D;d++){ dt+=(double)x[d]*y[d]; nx+=(double)x[d]*x[d]; }
              double cv=dt/sqrt(nx*ny); if(cv>ws[K-1]){ int j=K-1; while(j>0&&ws[j-1]<cv){ ws[j]=ws[j-1]; w[j]=w[j-1]; j--; } ws[j]=cv; w[j]=i; } }
          int m=kci_vec_search_ivf(cs,y,K,8,200,ids,sc); cos_hit+=hits(ids,m,w,K); }
      kci_vec_searcher_free(cs); kci_vec_free(c); }
    printf("  lists: all probed %.4f, nprobe 8 / R 200 %.4f (cosine index, same data: %.4f)\n",
           (double)iv_hit/(NQ*K), (double)iv_np_hit/(NQ*K), (double)cos_hit/(NQ*K));
    CHECK(iv_hit>=(rec ? NQ*K : NQ*K*99/100), "lists, all probed: L2 order (%d / %d)", iv_hit, NQ*K);
    if(rec) CHECK(iv_np_hit>=cos_hit*85/100, "lists, nprobe 8: recall %.3f, below 0.85 x the cosine index's %.3f",
                  (double)iv_np_hit/(NQ*K), (double)cos_hit/(NQ*K));

    /* graph + float re-rank */
    CHECK(kci_vec_build(v, 2, NULL)==0, "build");
    CHECK(kci_vec_build_ivf(v, 32, 2)==0, "build_ivf after the graph");
    kci_vec_searcher_free(s); s=kci_vec_searcher_new(v);
    int g_hit=0;
    for(int q=0;q<NQ;q++){ topk(X,N,NULL,Q+(size_t)q*D,D,K,want,wd);
        int m=kci_vec_search_rerank(s, Q+(size_t)q*D, K, 200, 100, ids, sc); g_hit+=hits(ids,m,want,K);
        if(m>0) CHECK(fabs(sc[0]-wd[0])/(wd[0]+1e-3)<1e-3 || (int)ids[0]!=want[0], "graph score is |q-x|^2"); }
    printf("  graph + re-rank: recall %.4f\n", (double)g_hit/(NQ*K));
    if(rec) CHECK(g_hit>=NQ*K*90/100, "graph recall %.3f", (double)g_hit/(NQ*K));

    /* save / load */
    const uint8_t key[]="test key for the l2 index";
    for(int enc=0;enc<2;enc++){
        const char *p = enc ? "_test_l2_enc.kv" : "_test_l2.kv"; char mg[9];
        CHECK(kci_vec_save(v, p, enc?key:NULL, enc?(int)sizeof key:0)==0, "save enc=%d", enc);
        CHECK(file_magic(p,mg)==0 && strcmp(mg, enc?"KCIVEC07":"KCIVEC06")==0, "magic %s", mg);
        KciVec *w=kci_vec_load(p, enc?key:NULL, enc?(int)sizeof key:0);
        CHECK(w && kci_vec_is_l2(w) && kci_vec_dim(w)==D, "load enc=%d", enc);
        if(w){ KciVecSearcher *t=kci_vec_searcher_new(w); int same=1;
            for(int q=0;q<NQ;q++){ uint64_t a[K], b[K]; float sa[K], sb[K];
                int m1=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 8, 200, a, sa), m2=kci_vec_search_ivf(t, Q+(size_t)q*D, K, 8, 200, b, sb);
                if(m1!=m2) same=0; for(int i=0;i<m1 && i<m2;i++) if(a[i]!=b[i] || sa[i]!=sb[i]) same=0; }
            CHECK(same, "loaded index answers the same (enc=%d)", enc);
            kci_vec_searcher_free(t); kci_vec_free(w); }
        remove(p); }

    /* insert / delete / compact */
    { float big[128]; for(int d=0;d<D;d++) big[d]=(float)(10.0*hi);   /* longer than anything added: refused */
      CHECK(kci_vec_insert(v, 50000, big)==-1, "insert longer than the scale refused");
      for(int d=0;d<D;d++) big[d]=(float)(lo*0.01/sqrt((double)D));   /* far shorter than the range fixed at the build */
      CHECK(kci_vec_insert(v, 50001, big)==-1, "insert far shorter than the range refused");
      float *Y=malloc(sizeof(float)*(N+40)*D); memcpy(Y,X,sizeof(float)*N*D);
      unsigned char *dead=calloc(N+40,1); int n=N;
      for(int i=0;i<40;i++){ float *y=Y+(size_t)n*D; mkvec(y,D); for(int d=0;d<D;d++) y[d]*=0.9f;   /* within the range of lengths seen at the build */
          CHECK(kci_vec_insert(v, (uint64_t)n, y)>0, "insert %d", i); n++; }
      for(int i=0;i<300;i++){ int id=i*7; CHECK(kci_vec_delete(v,(uint64_t)id)==0, "delete %d", id); dead[id]=1; }
      int h1=0, h2=0;
      for(int q=0;q<NQ;q++){ topk(Y,n,dead,Q+(size_t)q*D,D,K,want,wd);
          int m=kci_vec_search_exact(v, Q+(size_t)q*D, K, 2, ids, sc); (void)m;
          kci_vec_searcher_free(s); s=kci_vec_searcher_new(v);
          m=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 32, n, ids, sc); h1+=hits(ids,m,want,K); }
      CHECK(kci_vec_compact(v)==(uint32_t)(n-300), "compact");
      kci_vec_searcher_free(s); s=kci_vec_searcher_new(v);
      for(int q=0;q<NQ;q++){ topk(Y,n,dead,Q+(size_t)q*D,D,K,want,wd);
          int m=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 32, n, ids, sc); h2+=hits(ids,m,want,K); }
      printf("  after insert + delete: %.4f, after compact: %.4f\n", (double)h1/(NQ*K), (double)h2/(NQ*K));
      CHECK(h1>=(rec ? NQ*K : NQ*K*99/100), "insert + delete: L2 order (%d / %d)", h1, NQ*K);
      CHECK(h2>=(rec ? NQ*K : NQ*K*99/100), "compact: L2 order, rows not transformed twice (%d / %d)", h2, NQ*K);
      free(Y); free(dead); }
    kci_vec_searcher_free(s); kci_vec_free(v);

    /* max_norm given: a longer vector is refused at add */
    v=kci_vec_new_l2(D, 16, 200, 25.0f);
    { float x[128]; for(int d=0;d<D;d++) x[d]=0; x[0]=24.0f; CHECK(kci_vec_add(v,1,x)==1, "add within max_norm");
      x[0]=26.0f; CHECK(kci_vec_add(v,2,x)==-1, "add longer than max_norm refused"); }
    kci_vec_free(v);
    free(X); free(Q);
}

int main(void){
    CHECK(kci_vec_new_l2(0,16,200,0)==NULL && kci_vec_new_l2(8,16,200,-1)==NULL && kci_vec_new_l2(8,16,200,NAN)==NULL
          && kci_vec_new_l2(8,16,200,INFINITY)==NULL, "bad args refused");
    { KciVec *c=kci_vec_new(32,16,200); float x[32]; char mg[9];   /* cosine files keep their magic */
      kci_vec_keep_float(c,1); for(int i=0;i<200;i++){ mkvec(x,32); kci_vec_add(c,i,x); }
      CHECK(kci_vec_is_l2(c)==0, "cosine is not l2");
      kci_vec_build_ivf(c,8,1); kci_vec_save(c,"_test_l2_cos.kv",NULL,0);
      CHECK(file_magic("_test_l2_cos.kv",mg)==0 && strcmp(mg,"KCIVEC04")==0, "cosine magic %s", mg);
      remove("_test_l2_cos.kv"); kci_vec_free(c); }
    run_dim(25, 0.5, 2); run_dim(32, 0.5, 2); run_dim(100, 0.5, 2);
    run_dim(32, 0.05, 20);
    printf(g_fail ? "FAILED\n" : "PASSED\n");
    return g_fail;
}
