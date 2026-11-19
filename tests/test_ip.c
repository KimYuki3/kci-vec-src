/*
 test_ip.c — inner-product indexes: kci_vec_new_ip .
 *
 *   gcc -O2 -march=native -mprefer-vector-width=256 -I. tests/test_ip.c -lpthread -lm -o test_ip   (from the repo root)
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
static double dot(const float *a, const float *b, int D){ double s=0; for(int d=0;d<D;d++) s+=(double)a[d]*b[d]; return s; }
/* exact top-K by inner product over the live ids (dead[i] = 1 skips), largest first */
static void topk(const float *X, int N, const unsigned char *dead, const float *q, int D, int K, int *ids, double *ds){
    for(int j=0;j<K;j++){ ids[j]=-1; ds[j]=-1e300; }
    for(int i=0;i<N;i++){ if(dead && dead[i]) continue; double d=dot(X+(size_t)i*D,q,D);
        if(d>ds[K-1]){ int j=K-1; while(j>0 && ds[j-1]<d){ ds[j]=ds[j-1]; ids[j]=ids[j-1]; j--; } ds[j]=d; ids[j]=i; } }
}
static int hits(const uint64_t *got, int m, const int *want, int K){
    int h=0; for(int i=0;i<m;i++) for(int j=0;j<K;j++) if((int)got[i]==want[j]){ h++; break; } return h;
}
static int file_magic(const char *p, char out[9]){ FILE *f=fopen(p,"rb"); if(!f) return -1; size_t r=fread(out,1,8,f); fclose(f); out[8]=0; return r==8?0:-1; }

static int g_centre;
static void run_dim(int D, double lo, double hi){
    enum { N=6000, NQ=60, K=10 };
    g_lo=lo; g_hi=hi; int ex = hi/lo <= 4.01, rec = ex && !g_centre;
    printf("D=%d, lengths %.2f..%.2f%s\n", D, lo, hi, g_centre ? ", centred" : "");
    float *X=malloc(sizeof(float)*N*D), *Q=malloc(sizeof(float)*NQ*D);
    for(int i=0;i<N;i++) mkvec(X+(size_t)i*D, D);
    for(int i=0;i<NQ;i++) mkvec(Q+(size_t)i*D, D);
    KciVec *v=kci_vec_new_ip(D, 16, 200, 0);
    CHECK(v && kci_vec_metric(v)==2 && !kci_vec_is_l2(v) && kci_vec_dim(v)==D, "new_ip / metric / dim");
    if(g_centre) CHECK(kci_vec_set_ip_centre(v,1)==0, "set_ip_centre");
    kci_vec_reserve(v, N); kci_vec_keep_float(v, 1);
    for(int i=0;i<N;i++) CHECK(kci_vec_add(v, (uint64_t)i, X+(size_t)i*D)==i+1, "add %d", i);
    { int8_t c[128]={0}; CHECK(kci_vec_add_i8(v, 99999, c)==-1, "add_i8 refused for inner product"); }
    int want[K]; double wd[K]; uint64_t ids[K]; float sc[K];

    int ex_hit=0; double maxrel=0;
    for(int q=0;q<NQ;q++){ topk(X,N,NULL,Q+(size_t)q*D,D,K,want,wd);
        int m=kci_vec_search_exact(v, Q+(size_t)q*D, K, 2, ids, sc); ex_hit+=hits(ids,m,want,K);
        for(int i=0;i<m;i++){ double rel=fabs(sc[i]-wd[i])/(fabs(wd[i])+1e-3); if(rel>maxrel) maxrel=rel; } }
    printf("  exact: recall %.4f, max rel. error of q.x %.2e\n", (double)ex_hit/(NQ*K), maxrel);
    CHECK(ex_hit>=(ex ? NQ*K : NQ*K*99/100), "exact search: inner-product order (%d / %d)", ex_hit, NQ*K);
    CHECK(kci_vec_set_ip_centre(v,1)==-1, "set_ip_centre after the first build refused");
    CHECK(maxrel<(ex ? 1e-4 : 1e-2), "exact search: inner products (max rel. error %.2e)", maxrel);
    { float back[128]; CHECK(kci_vec_get_float(v, 17, back)==0, "get_float");
      double e=0, n=0; for(int d=0;d<D;d++){ e+=fabs(back[d]-X[17*D+d]); n+=fabs(X[17*D+d]); }
      CHECK(e/n<1e-5, "get_float gives the vector back (%.2e)", e/n); }

    /* lists: every list probed, full re-rank → exact */
    CHECK(kci_vec_build_ivf(v, 32, 2)==0, "build_ivf");
    KciVecSearcher *s=kci_vec_searcher_new(v);
    int iv_hit=0, iv_np_hit=0;
    for(int q=0;q<NQ;q++){ topk(X,N,NULL,Q+(size_t)q*D,D,K,want,wd);
        int m=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 32, N, ids, sc); iv_hit+=hits(ids,m,want,K);
        for(int i=1;i<m;i++) CHECK(sc[i]<=sc[i-1], "ivf scores descending");
        m=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 8, 200, ids, sc); iv_np_hit+=hits(ids,m,want,K); }
    printf("  lists: all probed %.4f, nprobe 8 / R 200 %.4f\n", (double)iv_hit/(NQ*K), (double)iv_np_hit/(NQ*K));
    CHECK(iv_hit>=(ex ? NQ*K : NQ*K*99/100), "lists, all probed: inner-product order (%d / %d)", iv_hit, NQ*K);

    /* graph + float re-rank, then the quantized graph */
    CHECK(kci_vec_build(v, 2, NULL)==0, "build");
    CHECK(kci_vec_build_ivf(v, 32, 2)==0, "build_ivf after the graph");
    kci_vec_searcher_free(s); s=kci_vec_searcher_new(v);
    int g_hit=0, qg_hit=0;
    for(int q=0;q<NQ;q++){ topk(X,N,NULL,Q+(size_t)q*D,D,K,want,wd);
        int m=kci_vec_search_rerank(s, Q+(size_t)q*D, K, 200, 100, ids, sc); g_hit+=hits(ids,m,want,K);
        if(m>0) CHECK(fabs(sc[0]-wd[0])/(fabs(wd[0])+1e-3)<1e-3 || (int)ids[0]!=want[0], "graph score is q.x"); }
    CHECK(kci_vec_build_qg(v, 1, 2)==0, "build_qg");
    CHECK(kci_vec_searcher_set_qg(s, 1)==0, "set_qg"); kci_vec_searcher_set_eps(s, 0.05f);
    for(int q=0;q<NQ;q++){ topk(X,N,NULL,Q+(size_t)q*D,D,K,want,wd);
        int m=kci_vec_search_rerank(s, Q+(size_t)q*D, K, 200, 100, ids, sc); qg_hit+=hits(ids,m,want,K); }
    kci_vec_searcher_set_qg(s, 0); kci_vec_searcher_set_eps(s, 0.0f);
    printf("  graph + re-rank: recall %.4f, quantized graph: %.4f\n", (double)g_hit/(NQ*K), (double)qg_hit/(NQ*K));
    if(rec){ CHECK(g_hit>=NQ*K*90/100, "graph recall %.3f", (double)g_hit/(NQ*K));
             CHECK(qg_hit>=NQ*K*90/100, "quantized graph recall %.3f", (double)qg_hit/(NQ*K)); }

    /* save / load */
    const uint8_t key[]="test key for the ip index";
    for(int enc=0;enc<2;enc++){
        const char *p = enc ? "_test_ip_enc.kv" : "_test_ip.kv"; char mg[9];
        CHECK(kci_vec_save(v, p, enc?key:NULL, enc?(int)sizeof key:0)==0, "save enc=%d", enc);
        CHECK(file_magic(p,mg)==0 && strcmp(mg, enc?"KCIVEC09":"KCIVEC08")==0, "magic %s", mg);
        if(enc) CHECK(kci_vec_load(p, (const uint8_t*)"wrong key", 9)==NULL, "wrong key refused");
        KciVec *w=kci_vec_load(p, enc?key:NULL, enc?(int)sizeof key:0);
        CHECK(w && kci_vec_metric(w)==2 && kci_vec_dim(w)==D, "load enc=%d", enc);
        if(w){ KciVecSearcher *t=kci_vec_searcher_new(w); int same=1;
            for(int q=0;q<NQ;q++){ uint64_t a[K], b[K]; float sa[K], sb[K];
                int m1=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 8, 200, a, sa), m2=kci_vec_search_ivf(t, Q+(size_t)q*D, K, 8, 200, b, sb);
                if(m1!=m2) same=0; for(int i=0;i<m1 && i<m2;i++) if(a[i]!=b[i] || sa[i]!=sb[i]) same=0; }
            CHECK(same, "loaded index answers the same (enc=%d)", enc);
            kci_vec_searcher_free(t); kci_vec_free(w); }
        remove(p); }
    /* image: attach (read-only) and thaw (writable copy) stay inner-product indexes */
    { const char *p="_test_ip.img";
      CHECK(kci_vec_image_save(v, p)==0, "image_save");
      KciVec *r=kci_vec_attach(p), *w=r ? kci_vec_thaw(r) : NULL;
      CHECK(r && kci_vec_metric(r)==2 && kci_vec_dim(r)==D, "attach");
      CHECK(w && kci_vec_metric(w)==2 && kci_vec_dim(w)==D, "thaw");
      if(r && w){ KciVecSearcher *t=kci_vec_searcher_new(r), *u=kci_vec_searcher_new(w); int same=1;
          for(int q=0;q<NQ;q++){ uint64_t a[K], b[K], c[K]; float sa[K], sb[K], sc3[K];
              int m1=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 8, 200, a, sa), m2=kci_vec_search_ivf(t, Q+(size_t)q*D, K, 8, 200, b, sb),
                  m3=kci_vec_search_ivf(u, Q+(size_t)q*D, K, 8, 200, c, sc3);
              if(m1!=m2 || m1!=m3) same=0; for(int i=0;i<m1 && i<m2 && i<m3;i++) if(a[i]!=b[i] || a[i]!=c[i]) same=0; }
          CHECK(same, "attached image and its thawed copy answer the same");
          kci_vec_searcher_free(t); kci_vec_searcher_free(u); }
      if(w) kci_vec_free(w);
      if(r) kci_vec_free(r);
      remove(p); }

    /* insert / delete / compact */
    { float big[128]; for(int d=0;d<D;d++) big[d]=(float)(10.0*hi);   /* longer than anything added: refused */
      CHECK(kci_vec_insert(v, 50000, big)==-1, "insert longer than the scale refused");
      float *Y=malloc(sizeof(float)*(N+41)*D); memcpy(Y,X,sizeof(float)*N*D);
      unsigned char *dead=calloc(N+41,1); int n=N;
      { float *y=Y+(size_t)n*D; for(int d=0;d<D;d++) y[d]=(float)(lo*0.01/sqrt((double)D));   /* far shorter: fits (never a top answer) */
        CHECK(kci_vec_insert(v, (uint64_t)n, y)>0, "insert far shorter accepted"); n++; }
      for(int i=0;i<40;i++){ float *y=Y+(size_t)n*D; mkvec(y,D); for(int d=0;d<D;d++) y[d]*=0.9f;   /* no longer than the longest at the build */
          CHECK(kci_vec_insert(v, (uint64_t)n, y)>0, "insert %d", i); n++; }
      for(int i=0;i<300;i++){ int id=i*7; CHECK(kci_vec_delete(v,(uint64_t)id)==0, "delete %d", id); dead[id]=1; }
      int h1=0, h2=0;
      kci_vec_searcher_free(s); s=kci_vec_searcher_new(v);
      for(int q=0;q<NQ;q++){ topk(Y,n,dead,Q+(size_t)q*D,D,K,want,wd);
          int m=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 32, n, ids, sc); h1+=hits(ids,m,want,K); }
      CHECK(kci_vec_compact(v)==(uint32_t)(n-300), "compact");
      kci_vec_searcher_free(s); s=kci_vec_searcher_new(v);
      for(int q=0;q<NQ;q++){ topk(Y,n,dead,Q+(size_t)q*D,D,K,want,wd);
          int m=kci_vec_search_ivf(s, Q+(size_t)q*D, K, 32, n, ids, sc); h2+=hits(ids,m,want,K); }
      printf("  after insert + delete: %.4f, after compact: %.4f\n", (double)h1/(NQ*K), (double)h2/(NQ*K));
      CHECK(h1>=(ex ? NQ*K : NQ*K*99/100), "insert + delete: inner-product order (%d / %d)", h1, NQ*K);
      CHECK(h2>=(ex ? NQ*K : NQ*K*99/100), "compact: inner-product order, rows not transformed twice (%d / %d)", h2, NQ*K);
      free(Y); free(dead); }
    kci_vec_searcher_free(s); kci_vec_free(v);

    /* max_norm given: a longer vector is refused at add */
    v=kci_vec_new_ip(D, 16, 200, 25.0f);
    { float x[128]; for(int d=0;d<D;d++) x[d]=0; x[0]=24.0f; CHECK(kci_vec_add(v,1,x)==1, "add within max_norm");
      x[0]=26.0f; CHECK(kci_vec_add(v,2,x)==-1, "add longer than max_norm refused"); }
    kci_vec_free(v);
    free(X); free(Q);
}

int main(void){
    CHECK(kci_vec_new_ip(0,16,200,0)==NULL && kci_vec_new_ip(8,16,200,-1)==NULL && kci_vec_new_ip(8,16,200,NAN)==NULL
          && kci_vec_new_ip(8,16,200,INFINITY)==NULL, "bad args refused");
    CHECK(kci_vec_metric(NULL)==-1, "metric(NULL)");
    { float x[32]; char mg[9];   /* cosine and L2 keep their metric and their magic */
      KciVec *c=kci_vec_new(32,16,200), *l=kci_vec_new_l2(32,16,200,0);
      kci_vec_keep_float(c,1); kci_vec_keep_float(l,1);
      for(int i=0;i<200;i++){ mkvec(x,32); kci_vec_add(c,i,x); kci_vec_add(l,i,x); }
      CHECK(kci_vec_metric(c)==0 && kci_vec_metric(l)==1 && kci_vec_is_l2(l)==1 && kci_vec_is_l2(c)==0, "cosine 0, L2 1");
      kci_vec_build_ivf(c,8,1); kci_vec_save(c,"_test_ip_cos.kv",NULL,0);
      kci_vec_build_ivf(l,8,1); kci_vec_save(l,"_test_ip_l2.kv",NULL,0);
      CHECK(file_magic("_test_ip_cos.kv",mg)==0 && strcmp(mg,"KCIVEC04")==0, "cosine magic %s", mg);
      CHECK(file_magic("_test_ip_l2.kv",mg)==0 && strcmp(mg,"KCIVEC06")==0, "L2 magic %s", mg);
      { KciVec *w=kci_vec_load("_test_ip_l2.kv",NULL,0); CHECK(w && kci_vec_metric(w)==1, "L2 file loads as L2"); if(w) kci_vec_free(w); }
      remove("_test_ip_cos.kv"); remove("_test_ip_l2.kv"); kci_vec_free(c); kci_vec_free(l); }
    run_dim(25, 0.5, 2); run_dim(32, 0.5, 2); run_dim(100, 0.5, 2);
    run_dim(32, 0.05, 20);
    g_centre=1; run_dim(32, 0.5, 2); run_dim(100, 0.5, 2);
    { KciVec *c=kci_vec_new(8,16,200); CHECK(kci_vec_set_ip_centre(c,1)==-1, "set_ip_centre on a cosine index refused"); kci_vec_free(c); }
    printf(g_fail ? "FAILED\n" : "PASSED\n");
    return g_fail;
}
