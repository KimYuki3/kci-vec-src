/*
 test_learn.c — lists learned from a query sample: kci_vec_learn / kci_vec_search_learned .
 *
 *   gcc -O2 -march=native -mprefer-vector-width=256 -I. tests/test_learn.c -lpthread -lm -o test_learn   (from the repo root)
 */
#include "kci_vec.c"
#include <stdio.h>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail = 1; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t g_rs = 88172645463325252ull;
static double urand(void){ g_rs ^= g_rs<<13; g_rs ^= g_rs>>7; g_rs ^= g_rs<<17; return (double)(g_rs>>11)/9007199254740992.0; }
static double grand(void){ double a=urand()+1e-12, b=urand(); return sqrt(-2*log(a))*cos(6.283185307179586*b); }

enum { N=6000, D=32, NS=3000, NQ=60, K=10, ND=8 };
static float DIR[ND][D];
static void mkvec(float *x){
    double n=0; for(int d=0;d<D;d++){ x[d]=(float)grand(); n+=(double)x[d]*x[d]; }
    double len=0.5*pow(4.0, urand()); n=sqrt(n);
    for(int d=0;d<D;d++) x[d]=(float)(x[d]/n*len);
}
static void mkquery(float *q){                       /* one of ND directions + noise, length 1 .. 3 */
    int c=(int)(urand()*ND); double n=0;
    for(int d=0;d<D;d++){ q[d]=(float)(DIR[c][d]+0.25*grand()); n+=(double)q[d]*q[d]; }
    double len=1.0+2.0*urand(); n=sqrt(n);
    for(int d=0;d<D;d++) q[d]=(float)(q[d]/n*len);
}
/* metric 0 cosine (larger better), 1 L2 (smaller better), 2 inner product (larger better); returns "larger = better" */
static double score(int metric, const float *x, const float *q){
    double dt=0, nx=0, nq=0, l2=0;
    for(int d=0;d<D;d++){ dt+=(double)x[d]*q[d]; nx+=(double)x[d]*x[d]; nq+=(double)q[d]*q[d]; double t=(double)x[d]-q[d]; l2+=t*t; }
    return metric==0 ? dt/sqrt(nx*nq) : metric==1 ? -l2 : dt;
}
static void topk(int metric, const float *X, int n, const unsigned char *dead, const float *q, int *ids, double *sc){
    for(int j=0;j<K;j++){ ids[j]=-1; sc[j]=-1e300; }
    for(int i=0;i<n;i++){ if(dead && dead[i]) continue; double s=score(metric, X+(size_t)i*D, q);
        if(s>sc[K-1]){ int j=K-1; while(j>0 && sc[j-1]<s){ sc[j]=sc[j-1]; ids[j]=ids[j-1]; j--; } sc[j]=s; ids[j]=i; } }
}
static int hits(const uint64_t *got, int m, const int *want){
    int h=0; for(int i=0;i<m;i++) for(int j=0;j<K;j++) if((int)got[i]==want[j]){ h++; break; } return h;
}

static void run(int metric){
    static const char *MN[]={"cosine","L2","inner product"};
    printf("%s\n", MN[metric]);
    float *X=malloc(sizeof(float)*(N+50)*D), *S=malloc(sizeof(float)*NS*D), *Q=malloc(sizeof(float)*NQ*D);
    uint64_t *A=malloc(sizeof(uint64_t)*NS*K); int want[K]; double ws[K]; uint64_t ids[K]; float sc[K];
    for(int i=0;i<N;i++) mkvec(X+(size_t)i*D);
    for(int i=0;i<NS;i++){ mkquery(S+(size_t)i*D); topk(metric,X,N,NULL,S+(size_t)i*D,want,ws); for(int k=0;k<K;k++) A[(size_t)i*K+k]=(uint64_t)want[k]; }
    for(int i=0;i<NQ;i++) mkquery(Q+(size_t)i*D);
    A[5]=999999999ull;                               /* an id that is not in the index: skipped */
    KciVec *v = metric==0 ? kci_vec_new(D,16,200) : metric==1 ? kci_vec_new_l2(D,16,200,0) : kci_vec_new_ip(D,16,200,0);
    kci_vec_reserve(v,N); kci_vec_keep_float(v,1);
    for(int i=0;i<N;i++) kci_vec_add(v,(uint64_t)i,X+(size_t)i*D);
    CHECK(kci_vec_learn(v,S,NS,A,K,16,0,2,NULL)==-1, "learn before any build refused");
    CHECK(kci_vec_build(v,2,NULL)==0, "build");
    KciVecSearcher *s=kci_vec_searcher_new(v);
    CHECK(kci_vec_search_learned(s,Q,K,4,500,0,ids,sc)==-2 && !kci_vec_has_learned(v), "nothing learned yet: -2");
    CHECK(kci_vec_learn(NULL,S,NS,A,K,16,0,2,NULL)==-1 && kci_vec_learn(v,NULL,NS,A,K,16,0,2,NULL)==-1
          && kci_vec_learn(v,S,0,A,K,16,0,2,NULL)==-1 && kci_vec_learn(v,S,NS,A,0,16,0,2,NULL)==-1, "bad arguments refused");
    float top5=-1;
    CHECK(kci_vec_learn(v,S,NS,A,K,16,0,2,&top5)==0 && kci_vec_has_learned(v), "learn");
    CHECK(top5>0.05f && top5<=1.0f, "concentration %.3f", top5);
    CHECK(kci_vec_search_learned(s,Q,K,4,500,-60,ids,sc)==-2, "ef < 0 before kci_vec_learn_pack: -2");
    CHECK(kci_vec_learn_pack(v)==0, "learn_pack");
    int h0=0, hg=0, hp=0, hk=0; double maxrel=0;
    for(int q=0;q<NQ;q++){ const float *y=Q+(size_t)q*D; topk(metric,X,N,NULL,y,want,ws);
        int m=kci_vec_search_learned(s,y,K,16,4000,0,ids,sc); h0+=hits(ids,m,want);
        for(int i=0;i<m;i++){ double e=score(metric,X+(size_t)ids[i]*D,y); if(metric==1) e=-e;
            double rel=fabs(sc[i]-e)/(fabs(e)+1e-3); if(rel>maxrel) maxrel=rel;
            if(i) CHECK(metric==1 ? sc[i]>=sc[i-1] : sc[i]<=sc[i-1], "scores best first"); }
        m=kci_vec_search_learned(s,y,K,16,4000,-60,ids,sc); hk+=hits(ids,m,want);
        for(int i=0;i<m;i++){ double e=score(metric,X+(size_t)ids[i]*D,y); if(metric==1) e=-e;
            CHECK(fabs(sc[i]-e)/(fabs(e)+1e-3)<1e-3, "packed: exact scores"); }
        m=kci_vec_search_learned(s,y,K,4,200,40,ids,sc); hg+=hits(ids,m,want);
        m=kci_vec_search_rerank(s,y,K,40,40,ids,sc); hp+=hits(ids,m,want); }
    printf("  top 5 %% of the vectors hold %.3f of the sample's answers\n", top5);
    printf("  lists only: recall %.4f (max rel. score error %.1e); lists as seeds, ef 40: %.4f; plain graph, ef 40: %.4f\n",
           (double)h0/(NQ*K), maxrel, (double)hg/(NQ*K), (double)hp/(NQ*K));
    CHECK(h0>=NQ*K*90/100, "lists only: recall %.3f", (double)h0/(NQ*K));
    printf("  packed lists (int8 scan, best 60 exact): recall %.4f\n", (double)hk/(NQ*K));
    CHECK(hk>=NQ*K*90/100 && hk>=h0-NQ*K*3/100, "packed lists: recall %.3f (lists only %.3f)", (double)hk/(NQ*K), (double)h0/(NQ*K));
    CHECK(maxrel<1e-3, "lists only: exact scores (%.1e)", maxrel);
    CHECK(hg>=hp-NQ*K/100 && hg>=NQ*K*93/100, "seeded graph %.3f vs plain %.3f", (double)hg/(NQ*K), (double)hp/(NQ*K));
    if(NS>=64){ long ha=0, hh=0; int same=1; uint64_t ids2[K]; float sc2[K];
        CHECK(kci_vec_searcher_set_learned_pages(s,-1)==-1 && kci_vec_searcher_set_learned_pages(NULL,1)==-1, "pages: bad arguments refused");
        CHECK(kci_vec_learn(v,S,NS,A,K,64,0,2,NULL)==0 && kci_vec_learn_pack(v)==0, "learn 64 clusters");
        for(int i=0;i<NQ;i++){ const float *y=Q+(size_t)i*D; topk(metric,X,N,NULL,y,want,ws);
            kci_vec_searcher_set_learned_pages(s,0);  int m=kci_vec_search_learned(s,y,K,4,4000,-60,ids,sc); ha+=hits(ids,m,want);
            kci_vec_searcher_set_learned_pages(s,50); int m2=kci_vec_search_learned(s,y,K,4,4000,-60,ids2,sc2);
            if(m2!=m) same=0; else for(int j=0;j<m;j++) if(ids[j]!=ids2[j]) same=0;
            kci_vec_searcher_set_learned_pages(s,4);  m=kci_vec_search_learned(s,y,K,4,4000,-60,ids,sc); hh+=hits(ids,m,want); }
        CHECK(same, "pages >= all pages: the plain choice");
        CHECK(hh>=ha-NQ*K*5/100, "4 of 8 pages: recall %.3f (all centroids %.3f)", (double)hh/(NQ*K), (double)ha/(NQ*K));
        kci_vec_searcher_set_learned_pages(s,0);
        CHECK(kci_vec_learn(v,S,NS,A,K,16,0,2,NULL)==0 && kci_vec_learn_pack(v)==0, "learn 16 clusters again"); }
    { long h1=0, hh=0, hx=0, hs=0, hb=0; uint64_t ids2[K]; float sc2[K];
      CHECK(kci_vec_searcher_set_learned_hot(s,-1)==-1 && kci_vec_searcher_set_learned_hot(NULL,1)==-1, "hot: bad arguments refused");
      for(int i=0;i<NQ;i++){ const float *y=Q+(size_t)i*D; topk(metric,X,N,NULL,y,want,ws);
          kci_vec_searcher_set_learned_hot(s,0);
          int m=kci_vec_search_learned(s,y,K,1,50,-60,ids,sc); h1+=hits(ids,m,want);
          kci_vec_searcher_set_learned_hot(s,1000000);                      /* more than there are: all of them */
          int m2=kci_vec_search_learned(s,y,K,1,50,-60,ids2,sc2); hh+=hits(ids2,m2,want);
          for(int j=0;j<m2;j++){ double e=score(metric,X+(size_t)ids2[j]*D,y); if(metric==1) e=-e; CHECK(fabs(sc2[j]-e)/(fabs(e)+1e-3)<1e-3, "hot, packed: exact scores"); }
          m2=kci_vec_search_learned(s,y,K,1,50,0,ids2,sc2); hx+=hits(ids2,m2,want);
          m2=kci_vec_search_learned(s,y,K,1,50,40,ids2,sc2); hs+=hits(ids2,m2,want);
          kci_vec_searcher_set_learned_hot(s,0);
          m2=kci_vec_search_learned(s,y,K,1,50,-60,ids2,sc2); hb+=hits(ids2,m2,want); }
      CHECK(hh>h1 && hx>=hh && hb==h1, "hot rows: packed %.3f -> %.3f, lists only %.3f, off again %.3f", (double)h1/(NQ*K), (double)hh/(NQ*K), (double)hx/(NQ*K), (double)hb/(NQ*K));
      CHECK(hs>=NQ*K*90/100, "hot rows as seeds: recall %.3f", (double)hs/(NQ*K)); }
    /* insert keeps the lists (the new vectors are just not in them); the answer still comes back */
    { int n=N; for(int i=0;i<50;i++){ float *y=X+(size_t)n*D; mkvec(y); for(int d=0;d<D;d++) y[d]*=0.9f;   /* within the lengths seen at the build (L2 / inner product refuse others) */ CHECK(kci_vec_insert(v,(uint64_t)n,y)>0, "insert %d", i); n++; }
      CHECK(kci_vec_has_learned(v), "insert keeps the lists");
      kci_vec_searcher_free(s); s=kci_vec_searcher_new(v);
      int h=0; for(int q=0;q<NQ;q++){ const float *y=Q+(size_t)q*D; topk(metric,X,n,NULL,y,want,ws);
          int m=kci_vec_search_learned(s,y,K,4,200,60,ids,sc); h+=hits(ids,m,want); }
      CHECK(h>=NQ*K*93/100, "after inserts, seeded graph: recall %.3f", (double)h/(NQ*K));
      CHECK(kci_vec_delete(v,3)==0, "delete");
      CHECK(kci_vec_compact(v)==(uint32_t)(n-1), "compact");
      kci_vec_searcher_free(s); s=kci_vec_searcher_new(v);
      CHECK(!kci_vec_has_learned(v) && kci_vec_search_learned(s,Q,K,4,200,0,ids,sc)==-2, "compact drops the lists");
      CHECK(kci_vec_learn(v,S,NS,A,K,16,0,2,NULL)==0 && kci_vec_search_learned(s,Q,K,4,200,0,ids,sc)>0, "learn again after compact");
      for(int i=0;i<kci_vec_search_learned(s,Q,K,16,4000,0,ids,sc);i++) CHECK(ids[i]!=3, "the deleted id is not returned");
      kci_vec_searcher_free(s); s=NULL;
      CHECK(kci_vec_build(v,2,NULL)==0 && !kci_vec_has_learned(v), "build drops the lists"); }
    kci_vec_free(v); free(X); free(S); free(Q); free(A);
}

int main(void){
    for(int c=0;c<ND;c++){ double n=0; for(int d=0;d<D;d++){ DIR[c][d]=(float)grand(); n+=(double)DIR[c][d]*DIR[c][d]; } n=sqrt(n); for(int d=0;d<D;d++) DIR[c][d]=(float)(DIR[c][d]/n); }
    run(2); run(0); run(1);
    printf(g_fail ? "FAILED\n" : "PASSED\n");
    return g_fail;
}
