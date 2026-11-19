/*
 test_vote.c — candidates from the answers of the nearest sample queries: kci_vec_vote_set / kci_vec_search_voted .
 *
 *   gcc -O2 -march=native -mprefer-vector-width=256 -I. tests/test_vote.c -lpthread -lm -o test_vote   (from the repo root)
 */
#include "kci_vec.c"
#include <stdio.h>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail = 1; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t g_rs = 88172645463325252ull;
static double urand(void){ g_rs ^= g_rs<<13; g_rs ^= g_rs>>7; g_rs ^= g_rs<<17; return (double)(g_rs>>11)/9007199254740992.0; }
static double grand(void){ double a=urand()+1e-12, b=urand(); return sqrt(-2*log(a))*cos(6.283185307179586*b); }

enum { N=6000, D=32, NS=4000, NQ=60, K=10, ND=8, T=40 };
static float DIR[ND][D];
static void mkvec(float *x){
    double n=0; for(int d=0;d<D;d++){ x[d]=(float)grand(); n+=(double)x[d]*x[d]; }
    double len=0.5*pow(4.0, urand()); n=sqrt(n);
    for(int d=0;d<D;d++) x[d]=(float)(x[d]/n*len);
}
static void mkquery(float *q){                       /* a mix of two of ND directions + noise, length 1 .. 3 */
    int c1=(int)(urand()*ND), c2=(int)(urand()*ND); double a=urand(), n=0;
    for(int d=0;d<D;d++){ q[d]=(float)(a*DIR[c1][d]+(1-a)*DIR[c2][d]+0.08*grand()); n+=(double)q[d]*q[d]; }
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
static void voters_of(const float *S, const float *q, uint32_t *vo){      /* the T sample queries nearest to q by cosine */
    double vs[T]; int nt=0;
    for(int i=0;i<NS;i++){ double s=score(0, S+(size_t)i*D, q);
        if(nt<T || s>vs[nt-1]){ int j = nt<T ? nt++ : nt-1; while(j>0 && vs[j-1]<s){ vs[j]=vs[j-1]; vo[j]=vo[j-1]; j--; } vs[j]=s; vo[j]=(uint32_t)i; } }
}

static void run(int metric){
    static const char *MN[]={"cosine","L2","inner product"};
    printf("%s\n", MN[metric]);
    float *X=malloc(sizeof(float)*(N+50)*D), *S=malloc(sizeof(float)*NS*D), *Q=malloc(sizeof(float)*NQ*D);
    uint64_t *A=malloc(sizeof(uint64_t)*NS*K); int want[K]; double ws[K]; uint64_t ids[K]; float sc[K]; uint32_t vo[T];
    for(int i=0;i<N;i++) mkvec(X+(size_t)i*D);
    for(int i=0;i<NS;i++){ mkquery(S+(size_t)i*D); topk(metric,X,N,NULL,S+(size_t)i*D,want,ws); for(int k=0;k<K;k++) A[(size_t)i*K+k]=(uint64_t)want[k]; }
    for(int i=0;i<NQ;i++) mkquery(Q+(size_t)i*D);
    A[5]=999999999ull;                               /* an id that is not in the index: skipped */
    KciVec *v = metric==0 ? kci_vec_new(D,16,200) : metric==1 ? kci_vec_new_l2(D,16,200,0) : kci_vec_new_ip(D,16,200,0);
    kci_vec_reserve(v,N+50); kci_vec_keep_float(v,1);
    for(int i=0;i<N;i++) kci_vec_add(v,(uint64_t)i,X+(size_t)i*D);
    CHECK(kci_vec_vote_set(v,A,NS,K)==-1, "vote_set before any build refused");
    CHECK(kci_vec_build(v,2,NULL)==0, "build");
    KciVecSearcher *s=kci_vec_searcher_new(v);
    voters_of(S,Q,vo);
    CHECK(kci_vec_search_voted(s,Q,vo,T,K,60,ids,sc)==-2, "search before vote_set is -2");
    CHECK(kci_vec_vote_set(NULL,A,NS,K)==-1 && kci_vec_vote_set(v,NULL,NS,K)==-1 && kci_vec_vote_set(v,A,0,K)==-1
          && kci_vec_vote_set(v,A,NS,0)==-1, "bad arguments refused");
    CHECK(kci_vec_vote_set(v,A,NS,K)==0, "vote_set");
    CHECK(kci_vec_search_voted(NULL,Q,vo,T,K,60,ids,sc)==-1 && kci_vec_search_voted(s,NULL,vo,T,K,60,ids,sc)==-1
          && kci_vec_search_voted(s,Q,NULL,T,K,60,ids,sc)==-1 && kci_vec_search_voted(s,Q,vo,0,K,60,ids,sc)==-1
          && kci_vec_search_voted(s,Q,vo,T,K,60,NULL,sc)==-1, "bad search arguments refused");
    int h=0, tot=0; double worst=0;
    for(int i=0;i<NQ;i++){ const float *q=Q+(size_t)i*D; topk(metric,X,N,NULL,q,want,ws); voters_of(S,q,vo);
        int r=kci_vec_search_voted(s,q,vo,T,K,60,ids,sc);
        CHECK(r==K, "query %d: %d results", i, r); if(r<0) r=0;
        h+=hits(ids,r,want); tot+=K;
        for(int x=0;x<r;x++){ double e=score(metric, X+(size_t)ids[x]*D, q); if(metric==1) e=-e;
            double er=fabs(sc[x]-e)/(fabs(e)+1e-3); if(er>worst) worst=er;
            if(x>0) CHECK(metric==1 ? sc[x]>=sc[x-1]-1e-6 : sc[x]<=sc[x-1]+1e-6, "query %d: not best first at %d", i, x); } }
    printf("  %d voters, 60 scored exactly: recall@%d %.3f, max rel. score error %.1e\n", T, K, (double)h/tot, worst);
    CHECK((double)h/tot>=0.90, "recall %.3f < 0.90", (double)h/tot);
    CHECK(worst<1e-3, "scores differ from the exact ones by %.1e", worst);
    /* short codes: a short vector per row whose product with a short query orders like the score. Here exactly so:
     * inner product x and q; cosine x/|x| and q; L2 (x, |x|^2) and (2q, -1), since -|q-x|^2 = 2q.x - |x|^2 - |q|^2. */
    { enum { SD=D+1 }; float *Z=malloc(sizeof(float)*N*SD), qs[SD]; uint64_t *I=malloc(sizeof(uint64_t)*N);
      for(int i=0;i<N;i++){ const float *x=X+(size_t)i*D; float *z=Z+(size_t)i*SD; double n2=0; for(int d=0;d<D;d++) n2+=(double)x[d]*x[d];
          for(int d=0;d<D;d++) z[d] = metric==0 ? (float)(x[d]/sqrt(n2)) : x[d];
          z[D] = metric==1 ? (float)n2 : 0.0f; I[i]=(uint64_t)(N-1-i); }
      for(int i=0;i<N/2;i++){ float t[SD]; memcpy(t,Z+(size_t)i*SD,sizeof t); memcpy(Z+(size_t)i*SD,Z+(size_t)(N-1-i)*SD,sizeof t); memcpy(Z+(size_t)(N-1-i)*SD,t,sizeof t); }   /* rows given in another order than added */
      voters_of(S,Q,vo); for(int d=0;d<D;d++) qs[d]=Q[d]; qs[D]=0;
      CHECK(kci_vec_search_voted_short(s,Q,qs,100,vo,T,K,60,ids,sc)==-2, "short search before vote_short is -2");
      CHECK(kci_vec_vote_short(v,I,Z,N-1,SD)==-1 && kci_vec_vote_short(v,I,Z,N,0)==-1 && kci_vec_vote_short(v,NULL,Z,N,SD)==-1, "bad vote_short arguments refused");
      { uint64_t keep0=I[0]; I[0]=I[1]; CHECK(kci_vec_vote_short(v,I,Z,N,SD)==-1, "an id given twice refused"); I[0]=keep0; }
      CHECK(kci_vec_vote_short(v,I,Z,N,SD)==0, "vote_short");
      CHECK(kci_vec_search_voted_short(s,Q,NULL,100,vo,T,K,60,ids,sc)==-1 && kci_vec_search_voted_short(s,Q,qs,0,vo,T,K,60,ids,sc)==-1, "bad short search arguments refused");
      int h2=0; double worst2=0;
      for(int i=0;i<NQ;i++){ const float *q=Q+(size_t)i*D; topk(metric,X,N,NULL,q,want,ws); voters_of(S,q,vo);
          for(int d=0;d<D;d++) qs[d] = metric==1 ? 2*q[d] : q[d];
          qs[D] = metric==1 ? -1.0f : 0.0f;
          int r=kci_vec_search_voted_short(s,q,qs,100,vo,T,K,60,ids,sc);
          CHECK(r==K, "short, query %d: %d results", i, r); if(r<0) r=0;
          h2+=hits(ids,r,want);
          for(int x=0;x<r;x++){ double e=score(metric, X+(size_t)ids[x]*D, q); if(metric==1) e=-e;
              double er=fabs(sc[x]-e)/(fabs(e)+1e-3); if(er>worst2) worst2=er; } }
      printf("  short codes, the best 100 kept: recall@%d %.3f, max rel. score error %.1e\n", K, (double)h2/tot, worst2);
      CHECK(h2>=h-tot*3/100, "short codes lose answers: %.3f against %.3f", (double)h2/tot, (double)h/tot);
      CHECK(worst2<1e-3, "short: scores differ from the exact ones by %.1e", worst2);
      free(Z); free(I); }
    { uint32_t bad[3]={NS, NS+7, 0xFFFFFFFFu};
      CHECK(kci_vec_search_voted(s,Q,bad,3,K,60,ids,sc)==0, "voters outside the sample give 0 results");
      uint32_t mix[2]={NS+1, 0};                     /* sample query 0 holds the unknown id (A[5]): 9 answers left */
      int r=kci_vec_search_voted(s,Q,mix,2,K,60,ids,sc);
      CHECK(r==K-1, "one voter with an unknown answer id: %d results, want %d", r, K-1); }
    /* insert keeps the table (the new vector is not in it) */
    { float *y=X+(size_t)N*D; mkvec(y); for(int d=0;d<D;d++) y[d]*=0.9f;   /* within the lengths seen at the build (L2 / inner product refuse others) */
      CHECK(kci_vec_insert(v,(uint64_t)N,y)>0, "insert"); }
    kci_vec_searcher_free(s); s=kci_vec_searcher_new(v);
    voters_of(S,Q,vo);
    CHECK(kci_vec_search_voted(s,Q,vo,T,K,60,ids,sc)==K, "search after insert");
    /* compact and build drop it */
    CHECK(kci_vec_delete(v,3)==0, "delete"); kci_vec_compact(v);
    kci_vec_searcher_free(s); s=kci_vec_searcher_new(v);
    CHECK(kci_vec_search_voted(s,Q,vo,T,K,60,ids,sc)==-2, "dropped by compact");
    CHECK(kci_vec_vote_set(v,A,NS,K)==0 && kci_vec_search_voted(s,Q,vo,T,K,60,ids,sc)>0, "set again after compact");
    CHECK(kci_vec_build(v,2,NULL)==0 && kci_vec_search_voted(s,Q,vo,T,K,60,ids,sc)==-2, "dropped by build");
    kci_vec_searcher_free(s); kci_vec_free(v); free(X); free(S); free(Q); free(A);
}

int main(void){
    for(int c=0;c<ND;c++){ double n=0; for(int d=0;d<D;d++){ DIR[c][d]=(float)grand(); n+=(double)DIR[c][d]*DIR[c][d]; }
        n=sqrt(n); for(int d=0;d<D;d++) DIR[c][d]=(float)(DIR[c][d]/n); }
    run(2); run(0); run(1);
    printf(g_fail ? "FAILED\n" : "PASSED\n");
    return g_fail;
}
