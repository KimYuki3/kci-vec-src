/*
 *   gcc -O1 -g -fsanitize=address,undefined -march=native -I. tests/test_eps.c kci_vec.c -lpthread -lm -o test_eps
 */
#include "kci_vec.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int dim=64;
/* pseudo-random vectors (splitmix64): near-orthogonal like real embeddings, so the radius has room to matter */
static void mkvec(uint64_t seed, float *q){
    uint64_t x = seed*0x9E3779B97F4A7C15ull + 12345;
    for(int d=0; d<dim; d++){
        x += 0x9E3779B97F4A7C15ull; uint64_t z=x;
        z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; z^=z>>31;
        q[d] = (float)((z>>11)*(1.0/9007199254740992.0)) - 0.5f;
    }
}
static int mod3(uint64_t id, void *ctx){ (void)ctx; return (id%3)==0; }

/* hits of ids against the exact top-10 for q */
static int hits(KciVec *v, const float *q, const uint64_t *ids, int n){
    uint64_t gt[10]; float gs[10];
    int g=kci_vec_search_exact(v,q,10,4,gt,gs);
    int h=0; for(int i=0;i<n;i++) for(int j=0;j<g;j++) if(ids[i]==gt[j]){ h++; break; }
    return h;
}

int main(void){
    float q[64];
    KciVec *v=kci_vec_new(dim,16,200); kci_vec_keep_float(v,1);
    for(uint64_t i=0;i<20000;i++){ mkvec(i,q); kci_vec_add(v,i,q); }
    double s; kci_vec_build(v,4,&s);
    KciVecSearcher *S=kci_vec_searcher_new(v);
    uint64_t a[10], b[10]; float sa[10], sb[10];
    int all=1;

    /* (1) bad arguments */
    int r1 = kci_vec_searcher_set_eps(NULL, 0.1f)==-1 && kci_vec_searcher_set_eps(S, NAN)==-1
          && kci_vec_searcher_set_eps(S, -1.0f)==0 && kci_vec_searcher_set_eps(S, 0.0f)==0;
    printf("bad args refused, <=0 accepted as off: %s\n", r1?"OK":"FAIL"); all&=r1;

    /* (2) eps 0 after eps 0.2 == never set: identical ids and scores, plain and rerank */
    int same=1;
    for(uint64_t t=0;t<50;t++){
        mkvec(100000+t*7,q);
        kci_vec_searcher_set_eps(S,0.0f);
        int na=kci_vec_search_r(S,q,10,40,a,sa);
        kci_vec_searcher_set_eps(S,0.2f); kci_vec_search_r(S,q,10,10,b,sb);   /* disturb the scratch */
        kci_vec_searcher_set_eps(S,0.0f);
        int nb=kci_vec_search_r(S,q,10,40,b,sb);
        if(na!=nb || memcmp(a,b,sizeof(uint64_t)*na) || memcmp(sa,sb,sizeof(float)*na)) same=0;
        na=kci_vec_search_rerank(S,q,10,40,40,a,sa);
        kci_vec_searcher_set_eps(S,0.2f); kci_vec_search_rerank(S,q,10,10,10,b,sb);
        kci_vec_searcher_set_eps(S,0.0f);
        nb=kci_vec_search_rerank(S,q,10,40,40,b,sb);
        if(na!=nb || memcmp(a,b,sizeof(uint64_t)*na) || memcmp(sa,sb,sizeof(float)*na)) same=0;
    }
    printf("eps 0 == plain HNSW (ids and scores): %s\n", same?"OK":"FAIL"); all&=same;

    /* (3) ef 10 over 200 queries: eps 0.2 within 0.005 of plain ef 10 or better, eps 0.5 strictly better; always k results */
    int h0=0, h1=0, h2=0, kfull=1;
    for(uint64_t t=0;t<200;t++){
        mkvec(200000+t*13,q);
        kci_vec_searcher_set_eps(S,0.0f); int n0=kci_vec_search_r(S,q,10,10,a,sa); h0+=hits(v,q,a,n0);
        kci_vec_searcher_set_eps(S,0.2f); int n1=kci_vec_search_r(S,q,10,10,b,sb); h1+=hits(v,q,b,n1);
        kci_vec_searcher_set_eps(S,0.5f); int n2=kci_vec_search_r(S,q,10,10,b,sb); h2+=hits(v,q,b,n2);
        if(n1!=10 || n2!=10) kfull=0;
    }
    int r3 = kfull && h1>=h0-10 && h2>h0;
    printf("ef 10: recall plain %.3f, eps 0.2 %.3f, eps 0.5 %.3f, k results %s -> %s\n",
           h0/2000.0, h1/2000.0, h2/2000.0, kfull?"always":"NOT always", r3?"OK":"FAIL"); all&=r3;

    /* (4) filtered search under eps: pure, finds the matching query's own id */
    kci_vec_searcher_set_eps(S,0.2f);
    int purity=1, selffind=0, trials=0;
    for(uint64_t base=0; base<20000; base+=1500){ mkvec(base,q); trials++;
        int n=kci_vec_search_filtered(S,q,10,10,10,mod3,NULL,a,sa);
        for(int i=0;i<n;i++) if(a[i]%3!=0) purity=0;
        for(int i=0;i<n;i++) if(a[i]==base){ selffind++; break; } }
    int r4 = purity && selffind==trials;
    printf("filtered under eps: purity %s, self-find %d/%d -> %s\n", purity?"OK":"FAIL", selffind, trials, r4?"OK":"FAIL"); all&=r4;

    kci_vec_searcher_free(S); kci_vec_free(v);
    printf("\n%s\n", all?"ALL PASS":"SOME FAILED");
    return all?0:1;
}
