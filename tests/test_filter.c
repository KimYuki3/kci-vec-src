/*
 Filtered search: results must all pass the predicate, and matching nearest
 * neighbours are still found (incl. a highly selective single-id filter). Run
 * under ASan/UBSan.
 */
#include "kci_vec.h"
#include <stdio.h>
#include <math.h>

static int dim=64;
static void mkvec(uint64_t seed, float *q){
    for(int d=0; d<dim; d++) q[d] = sinf((float)(seed*131 + d*7)) + 0.01f*d;
}
static int mod3(uint64_t id, void *ctx){ (void)ctx; return (id%3)==0; }
static int only(uint64_t id, void *ctx){ return id == *(uint64_t*)ctx; }

int main(void){
    float q[64];
    KciVec *v=kci_vec_new(dim,16,200); kci_vec_keep_float(v,1);
    for(uint64_t i=0;i<3000;i++){ mkvec(i,q); kci_vec_add(v,i,q); }
    double s; kci_vec_build(v,4,&s);
    KciVecSearcher *S=kci_vec_searcher_new(v);
    uint64_t ids[20]; float sc[20];

    /* (1) mod-3 filter: every result must satisfy id%3==0, and a matching query's
     * own id should come back. */
    int purity=1, selffind=0, trials=0;
    for(uint64_t base=0; base<3000; base+=300){    /* base%3==0 */
        mkvec(base,q); trials++;
        int n=kci_vec_search_filtered(S,q,10,100,50,mod3,NULL,ids,sc);
        for(int i=0;i<n;i++) if(ids[i]%3!=0) purity=0;
        for(int i=0;i<n;i++) if(ids[i]==base){ selffind++; break; }
    }
    printf("mod3 filter: purity %s ; self-find %d/%d\n", purity?"OK":"FAIL", selffind, trials);

    /* (2) highly selective filter: only id==1234 eligible. Must return exactly it. */
    uint64_t target=1234; mkvec(7,q);   /* arbitrary query */
    int n2=kci_vec_search_filtered(S,q,10,100,50,only,&target,ids,sc);
    int sel_ok = (n2==1 && ids[0]==target);
    printf("selective filter: n=%d top=%llu -> %s\n", n2,(unsigned long long)ids[0], sel_ok?"OK":"FAIL");

    /* (3) NULL filter == plain rerank (self-find) */
    mkvec(2000,q);
    int n3=kci_vec_search_filtered(S,q,10,100,50,NULL,NULL,ids,sc);
    int null_ok=0; for(int i=0;i<n3;i++) if(ids[i]==2000){ null_ok=1; break; }
    printf("NULL filter == rerank: %s\n", null_ok?"OK":"FAIL");

    kci_vec_searcher_free(S); kci_vec_free(v);
    int all = purity && (selffind==trials) && sel_ok && null_ok;
    printf("\n%s\n", all?"ALL PASS":"SOME FAILED");
    return all?0:1;
}
