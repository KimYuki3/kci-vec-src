/*
 Incremental update round-trip: add→build→insert→delete→compact→save/load.
 * Verifies tombstoned ids vanish from results, inserted ids are findable, compact
 * preserves live set, and the snapshot round-trips. Run under ASan/UBSan.
 */
#include "kci_vec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int dim=64;
static void mkvec(uint64_t seed, float *q){
    for(int d=0; d<dim; d++) q[d] = sinf((float)(seed*131 + d*7)) + 0.01f*d;
}
static int found(KciVec *v, uint64_t want, const float *q){
    uint64_t ids[20]; float sc[20];
    KciVecSearcher *s=kci_vec_searcher_new(v);
    int n=kci_vec_search_rerank(s, q, 20, 100, 50, ids, sc);
    int hit=0; for(int i=0;i<n;i++) if(ids[i]==want){ hit=1; break; }
    kci_vec_searcher_free(s);
    return hit;
}

int main(void){
    float q[64];
    KciVec *v=kci_vec_new(dim,16,200); kci_vec_keep_float(v,1);
    for(uint64_t i=0;i<2000;i++){ mkvec(i,q); kci_vec_add(v,i,q); }
    double s; kci_vec_build(v,4,&s);
    printf("built count=%u\n", kci_vec_count(v));

    /* self-find every original id (rerank should return it as its own NN) */
    int miss=0; for(uint64_t i=0;i<2000;i+=50){ mkvec(i,q); if(!found(v,i,q)) miss++; }
    printf("self-find original: %s (%d misses of 40)\n", miss==0?"OK":"FAIL", miss);

    /* insert new ids 9000..9099 and confirm findable */
    for(uint64_t i=9000;i<9100;i++){ mkvec(i,q); int64_t r=kci_vec_insert(v,i,q); if(r<0){printf("insert FAIL %lld\n",(long long)r);return 1;} }
    printf("after insert count=%u\n", kci_vec_count(v));
    int ins_ok=1; for(uint64_t i=9000;i<9100;i+=10){ mkvec(i,q); if(!found(v,i,q)) ins_ok=0; }
    printf("find inserted: %s\n", ins_ok?"OK":"FAIL");

    /* delete ids 100..199, confirm they vanish and others remain */
    for(uint64_t i=100;i<200;i++) if(kci_vec_delete(v,i)!=0){printf("delete FAIL %llu\n",(unsigned long long)i);return 1;}
    int del_gone=1, others=1;
    for(uint64_t i=100;i<200;i+=10){ mkvec(i,q); if(found(v,i,q)) del_gone=0; }
    for(uint64_t i=300;i<400;i+=10){ mkvec(i,q); if(!found(v,i,q)) others=0; }
    printf("deleted vanished: %s ; survivors found: %s\n", del_gone?"OK":"FAIL", others?"OK":"FAIL");
    if(kci_vec_delete(v,99999)!=-1){ printf("delete-missing should be -1 FAIL\n"); return 1; }

    /* compact: live = 2000 + 100 inserted - 100 deleted = 2000 */
    uint32_t live=kci_vec_compact(v);
    printf("compact live=%u (expect 2000)\n", live);
    int post=1;
    for(uint64_t i=100;i<200;i+=10){ mkvec(i,q); if(found(v,i,q)) post=0; }      /* still gone */
    for(uint64_t i=9000;i<9100;i+=10){ mkvec(i,q); if(!found(v,i,q)) post=0; }   /* inserted survive */
    printf("post-compact correct: %s\n", post?"OK":"FAIL");

    /* save/load round-trip */
    kci_vec_save(v,"_inc.kv",NULL,0);
    KciVec *w=kci_vec_load("_inc.kv",NULL,0);
    if(!w){ printf("load FAIL\n"); return 1; }
    int rt=1; for(uint64_t i=9000;i<9100;i+=10){ mkvec(i,q); if(!found(w,i,q)) rt=0; }
    printf("reload finds inserted: %s ; reloaded count=%u\n", rt?"OK":"FAIL", kci_vec_count(w));

    int ld=1;
    for(uint64_t i=9500;i<9600;i++){ mkvec(i,q); if(kci_vec_insert(w,i,q)<0) ld=0; }
    for(uint64_t i=9500;i<9600;i+=10){ mkvec(i,q); if(!found(w,i,q)) ld=0; }
    for(uint64_t i=9000;i<9100;i+=10){ mkvec(i,q); if(!found(w,i,q)) ld=0; }
    printf("insert after load: %s ; count=%u\n", ld?"OK":"FAIL", kci_vec_count(w));
    remove("_inc.kv");
    kci_vec_free(v); kci_vec_free(w);
    int allok = (miss==0)&&ins_ok&&del_gone&&others&&(live==2000)&&post&&rt&&ld;
    printf("\n%s\n", allok?"ALL PASS":"SOME FAILED");
    return allok?0:1;
}
