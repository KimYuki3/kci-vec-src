/*
 test_insert_reach.c — every vector inserted into a built (and reloaded) graph is found by a search for itself.
 *
 *   gcc -O2 -march=native -I. tests/test_insert_reach.c -lpthread -lm -o test_insert_reach   (run from the repo root)
 */
#include <stdlib.h>
static unsigned int t_seed = 1;
static void t_srand(unsigned int s){ t_seed = s; }
static int  t_rand(void){ t_seed = t_seed * 214013u + 2531011u; return (int)((t_seed >> 16) & 0x7fff); }
#undef RAND_MAX
#define RAND_MAX 32767
#define rand  t_rand
#define srand t_srand
#include "../kci_vec.c"
#include <stdio.h>

static int dim = 64;
static void mkvec(uint64_t seed, float *q){
    for(int d = 0; d < dim; d++) q[d] = sinf((float)(seed * 131 + d * 7)) + 0.01f * d;
}
static int found(KciVec *v, uint64_t want, const float *q){
    uint64_t ids[20]; float sc[20];
    KciVecSearcher *s = kci_vec_searcher_new(v);
    int n = kci_vec_search_rerank(s, q, 20, 100, 50, ids, sc), hit = 0;
    for(int i = 0; i < n; i++) if(ids[i] == want){ hit = 1; break; }
    kci_vec_searcher_free(s);
    return hit;
}

int main(void){
    float q[64]; int bad = 0;
    for(int pass = 0; pass < 3; pass++){           /* the 4-thread build is not deterministic: three graphs */
        KciVec *v = kci_vec_new(dim, 16, 200); kci_vec_keep_float(v, 1);
        for(uint64_t i = 0; i < 2000; i++){ mkvec(i, q); kci_vec_add(v, i, q); }
        double s; if(kci_vec_build(v, 4, &s) != 0){ printf("build FAIL\n"); return 1; }
        for(uint64_t i = 9000; i < 9100; i++){ mkvec(i, q); if(kci_vec_insert(v, i, q) < 0){ printf("insert FAIL\n"); return 1; } }
        for(uint64_t i = 100; i < 200; i++) kci_vec_delete(v, i);
        kci_vec_compact(v);
        if(kci_vec_save(v, "_reach.kv", NULL, 0) != 0){ printf("save FAIL\n"); return 1; }
        kci_vec_free(v);
        KciVec *w = kci_vec_load("_reach.kv", NULL, 0);
        if(!w){ printf("load FAIL\n"); return 1; }
        for(uint64_t i = 9500; i < 9600; i++){ mkvec(i, q); if(kci_vec_insert(w, i, q) < 0){ printf("insert after load FAIL\n"); return 1; } }
        int miss = 0, lv = 0;
        for(uint64_t i = 9500; i < 9600; i++){ mkvec(i, q); if(!found(w, i, q)) miss++; if(w->level[h_get(w, i)] > 0) lv++; }
        int old = 0;
        for(uint64_t i = 9000; i < 9100; i++){ mkvec(i, q); if(!found(w, i, q)) old++; }
        printf("pass %d: inserted after load, not found by their own query: %d of 100 (%d of them on an upper layer); "
               "inserted before the save: %d of 100  %s\n", pass + 1, miss, lv, old, miss == 0 && old == 0 ? "OK" : "FAIL");
        if(miss || old) bad = 1;
        kci_vec_free(w);
    }
    remove("_reach.kv");
    printf("\n%s\n", bad ? "SOME FAILED" : "ALL PASS");
    return bad;
}
