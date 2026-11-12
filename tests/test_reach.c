/*
 test_reach.c — every vector must be findable by searching for itself.
 *
 * A parallel kci_vec_build connected each node top-down, so a
 * concurrent insert could descend to a node whose layer-0 links did not exist
 * yet, find only 2-3 candidates there and keep just those. Such nodes had 1-2
 * incoming edges from far away, or none, and were not found whatever ef
 * (2,000 random 32-dim vectors, 16 threads: 7 never found; 20,000: 27).
 * Build with 16 threads on purpose, also after compact().
 *
 *   gcc -O2 -march=native -I. tests/test_reach.c kci_vec.c -lpthread -lm -o test_reach
 */
#include "kci_vec.h"
#include <stdio.h>
#include <stdint.h>

#define DIM 32

static void mkvec(uint64_t id, float *v) {
    uint64_t s = id * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL + 1;
    for (int i = 0; i < DIM; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        v[i] = (float)((int64_t)(s % 2001) - 1000) / 1000.0f;
    }
    v[0] += 0.001f;
}

/* number of live ids in [lo, hi) (step) that do not find themselves in the top 10 */
static int self_miss(KciVec *v, uint64_t lo, uint64_t hi, uint64_t step) {
    KciVecSearcher *s = kci_vec_searcher_new(v);
    float f[DIM]; uint64_t ids[10]; float sc[10]; int miss = 0;
    for (uint64_t id = lo; id < hi; id += step) {
        mkvec(id, f);
        int n = kci_vec_search_rerank(s, f, 10, 200, 200, ids, sc), hit = 0;
        for (int i = 0; i < n; i++) hit |= ids[i] == id;
        miss += !hit;
    }
    kci_vec_searcher_free(s);
    return miss;
}

int main(void) {
    int ok = 1;
    for (int round = 0; round < 3; round++) {
        uint32_t N = round == 2 ? 20000 : 2000;
        KciVec *v = kci_vec_new(DIM, 16, 200);
        kci_vec_keep_float(v, 1);
        float f[DIM];
        for (uint64_t id = 1; id <= N; id++) { mkvec(id + round * 1000000ULL, f); kci_vec_add(v, id + round * 1000000ULL, f); }
        kci_vec_build(v, 16, NULL);
        uint64_t base = round * 1000000ULL;
        int m = self_miss(v, base + 1, base + N + 1, 1);
        printf("build N=%u, 16 threads: self not in top 10 = %d\n", N, m);
        ok &= m == 0;
        if (round == 0) {      /* delete every 3rd, compact (= parallel rebuild) */
            for (uint64_t id = 1; id <= N; id += 3) kci_vec_delete(v, id);
            kci_vec_compact(v);
            int mc = 0;
            for (uint64_t id = 2; id <= N; id += 3) mc += self_miss(v, id, id + 2 <= N + 1 ? id + 2 : N + 1, 1);
            printf("after compact: self not in top 10 = %d\n", mc);
            ok &= mc == 0;
        }
        kci_vec_free(v);
    }
    printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
