/*
 test_fast_build.c — kci_vec_build_fast makes a graph of the same kind as kci_vec_build.
 *
 *   gcc -O2 -march=native -I. tests/test_fast_build.c kci_vec.c -lpthread -lm -o test_fast_build
 */
#include "kci_vec.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 48
#define NC 300          /* clusters */
#define NQ 200

static uint64_t rs = 0x1234567887654321ULL;
static float rnd1(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((int64_t)(rs % 2001) - 1000) / 1000.0f; }
static float cen[NC][DIM];
/* row id: a cluster centre plus noise; metric 2 (inner product) gets lengths between 0.5 and 1.5 */
static void mkvec(uint64_t id, int metric, float *v) {
    uint64_t s = id * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL + 1;
    const float *c = cen[id % NC];
    for (int i = 0; i < DIM; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = c[i] + 0.9f * (float)((int64_t)(s % 2001) - 1000) / 1000.0f; }
    if (metric == 2) { float f = 0.5f + (float)(id % 101) / 100.0f; for (int i = 0; i < DIM; i++) v[i] *= f; }
}
static KciVec *mk(int metric) {
    return metric == 1 ? kci_vec_new_l2(DIM, 16, 100, 0.0f) : metric == 2 ? kci_vec_new_ip(DIM, 16, 100, 0.0f) : kci_vec_new(DIM, 16, 100);
}
static double recall(KciVec *v, int metric) {
    KciVecSearcher *s = kci_vec_searcher_new(v); long hit = 0, tot = 0; float q[DIM], sc[10], sg[10]; uint64_t a[10], g[10];
    for (int qi = 0; qi < NQ; qi++) {
        mkvec(1000000000ULL + (uint64_t)qi * 7919, metric, q);
        int ng = kci_vec_search_exact(v, q, 10, 4, g, sg), na = kci_vec_search_rerank(s, q, 10, 100, 100, a, sc);
        for (int i = 0; i < ng; i++) { tot++; for (int j = 0; j < na; j++) if (a[j] == g[i]) { hit++; break; } }
    }
    kci_vec_searcher_free(s);
    return tot ? (double)hit / (double)tot : 0.0;
}
/* rows of [1, n] (every step-th) that do not find themselves in the top 10 */
static int self_miss(KciVec *v, int metric, uint64_t n, uint64_t step) {
    KciVecSearcher *s = kci_vec_searcher_new(v); float f[DIM], sc[10]; uint64_t ids[10]; int miss = 0;
    for (uint64_t id = 1; id <= n; id += step) {
        mkvec(id, metric, f);
        int m = kci_vec_search_rerank(s, f, 10, 200, 200, ids, sc), hit = 0;
        for (int i = 0; i < m; i++) hit |= ids[i] == id;
        miss += !hit;
    }
    kci_vec_searcher_free(s);
    return miss;
}

int main(void) {
    int ok = 1; float f[DIM];
    for (int c = 0; c < NC; c++) for (int i = 0; i < DIM; i++) cen[c][i] = rnd1();
    static const char *name[3] = { "cosine", "L2", "inner product" };
    const uint32_t N = 40000;

    for (int metric = 0; metric < 3; metric++) {
        double r[2]; int miss = 0;
        for (int fast = 0; fast < 2; fast++) {
            KciVec *v = mk(metric); if (!v) { printf("new failed\n"); return 1; }
            kci_vec_keep_float(v, 1);
            for (uint64_t id = 1; id <= N; id++) { mkvec(id, metric, f); if (kci_vec_add(v, id, f) < 0) { printf("add failed\n"); return 1; } }
            double secs = 0; int rc = fast ? kci_vec_build_fast(v, 8, &secs) : kci_vec_build(v, 8, &secs);
            if (rc != 0) { printf("build rc %d\n", rc); return 1; }
            r[fast] = recall(v, metric);
            if (fast) {
                /* inner product: a short row is nobody's nearest by the product, so "finds itself" is not asked of it */
                if (metric != 2) miss = self_miss(v, metric, N, 7);
                if (metric == 0) {          /* save / load, then insert and delete on the loaded graph */
                    static const uint8_t key[4] = { 1, 2, 3, 4 }; const char *path = "test_fast_build.kvec";
                    float q[DIM], s1[10], s2[10]; uint64_t a1[10], a2[10]; mkvec(1000000007ULL, 0, q);
                    KciVecSearcher *s = kci_vec_searcher_new(v); int n1 = kci_vec_search_rerank(s, q, 10, 100, 100, a1, s1); kci_vec_searcher_free(s);
                    int src = kci_vec_save(v, path, key, 4); KciVec *w = src == 0 ? kci_vec_load(path, key, 4) : NULL;
                    if (!w) { printf("save / load failed (save rc %d)\n", src); ok = 0; }
                    else {
                        s = kci_vec_searcher_new(w); int n2 = kci_vec_search_rerank(s, q, 10, 100, 100, a2, s2); kci_vec_searcher_free(s);
                        int same = n1 == n2 && memcmp(a1, a2, sizeof(uint64_t) * (size_t)n1) == 0;
                        printf("  save / load: the same 10 answers = %s\n", same ? "yes" : "NO"); ok &= same;
                        int bad = 0;
                        for (uint64_t id = N + 1; id <= N + 300; id++) { mkvec(id, 0, f); if (kci_vec_insert(w, id, f) < 0) bad++; }
                        for (uint64_t id = 5; id <= N; id += 50) if (kci_vec_delete(w, id) != 0) bad++;
                        s = kci_vec_searcher_new(w); int im = 0, dm = 0;
                        for (uint64_t id = N + 1; id <= N + 300; id++) { mkvec(id, 0, f); int m = kci_vec_search_rerank(s, f, 10, 200, 200, a2, s2), hit = 0; for (int i = 0; i < m; i++) hit |= a2[i] == id; im += !hit; }
                        for (uint64_t id = 5; id <= N; id += 50) { mkvec(id, 0, f); int m = kci_vec_search_rerank(s, f, 10, 200, 200, a2, s2); for (int i = 0; i < m; i++) dm += a2[i] == id; }
                        kci_vec_searcher_free(s);
                        printf("  after load: insert / delete errors %d, inserted rows not found %d of 300, deleted rows returned %d\n", bad, im, dm);
                        ok &= bad == 0 && im == 0 && dm == 0;
                        kci_vec_free(w);
                    }
                    remove(path);
                }
            }
            kci_vec_free(v);
        }
        printf("%s, %u rows: recall@10 at ef 100: build %.4f, build_fast %.4f; rows not finding themselves %d\n", name[metric], N, r[0], r[1], miss);
        ok &= r[1] >= 0.90 && r[1] >= r[0] - 0.05 && miss == 0;
    }

    {   /* 3,000 copies of one vector among the rows: no crash, the other rows are still found */
        KciVec *v = kci_vec_new(DIM, 16, 100); kci_vec_keep_float(v, 1);
        for (uint64_t id = 1; id <= N; id++) { mkvec(id % 13 == 0 && id <= 39000 ? 13 : id, 0, f); kci_vec_add(v, id, f); }
        int rc = kci_vec_build_fast(v, 8, NULL); double r = rc == 0 ? recall(v, 0) : 0.0;
        printf("with 3,000 copies of one vector: rc %d, recall@10 %.4f\n", rc, r);
        ok &= rc == 0 && r >= 0.90;
        kci_vec_free(v);
    }
    {   /* fewer rows than the fast build takes: it is the ordinary build */
        KciVec *v = kci_vec_new(DIM, 16, 100); kci_vec_keep_float(v, 1);
        for (uint64_t id = 1; id <= 3000; id++) { mkvec(id, 0, f); kci_vec_add(v, id, f); }
        int rc = kci_vec_build_fast(v, 8, NULL), m = rc == 0 ? self_miss(v, 0, 3000, 1) : -1;
        printf("3,000 rows: rc %d, rows not finding themselves %d\n", rc, m);
        ok &= rc == 0 && m == 0;
        kci_vec_free(v);
        ok &= kci_vec_build_fast(NULL, 1, NULL) == -1;
    }
    printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
