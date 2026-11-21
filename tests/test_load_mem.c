/*

 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "kci_vec.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)
static uint64_t rs = 88172645463325252ULL;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) & 0xFFFFFF) / 8388608.0f - 1.0f; }
static long rss_kb(void) {
#ifdef __linux__
    FILE *f = fopen("/proc/self/status", "r"); char ln[256]; long r = -1; if (!f) return -1;
    while (fgets(ln, sizeof ln, f)) if (sscanf(ln, "VmRSS: %ld", &r) == 1) break;
    fclose(f); return r;
#else
    return -1;
#endif
}

int main(void) {
    const int N = 300000, D = 64, NQ = 50, K = 10; const char *path = "test_load_mem.kvec";
    float *X = malloc((size_t)N * D * sizeof(float)), *Q = malloc((size_t)NQ * D * sizeof(float));
    for (size_t i = 0; i < (size_t)N * D; i++) X[i] = rnd();
    for (size_t i = 0; i < (size_t)NQ * D; i++) Q[i] = rnd();
    KciVec *v = kci_vec_new(D, 8, 60); kci_vec_keep_float(v, 1);
    for (int i = 0; i < N; i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i * D);
    double secs = 0; CHECK(kci_vec_build(v, 8, &secs) == 0, "build %d x %d", N, D);
    uint64_t before[50 * 10]; float sc[16]; KciVecSearcher *s = kci_vec_searcher_new(v);
    for (int q = 0; q < NQ; q++) kci_vec_search_rerank(s, Q + (size_t)q * D, K, 80, 80, before + (size_t)q * K, sc);
    kci_vec_searcher_free(s);
    CHECK(kci_vec_save(v, path, NULL, 0) == 0, "save");
    kci_vec_free(v); free(X);

    long r0 = rss_kb();
    v = kci_vec_load(path, NULL, 0);
    long r1 = rss_kb();
    CHECK(v != NULL, "load"); if (!v) return 1;
    CHECK(kci_vec_count(v) == N, "count %lld", (long long)kci_vec_count(v));
    if (r0 > 0 && r1 > 0) {
        double used = (double)N * D * 4 + (double)N * D + (double)N * 8 + (double)N * (16 + 1) * 4 + (double)N * 16;
        double spare = (double)(524288 - N) * D * 4;
        double grew = (double)(r1 - r0) * 1024.0;
        printf("   the load added %.1f MB resident; the index itself is about %.1f MB; the spare float rows would be %.1f MB more\n", grew / 1048576, used / 1048576, spare / 1048576);
        CHECK(grew < used + spare * 0.5, "the spare float rows are not resident after the load");
    } else printf("ok: (resident memory not read on this system: skipped)\n");

    s = kci_vec_searcher_new(v); int same = 0; uint64_t now_[10];
    for (int q = 0; q < NQ; q++) { kci_vec_search_rerank(s, Q + (size_t)q * D, K, 80, 80, now_, sc); same += memcmp(now_, before + (size_t)q * K, sizeof now_) == 0; }
    CHECK(same == NQ, "the same answers after the load (%d of %d queries)", same, NQ);

    /* inserts after the load land in the spare rows: their floats must come back as given, and they must be found */
    int okins = 1, found = 0; float y[64];
    for (int i = 0; i < 2000; i++) { for (int d = 0; d < D; d++) y[d] = rnd();
        if (kci_vec_insert(v, (uint64_t)(N + i), y) < 0) { okins = 0; break; }
        if (i % 500 == 0) { uint64_t o[4]; int c = kci_vec_search_rerank(s, y, 1, 100, 100, o, sc);   /* the float pass scores it with its own stored floats */
            int hit = c > 0 && o[0] == (uint64_t)(N + i); found += hit; if (hit && !(sc[0] > 0.9999f)) okins = 0; } }
    CHECK(okins, "2,000 inserts after the load, each scored 1 against its own stored floats");
    CHECK(found == 4, "an inserted vector is its own nearest (%d of 4)", found);
    kci_vec_searcher_free(s);
    CHECK(kci_vec_save(v, path, NULL, 0) == 0, "save again after the inserts");
    kci_vec_free(v);
    v = kci_vec_load(path, NULL, 0); CHECK(v && kci_vec_count(v) == N + 2000, "load again: %lld vectors", v ? (long long)kci_vec_count(v) : -1LL);
    if (v) kci_vec_free(v);
    remove(path); free(Q);
    printf(fails ? "test_load_mem: %d FAILED\n" : "test_load_mem: all OK\n", fails);
    return fails != 0;
}
