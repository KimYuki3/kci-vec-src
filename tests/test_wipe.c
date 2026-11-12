/*
 *   gcc -O1 -g -march=native -o test_wipe tests/test_wipe.c -Wl,--wrap=free -lpthread -lm && ./test_wipe
 */
#include "../kci_vec.c"
#include <malloc.h>
#include <stdio.h>

void __real_free(void *p);
static long g_hits = 0, g_frees = 0, g_scanning = 1;
static const char *g_stage = "";
static float g_m0 = 0.6f, g_m1 = 0.8f;

void __wrap_free(void *p) {
    if (p && g_scanning) {
        size_t n = malloc_usable_size(p);
        const unsigned char *b = p;
        uint32_t a, c;
        memcpy(&a, &g_m0, 4); memcpy(&c, &g_m1, 4);
        for (size_t i = 0; i + 8 <= n; i += 4) {
            uint32_t x, y;
            memcpy(&x, b + i, 4); memcpy(&y, b + i + 4, 4);
            if (x == a && y == c) {
                if (g_hits < 10) fprintf(stderr, "  leak: %zu-byte block freed with a marker row (stage %s)\n", n, g_stage);
                g_hits++;
                break;
            }
        }
        g_frees++;
    }
    __real_free(p);
}

int main(void) {
    int D = 32, N = 6000, fails = 0;
    float *X = malloc(sizeof(float) * D * N);
    uint64_t s = 99;
    for (int i = 0; i < N; i++) {
        float *x = X + (size_t)i * D;
        if (i % 50 == 0) { memset(x, 0, D * sizeof(float)); x[0] = 3; x[1] = 4; continue; }   /* marker */
        for (int d = 0; d < D; d++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; x[d] = (float)((int)(s % 2001) - 1000) / 1000.f; }
    }
    const uint8_t key[] = "k";
    struct { const char *name; int graph; } runs[] = { { "lists", 0 }, { "graph + lists", 1 } };
    for (int r = 0; r < 2; r++) {
        g_hits = 0;
        g_stage = "add"; KciVec *v = kci_vec_new(D, 16, 100);
        kci_vec_keep_float(v, 1);
        for (int i = 0; i < N / 2; i++) kci_vec_add(v, 1000 + (uint64_t)i, X + (size_t)i * D);
        g_stage = "build"; if (runs[r].graph) kci_vec_build(v, 2, NULL);
        kci_vec_build_ivf(v, 32, 2);
#ifndef TEST_NO_PROTECT   /* -DTEST_NO_PROTECT: for a kci_vec.c without kci_vec_protect */
        g_stage = "protect"; kci_vec_protect(v);
#endif
        g_stage = "insert"; for (int i = N / 2; i < N; i++) kci_vec_insert(v, 1000 + (uint64_t)i, X + (size_t)i * D);   /* grows past cap? */
        g_stage = "reserve"; kci_vec_reserve(v, 200000);                 /* force the buffers to move */
        g_stage = "delete"; for (int i = 0; i < N; i += 3) kci_vec_delete(v, 1000 + (uint64_t)i);
        g_stage = "compact"; kci_vec_compact(v);
        g_stage = "search"; { KciVecSearcher *sr = kci_vec_searcher_new(v); uint64_t ids[10]; float sc[10];
            kci_vec_search_ivf(sr, X + 50 * D, 10, 8, 50, ids, sc); kci_vec_searcher_free(sr); }
        g_stage = "save";
#ifdef KCI_VEC_SAVE_NO_FLOAT
        kci_vec_save_ex(v, "_t_wipe.kv", key, 1, KCI_VEC_SAVE_NO_FLOAT);
#else
        kci_vec_save(v, "_t_wipe.kv", key, 1);
#endif
        g_stage = "free"; kci_vec_free(v);
        g_stage = "load"; KciVec *w = kci_vec_load("_t_wipe.kv", key, 1); remove("_t_wipe.kv");
        if (w) {
#ifdef KCI_VEC_SAVE_NO_FLOAT
            for (int i = 0; i < N; i++) if (i % 3) kci_vec_set_float(w, 1000 + (uint64_t)i, X + (size_t)i * D);
#endif
            g_stage = "free (loaded)"; kci_vec_free(w); }
        printf("%-14s: %ld freed blocks still held a marker row%s\n", runs[r].name, g_hits, g_hits ? "  FAIL" : "");
        fails += g_hits != 0;
    }
    g_scanning = 0;
    free(X);
    printf("%s\n", fails ? "FAIL" : "ALL PASS");
    return fails;
}
