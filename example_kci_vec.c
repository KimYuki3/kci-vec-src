/*
 * example_kci_vec.c — minimal end-to-end usage of the kci_vec library.
 *
 * Demonstrates: create → add vectors → build (multithreaded) → search →
 * encrypted snapshot save → load (key required) → verify identical results.
 *
 * Build & run (after building the library — see build_kci_vec.sh):
 *   clang -O2 -I. example_kci_vec.c -L. -lkci_vec -lpthread -lm -o example   # shared
 *   ./example
 * or static:
 *   clang -O2 -I. example_kci_vec.c libkci_vec.a -lpthread -lm -o example
 */
#include "kci_vec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float frand(unsigned *s){ *s = *s*1103515245u + 12345u; return ((*s>>9)&0x7fffff)/8388608.0f - 0.5f; }

int main(void) {
    const int dim = 128, N = 20000, K = 5;
    unsigned seed = 42;

    /* 1. create an index (dim, HNSW M=16, efConstruction=200) */
    KciVec *v = kci_vec_new(dim, 16, 200);
    if (!v) { fprintf(stderr, "new failed\n"); return 1; }

    /* 2. add N random vectors; id is the caller's opaque key */
    float *vec = malloc(dim * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dim; d++) vec[d] = frand(&seed);
        kci_vec_add(v, (uint64_t)(1000 + i), vec);
    }
    printf("added %u vectors (dim %d)\n", kci_vec_count(v), kci_vec_dim(v));

    /* 3. build the graph (0 threads = auto) */
    double secs = 0;
    if (kci_vec_build(v, 0, &secs) != 0) { fprintf(stderr, "build failed\n"); return 1; }
    printf("built in %.2fs\n", secs);

    /* 4. search: a query vector (reuse vec as the query) */
    uint64_t ids[K]; float sc[K];
    int n = kci_vec_search(v, vec, K, 100, ids, sc);
    printf("search top-%d:\n", n);
    for (int i = 0; i < n; i++) printf("  id=%llu  cosine=%.4f\n", (unsigned long long)ids[i], sc[i]);

    /* 5. save an ENCRYPTED snapshot (ChaCha20 (RFC 8439)). The key is required to load. */
    const char *key = "my-secret-key";
    if (kci_vec_save(v, "index.kv", (const uint8_t*)key, (int)strlen(key)) != 0) {
        fprintf(stderr, "save failed\n"); return 1;
    }
    printf("saved encrypted snapshot index.kv\n");
    kci_vec_free(v);

    /* 6. load with the key and verify the same top-k */
    KciVec *v2 = kci_vec_load("index.kv", (const uint8_t*)key, (int)strlen(key));
    if (!v2) { fprintf(stderr, "load failed (wrong key?)\n"); return 1; }
    uint64_t ids2[K]; float sc2[K];
    int n2 = kci_vec_search(v2, vec, K, 100, ids2, sc2);
    int match = (n2 == n);
    for (int i = 0; i < n && match; i++) if (ids2[i] != ids[i]) match = 0;
    printf("loaded encrypted snapshot: %u vectors, search-match-after-decrypt=%s\n",
           kci_vec_count(v2), match ? "YES" : "NO");

    /* 7. wrong key is rejected */
    KciVec *bad = kci_vec_load("index.kv", (const uint8_t*)"wrong", 5);
    printf("wrong-key-rejected=%s\n", bad == NULL ? "YES" : "NO");

    kci_vec_free(v2);
    free(vec);
    return match ? 0 : 1;
}
