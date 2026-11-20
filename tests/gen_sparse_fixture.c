/*
 * gen_sparse_fixture.c — writes the files tests/test_sparse.c (old_files) reads back, to prove that a file
 * written by an earlier version still loads. Run it with the version whose format is to be kept:
 *
 *   gcc -O2 -I. tests/gen_sparse_fixture.c kci_sparse.c -lpthread -lm -o gen_sparse_fixture
 *   ./gen_sparse_fixture tests/data/sparse_v1          -> sparse_v1_plain.kcs, sparse_v1_key.kcs
 *
 * New. tests/data/sparse_v1_*.kcs were written by commit 9a65e47 (format KCISPAR1 as first
 * committed in c31fd7a: full rows, summaries as runs of (group u16, byte u8) with 32-bit run offsets).
 * The rows come from fx_row() below; the test has the same function and builds the same index again.
 */
#include "kci_sparse.h"
#include <stdio.h>
#include <string.h>

#define FX_DIM 600
#define FX_N   200
#define FX_KEY "fixture key 2026"

/* row d: 8-40 pairs, dimensions ascending, weights 0.05-2.0, every 6th row with some negative weights */
static int fx_row(int d, uint32_t *idx, float *val){
    uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t)(d + 1) + 12345;
    int n = 0; uint32_t k = 0;
    for (;;){
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        k += 1 + (uint32_t)(x >> 33) % 60;
        if (k >= FX_DIM || n >= 40) break;
        float v = 0.05f + (float)((x >> 12) % 1950) / 1000.0f;
        if (d % 6 == 0 && ((x >> 5) & 3) == 0) v = -v;
        idx[n] = k; val[n] = v; n++;
    }
    return n;
}

int main(int argc, char **argv){
    if (argc < 2){ fprintf(stderr, "usage: %s out_prefix\n", argv[0]); return 2; }
    uint32_t idx[64]; float val[64]; char path[1024];
    KciSparse *s = kci_sparse_new(FX_DIM);
    if (!s) return 1;
    for (int d = 0; d < FX_N; d++){ int n = fx_row(d, idx, val); if (kci_sparse_add(s, 100 + (uint64_t)d, idx, val, n) < 0) return 1; }
    KciSparseBlockParams bp; memset(&bp, 0, sizeof bp); bp.list_len = 40; bp.block_size = 5; bp.threads = 1;
    if (kci_sparse_build(s, 1, NULL) != 0 || kci_sparse_build_blocks(s, &bp, NULL) != 0 || kci_sparse_build_knn(s, 6, 0, 0, 1, NULL) != 0) return 1;
    for (int d = 0; d < 5; d++) kci_sparse_delete(s, 100 + (uint64_t)(d * 37));
    for (int d = FX_N; d < FX_N + 3; d++){ int n = fx_row(d, idx, val); if (kci_sparse_insert(s, 100 + (uint64_t)d, idx, val, n) < 0) return 1; }
    snprintf(path, sizeof path, "%s_plain.kcs", argv[1]);
    if (kci_sparse_save(s, path, NULL, 0) != 0) return 1;
    snprintf(path, sizeof path, "%s_key.kcs", argv[1]);
    if (kci_sparse_save(s, path, (const uint8_t *)FX_KEY, (int)strlen(FX_KEY)) != 0) return 1;
    printf("wrote %s_plain.kcs and %s_key.kcs: %u rows, dim %d, %u waiting, %d neighbours\n", argv[1], argv[1],
           kci_sparse_count(s), FX_DIM, kci_sparse_pending(s), kci_sparse_knn(s));
    kci_sparse_free(s);
    return 0;
}
