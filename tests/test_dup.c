/*
 test_dup.c — a group of identical vectors must not become a dead end of the graph.
 *
 *   gcc -O2 -march=native -I. tests/test_dup.c kci_vec.c -lpthread -lm -o test_dup
 */
#include "kci_vec.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define DIM 64
#define N   20000
#define NDUP 60
#define NQ  500

static uint64_t rs = 0x2545F4914F6CDD1DULL;
static float rnd(void){            /* roughly normal: sum of 4 uniforms */
    float a = 0;
    for (int i = 0; i < 4; i++){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; a += (float)(rs % 20001) / 10000.0f - 1.0f; }
    return a * 0.866f;
}
static float centre[DIM];
static void mk(float *v, float spread){
    float n = 0;
    for (int i = 0; i < DIM; i++){ v[i] = centre[i] + spread * rnd(); n += v[i] * v[i]; }
    n = 1.0f / sqrtf(n);
    for (int i = 0; i < DIM; i++) v[i] *= n;
}
static float dot(const float *a, const float *b){ float s = 0; for (int i = 0; i < DIM; i++) s += a[i] * b[i]; return s; }
static int cmp_desc(const void *a, const void *b){ float x = *(const float*)a, y = *(const float*)b; return x < y ? 1 : x > y ? -1 : 0; }

static int run(int threads, int M){
    int total = N + NDUP, ok = 1;
    float *base = malloc((size_t)total * DIM * sizeof *base), *q = malloc((size_t)NQ * DIM * sizeof *q), *sc = malloc((size_t)total * sizeof *sc);
    if (!base || !q || !sc) return 0;
    rs = 0x2545F4914F6CDD1DULL;
    { float n = 0; for (int i = 0; i < DIM; i++){ centre[i] = rnd(); n += centre[i] * centre[i]; } n = 1.0f / sqrtf(n); for (int i = 0; i < DIM; i++) centre[i] *= n; }
    for (int i = 0; i < total; i++){
        /* the repeated rows are spread over the insertion order, as empty documents are in a corpus */
        if (i % (total / NDUP) == 7 && i / (total / NDUP) < NDUP) mk(base + (size_t)i * DIM, 0.0f);
        else mk(base + (size_t)i * DIM, 0.12f);
    }
    for (int i = 0; i < NQ; i++) mk(q + (size_t)i * DIM, 0.12f);
    KciVec *v = kci_vec_new(DIM, M, 200);
    kci_vec_keep_float(v, 1);
    for (int i = 0; i < total; i++) kci_vec_add(v, (uint64_t)i + 1, base + (size_t)i * DIM);
    kci_vec_build(v, threads, NULL);
    KciVecSearcher *s = kci_vec_searcher_new(v);
    uint64_t ids[10]; float out[10]; double sum = 0; int zero = 0, low = 0;
    for (int qi = 0; qi < NQ; qi++){
        const float *qq = q + (size_t)qi * DIM;
        for (int i = 0; i < total; i++) sc[i] = dot(qq, base + (size_t)i * DIM);
        int n = kci_vec_search_rerank(s, qq, 10, 200, 200, ids, out), hit = 0;
        float got[10];
        for (int i = 0; i < n; i++) got[i] = sc[ids[i] - 1];
        qsort(sc, (size_t)total, sizeof *sc, cmp_desc);
        for (int i = 0; i < n; i++) hit += got[i] >= sc[9] - 1e-5f;
        sum += hit / 10.0; zero += hit == 0; low += hit < 5;
    }
    printf("M %d, %d thread(s): top 10 against exact %.4f, queries with 0 of 10: %d, with fewer than 5: %d\n", M, threads, sum / NQ, zero, low);
    ok &= zero == 0 && low <= 5 && sum / NQ >= 0.92;
    /* the group itself: its vector as the query must return rows of the group (score 1) */
    {
        float c[DIM]; mk(c, 0.0f);
        int n = kci_vec_search_rerank(s, c, 10, 200, 200, ids, out), same = 0;
        for (int i = 0; i < n; i++) same += dot(c, base + (size_t)(ids[i] - 1) * DIM) > 0.9999f;
        printf("  the repeated vector as the query: %d of 10 are rows of the group\n", same);
        ok &= same == 10;
    }
    kci_vec_searcher_free(s); kci_vec_free(v);
    free(base); free(q); free(sc);
    return ok;
}

int main(void){
    int ok = 1;
    ok &= run(1, 16);
    ok &= run(8, 16);

    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
