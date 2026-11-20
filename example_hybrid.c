/*
 * example_hybrid.c — hybrid search: a dense index and a sparse index over the
 * same documents, merged into one ranking.
 *
 *   gcc -O2 -march=native -I. example_hybrid.c kci_vec.c kci_sparse.c -lpthread -lm -o example_hybrid
 *
 * In real use the dense vectors come from an embedding model and the sparse
 * ones from a sparse model (SPLADE) or from word weights (BM25); the calls
 * below stay the same. Give a document the same id on both sides.
 * New.
 */
#include "kci_vec.h"
#include "kci_sparse.h"
#include <stdio.h>
#include <math.h>

#define DIM      32
#define SUBJECTS 6
#define PARTS    8
#define DEPTH    8      /* how many candidates to take from each side: a few times K */
#define K        3

static uint64_t rs = 0x2545F4914F6CDD1Dull;
static float rndf(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) % 1000000) / 1000000.0f - 0.5f; }
static void unit(float *v){ double n = 0; for (int i = 0; i < DIM; i++) n += (double)v[i] * v[i]; n = sqrt(n); for (int i = 0; i < DIM; i++) v[i] = (float)(v[i] / n); }

static float subject[SUBJECTS][DIM];
static void about(int s, float *out){   /* a vector "about subject s" */
    for (int i = 0; i < DIM; i++) out[i] = subject[s][i] + 0.15f * rndf();
    unit(out);
}
static void show(const char *title, const uint64_t *ids, int n){
    printf("%-12s", title);
    for (int i = 0; i < n && i < K; i++) printf("  subject %llu part %llu", (unsigned long long)(ids[i] / 100), (unsigned long long)(ids[i] % 100));
    printf("\n");
}

int main(void){
    for (int s = 0; s < SUBJECTS; s++){ for (int i = 0; i < DIM; i++) subject[s][i] = rndf(); unit(subject[s]); }

    /* 1. make both indexes and add every document to both, under the same id */
    KciVec    *dense  = kci_vec_new(DIM, 16, 200);
    KciSparse *sparse = kci_sparse_new(PARTS);          /* vocabulary: the part-number words 0..PARTS-1 */
    if (!dense || !sparse) return 1;
    float v[DIM];
    for (int s = 0; s < SUBJECTS; s++)
        for (int p = 0; p < PARTS; p++){
            uint64_t id = (uint64_t)s * 100 + (uint64_t)p;
            about(s, v);
            uint32_t word[1] = { (uint32_t)p }; float weight[1] = { 1.0f };
            if (kci_vec_add(dense, id, v) < 0 || kci_sparse_add(sparse, id, word, weight, 1) < 0) return 1;
        }
    if (kci_vec_build(dense, 0, NULL) != 0 || kci_sparse_build(sparse, 0, NULL) != 0) return 1;

    /* 2. one searcher per index (per thread) */
    KciVecSearcher    *ds = kci_vec_searcher_new(dense);
    KciSparseSearcher *ss = kci_sparse_searcher_new(sparse);
    if (!ds || !ss) return 1;

    /* 3. the query, in both forms: "about subject 2" and "has part number 5" */
    about(2, v);
    uint32_t qword[1] = { 5 }; float qweight[1] = { 1.0f };

    /* 4. search each side for DEPTH candidates */
    uint64_t dids[DEPTH], sids[DEPTH], out[K]; float score[K];
    int nd = kci_vec_search_r(ds, v, DEPTH, 100, dids, NULL);
    int ns = kci_sparse_search_exact(ss, qword, qweight, 1, DEPTH, sids, NULL);
    if (nd < 0 || ns < 0) return 1;

    /* 5. merge the two rankings (weights NULL = both sides count the same) */
    const uint64_t *lists[2] = { dids, sids }; int lens[2] = { nd, ns };
    int n = kci_hybrid_rrf(lists, lens, NULL, 2, 60.0f, K, out, score);
    if (n < 0) return 1;

    printf("query: subject 2, part 5\n");
    show("dense only", dids, nd);
    show("sparse only", sids, ns);
    show("hybrid", out, n);
    printf("%s\n", n > 0 && out[0] == 205 ? "hybrid puts subject 2 part 5 first" : "unexpected: subject 2 part 5 is not first");

    kci_vec_searcher_free(ds); kci_sparse_searcher_free(ss);
    kci_vec_free(dense); kci_sparse_free(sparse);
    return n > 0 && out[0] == 205 ? 0 : 1;
}
