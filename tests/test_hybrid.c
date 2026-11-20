/*
 Hybrid search end to end: a dense index (kci_vec) and a sparse index
 * (kci_sparse) over the same documents, same ids, merged by kci_hybrid_rrf.
 *
 * The data is made so that neither side can answer alone:
 *   - TOPICS topics x PER documents. A document's dense vector is its topic's
 *     centre plus noise, so the dense side knows the topic but not which
 *     document of the topic.
 *   - Document j of every topic carries code word j, so the sparse side knows
 *     the code but not the topic (TOPICS documents share each code).
 * A query asks for "topic t, code j". Exactly one document fits both. Taking
 * PER candidates from the dense side (= one topic) it is the only document in
 * both lists, so the merged ranking must put it first. Run under ASan/UBSan.
 */
#include "kci_vec.h"
#include "kci_sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM    64
#define TOPICS 20
#define PER    50
#define N      (TOPICS * PER)
#define VOCAB  (PER + TOPICS)     /* code words 0..PER-1, then one common word per topic */
#define DEPTH  PER                /* candidates taken from each side. Not more than PER: the dense list would
                                     then reach into other topics, and a document of another topic with the
                                     right code, first on the sparse side, can then outrank the wanted one */
#define K      10

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static float rndf(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) % 1000000) / 1000000.0f; }
static float gauss(void){ float s = 0; for (int i = 0; i < 6; i++) s += rndf(); return (s - 3.0f) * 1.41f; }
static void unit(float *v){ double n = 0; for (int i = 0; i < DIM; i++) n += (double)v[i] * v[i]; n = sqrt(n); for (int i = 0; i < DIM; i++) v[i] = (float)(v[i] / n); }

static float centre[TOPICS][DIM];
static void topic_vec(int t, float noise, float *out){
    for (int i = 0; i < DIM; i++) out[i] = centre[t][i] + noise * gauss();
    unit(out);
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

int main(void){
    for (int t = 0; t < TOPICS; t++){ for (int i = 0; i < DIM; i++) centre[t][i] = gauss(); unit(centre[t]); }

    KciVec *dense = kci_vec_new(DIM, 16, 200);
    KciSparse *sparse = kci_sparse_new(VOCAB);
    CHECK(dense && sparse, "new");
    float v[DIM];
    for (int t = 0; t < TOPICS; t++)
        for (int j = 0; j < PER; j++){
            uint64_t id = 100000 + (uint64_t)t * 1000 + (uint64_t)j;      /* the same id on both sides */
            topic_vec(t, 0.08f, v);
            CHECK(kci_vec_add(dense, id, v) > 0, "dense add");
            uint32_t idx[2] = { (uint32_t)j, (uint32_t)(PER + t) };       /* its code word, its topic's common word */
            float    val[2] = { 2.0f, 0.3f };
            CHECK(kci_sparse_add(sparse, id, idx, val, 2) > 0, "sparse add");
        }
    CHECK(kci_vec_build(dense, 4, NULL) == 0, "dense build");
    CHECK(kci_sparse_build(sparse, 4, NULL) == 0, "sparse build");
    KciVecSearcher *ds = kci_vec_searcher_new(dense);
    KciSparseSearcher *ss = kci_sparse_searcher_new(sparse);
    CHECK(ds && ss, "searchers");

    int top1_dense = 0, top1_sparse = 0, top1_hybrid = 0, both_sides = 0, nq = 0, bad = 0, off_topic = 0;
    uint64_t dids[DEPTH], sids[DEPTH], out[K]; float sc[K];
    for (int t = 0; t < TOPICS; t++)
        for (int j = 0; j < PER; j += 5){
            uint64_t target = 100000 + (uint64_t)t * 1000 + (uint64_t)j;
            topic_vec(t, 0.08f, v);
            uint32_t qi[1] = { (uint32_t)j }; float qv[1] = { 1.0f };
            int nd = kci_vec_search_r(ds, v, DEPTH, 200, dids, NULL);
            int ns = kci_sparse_search_exact(ss, qi, qv, 1, DEPTH, sids, NULL);
            if (nd <= 0 || ns != TOPICS){ bad++; continue; }
            const uint64_t *lists[2] = { dids, sids }; int lens[2] = { nd, ns };
            int n = kci_hybrid_rrf(lists, lens, NULL, 2, 60.0f, K, out, sc);
            if (n != K){ bad++; continue; }
            for (int i = 1; i < n; i++) if (sc[i] > sc[i-1]) bad++;
            int in_d = 0, in_s = 0;
            for (int i = 0; i < nd; i++) if ((dids[i] - 100000) / 1000 != (uint64_t)t) off_topic++;
            for (int i = 0; i < nd; i++) if (dids[i] == target) in_d = 1;
            for (int i = 0; i < ns; i++) if (sids[i] == target) in_s = 1;
            both_sides += in_d && in_s;
            top1_dense  += dids[0] == target;
            top1_sparse += sids[0] == target;
            top1_hybrid += out[0] == target;
            nq++;
        }
    printf("queries %d : wanted document first — dense only %d, sparse only %d, hybrid %d (in both candidate lists %d)\n",
           nq, top1_dense, top1_sparse, top1_hybrid, both_sides);
    CHECK(bad == 0, "%d queries returned a wrong count or unordered scores", bad);
    CHECK(nq == TOPICS * (PER / 5), "queries run");
    CHECK(off_topic == 0, "the dense side returned %d documents of another topic (the data is meant to keep topics apart)", off_topic);
    /* the wanted document is the only one in both lists, so whenever both sides have it, it must win */
    CHECK(top1_hybrid == both_sides, "hybrid top-1 %d != in-both-lists %d", top1_hybrid, both_sides);
    CHECK(both_sides >= nq * 99 / 100, "the wanted document should be within the candidates of both sides");
    CHECK(top1_dense < nq / 4 && top1_sparse < nq / 4, "the data is meant to defeat each side alone");

    /* weights: with the dense side switched off the merged order is the sparse order, and the other way round */
    {
        topic_vec(3, 0.08f, v);
        uint32_t qi[1] = { 7 }; float qv[1] = { 1.0f };
        int nd = kci_vec_search_r(ds, v, DEPTH, 200, dids, NULL);
        int ns = kci_sparse_search_exact(ss, qi, qv, 1, DEPTH, sids, NULL);
        const uint64_t *lists[2] = { dids, sids }; int lens[2] = { nd, ns };
        float w_sparse[2] = { 0.0f, 1.0f }, w_dense[2] = { 1.0f, 0.0f };
        int n = kci_hybrid_rrf(lists, lens, w_sparse, 2, 60.0f, K, out, NULL);
        int same = n == K; for (int i = 0; i < n && same; i++) same = out[i] == sids[i];
        CHECK(same, "weight 0 on the dense side must give the sparse order");
        n = kci_hybrid_rrf(lists, lens, w_dense, 2, 60.0f, K, out, NULL);
        same = n == K; for (int i = 0; i < n && same; i++) same = out[i] == dids[i];
        CHECK(same, "weight 0 on the sparse side must give the dense order");
    }

    kci_vec_searcher_free(ds); kci_sparse_searcher_free(ss);
    kci_vec_free(dense); kci_sparse_free(sparse);
    printf(fails ? "test_hybrid: %d FAILED\n" : "test_hybrid: all OK\n", fails);
    return fails ? 1 : 0;
}
