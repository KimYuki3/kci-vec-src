/*
 test_ivf.c — partitioned 4-bit table-lookup index (kci_vec_build_ivf / _search_ivf).
 */
#include "kci_vec.c"
#include <stdio.h>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail = 1; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint64_t g_s = 88172645463325252ULL;
static uint64_t rnd(void) { g_s ^= g_s << 13; g_s ^= g_s >> 7; g_s ^= g_s << 17; return g_s; }
static float frnd(void) { return (float)((double)(rnd() >> 11) / 9007199254740992.0) * 2.0f - 1.0f; }

/* 1. kv_scan_list vs a plain sum of table lookups */
static int cmp_float(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
static void test_scan(int M, int nvec, int R, int high) {
    struct KvIvf f; memset(&f, 0, sizeof f); f.M = M; f.bb = (size_t)M * 16;
    KvList L; memset(&L, 0, sizeof L);
    uint8_t *codes = malloc((size_t)nvec * M);
    for (int i = 0; i < nvec; i++) { for (int m = 0; m < M; m++) codes[i * M + m] = (uint8_t)(rnd() & 15); kv_list_push(&L, &f, (uint32_t)i, codes + i * M, NULL); }
    int cap = 65000 / M + 1; if (cap > KV_LUT_TOP + 1) cap = KV_LUT_TOP + 1;
    /* high: values in the upper half, so most sums are above 32,767 (a signed 16-bit compare got them wrong) */
    int lo = high ? cap / 2 : 0;
    uint8_t *lut = malloc((size_t)(M + 4) * 16); for (int i = 0; i < M * 16; i++) lut[i] = (uint8_t)(lo + rnd() % (cap - lo)); memset(lut + M * 16, 0, 64);
    float *want = malloc(nvec * sizeof(float)), *wsort = malloc(nvec * sizeof(float));
    for (int i = 0; i < nvec; i++) { uint32_t s = 0; for (int m = 0; m < M; m++) s += lut[m * 16 + codes[i * M + m]]; want[i] = wsort[i] = -(float)s; }
    qsort(wsort, nvec, sizeof(float), cmp_float);
    int Rr = R < nvec ? R : nvec, scap = 3 * Rr + 64;
    He *buf = malloc((size_t)(2 * nvec + scap + 64) * sizeof(He));
    for (int way = 0; way < 2; way++) {
        KvSl S = {buf, 0, scap, R, 1e30f};
        kv_scan_list(&f, &L, lut, 0.0f, 1.0f, &S, way ? 0u : UINT32_MAX, NULL, 0, NULL, NULL, NULL);
        kv_sl_compact(&S);
        float *got = malloc((size_t)(S.n + 1) * sizeof(float));
        int ok = S.n >= Rr && S.n <= Rr + scap / 2, idok = 1;
        for (int i = 0; i < S.n; i++) { got[i] = S.a[i].d; if (S.a[i].id >= (uint32_t)nvec || want[S.a[i].id] != S.a[i].d) idok = 0; }
        qsort(got, S.n, sizeof(float), cmp_float);
        for (int i = 0; i < Rr && ok; i++) ok = got[i] == wsort[i];
        CHECK(ok && idok, "table sums M=%d nvec=%d R=%d high=%d way=%d: shortlist differs from the plain sum (n=%d, ids %s)",
              M, nvec, R, high, way, S.n, idok ? "ok" : "wrong");
        free(got); }
    { for (int i = 0; i < nvec; i++) kv_list_push(&L, &f, (uint32_t)i, codes + i * M, NULL);
      uint16_t *vis = calloc(nvec, sizeof(uint16_t));
      KvSl S = {buf, 0, scap, R, 1e30f};
      kv_scan_list(&f, &L, lut, 0.0f, 1.0f, &S, UINT32_MAX, vis, 1, NULL, NULL, NULL);
      kv_sl_compact(&S);
      int dup = 0; for (int i = 0; i < S.n; i++) for (int j = i + 1; j < S.n; j++) dup += S.a[i].id == S.a[j].id;
      CHECK(dup == 0 && S.n >= Rr && S.n <= Rr + scap / 2, "marks M=%d nvec=%d R=%d: %d duplicate ids, %d in shortlist", M, nvec, R, dup, S.n);
      free(vis); }
    free(want); free(wsort); free(codes); free(L.codes); free(L.ids); free(buf); free(lut);
}

/* clustered data: 64 centres, unit-ish noise */
#define N0 20000
static int g_nlist = 100;   /* lists per index; 512 exercises the two-level centroid scoring (blocks of 16 are pre-selected) */
static float *mkdata(int n, int D, uint64_t seed) {
    g_s = seed; float *c = malloc(64 * D * sizeof(float)), *x = malloc((size_t)n * D * sizeof(float));
    for (int i = 0; i < 64 * D; i++) c[i] = frnd();
    for (int i = 0; i < n; i++) { int k = (int)(rnd() % 64); for (int d = 0; d < D; d++) x[(size_t)i * D + d] = c[k * D + d] + 0.6f * frnd(); }
    free(c); return x;
}
static float *mkdata_r(int n, int D, uint64_t seed, int r) {
    g_s = seed; float *c = malloc(64 * D * sizeof(float)), *A = malloc((size_t)D * r * sizeof(float)), *x = malloc((size_t)n * D * sizeof(float));
    for (int i = 0; i < 64 * D; i++) c[i] = frnd();
    for (int i = 0; i < D * r; i++) A[i] = frnd();
    for (int i = 0; i < n; i++) { int k = (int)(rnd() % 64); float z[r]; for (int j = 0; j < r; j++) z[j] = frnd();
        for (int d = 0; d < D; d++) { float s2 = 0; for (int j = 0; j < r; j++) s2 += A[(size_t)d * r + j] * z[j]; x[(size_t)i * D + d] = c[k * D + d] + 1.8f * s2 / sqrtf((float)r) + 0.05f * frnd(); } }
    free(c); free(A); return x;
}
static void unit(float *x, int D) { double s = 0; for (int d = 0; d < D; d++) s += (double)x[d] * x[d]; float r = (float)(1 / sqrt(s)); for (int d = 0; d < D; d++) x[d] *= r; }

static double recall(KciVec *v, const float *X, int n, int D, int nq, int nprobe, int R, const uint8_t *dead) {
    KciVecSearcher *s = kci_vec_searcher_new(v); float *q = malloc(D * sizeof(float)), *u = malloc(D * sizeof(float));
    uint64_t ids[10]; float sc[10]; double hit = 0; g_s = 4242;
    for (int t = 0; t < nq; t++) {
        int r = (int)(rnd() % n); for (int d = 0; d < D; d++) q[d] = X[(size_t)r * D + d] + 0.3f * frnd();
        memcpy(u, q, D * sizeof(float)); unit(u, D);
        float best[10]; int bid[10]; for (int j = 0; j < 10; j++) { best[j] = -1e30f; bid[j] = -1; }
        for (int i = 0; i < n; i++) { if (dead && dead[i]) continue; float xx[D]; memcpy(xx, X + (size_t)i * D, D * sizeof(float)); unit(xx, D);
            float s2 = 0; for (int d = 0; d < D; d++) s2 += u[d] * xx[d];
            if (s2 > best[9]) { int j = 9; while (j > 0 && best[j - 1] < s2) { best[j] = best[j - 1]; bid[j] = bid[j - 1]; j--; } best[j] = s2; bid[j] = i; } }
        int k = kci_vec_search_ivf(s, q, 10, nprobe, R, ids, sc);
        for (int a = 0; a < k; a++) for (int b = 0; b < 10; b++) if ((int)ids[a] == bid[b]) { hit++; break; }
    }
    free(q); free(u); kci_vec_searcher_free(s);
    return hit / (nq * 10.0);
}
/* number of ids in [0,n) (not dead) that do not find themselves with all lists probed */
static int self_miss(KciVec *v, const float *X, int n, int D, const uint8_t *dead, int step) {
    KciVecSearcher *s = kci_vec_searcher_new(v); uint64_t ids[5]; float sc[5]; int miss = 0;
    for (int i = 0; i < n; i += step) { if (dead && dead[i]) continue;
        /* low-rank lists are a lossy filter: give them a wider shortlist (400 of 20,000) */
        int k = kci_vec_search_ivf(s, X + (size_t)i * D, 5, 1 << 30, v->ivf->lr ? 400 : 50, ids, sc), hit = 0;
        for (int a = 0; a < k; a++) hit |= (int)ids[a] == i; miss += !hit; }
    kci_vec_searcher_free(s); return miss;
}
static int same_answers(KciVec *a, KciVec *b, const float *X, int D) {
    KciVecSearcher *sa = kci_vec_searcher_new(a), *sb = kci_vec_searcher_new(b);
    uint64_t ia[10], ib[10]; float fa[10], fb[10]; int ok = 1;
    for (int t = 0; t < 200 && ok; t++) { int na = kci_vec_search_ivf(sa, X + (size_t)t * 37 * D, 10, 8, 40, ia, fa), nb = kci_vec_search_ivf(sb, X + (size_t)t * 37 * D, 10, 8, 40, ib, fb);
        ok = na == nb && memcmp(ia, ib, na * sizeof(uint64_t)) == 0; }
    kci_vec_searcher_free(sa); kci_vec_searcher_free(sb); return ok;
}

/* every list entry counted (both copies when the index spills) */
static size_t list_entries(KciVec *v) { size_t n = 0; for (int c = 0; c < v->ivf->nlist; c++) n += v->ivf->lists[c].n; return n; }
static int even_id(uint64_t id, void *ctx) { (void)ctx; return (id & 1) == 0; }   /* filter for the filtered-search check */
/* duplicate ids in the answers of 300 queries with every list probed (a spilled vector sits in two lists) */
static int dup_answers(KciVec *v, const float *X, int D) {
    KciVecSearcher *s = kci_vec_searcher_new(v); uint64_t ids[10]; float sc[10]; int dup = 0;
    for (int t = 0; t < 300; t++) { int k = kci_vec_search_ivf(s, X + (size_t)t * 53 * D, 10, 1 << 30, 100, ids, sc);
        for (int a = 0; a < k; a++) for (int b = a + 1; b < k; b++) dup += ids[a] == ids[b]; }
    kci_vec_searcher_free(s); return dup;
}

static void test_index(int D, int with_graph, float soar, int code_dims, int lowrank) {
    printf("index D=%d %s%s%s%s\n", D, with_graph ? "graph + lists" : "lists only", soar > 0 ? ", two lists per vector (SOAR)" : "",
           code_dims ? ", coded dims" : "", lowrank ? ", low-rank lists" : "");
    /* coded dims / low rank: the 500 vectors inserted later must come from the same clusters and subspace, so draw them together */
    int rk = code_dims ? code_dims : lowrank;
    float *X = rk ? mkdata_r(N0 + 500, D, 1234 + D, rk) : mkdata(N0, D, 1234 + D);
    KciVec *v = kci_vec_new(D, 16, 100); kci_vec_keep_float(v, 1);
    for (int i = 0; i < N0; i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i * D);
    if (with_graph) CHECK(kci_vec_build(v, 0, NULL) == 0, "graph build");
    KciVecIvfParams P = { g_nlist, 0, soar, code_dims, lowrank };
    CHECK(kci_vec_build_ivf_p(v, &P) == 0, "build_ivf");
    { int Dc = code_dims ? code_dims : D, dp = Dc <= KV_LOWDIM ? 1 : 2;
      CHECK(v->ivf->d1 == code_dims && v->ivf->lr == lowrank && v->ivf->dp == dp && v->ivf->M == (lowrank ? 0 : (dp == 2 ? (Dc + 1) / 2 : Dc)),
            "coded dims %d low rank %d pieces %d dims per code %d", v->ivf->d1, v->ivf->lr, v->ivf->M, v->ivf->dp); }
    CHECK(list_entries(v) == (soar > 0 ? 2u : 1u) * N0, "list entries %zu (expected %u per vector)", list_entries(v), soar > 0 ? 2 : 1);
    double r = recall(v, X, N0, D, 200, 20, 100, NULL);
    printf("  recall@10 (nprobe=20 of %d, rerank 100) = %.3f\n", g_nlist, r);
    CHECK(r >= 0.90, "recall@10 %.3f < 0.90", r);
    int sm = self_miss(v, X, N0, D, NULL, 7);
    CHECK(sm == 0, "%d vectors do not find themselves", sm);
    { int dup = dup_answers(v, X, D); CHECK(dup == 0, "%d duplicate ids in answers", dup); }
    { KciVecSearcher *sf = kci_vec_searcher_new(v), *si = kci_vec_searcher_new(v); kci_vec_searcher_set_rerank_int8(si, 1);
      uint64_t a[10], b[10]; float fa[10], fb[10]; double ov = 0; int dup = 0;
      for (int t = 0; t < 200; t++) { int na = kci_vec_search_ivf(sf, X + (size_t)t * 37 * D, 10, 20, 100, a, fa), nb = kci_vec_search_ivf(si, X + (size_t)t * 37 * D, 10, 20, 100, b, fb);
          for (int i = 0; i < na; i++) for (int j = 0; j < nb; j++) ov += a[i] == b[j];
          for (int i = 0; i < nb; i++) for (int j = i + 1; j < nb; j++) dup += b[i] == b[j]; }
      ov /= 200 * 10.0; CHECK(ov >= 0.9 && dup == 0, "int8 re-rank: overlap with float %.3f, %d duplicates", ov, dup);
      kci_vec_searcher_free(sf); kci_vec_searcher_free(si); }
    { KciVecSearcher *s0 = kci_vec_searcher_new(v), *sw = kci_vec_searcher_new(v), *sm2 = kci_vec_searcher_new(v), *sn = kci_vec_searcher_new(v);
      CHECK(kci_vec_searcher_set_probe_gap(sw, 10.0f, 1) == 0 && kci_vec_searcher_set_probe_gap(sm2, 0.01f, 20) == 0 &&
            kci_vec_searcher_set_probe_gap(sn, 0.05f, 2) == 0 && kci_vec_searcher_set_probe_gap(sn, NAN, 2) == -1, "set_probe_gap");
      uint64_t a[10], b[10], c[10], d[10]; float fa[10], fb[10], fc[10], fd[10]; int same = 1, bad = 0;
      for (int t = 0; t < 200; t++) { const float *q = X + (size_t)t * 37 * D;
          int na = kci_vec_search_ivf(s0, q, 10, 20, 100, a, fa), nb = kci_vec_search_ivf(sw, q, 10, 20, 100, b, fb);
          int nc = kci_vec_search_ivf(sm2, q, 10, 20, 100, c, fc), nd = kci_vec_search_ivf(sn, q, 10, 20, 100, d, fd);
          if (na != nb || na != nc) same = 0;
          for (int i = 0; i < na && same; i++) same = a[i] == b[i] && a[i] == c[i];
          if (nd != 10) bad++;
          for (int i = 0; i < nd; i++) for (int j = i + 1; j < nd; j++) bad += d[i] == d[j]; }
      CHECK(same && bad == 0, "probe gap: wide gap / min_probe >= nprobe %s the plain answers, narrow gap %d short or duplicate answers",
            same ? "match" : "differ from", bad);
      kci_vec_searcher_free(s0); kci_vec_searcher_free(sw); kci_vec_searcher_free(sm2); kci_vec_searcher_free(sn); }
    { KciVecSearcher *s = kci_vec_searcher_new(v); uint64_t ids[10]; float sc[10]; int odd = 0; double hit = 0; g_s = 777;
      float *q = malloc(D * sizeof(float)), *u = malloc(D * sizeof(float));
      for (int t = 0; t < 100; t++) {
          int r = (int)(rnd() % N0); for (int d = 0; d < D; d++) q[d] = X[(size_t)r * D + d] + 0.3f * frnd();
          memcpy(u, q, D * sizeof(float)); unit(u, D);
          float best[10]; int bid[10]; for (int j = 0; j < 10; j++) { best[j] = -1e30f; bid[j] = -1; }
          for (int i = 0; i < N0; i += 2) { float xx[D]; memcpy(xx, X + (size_t)i * D, D * sizeof(float)); unit(xx, D);
              float s2 = 0; for (int d = 0; d < D; d++) s2 += u[d] * xx[d];
              if (s2 > best[9]) { int j = 9; while (j > 0 && best[j - 1] < s2) { best[j] = best[j - 1]; bid[j] = bid[j - 1]; j--; } best[j] = s2; bid[j] = i; } }
          int k = kci_vec_search_ivf_filtered(s, q, 10, 20, 200, even_id, NULL, ids, sc);
          for (int a = 0; a < k; a++) { odd += (int)(ids[a] & 1); for (int b = 0; b < 10; b++) if ((int)ids[a] == bid[b]) { hit++; break; } } }
      hit /= 1000.0; CHECK(odd == 0 && hit >= 0.85, "filtered list search: %d odd ids returned, recall@10 over even rows %.3f", odd, hit);
      free(q); free(u); kci_vec_searcher_free(s); }
    /* delete 300, insert 500 new ones */
    uint8_t *dead = calloc(N0 + 500, 1);
    for (int i = 0; i < N0; i += 67) { CHECK(kci_vec_delete(v, (uint64_t)i) == 0, "delete %d", i); dead[i] = 1; }
    float *Y = rk ? X + (size_t)N0 * D : mkdata(500, D, 999 + D); float *XY = malloc((size_t)(N0 + 500) * D * sizeof(float));
    memcpy(XY, X, (size_t)N0 * D * sizeof(float)); memcpy(XY + (size_t)N0 * D, Y, (size_t)500 * D * sizeof(float));
    for (int i = 0; i < 500; i++) CHECK(kci_vec_insert(v, (uint64_t)(N0 + i), Y + (size_t)i * D) > 0, "insert %d", i);
    sm = self_miss(v, XY, N0 + 500, D, dead, 7);
    CHECK(sm == 0, "after delete/insert: %d vectors do not find themselves", sm);
    { KciVecSearcher *s = kci_vec_searcher_new(v); uint64_t ids[10]; float sc[10]; int seen = 0;
      for (int i = 0; i < N0; i += 67) { int k = kci_vec_search_ivf(s, X + (size_t)i * D, 10, 1 << 30, 50, ids, sc);
          for (int a = 0; a < k; a++) seen += dead[ids[a]]; }
      kci_vec_searcher_free(s); CHECK(seen == 0, "%d deleted vectors returned", seen); }
    /* save / load: plain and encrypted */
    const uint8_t *key = (const uint8_t *)"ivf key";
    for (int enc = 0; enc < 2; enc++) {
        CHECK(kci_vec_save(v, "_ivf.kv", enc ? key : NULL, enc ? 7 : 0) == 0, "save enc=%d", enc);
        KciVec *w = kci_vec_load("_ivf.kv", enc ? key : NULL, enc ? 7 : 0);
        CHECK(w != NULL, "load enc=%d", enc);
        if (w) { CHECK(same_answers(v, w, XY, D), "loaded index (enc=%d) answers differently", enc);
            CHECK(w->ivf->soar == v->ivf->soar && list_entries(w) == list_entries(v), "loaded index (enc=%d): soar %.3f entries %zu", enc, w->ivf->soar, list_entries(w));
            CHECK(w->ivf->d1 == v->ivf->d1 && w->ivf->M == v->ivf->M && w->ivf->lr == v->ivf->lr, "loaded index (enc=%d): coded dims %d low rank %d", enc, w->ivf->d1, w->ivf->lr);
            CHECK(w->ivf->dp == v->ivf->dp && w->ivf->avqt == v->ivf->avqt && w->ivf->eta == v->ivf->eta,
                  "loaded index (enc=%d): dims per code %d / %d, AVQ threshold %.4f / %.4f", enc, w->ivf->dp, v->ivf->dp, w->ivf->avqt, v->ivf->avqt);
            CHECK(dup_answers(w, XY, D) == 0, "loaded index (enc=%d): duplicate ids", enc);
            CHECK(self_miss(w, XY, N0 + 500, D, dead, 13) == 0, "loaded index (enc=%d): self miss", enc);
            CHECK(kci_vec_insert(w, 777777, Y) > 0, "insert into loaded index (enc=%d)", enc);
            kci_vec_free(w); }
    }
    remove("_ivf.kv");
    /* compact */
    uint32_t live = kci_vec_compact(v);
    CHECK(live == N0 + 500 - (N0 + 66) / 67, "compact live=%u", live);
    sm = self_miss(v, XY, N0 + 500, D, dead, 11);
    CHECK(sm == 0, "after compact: %d vectors do not find themselves", sm);
    CHECK(with_graph == kci_vec_is_built(v), "graph state after compact");
    kci_vec_free(v); free(X); if (!rk) free(Y); free(XY); free(dead);
}

static void test_exact(void) {
    int D = 48, N = 2000;
    KciVec *v = kci_vec_new(D, 16, 100);
    kci_vec_keep_float(v, 1);
    float *X = malloc(sizeof(float) * D * N);
    uint64_t s = 7;
    for (int i = 0; i < N * D; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; X[i] = (float)((int)(s % 2001) - 1000) / 1000.f; }
    for (int i = 0; i < N; i++) {
        float n = 0; for (int d = 0; d < D; d++) n += X[i * D + d] * X[i * D + d];
        n = sqrtf(n); for (int d = 0; d < D; d++) X[i * D + d] /= n;
        kci_vec_add(v, 5000 + (uint64_t)i, X + (size_t)i * D);
    }
    kci_vec_build_ivf(v, 32, 1);
    uint64_t ids[10]; float sc[10];
    int bad = 0;
    for (int qi = 0; qi < 50; qi++) {
        const float *q = X + (size_t)(qi * 37 % N) * D;
        int n = kci_vec_search_exact(v, q, 10, 1, ids, sc);
        float best = -2; int bi = -1;   /* brute force in floats */
        for (int i = 0; i < N; i++) { float d = 0; for (int k = 0; k < D; k++) d += q[k] * X[(size_t)i * D + k]; if (d > best) { best = d; bi = i; } }
        if (n != 10 || ids[0] != 5000 + (uint64_t)bi || fabsf(sc[0] - best) > 1e-4f || sc[0] > 1.0001f) bad++;
    }
    CHECK(bad == 0, "exact on a lists-only index: %d of 50 wrong", bad);
    kci_vec_delete(v, 5000 + 37);
    int n = kci_vec_search_exact(v, X + 37 * D, 10, 1, ids, sc);
    int seen = 0;
    for (int i = 0; i < n; i++) seen |= ids[i] == 5000 + 37;
    CHECK(n == 10 && !seen, "exact: deleted vector returned (n=%d)", n);
    free(X);
    kci_vec_free(v);
    printf("exact search: %s\n", bad || seen || n != 10 ? "FAIL" : "OK");
}

static void test_reload(int graph) {
    int D = 32, N = 3000;
    KciVec *v = kci_vec_new(D, 16, 100);
    kci_vec_keep_float(v, 1);
    float *X = malloc(sizeof(float) * D * (N + 10));
    uint64_t s = 11;
    for (int i = 0; i < (N + 10) * D; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; X[i] = (float)((int)(s % 2001) - 1000) / 1000.f; }
    for (int i = 0; i < N; i++) kci_vec_add(v, 100 + (uint64_t)i, X + (size_t)i * D);
    if (graph) kci_vec_build(v, 2, NULL);
    kci_vec_build_ivf(v, 40, 2);
    /* replace id 105 with a new vector (row N+1), then save with floats / without */
    kci_vec_delete(v, 105);
    kci_vec_insert(v, 105, X + (size_t)(N + 1) * D);
    const char *p1 = "_t_reload_f.kv", *p2 = "_t_reload_nf.kv";
    const uint8_t key[] = "k";
    CHECK(kci_vec_save(v, p1, key, 1) == 0 && kci_vec_save_ex(v, p2, key, 1, KCI_VEC_SAVE_NO_FLOAT) == 0, "save");
    KciVec *a = kci_vec_load(p1, key, 1), *b = kci_vec_load(p2, key, 1);
    CHECK(a && b, "load");
    if (!a || !b) { free(X); kci_vec_free(v); kci_vec_free(a); kci_vec_free(b); return; }
    FILE *fa = fopen(p1, "rb"), *fb = fopen(p2, "rb");
    fseek(fa, 0, SEEK_END); fseek(fb, 0, SEEK_END);
    long la = ftell(fa), lb = ftell(fb); fclose(fa); fclose(fb);
    CHECK(lb < la - (long)N * D * 4 + 4096, "no-float file is smaller by the floats (%ld vs %ld)", lb, la);
    /* fill b's floats from the source of truth (what an application does from its own store) */
    int bad_set = 0;
    for (int i = 0; i < N; i++) if (i != 5) bad_set += kci_vec_set_float(b, 100 + (uint64_t)i, X + (size_t)i * D) != 0;
    bad_set += kci_vec_set_float(b, 105, X + (size_t)(N + 1) * D) != 0;
    CHECK(bad_set == 0 && kci_vec_set_float(b, 999999, X) == -1, "set_float: every id found, unknown id refused");
    KciVecSearcher *sa = kci_vec_searcher_new(a), *sb = kci_vec_searcher_new(b);
    uint64_t ia[10], ib[10]; float fa2[10], fb2[10];
    int diff = 0;
    for (int q = 0; q < 50; q++) {
        const float *qq = X + (size_t)(q * 53 % N) * D;
        int na = kci_vec_search_ivf(sa, qq, 10, 40, 100, ia, fa2), nb = kci_vec_search_ivf(sb, qq, 10, 40, 100, ib, fb2);
        diff += na != nb || memcmp(ia, ib, sizeof(uint64_t) * (size_t)(na > 0 ? na : 0)) != 0;
    }
    CHECK(diff == 0, "no-float load + set_float answers like the full load (%d of 50 differ)", diff);
    /* delete the replaced id after the load: it must be gone */
    for (int t = 0; t < 2; t++) {
        KciVec *w = t ? b : a; KciVecSearcher *sw = t ? sb : sa;
        CHECK(kci_vec_delete(w, 105) == 0, "delete replaced id after load");
        kci_vec_searcher_free(sw);
        sw = kci_vec_searcher_new(w);
        int n = kci_vec_search_ivf(sw, X + (size_t)(N + 1) * D, 10, 40, 100, ia, fa2), seen = 0;
        for (int i = 0; i < n; i++) seen |= ia[i] == 105;
        CHECK(!seen, "replaced + reloaded + deleted id not returned (%s)", t ? "no-float file" : "full file");
        kci_vec_searcher_free(sw);
        if (t) sb = NULL; else sa = NULL;
    }
    remove(p1); remove(p2);
    free(X); kci_vec_free(v); kci_vec_free(a); kci_vec_free(b);
    printf("reload (%s): %s\n", graph ? "graph + lists" : "lists", g_fail ? "FAIL" : "OK");
}

static void test_rawkey(void) {
    int D = 16, N = 500;
    KciVec *v = kci_vec_new(D, 16, 100);
    kci_vec_keep_float(v, 1);
    float x[16];
    for (int i = 0; i < N; i++) { for (int d = 0; d < D; d++) x[d] = (float)((i * 31 + d * 17) % 97) - 48.f; kci_vec_add(v, (uint64_t)i, x); }
    kci_vec_build_ivf(v, 8, 1);
    uint8_t k1[32], k2[32];
    for (int i = 0; i < 32; i++) { k1[i] = (uint8_t)(i * 7 + 1); k2[i] = (uint8_t)(i * 7 + 2); }
    CHECK(kci_vec_save_ex(v, "_t_raw.kv", k1, 32, KCI_VEC_SAVE_RAW_KEY) == 0, "raw-key save");
    CHECK(kci_vec_save_ex(v, "_t_raw2.kv", k1, 16, KCI_VEC_SAVE_RAW_KEY) == -1, "raw key must be 32 bytes");
    KciVec *a = kci_vec_load_ex("_t_raw.kv", k1, 32, KCI_VEC_LOAD_RAW_KEY);
    CHECK(a && kci_vec_count(a) == (uint32_t)N, "raw-key load");
    KciVec *b = kci_vec_load("_t_raw.kv", k1, 32);
    CHECK(!b, "raw-key file refused as a passphrase file");
    KciVec *c = kci_vec_load_ex("_t_raw.kv", k2, 32, KCI_VEC_LOAD_RAW_KEY);
    CHECK(!c, "raw-key file refused with another key");
    kci_vec_save(v, "_t_pass.kv", (const uint8_t *)"pw", 2);
    KciVec *e = kci_vec_load_ex("_t_pass.kv", (const uint8_t *)"pw", 2, KCI_VEC_LOAD_RAW_KEY);
    CHECK(!e, "passphrase file refused as a raw-key file");
    remove("_t_raw.kv"); remove("_t_raw2.kv"); remove("_t_pass.kv");
    kci_vec_free(v); kci_vec_free(a); kci_vec_free(b); kci_vec_free(c); kci_vec_free(e);
    printf("raw key: %s\n", g_fail ? "FAIL" : "OK");
}

static size_t live_entries(KciVec *v) { size_t n = 0; for (int c = 0; c < v->ivf->nlist; c++) { KvList *L = &v->ivf->lists[c];
    for (uint32_t j = 0; j < L->n; j++) n += L->ids[j] != UINT32_MAX; } return n; }
static void test_forget(int D, float soar, int lowrank) {
    int N = 8000, fail0 = g_fail;   /* same_answers reads rows up to 7,400 */
    float *X = mkdata(N, D, 777 + D), *Y = mkdata(1, D, 999);
    KciVec *v = kci_vec_new(D, 16, 100);
    kci_vec_keep_float(v, 1);
    for (int i = 0; i < N; i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i * D);
    KciVecIvfParams P = { 30, 1, soar, 0, lowrank };
    CHECK(kci_vec_build_ivf_p(v, &P) == 0, "forget: build");
    CHECK(kci_vec_delete(v, 5) == 0 && kci_vec_insert(v, 5, Y) > 0, "forget: replace id 5 (old row tombstoned)");
    uint32_t per = soar > 0 ? 2u : 1u;
    if (lowrank) {                                  /* per-list bases cannot be recomputed here: the caller rebuilds */
        CHECK(kci_vec_forget(v, 5) == -2, "forget: low-rank lists -> -2");
        kci_vec_free(v); free(X); free(Y);
        printf("forget D=%d soar=%.1f lowrank=%d: %s\n", D, soar, lowrank, g_fail != fail0 ? "FAIL" : "OK");
        return;
    }
    CHECK(kci_vec_forget(v, 5) == 2, "forget: both rows of id 5 removed");
    CHECK(kci_vec_count(v) == (uint32_t)N - 1 && live_entries(v) == per * (N - 1), "forget: rows %u, list entries %zu",
          kci_vec_count(v), live_entries(v));
    float a[D], b[D], r[D]; memcpy(a, X + 5 * (size_t)D, sizeof a); unit(a, D); memcpy(b, Y, sizeof b); unit(b, D);
    int left = 0, badref = 0;
    for (uint32_t i = 0; i < kci_vec_count(v); i++) { kci_vec_get_float(v, i, r);
        left += !memcmp(r, a, sizeof r) || !memcmp(r, b, sizeof r) || v->vid[i] == 5; }
    for (int c = 0; c < v->ivf->nlist; c++) { KvList *L = &v->ivf->lists[c];
        for (uint32_t j = 0; j < L->n; j++) badref += L->ids[j] != UINT32_MAX && L->ids[j] >= kci_vec_count(v); }
    CHECK(left == 0 && badref == 0, "forget: %d rows still hold the vector or the id, %d list entries point past the rows", left, badref);
    { int close_c = 0;                              /* (review) no centroid left next to the forgotten vectors */
      for (int c = 0; c < v->ivf->nlist; c++) { const float *ce = v->ivf->cent + (size_t)c * D; double da = 0, db = 0;
          for (int d = 0; d < D; d++) { da += ce[d] * a[d]; db += ce[d] * b[d]; }
          close_c += da > 0.9999 || db > 0.9999; }
      CHECK(close_c == 0, "forget: %d centroids equal to a forgotten vector", close_c); }
    uint8_t *dead = calloc(N, 1); dead[5] = 1;
    int sm = self_miss(v, X, N, D, dead, 7);
    CHECK(sm == 0, "forget: %d other vectors do not find themselves", sm);
    { KciVecSearcher *s2 = kci_vec_searcher_new(v); uint64_t ids[10]; float sc[10];
      int k = kci_vec_search_ivf(s2, Y, 10, 1 << 30, 100, ids, sc), seen = 0;
      for (int j = 0; j < k; j++) seen += ids[j] == 5;
      kci_vec_searcher_free(s2); CHECK(seen == 0, "forget: id 5 still returned"); }
    CHECK(kci_vec_forget(v, 5) == 0, "forget: absent id -> 0");
    uint8_t key[32]; for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i + 3);
    CHECK(kci_vec_save_ex(v, "_t_forget.kv", key, 32, KCI_VEC_SAVE_RAW_KEY) == 0, "forget: save");
    KciVec *w = kci_vec_load_ex("_t_forget.kv", key, 32, KCI_VEC_LOAD_RAW_KEY);
    CHECK(w && kci_vec_count(w) == (uint32_t)N - 1 && same_answers(v, w, X, D), "forget: reloaded index has %u rows, same answers",
          w ? kci_vec_count(w) : 0);
    CHECK(kci_vec_insert(v, 5, Y) > 0, "forget: insert again");
    { KciVecSearcher *s2 = kci_vec_searcher_new(v); uint64_t ids[3]; float sc[3];
      int k = kci_vec_search_ivf(s2, Y, 3, 1 << 30, 100, ids, sc);
      kci_vec_searcher_free(s2); CHECK(k > 0 && ids[0] == 5, "forget: re-inserted id 5 found first"); }
    remove("_t_forget.kv"); kci_vec_free(w); kci_vec_free(v); free(dead); free(X); free(Y);
    KciVec *g = kci_vec_new(16, 16, 100); float x[16];
    kci_vec_keep_float(g, 1);
    for (int i = 0; i < 300; i++) { for (int d = 0; d < 16; d++) x[d] = frnd(); kci_vec_add(g, (uint64_t)i, x); }
    kci_vec_build(g, 0, NULL);
    CHECK(kci_vec_forget(g, 3) == -2, "forget: graph index -> -2");
    kci_vec_free(g);
    printf("forget D=%d soar=%.1f lowrank=%d: %s\n", D, soar, lowrank, g_fail != fail0 ? "FAIL" : "OK");
}

typedef struct { const float *X; int n, D; uint64_t gone; int calls; } FSrc;
static int fsrc(uint64_t id, float *out, void *ctx) {
    FSrc *c = ctx; c->calls++;
    if (id == c->gone || id >= (uint64_t)c->n) return 1;
    memcpy(out, c->X + (size_t)id * c->D, c->D * sizeof(float));
    return 0;
}
static void test_float_source(void) {
    int D = 64, N = 8000, fail0 = g_fail;
    float *X = mkdata(N + 1, D, 4242);
    KciVec *v = kci_vec_new(D, 16, 100);
    kci_vec_keep_float(v, 1);
    for (int i = 0; i < N; i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i * D);
    KciVecIvfParams P = { 40, 1, 0, 0, 0 };
    CHECK(kci_vec_build_ivf_p(v, &P) == 0, "float source: build");
    uint64_t ia[200][10]; float fa[200][10]; int na[200];
    KciVecSearcher *s1 = kci_vec_searcher_new(v);
    for (int t = 0; t < 200; t++) na[t] = kci_vec_search_ivf(s1, X + (size_t)t * 37 * D, 10, 8, 40, ia[t], fa[t]);
    kci_vec_searcher_free(s1);
    FSrc c = { X, N + 1, D, UINT64_MAX, 0 };
    CHECK(kci_vec_set_float_source(v, fsrc, &c, NULL) == 0 && v->vf == NULL && v->keepf == 0, "float source: set, floats dropped");
    KciVecSearcher *s2 = kci_vec_searcher_new(v);
    int same = 1; uint64_t ib[10]; float fb[10];
    for (int t = 0; t < 200 && same; t++) { int nb = kci_vec_search_ivf(s2, X + (size_t)t * 37 * D, 10, 8, 40, ib, fb);
        same = nb == na[t] && memcmp(ib, ia[t], nb * sizeof(uint64_t)) == 0;
        for (int j = 0; j < nb && same; j++) same = fabsf(fb[j] - fa[t][j]) < 1e-5f; }
    CHECK(same && c.calls > 0, "float source: the same answers and scores as with resident floats (%d fetches)", c.calls);
    CHECK(kci_vec_insert(v, (uint64_t)N, X + (size_t)N * D) > 0, "float source: insert without resident floats");
    int k = kci_vec_search_ivf(s2, X + (size_t)N * D, 3, 1 << 30, 50, ib, fb);
    CHECK(k > 0 && ib[0] == (uint64_t)N, "float source: the inserted vector finds itself");
    c.gone = 5;                                     /* the source no longer has id 5: it must not come back */
    k = kci_vec_search_ivf(s2, X + (size_t)5 * D, 3, 1 << 30, 50, ib, fb);
    CHECK(k > 0 && ib[0] != 5, "float source: an id the source lacks ranks last");
    c.gone = UINT64_MAX;
    CHECK(kci_vec_forget(v, 7) == 1, "float source: forget");
    kci_vec_searcher_free(s2);
    uint8_t key[32]; for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i * 3);
    CHECK(kci_vec_save_ex(v, "_t_fsrc.kv", key, 32, KCI_VEC_SAVE_RAW_KEY) == 0, "float source: save (no floats to write)");
    KciVec *w = kci_vec_load_ex("_t_fsrc.kv", key, 32, KCI_VEC_LOAD_RAW_KEY);
    CHECK(w && w->vf == NULL && kci_vec_set_float_source(w, fsrc, &c, NULL) == 0, "float source: load, set again");
    if (w) { KciVecSearcher *s3 = kci_vec_searcher_new(w);
        k = kci_vec_search_ivf(s3, X + (size_t)100 * D, 3, 1 << 30, 50, ib, fb);
        CHECK(k > 0 && ib[0] == 100, "float source: the loaded index answers"); kci_vec_searcher_free(s3); }
    remove("_t_fsrc.kv"); kci_vec_free(w); kci_vec_free(v); free(X);
    printf("float source: %s\n", g_fail != fail0 ? "FAIL" : "OK");
}

static void test_forget_centroid(void) {
    int D = 16, fail0 = g_fail; float x[16];
    KciVec *v = kci_vec_new(D, 16, 100);
    kci_vec_keep_float(v, 1);
    for (int i = 0; i < 4; i++) { for (int d = 0; d < D; d++) x[d] = (d == i * 4) ? 1.0f : 0.01f * d; kci_vec_add(v, (uint64_t)i, x); }
    KciVecIvfParams P = { 4, 1, 0, 0, 0 };
    CHECK(kci_vec_build_ivf_p(v, &P) == 0, "centroid: build 4 lists of 1");
    float a[16]; for (int d = 0; d < D; d++) a[d] = (d == 8) ? 1.0f : 0.01f * d; unit(a, D);
    int before = 0, after = 0;
    for (int c = 0; c < 4; c++) { double dt = 0; for (int d = 0; d < D; d++) dt += v->ivf->cent[c * D + d] * a[d]; before += dt > 0.9999; }
    CHECK(kci_vec_forget(v, 2) == 1, "centroid: forget id 2");
    for (int c = 0; c < 4; c++) { double dt = 0; for (int d = 0; d < D; d++) dt += v->ivf->cent[c * D + d] * a[d]; after += dt > 0.9999; }
    CHECK(before == 1 && after == 0, "centroid: id 2's centroid was there (%d) and is gone (%d)", before, after);
    KciVecSearcher *s = kci_vec_searcher_new(v); uint64_t ids[4]; float sc[4];
    for (int d = 0; d < D; d++) x[d] = (d == 4) ? 1.0f : 0.01f * d;
    int k = kci_vec_search_ivf(s, x, 1, 4, 4, ids, sc);
    CHECK(k == 1 && ids[0] == 1, "centroid: the others still found");
    kci_vec_searcher_free(s); kci_vec_free(v);
    printf("forget centroid: %s\n", g_fail != fail0 ? "FAIL" : "OK");
}

static void test_codes(void) {
    printf("codes: dims per code / AVQ threshold\n");
    struct { int D, dp_set; float t_set; int dp; float t; } cs[] = {
        { 25, 0, 0, 1, KV_LOWDIM_T }, { 32, 0, 0, 1, KV_LOWDIM_T }, { 33, 0, 0, 2, (float)KV_IVF_T }, { 100, 0, 0, 2, (float)KV_IVF_T },
        { 25, 2, 0, 2, KV_LOWDIM_T }, { 100, 1, 0.3f, 1, 19.0f / 64 }, { 64, 2, 0.55f, 2, 35.0f / 64 } };
    for (int t = 0; t < 7; t++) {
        int D = cs[t].D; float *X = mkdata(8000, D, 77 + D);
        KciVec *v = kci_vec_new(D, 16, 100); kci_vec_keep_float(v, 1);
        for (int i = 0; i < 8000; i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i * D);
        if (cs[t].dp_set || cs[t].t_set > 0) CHECK(kci_vec_set_ivf_codes(v, cs[t].dp_set, cs[t].t_set) == 0, "set_ivf_codes");
        CHECK(kci_vec_build_ivf(v, 32, 0) == 0, "build D=%d", D);
        int Mx = cs[t].dp == 2 ? (D + 1) / 2 : D;
        CHECK(v->ivf->dp == cs[t].dp && v->ivf->M == Mx && fabsf(v->ivf->avqt - cs[t].t) < 1e-6f,
              "D=%d set (%d, %.2f): dims per code %d (want %d), pieces %d, threshold %.4f (want %.4f)",
              D, cs[t].dp_set, cs[t].t_set, v->ivf->dp, cs[t].dp, v->ivf->M, v->ivf->avqt, cs[t].t);
        double r = recall(v, X, 8000, D, 100, 16, 100, NULL);
        CHECK(r >= 0.85, "D=%d dims per code %d: recall@10 %.3f", D, v->ivf->dp, r);
        CHECK(kci_vec_save(v, "_codes.kv", NULL, 0) == 0, "save");
        KciVec *w = kci_vec_load("_codes.kv", NULL, 0);
        CHECK(w && w->ivf && w->ivf->dp == v->ivf->dp && w->ivf->avqt == v->ivf->avqt && same_answers(v, w, X, D), "D=%d: load keeps the codes", D);
        if (w) { for (int i = 0; i < 8000; i += 5) kci_vec_delete(w, (uint64_t)i);
            kci_vec_compact(w);
            CHECK(w->ivf && w->ivf->dp == v->ivf->dp && w->ivf->avqt == v->ivf->avqt, "D=%d: compact after load keeps the codes", D);
            kci_vec_free(w); }
        remove("_codes.kv"); kci_vec_free(v); free(X);
    }
    { int D = 32; float *X = mkdata(8000, D, 91);
      KciVec *v = kci_vec_new(D, 16, 100); kci_vec_keep_float(v, 1);
      for (int i = 0; i < 8000; i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i * D);
      CHECK(kci_vec_set_ivf_training(v, 30, UINT32_MAX) == 0 && kci_vec_build_ivf(v, 32, 0) == 0, "build with 30 iterations on all vectors");
      double r = recall(v, X, 8000, D, 100, 16, 100, NULL);
      CHECK(r >= 0.85, "training 30 x all: recall@10 %.3f", r);
      CHECK(kci_vec_set_ivf_training(v, 1, 500) == 0 && kci_vec_build_ivf(v, 32, 0) == 0, "build with 1 iteration on 500 vectors");
      CHECK(kci_vec_set_ivf_training(v, -1, 0) == -1 && kci_vec_set_ivf_training(v, 1001, 0) == -1 && kci_vec_set_ivf_training(NULL, 10, 0) == -1,
            "set_ivf_training: bad args refused");
      kci_vec_free(v); free(X); }
    KciVec *v = kci_vec_new(16, 16, 100);
    CHECK(kci_vec_set_ivf_codes(v, 3, 0) == -1 && kci_vec_set_ivf_codes(v, -1, 0) == -1 && kci_vec_set_ivf_codes(v, 1, -0.1f) == -1
          && kci_vec_set_ivf_codes(v, 1, 0.99f) == -1 && kci_vec_set_ivf_codes(v, 1, NAN) == -1 && kci_vec_set_ivf_codes(NULL, 1, 0) == -1,
          "set_ivf_codes: bad args refused");
    kci_vec_free(v);
}

int main(void) {
#if defined(__AVX512BW__)
    printf("scan path: AVX-512\n");
#elif defined(__AVX2__)
    printf("scan path: AVX2\n");
#else
    printf("scan path: plain C\n");
#endif
    int Ms[] = {1, 2, 3, 5, 8, 50, 51, 64, 128, 384, 1024, 2560}; int ns[] = {1, 31, 32, 33, 100, 1000};
    for (int a = 0; a < 12; a++) for (int b = 0; b < 6; b++) for (int h = 0; h < 2; h++) { test_scan(Ms[a], ns[b], 1 << 20, h); test_scan(Ms[a], ns[b], 5, h); }
    printf("table sums: %s\n", g_fail ? "FAIL" : "OK");
    test_index(64, 0, 0, 0, 0);
    test_index(33, 1, 0, 0, 0);
    test_index(768, 0, 0, 0, 0);
    test_index(64, 0, 1.5f, 0, 0);
    test_index(33, 1, 1.5f, 0, 0);
    test_index(64, 0, 0, 32, 0);
    test_index(64, 1, 0, 32, 0);     /* small dims (33, 48) coded at 1/2 lose too much for the self check: code_dims is for 384+ */
    test_index(768, 0, 1.5f, 192, 0);
    test_index(64, 0, 0, 0, 32);
    test_index(768, 0, 1.5f, 0, 64); /* with SOAR (two lists, two coefficient sets per vector) */
    test_index(64, 1, 0, 0, 16);     /* graph first (renumbered rows), then low-rank lists */
    test_index(768, 0, 0, 256, 64);  /* low rank inside a projected space (code_dims 256 of 768): smaller bases */
    test_index(25, 0, 0, 0, 0);
    test_index(25, 1, 1.5f, 0, 0);
    test_codes();
    g_nlist = 512;
    test_index(64, 0, 0, 0, 0);
    test_index(64, 0, 1.5f, 0, 0);
    g_nlist = 100;
    test_exact();
    test_rawkey();
    test_forget(64, 0, 0);
    test_forget(64, 1.5f, 0);
    test_forget(64, 0, 32);
    test_float_source();
    test_forget_centroid();
    test_reload(0);
    test_reload(1);
    printf("%s\n", g_fail ? "FAIL" : "ALL PASS");
    return g_fail;
}
