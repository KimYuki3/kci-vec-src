/* kci_vec_judge / kci_vec_judge_queries, kci_vec_graph_recall, kci_vec_row_id.
 *   gcc -O2 -march=native -I. tests/test_judge.c -lpthread -lm -o test_judge   (from the repository root) */
#include "kci_vec.c"
#include <stdio.h>

static int fails = 0;
#define CHECK(c, ...) do { if(!(c)){ fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while(0)

enum { N = 30000, D = 32, K = 10, NQ = 400, NC = 150 };
static float *X, *Q;
static unsigned rs = 24680;
static float rnd(void){ rs = rs*1664525u + 1013904223u; return (float)(rs>>8)/(float)(1u<<24) - 0.5f; }
static float gauss(void){ float a = 0; for(int i=0;i<6;i++) a += rnd(); return a*1.41f; }

/* Half of the vectors lie in NC tight groups (a query there has its neighbours close by and well apart from the rest:
 * easy), half are spread without structure (hard). The queries are drawn the same way, so both kinds occur. */
static void make(float *out, int n, const float *cen){
    for(int i=0;i<n;i++){ float *x = out + (size_t)i*D;
        if(i & 1){ const float *c = cen + (size_t)((rs>>10) % NC)*D; rnd(); for(int d=0;d<D;d++) x[d] = c[d] + 0.05f*gauss(); }
        else for(int d=0;d<D;d++) x[d] = gauss(); }
}
static double recall(KciVec *v, KciVecSearcher *s, int ef){        /* against the exact scan */
    uint64_t a[K], b[K]; long hit = 0;
    for(int q=0;q<NQ;q++){ const float *y = Q + (size_t)q*D;
        int n = kci_vec_search_rerank(s, y, K, ef, ef > 0 ? ef : K, a, NULL), m = kci_vec_search_exact(v, y, K, 4, b, NULL);
        for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(a[i]==b[j]){ hit++; break; } }
    return (double)hit/(NQ*K);
}

static void run(int l2){
    const char *name = l2 ? "l2" : "cosine";
    float *cen = malloc(sizeof(float)*NC*D);
    for(int i=0;i<NC*D;i++) cen[i] = 2.0f*gauss();
    make(X, N, cen); make(Q, NQ, cen);
    KciVec *v = l2 ? kci_vec_new_l2(D, 16, 100, 0.0f) : kci_vec_new(D, 16, 100);
    kci_vec_keep_float(v, 1);
    for(int i=0;i<N;i++) kci_vec_add(v, (uint64_t)i + 500, X + (size_t)i*D);
    CHECK(kci_vec_judge(v, K, 0.99f, 0, NULL) == -1, "%s: not built -> -1", name);
    { int bad = 0; float g[D];
      for(uint32_t i=0;i<N;i+=531){ bad += kci_vec_row_id(v, i) != (uint64_t)i + 500; if(l2){ kci_vec_get_float(v, i, g); bad += fabsf(g[3] - X[(size_t)i*D+3]) > 1e-4f; } }
      CHECK(bad == 0 && kci_vec_row_id(v, N) == UINT64_MAX && kci_vec_row_id(NULL, 0) == UINT64_MAX, "%s: kci_vec_row_id before the build (%d wrong)", name, bad); }
    CHECK(kci_vec_build(v, 8, NULL) == 0, "%s: build", name);
    CHECK(kci_vec_judged(v) == 0, "%s: no rule before kci_vec_judge", name);
    /* after the build rows may have moved: the id still belongs to the vector get_float returns for that row */
    { int bad = 0; float g[D];
      for(uint32_t i=0;i<N;i+=531){ uint64_t id = kci_vec_row_id(v, i); if(id < 500 || id >= (uint64_t)N + 500){ bad++; continue; }
          if(l2){ kci_vec_get_float(v, i, g); bad += fabsf(g[3] - X[(size_t)(id-500)*D+3]) > 1e-4f; } }
      CHECK(bad == 0, "%s: kci_vec_row_id after the build (%d wrong)", name, bad); }
    CHECK(kci_vec_judge(v, 1, 0.99f, 0, NULL) == -1 && kci_vec_judge(v, 65, 0.99f, 0, NULL) == -1, "%s: k 1 / 65 -> -1", name);
    KciVecSearcher *s = kci_vec_searcher_new(v);

    /* 1. a walk that goes on gives what a walk of that width from the start gives */
    { long same = 0, tot = 0; uint32_t a[K], b[K]; const float *q;
      for(int i=0;i<NQ;i++){ double qn2 = 0; q = Q + (size_t)i*D; (void)kv_l2_qs(s, &q, &qn2);
          enc_query(v, q, s->qf, s->q);
          int m = kv_jg_first(s, 32); CHECK(m == 32, "%s: the first pass holds %d rows, not 32", name, m);
          m = kv_jg_more(s, 100); m = kv_jg_more(s, 300); CHECK(m == 300, "%s: the continued walk holds %d rows, not 300", name, m);
          int na = kv_jg_topk(s->s.tmp, m, K, a);
          m = kv_jg_first(s, 300); int nb = kv_jg_topk(s->s.tmp, m, K, b);
          CHECK(na == K && nb == K, "%s: top k of the walks", name);
          for(int x=0;x<na;x++) for(int y=0;y<nb;y++) if(a[x]==b[y]){ same++; break; }
          tot += K; }
      printf("%s: 32 -> 100 -> 300 continued against 300 from the start: %.4f of the top %d the same\n", name, (double)same/tot, K);
      CHECK((double)same/tot >= 0.985, "%s: the continued walk differs from the direct one (%.4f)", name, (double)same/tot); }

    /* 2. a masked row is not entered: the stored vector asked as a question does not come back */
    { int back = 0; uint32_t a[K]; float qb[D];
      for(uint32_t row=7; row<N; row+=997){ const float *q = qb; double qn2 = 0;
          if(row == v->ep) continue;
          CHECK(kci_vec_get_float(v, row, qb) == 0, "%s: get_float", name); (void)kv_l2_qs(s, &q, &qn2);
          enc_query(v, q, s->qf, s->q); s->s.jskip = row + 1;
          int m = kv_jg_first(s, 32); m = kv_jg_more(s, 200); s->s.jskip = 0;
          int n = kv_jg_topk(s->s.tmp, m, K, a);
          for(int i=0;i<n;i++) back += a[i] == row; }
      CHECK(back == 0, "%s: a masked row came back %d times", name, back); }

    { static const int efs[5] = { 12, 24, 50, 200, 800 }; float r[5] = { -1, -1, -1, -1, -1 };
      int ns = kci_vec_graph_recall(v, K, 300, efs, 5, 4, r);
      CHECK(ns >= 250 && ns <= 300, "%s: kci_vec_graph_recall used %d samples", name, ns);
      for(int j=0;j<5;j++) CHECK(r[j] > 0.2f && r[j] <= 1.0f && (j == 0 || r[j] >= r[j-1] - 0.005f), "%s: recall %.4f at width %d", name, r[j], efs[j]);
      /* the l2 data of this test is hard for a graph */
      CHECK(r[4] >= (l2 ? 0.85f : 0.97f), "%s: masked recall at width 800 is %.4f", name, r[4]);
      long hit = 0, tot = 0; uint64_t a[K+1], b[K+1]; float qb[D];
      for(uint32_t row=3; row<N; row+=101){ if(kci_vec_get_float(v, row, qb) != 0) continue;
          int na = kci_vec_search_rerank(s, qb, K+1, 12, 12, a, NULL), nb = kci_vec_search_exact(v, qb, K+1, 4, b, NULL);
          for(int i=0;i<na;i++){ if(a[i] == v->vid[row]) continue; for(int j=0;j<nb;j++) if(b[j] == a[i]){ hit++; break; } }
          tot += K; }
      printf("%s: recall of stored vectors at width 12: %.4f searched plainly, %.4f with the walk kept out (width 800: %.4f)\n",
             name, (double)hit/tot, r[0], r[4]);
      CHECK(r[0] <= (double)hit/tot + 0.01, "%s: the masked walk is not harder than the plain one (%.4f, %.4f)", name, r[0], (double)hit/tot);
      static const int bad[2] = { 50, 24 }, low[1] = { K };
      CHECK(kci_vec_graph_recall(v, K, 300, bad, 2, 4, r) == -1 && kci_vec_graph_recall(v, K, 300, low, 1, 4, r) == -1
            && kci_vec_graph_recall(v, 65, 300, efs, 5, 4, r) == -1 && kci_vec_graph_recall(v, K, 300, efs, 5, 4, NULL) == -1
            && kci_vec_graph_recall(NULL, K, 300, efs, 5, 4, r) == -1, "%s: kci_vec_graph_recall refusals", name); }

    /* 3. the rule */
    uint64_t before[NQ][K], after[NQ][K];
    for(int q=0;q<NQ;q++) kci_vec_search_rerank(s, Q + (size_t)q*D, K, 200, 200, before[q], NULL);
    double r_first = recall(v, s, 32), r_wide = recall(v, s, 1600);
    float gain = -1; int on = kci_vec_judge(v, K, 0.99f, 0, &gain);
    CHECK(on == 0 || on == 1, "%s: kci_vec_judge returned %d", name, on);
    CHECK(kci_vec_judged(v) == on, "%s: kci_vec_judged %d != %d", name, kci_vec_judged(v), on);
    CHECK(gain > 0.3f && gain < 100.0f, "%s: gain %.3f", name, gain);
    CHECK(v->jg && v->jg->ef0 == 32 && v->jg->k == K && v->jg->wmax == 1600, "%s: the rule's widths", name);
    { int up = 1, zero = 0; for(int c=0;c<KV_JG_NC;c++){ int w = v->jg->width[c]; zero += w == 0;
          CHECK(w == 0 || (w > 32 && w <= 1600), "%s: class %d has width %d", name, c, w);
          if(c && w < v->jg->width[c-1] / 4) up = 0; }
      CHECK(up, "%s: the widths do not grow with the classes", name);
      printf("%s: rule %s, gain on its check half %.2fx, recall there %.4f, classes answered by the first pass %d of %d\n",
             name, on ? "in use" : "off", gain, v->jg->rec, zero, KV_JG_NC); }
    for(int q=0;q<NQ;q++) kci_vec_search_rerank(s, Q + (size_t)q*D, K, 200, 200, after[q], NULL);
    CHECK(memcmp(before, after, sizeof before) == 0, "%s: a search with an explicit ef changed after kci_vec_judge", name);
    v->jg->on = 1;
    double r_j = recall(v, s, 0);
    printf("%s: recall against the exact scan: width 32 %.4f, width 1600 %.4f, judged %.4f\n", name, r_first, r_wide, r_j);
    CHECK(r_j >= r_wide - 0.03, "%s: the judged search lost recall (%.4f against %.4f at 1600)", name, r_j, r_wide);
    CHECK(r_j > r_first + 0.01 || r_first > 0.97, "%s: the judged search is no better than its first pass (%.4f, %.4f)", name, r_j, r_first);
    /* a cosine index: the same query three times as long gets the same answer */
    if(!l2){ int diff = 0; uint64_t a[K], b[K]; float q3[D];
        for(int q=0;q<NQ;q++){ for(int d=0;d<D;d++) q3[d] = 3.0f*Q[(size_t)q*D+d];
            int n1 = kci_vec_search_rerank(s, Q + (size_t)q*D, K, 0, K, a, NULL), n2 = kci_vec_search_rerank(s, q3, K, 0, K, b, NULL);
            diff += n1 != n2 || memcmp(a, b, sizeof(uint64_t)*(size_t)n1) != 0; }
        CHECK(diff <= NQ/20, "%s: %d of %d queries answered differently when 3 times as long", name, diff, NQ);
        CHECK(v->jg->sd[5] > 1e29f, "%s: the query's length is used as a sign although the samples had one length", name); }
    /* k above the rule's k: not judged, the old meaning of ef <= 0 */
    { uint64_t a[20]; int n = kci_vec_search_rerank(s, Q, 20, 0, 20, a, NULL); CHECK(n == 20, "%s: k 20 with ef 0 gave %d", name, n); }
    /* the caller's queries */
    { float g2 = -1; int r = kci_vec_judge_queries(v, Q, NQ, K, 0.99f, &g2);
      CHECK(r == 0 || r == 1, "%s: kci_vec_judge_queries returned %d", name, r);
      CHECK(kci_vec_judge_queries(v, Q, 100, K, 0.99f, NULL) == -1 && kci_vec_judge_queries(v, NULL, NQ, K, 0.99f, NULL) == -1, "%s: too few / no queries -> -1", name); }
    CHECK(kci_vec_judge_off(v) == 0 && kci_vec_judged(v) == 0 && v->jg == NULL, "%s: kci_vec_judge_off", name);
    /* a build drops the rule */
    CHECK(kci_vec_judge(v, K, 0.99f, 400, NULL) >= 0 && v->jg != NULL, "%s: judge with 400 samples", name);
    kci_vec_searcher_free(s);
    CHECK(kci_vec_build(v, 8, NULL) == 0 && v->jg == NULL && kci_vec_judged(v) == 0, "%s: the rule survived a build", name);
    kci_vec_free(v); free(cen);
}

int main(void){
    X = malloc(sizeof(float)*(size_t)N*D); Q = malloc(sizeof(float)*(size_t)NQ*D);
    run(0); run(1);
    /* refused: an inner-product index, an index too small to judge */
    { KciVec *v = kci_vec_new_ip(D, 16, 100, 0.0f); kci_vec_keep_float(v, 1);
      for(int i=0;i<3000;i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i*D);
      CHECK(kci_vec_build(v, 4, NULL) == 0 && kci_vec_judge(v, K, 0.99f, 0, NULL) == -1, "inner product -> -1");
      kci_vec_free(v);
      v = kci_vec_new(D, 16, 100); kci_vec_keep_float(v, 1);
      for(int i=0;i<300;i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i*D);
      CHECK(kci_vec_build(v, 4, NULL) == 0 && kci_vec_judge(v, K, 0.99f, 0, NULL) == -1, "300 vectors -> -1");
      kci_vec_free(v);
      CHECK(kci_vec_judge(NULL, K, 0.99f, 0, NULL) == -1 && kci_vec_judged(NULL) == 0 && kci_vec_judge_off(NULL) == -1, "NULL"); }
    free(X); free(Q);
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails != 0;
}
