/* kci_vec_calibrate: the search width chosen from the index's own vectors or from given questions, kept across save / load.
 *   gcc -O2 -march=native -I. tests/test_calibrate.c kci_vec.c -lpthread -lm -o test_calibrate && ./test_calibrate */
#include "kci_vec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if(!(c)){ fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while(0)

enum { N = 40000, DMAX = 128, K = 10, NQ = 100 };
static int D;
static float *X, Q[NQ][DMAX];
static unsigned rs = 97531;
static float rnd(void){ rs = rs*1664525u + 1013904223u; return (float)(rs>>8)/(float)(1u<<24) - 0.5f; }

static double recall_at(KciVec *v, int ef){
    KciVecSearcher *s = kci_vec_searcher_new(v); uint64_t a[K], b[K]; long hit = 0;
    for(int q=0;q<NQ;q++){
        int n = kci_vec_search_rerank(s, Q[q], K, ef, 200, a, NULL), m = kci_vec_search_exact(v, Q[q], K, 4, b, NULL);
        for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(a[i]==b[j]){ hit++; break; }
    }
    kci_vec_searcher_free(s); return (double)hit/(NQ*K);
}

static void run(int l2, int hard){
    char name[32]; snprintf(name, sizeof name, "%s/%s", l2 ? "l2" : "cosine", hard ? "hard" : "easy");
    /* independent noise in every dim: 16 dims are easy for a graph, 128 dims are hard */
    D = hard ? 128 : 16;
    for(size_t i=0;i<(size_t)N*D;i++) X[i] = rnd();
    for(int q=0;q<NQ;q++) for(int d=0;d<D;d++) Q[q][d] = rnd();
    KciVec *v = l2 ? kci_vec_new_l2(D, 16, 100, 0.0f) : kci_vec_new(D, 16, 100);
    kci_vec_keep_float(v, 1);
    for(int i=0;i<N;i++) kci_vec_add(v, (uint64_t)i + 1000, X + (size_t)i*D);
    CHECK(kci_vec_default_ef(v) == 0, "%s: a new index has no default width", name);
    CHECK(kci_vec_calibrate(v, 0, 200, 0, 0, 0, NULL) == -1, "%s: not built -> -1", name);
    CHECK(kci_vec_build(v, 8, NULL) == 0, "%s: build", name);
    float rec = -1; int ef = kci_vec_calibrate(v, K, 200, 0.98f, 128, 0, &rec);
    CHECK(ef == 200 || ef == 400 || ef == 800 || ef == 1600, "%s: ef %d is not one of the four widths", name, ef);
    CHECK(kci_vec_default_ef(v) == ef, "%s: default_ef %d != %d", name, kci_vec_default_ef(v), ef);
    CHECK(kci_vec_default_rerank(v) == 0 || kci_vec_default_rerank(v) == ef, "%s: default_rerank %d", name, kci_vec_default_rerank(v));
    CHECK(rec >= 0.98f || ef == 1600, "%s: recall %.4f below the target at ef %d", name, rec, ef);
    double r200 = recall_at(v, 200), rpick = recall_at(v, ef);
    printf("%s: calibrated ef %d (sample recall %.4f); other questions: recall %.4f at ef 200, %.4f at the calibrated width\n", name, ef, rec, r200, rpick);
    if(hard){ CHECK(ef > 200, "%s: the hard data kept ef 200", name);
              CHECK(rpick > r200 + 0.02, "%s: the calibrated width did not raise the recall (%.4f -> %.4f)", name, r200, rpick);
              CHECK(rpick >= 0.95, "%s: recall %.4f on other questions at the calibrated width", name, rpick); }
    else      CHECK(ef == 200 && rpick >= 0.98, "%s: the easy data needed ef %d, recall %.4f", name, ef, rpick);
    /* the same call gives the same answer (the sample is fixed) */
    CHECK(kci_vec_calibrate(v, K, 200, 0.98f, 128, 0, NULL) == ef, "%s: a second calibration differs", name);
    for(int enc=0; enc<2; enc++){
        const char *p = "t_calibrate.kvec";
        CHECK(kci_vec_save(v, p, enc ? (const uint8_t*)"key" : NULL, enc ? 3 : 0) == 0, "%s: save", name);
        KciVec *w = kci_vec_load(p, enc ? (const uint8_t*)"key" : NULL, enc ? 3 : 0);
        CHECK(w != NULL, "%s: load (enc %d)", name, enc);
        if(w){ CHECK(kci_vec_default_ef(w) == ef, "%s: loaded default_ef %d != %d (enc %d)", name, kci_vec_default_ef(w), ef, enc);
               double rl = recall_at(w, ef); CHECK(rl > rpick - 0.01, "%s: loaded recall %.4f against %.4f", name, rl, rpick);
               kci_vec_free(w); }
        remove(p);
    }
    CHECK(kci_vec_set_default_ef(v, 0) == 0 && kci_vec_default_ef(v) == 0 && kci_vec_set_default_ef(v, 70000) == -1, "%s: set_default_ef", name);
    kci_vec_free(v);
}

static void groups(void){
    enum { GD = 48, NG = 60, PER = 400, GN = NG*PER };
    float *G = malloc(sizeof(float)*(size_t)GN*GD), (*C)[GD] = malloc(sizeof(float)*NG*GD), q[GD]; uint64_t a[K], b[K];
    for(int c=0;c<NG;c++) for(int d=0;d<GD;d++) C[c][d] = rnd();
    for(int i=0;i<GN;i++) for(int d=0;d<GD;d++) G[(size_t)i*GD+d] = C[i%NG][d] + 0.02f*rnd();
    for(int l2=0;l2<2;l2++){
        KciVec *v = l2 ? kci_vec_new_l2(GD, 16, 200, 0.0f) : kci_vec_new(GD, 16, 200);
        kci_vec_keep_float(v, 1);
        for(int i=0;i<GN;i++) kci_vec_add(v, (uint64_t)i, G + (size_t)i*GD);
        kci_vec_build(v, 8, NULL);
        float rec = 0; int ef = kci_vec_calibrate(v, K, 200, 0.98f, 128, 0, &rec), rr = kci_vec_default_rerank(v);
        KciVecSearcher *s = kci_vec_searcher_new(v); long h0 = 0, h1 = 0;
        for(int t=0;t<NQ;t++){
            for(int d=0;d<GD;d++) q[d] = C[t*7%NG][d] + 0.02f*rnd();
            int m = kci_vec_search_exact(v, q, K, 4, b, NULL), n = kci_vec_search_rerank(s, q, K, ef, 200, a, NULL);
            for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(a[i]==b[j]){ h0++; break; }
            n = kci_vec_search_rerank(s, q, K, ef, rr > 200 ? rr : 200, a, NULL);
            for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(a[i]==b[j]){ h1++; break; }
        }
        double r0 = (double)h0/(NQ*K), r1 = (double)h1/(NQ*K);
        printf("%s/groups of %d: calibrated ef %d, rerank %d (sample recall %.4f); other questions: %.4f with rerank 200, %.4f with the calibrated one\n",
               l2 ? "l2" : "cosine", PER, ef, rr, rec, r0, r1);
        CHECK(rr == ef && ef >= 400, "groups: the float pass was not widened (ef %d, rerank %d)", ef, rr);
        CHECK(r1 >= 0.95 && r1 > r0 + 0.1, "groups: recall %.4f -> %.4f", r0, r1);
        CHECK(kci_vec_save(v, "t_calibrate.kvec", NULL, 0) == 0, "groups: save");
        KciVec *w = kci_vec_load("t_calibrate.kvec", NULL, 0);
        CHECK(w && kci_vec_default_ef(w) == ef && kci_vec_default_rerank(w) == rr, "groups: load keeps ef and rerank");
        if(w) kci_vec_free(w);
        remove("t_calibrate.kvec");
        CHECK(kci_vec_set_default_ef(v, 300) == 0 && kci_vec_default_rerank(v) == 0, "groups: set_default_ef forgets the rerank");
        kci_vec_searcher_free(s); kci_vec_free(v);
    }
    free(G); free(C);
}

static int only3(uint64_t id, void *ctx){ const uint64_t *a = ctx; return id==a[0] || id==a[1] || id==a[2]; }
static void more(void){
    enum { MD = 64, MN = 30000 };
    float *Y = malloc(sizeof(float)*(size_t)MN*MD), (*P)[MD] = malloc(sizeof(float)*NQ*MD); uint64_t a[K], b[K];
    for(size_t i=0;i<(size_t)MN*MD;i++) Y[i] = rnd() + 0.4f;                /* off the origin: the questions below are not */
    for(int q=0;q<NQ;q++) for(int d=0;d<MD;d++) P[q][d] = rnd();
    for(int ip=0; ip<2; ip++){
        const char *name = ip ? "ip" : "cosine";
        KciVec *v = ip ? kci_vec_new_ip(MD, 16, 100, 0.0f) : kci_vec_new(MD, 16, 100);
        kci_vec_keep_float(v, 1);
        for(int i=0;i<MN;i++) kci_vec_add(v, (uint64_t)i, Y + (size_t)i*MD);
        CHECK(kci_vec_build(v, 8, NULL) == 0, "%s: build", name);
        CHECK(kci_vec_unreached(v) == 0, "%s: %u rows unreached", name, kci_vec_unreached(v));
        KciVecSearcher *s = kci_vec_searcher_new(v); long h0 = 0, h1 = 0, hz = 0;
        for(int q=0;q<NQ;q++){ int m = kci_vec_search_exact(v, P[q], K, 4, b, NULL), n = kci_vec_search_rerank(s, P[q], K, 0, 200, a, NULL);
            for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(a[i]==b[j]){ h0++; break; } }
        float rs_ = -1, rq = -1; int es = kci_vec_calibrate(v, K, 200, 0.98f, 128, 0, &rs_);
        int eq = kci_vec_calibrate_queries(v, &P[0][0], NQ, K, 200, 0.98f, 0, &rq);
        CHECK(es >= 200 && eq >= 200 && kci_vec_default_ef(v) == eq, "%s: calibrate %d, with questions %d", name, es, eq);
        for(int q=0;q<NQ;q++){ int m = kci_vec_search_exact(v, P[q], K, 4, b, NULL);
            int n = kci_vec_search_rerank(s, P[q], K, eq, kci_vec_default_rerank(v) > 200 ? kci_vec_default_rerank(v) : 200, a, NULL);
            for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(a[i]==b[j]){ h1++; break; }
            n = kci_vec_search_rerank(s, P[q], K, 0, 200, a, NULL);            /* ef 0: the same width, taken from the index */
            for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(a[i]==b[j]){ hz++; break; } }
        printf("%s/own questions: stored vectors as questions -> ef %d (%.4f); the questions themselves -> ef %d, rerank %d (%.4f); recall %.4f uncalibrated, %.4f calibrated, %.4f with ef 0\n",
               name, es, rs_, eq, kci_vec_default_rerank(v), rq, (double)h0/(NQ*K), (double)h1/(NQ*K), (double)hz/(NQ*K));
        CHECK(h1 >= h0 && hz == h1, "%s: ef 0 does not use the calibrated width (%ld, %ld, %ld)", name, h0, h1, hz);
        CHECK((double)h1/(NQ*K) >= rq - 0.02, "%s: recall %.4f against the measured %.4f", name, (double)h1/(NQ*K), rq);
        CHECK(kci_vec_calibrate_queries(v, &P[0][0], 3, K, 200, 0.98f, 0, NULL) == -1 && kci_vec_calibrate_queries(v, NULL, NQ, K, 200, 0.98f, 0, NULL) == -1, "%s: too few / no questions", name);
        /* a filter passing 3 ids returns the 3, best first, on inner product too */
        uint64_t three[3] = { 11, 15000, 29999 }; int bad = 0;
        for(int q=0;q<20;q++){ uint64_t g[K], e[K];
            int n = kci_vec_search_filtered(s, P[q], K, 0, 200, only3, three, g, NULL), m = kci_vec_search_exact_filtered(v, P[q], K, 1, only3, three, e, NULL);
            bad += !(n == 3 && m == 3 && g[0]==e[0] && g[1]==e[1] && g[2]==e[2]); }
        CHECK(bad == 0, "%s: a filter passing 3 ids: %d of 20 questions wrong", name, bad);
        kci_vec_searcher_free(s); kci_vec_free(v);
    }
    CHECK(kci_vec_unreached(NULL) == 0, "unreached(NULL)");
    free(Y); free(P);
}

int main(void){
    X = malloc(sizeof(float)*(size_t)N*DMAX); if(!X) return 2;
    run(0, 0); run(1, 0);
    run(0, 1); run(1, 1);
    /* no floats: refused; an index that was never calibrated saves and loads with 0 */
    KciVec *v = kci_vec_new(D, 16, 100);
    for(int i=0;i<2000;i++) kci_vec_add(v, (uint64_t)i, X + (size_t)i*D);
    kci_vec_build(v, 4, NULL);
    CHECK(kci_vec_calibrate(v, K, 200, 0.98f, 64, 0, NULL) == -1, "no floats -> -1");
    CHECK(kci_vec_save(v, "t_calibrate.kvec", NULL, 0) == 0, "save");
    KciVec *w = kci_vec_load("t_calibrate.kvec", NULL, 0);
    CHECK(w && kci_vec_default_ef(w) == 0, "an uncalibrated index loads with default_ef 0");
    if(w) kci_vec_free(w);
    remove("t_calibrate.kvec"); kci_vec_free(v);
    CHECK(kci_vec_calibrate(NULL, K, 200, 0.98f, 64, 0, NULL) == -1 && kci_vec_default_ef(NULL) == 0 && kci_vec_default_rerank(NULL) == 0, "NULL");
    groups();
    more();
    free(X);
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails != 0;
}
