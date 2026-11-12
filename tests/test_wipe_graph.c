/*
 *   gcc -O1 -g -march=native -o test_wipe_graph tests/test_wipe_graph.c -Wl,--wrap=free -lpthread -lm && ./test_wipe_graph
 */
#include "../kci_vec.c"
#include <stdio.h>

void __real_free(void *p);
#define NW 64
static struct { void *p; size_t n; int seen, zero; const char *what; } g_w[NW];
static int g_nw = 0;

void __wrap_free(void *p) {
    for (int i = 0; p && i < g_nw; i++) if (g_w[i].p == p && !g_w[i].seen) {
        const unsigned char *b = p; int z = 1;
        for (size_t k = 0; k < g_w[i].n; k++) if (b[k]) { z = 0; break; }
        g_w[i].seen = 1; g_w[i].zero = z;
    }
    __real_free(p);
}

static void watch(void *p, size_t n, const char *what) {
    if (p && n && g_nw < NW) { g_w[g_nw].p = p; g_w[g_nw].n = n; g_w[g_nw].seen = 0; g_w[g_nw].zero = 0; g_w[g_nw].what = what; g_nw++; }
}

static int report(const char *stage) {
    int bad = 0;
    for (int i = 0; i < g_nw; i++) {
        if (!g_w[i].seen) { printf("FAIL %s: %s was not freed\n", stage, g_w[i].what); bad++; }
        else if (!g_w[i].zero) { printf("FAIL %s: %s freed with data in it\n", stage, g_w[i].what); bad++; }
    }
    if (!bad) printf("ok   %s: %d graph blocks, each wiped before free\n", stage, g_nw);
    g_nw = 0;
    return bad;
}

static void mkvec(uint64_t i, int dim, float *q) {
    uint64_t x = i * 2654435761u + 1;
    for (int d = 0; d < dim; d++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; q[d] = (float)((x % 2001) / 1000.0 - 1.0); }
}

static void watch_graph(KciVec *v, const char *stage) {
    (void)stage;
    watch(v->l0, (size_t)v->gcap * (v->M0 + 1) * sizeof(uint32_t), "l0 (neighbour lists)");
    watch(v->level, v->gcap, "level");
    watch(v->dead, v->gcap, "dead");
    watch(v->lock, v->gcap, "lock");
}

int main(void) {
    const int dim = 16;
    float q[16];
    int fails = 0;
    KciVec *v = kci_vec_new(dim, 8, 64);
    kci_vec_keep_float(v, 1);
    for (uint64_t i = 0; i < 300; i++) { mkvec(i, dim, q); kci_vec_add(v, i, q); }
    double s; kci_vec_build(v, 2, &s);
    uint32_t cap0 = v->gcap;
    watch_graph(v, "grow");
    watch(v->lu, (size_t)v->gcap * sizeof(uint32_t *), "lu (pointer array)");
    for (uint64_t i = 300; v->gcap == cap0 && i < 100000; i++) { mkvec(i, dim, q); kci_vec_insert(v, i, q); }
    printf("     graph capacity %u -> %u\n", cap0, v->gcap);
    fails += report("insert past the capacity (was realloc)");
    for (uint64_t i = 0; i < 200; i += 3) kci_vec_delete(v, i);
    watch_graph(v, "compact");
    watch(v->lu, (size_t)v->gcap * sizeof(uint32_t *), "lu (pointer array)");
    kci_vec_compact(v);
    fails += report("compact (the old graph order)");
    for (uint32_t i = 0; i < v->gN && g_nw < 8; i++)
        if (v->level[i] > 0 && v->lu[i]) watch(v->lu[i], (size_t)v->level[i] * (v->M + 1) * sizeof(uint32_t), "lu[i] (upper links)");
    uint64_t pb[6];
    kci_vec_plain_bytes(v, pb);
    int okb = pb[0] == (uint64_t)v->cap * dim && pb[1] == (uint64_t)v->cap * dim * 4 && pb[2] == 0 &&
              pb[3] >= (uint64_t)v->gcap * (v->M0 + 1) * 4 && pb[4] >= (uint64_t)v->cap * 8 &&
              pb[5] == pb[0] + pb[1] + pb[2] + pb[3] + pb[4];
    printf("%s plain bytes (graph index): codes %llu, floats %llu, derived %llu, graph %llu, ids %llu, total %llu\n",
           okb ? "ok  " : "FAIL", (unsigned long long)pb[0], (unsigned long long)pb[1], (unsigned long long)pb[2],
           (unsigned long long)pb[3], (unsigned long long)pb[4], (unsigned long long)pb[5]);
    fails += !okb;
    watch_graph(v, "free");
    kci_vec_free(v);
    fails += report("kci_vec_free");

    KciVec *w = kci_vec_new(dim, 16, 200);
    kci_vec_keep_float(w, 1);
    for (uint64_t i = 0; i < 2000; i++) { mkvec(i, dim, q); kci_vec_add(w, i, q); }
    if (kci_vec_build_ivf(w, 16, 2) != 0) { printf("FAIL build_ivf\n"); return 1; }
    struct KvIvf *f = w->ivf;
    kci_vec_plain_bytes(w, pb);
    uint64_t codes = 0;
    for (int c = 0; c < f->nlist; c++) codes += (uint64_t)f->lists[c].capb * f->bb;
    okb = pb[0] == (uint64_t)w->cap * dim + codes && pb[2] >= (uint64_t)f->nlist * dim * 4 && pb[3] <= w->gcap &&   /* tombstones only */
          pb[5] == pb[0] + pb[1] + pb[2] + pb[3] + pb[4];
    printf("%s plain bytes (lists index): codes %llu (4-bit %llu), floats %llu, derived %llu, ids %llu, total %llu\n",
           okb ? "ok  " : "FAIL", (unsigned long long)pb[0], (unsigned long long)codes, (unsigned long long)pb[1],
           (unsigned long long)pb[2], (unsigned long long)pb[4], (unsigned long long)pb[5]);
    fails += !okb;
    watch(f->cent, (size_t)f->nlist * dim * sizeof(float), "ivf centroids");
    watch(f->pq, (size_t)(f->M + 1) * 32 * sizeof(float), "ivf codebook");
    watch(f->c8, (size_t)(f->nlistp / 16) * f->D4 * 64, "ivf centroid table");
    kci_vec_free(w);
    fails += report("kci_vec_free (lists index)");
    printf(fails ? "%d FAILED\n" : "ALL PASS\n", fails);
    return fails != 0;
}
