/*
 test_dedup.c — kci_vec_set_dedup: identical vectors are kept once, their ids beside the row .
 *
 *   gcc -O2 -march=native -I. tests/test_dedup.c kci_vec.c -lpthread -lm -o test_dedup     (from the repo root)
 */
#include "kci_vec.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DIM 48
#define ND  12000          /* distinct */
#define NQ  300
static int fails = 0;
#define CHECK(c, ...) do{ if(!(c)){ fails++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } }while(0)

static uint64_t rs = 0x9E3779B97F4A7C15ULL;
static float rnd(void){ float a = 0; for (int i = 0; i < 4; i++){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; a += (float)(rs % 20001) / 10000.0f - 1.0f; } return a * 0.866f; }
static float centre[DIM];
static void mk(float *v, float spread){ float n = 0; for (int i = 0; i < DIM; i++){ v[i] = centre[i] + spread * rnd(); n += v[i] * v[i]; } n = 1.0f / sqrtf(n); for (int i = 0; i < DIM; i++) v[i] *= n; }
static float dot(const float *a, const float *b){ float s = 0; for (int i = 0; i < DIM; i++) s += a[i] * b[i]; return s; }
static int cmp_desc(const void *a, const void *b){ float x = *(const float*)a, y = *(const float*)b; return x < y ? 1 : x > y ? -1 : 0; }

static const int GS[] = { 700, 300, 120, 64, 40, 33, 20, 9, 5, 3, 2, 2 };   /* 1,298 rows; the rest in groups of 2 .. 7 */
static int even_only(uint64_t id, void *ctx){ (void)ctx; return (id & 1) == 0; }
static int not_id(uint64_t id, void *ctx){ return id != *(uint64_t*)ctx; }

static double recall(KciVecSearcher *s, const float *base, int total, const float *q, int ef, int rerank, float *sc){
    uint64_t ids[10]; float out[10]; double sum = 0;
    for (int qi = 0; qi < NQ; qi++){ const float *qq = q + (size_t)qi * DIM; int hit = 0;
        for (int i = 0; i < total; i++) sc[i] = dot(qq, base + (size_t)i * DIM);
        int n = rerank ? kci_vec_search_rerank(s, qq, 10, ef, rerank, ids, out) : kci_vec_search_r(s, qq, 10, ef, ids, out);
        float got[10]; for (int i = 0; i < n; i++) got[i] = ids[i] <= (uint64_t)total ? sc[ids[i] - 1] : -2.0f;   /* ids inserted later: not rows of base */
        qsort(sc, (size_t)total, sizeof *sc, cmp_desc);
        for (int i = 0; i < n; i++) hit += got[i] >= sc[9] - 2e-3f;       /* int8 scores in kci_vec_search_r: a little slack */
        sum += hit / 10.0; }
    return sum / NQ;
}

int main(void){
    int ngroups = 0, dups = 0, total;
    int *gsize = malloc(sizeof(int) * 4000);
    for (size_t i = 0; i < sizeof GS / sizeof GS[0]; i++){ gsize[ngroups++] = GS[i]; dups += GS[i]; }
    while (dups < 2500){ int g = 2 + (int)(rs % 6); rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; gsize[ngroups++] = g; dups += g; }
    total = ND + dups;
    float *base = malloc((size_t)total * DIM * sizeof *base), *q = malloc((size_t)NQ * DIM * sizeof *q), *sc = malloc((size_t)(total + 16) * sizeof *sc);
    float *gvec = malloc((size_t)ngroups * DIM * sizeof *gvec); int *grp = malloc(sizeof(int) * (size_t)total);   /* grp[row] = its group, or -1 */
    { float n = 0; for (int i = 0; i < DIM; i++){ centre[i] = rnd(); n += centre[i] * centre[i]; } n = 1.0f / sqrtf(n); for (int i = 0; i < DIM; i++) centre[i] *= n; }
    for (int g = 0; g < ngroups; g++) mk(gvec + (size_t)g * DIM, 0.15f);
    /* which rows are copies: spread over the order of insertion */
    { int *left = malloc(sizeof(int) * (size_t)ngroups), remaining = dups, g = 0; memcpy(left, gsize, sizeof(int) * (size_t)ngroups);
      for (int i = 0; i < total; i++){ grp[i] = -1;
          if (remaining > 0 && (long long)i * dups / total >= dups - remaining){ while (left[g] == 0) g = (g + 1) % ngroups; grp[i] = g; left[g]--; remaining--; g = (g + 1) % ngroups; } }
      for (int i = total - 1; remaining > 0 && i >= 0; i--) if (grp[i] < 0){ while (left[g] == 0) g = (g + 1) % ngroups; grp[i] = g; left[g]--; remaining--; }
      free(left); }
    for (int i = 0; i < total; i++){ if (grp[i] >= 0) memcpy(base + (size_t)i * DIM, gvec + (size_t)grp[i] * DIM, sizeof(float) * DIM); else mk(base + (size_t)i * DIM, 0.15f); }
    for (int i = 0; i < NQ; i++) mk(q + (size_t)i * DIM, 0.15f);

    KciVec *off = kci_vec_new(DIM, 16, 200), *on = kci_vec_new(DIM, 16, 200);
    kci_vec_keep_float(off, 1); kci_vec_keep_float(on, 1);
    CHECK(kci_vec_set_dedup(on, 1) == 0 && kci_vec_set_dedup(NULL, 1) == -1, "set_dedup");
    for (int i = 0; i < total; i++){ kci_vec_add(off, (uint64_t)i + 1, base + (size_t)i * DIM); kci_vec_add(on, (uint64_t)i + 1, base + (size_t)i * DIM); }
    CHECK(kci_vec_build(off, 4, NULL) == 0 && kci_vec_build(on, 4, NULL) == 0, "build");
    uint64_t st[8], so[8];
    CHECK(kci_vec_dup_stats(on, st) == 0 && kci_vec_dup_stats(off, so) == 0, "dup_stats");
    printf("without dedup: rows %llu, groups %llu, rows in groups %llu, largest %llu | with: rows %llu, groups %llu, ids beside rows %llu, rows with company %llu\n",
           (unsigned long long)so[0], (unsigned long long)so[1], (unsigned long long)so[2], (unsigned long long)so[3],
           (unsigned long long)st[0], (unsigned long long)st[1], (unsigned long long)st[6], (unsigned long long)st[7]);
    CHECK(so[0] == (uint64_t)total && so[1] == (uint64_t)ngroups && so[2] == (uint64_t)dups && so[3] == 700 && so[6] == 0, "the index without dedup holds every row");
    CHECK(st[0] == (uint64_t)(ND + ngroups) && st[1] == 0 && st[6] == (uint64_t)(dups - ngroups) && st[7] == (uint64_t)ngroups, "with dedup: one row per distinct vector, the other ids beside");
    CHECK(kci_vec_count(on) == (uint32_t)total && kci_vec_count(off) == (uint32_t)total, "count counts ids (%u, %u)", kci_vec_count(on), kci_vec_count(off));

    KciVecSearcher *s0 = kci_vec_searcher_new(off), *s1 = kci_vec_searcher_new(on);
    for (int ef = 20; ef <= 200; ef *= 10){
        double r0 = recall(s0, base, total, q, ef, 0, sc), r1 = recall(s1, base, total, q, ef, 0, sc);
        double f0 = recall(s0, base, total, q, ef, ef, sc), f1 = recall(s1, base, total, q, ef, ef, sc);
        printf("ef %3d: recall@10 int8 %.4f -> %.4f with dedup; float re-rank %.4f -> %.4f\n", ef, r0, r1, f0, f1);
        CHECK(r1 >= r0 - 0.01 && f1 >= f0 - 0.01, "dedup does not lose recall at ef %d", ef);
        if (ef == 200) CHECK(f1 >= 0.97, "re-ranked recall at ef 200 (%.4f)", f1); }

    /* exact search, and the lists: ids as without dedup */
    { uint64_t a[10], b[10]; float sa[10], sb[10]; int same = 1;
      for (int qi = 0; qi < 50; qi++){ const float *qq = q + (size_t)qi * DIM;
          int n0 = kci_vec_search_exact(off, qq, 10, 2, a, sa), n1 = kci_vec_search_exact(on, qq, 10, 2, b, sb);
          if (n0 != 10 || n1 != 10) same = 0;
          for (int i = 0; i < 10 && same; i++) if (fabsf(sa[i] - sb[i]) > 1e-5f) same = 0; }
      CHECK(same, "exact search: the same 10 scores with and without dedup");
      CHECK(kci_vec_build_ivf(on, 32, 2) == 0, "lists on the deduped rows");
      kci_vec_searcher_free(s1); s1 = kci_vec_searcher_new(on);
      float *g0 = gvec; int n = kci_vec_search_ivf(s1, g0, 10, 32, 500, b, sb), in = 0;
      for (int i = 0; i < n; i++) in += grp[b[i] - 1] == 0;
      CHECK(n == 10 && in == 10, "lists: the largest group's vector returns 10 ids of the group (%d of %d)", in, n); }

    /* a group's vector as the query; unique */
    { uint64_t ids[10]; float out[10];
      for (int g = 0; g < 12; g++){
          int n = kci_vec_search_rerank(s1, gvec + (size_t)g * DIM, 10, 100, 100, ids, out), in = 0, want = gsize[g] < 10 ? gsize[g] : 10;
          for (int i = 0; i < n; i++) in += grp[ids[i] - 1] == g && out[i] > 0.9999f;
          CHECK(n == 10 && in == want, "group %d (size %d) as the query: %d of its ids first (got %d)", g, gsize[g], want, in);
          for (int i = 1; i < n; i++) CHECK(out[i - 1] >= out[i] - 1e-6f, "scores in order"); }
      CHECK(kci_vec_searcher_set_unique(s1, 1) == 0, "set_unique");
      int n = kci_vec_search_rerank(s1, gvec, 10, 100, 100, ids, out), in = 0, distinct = 1;
      for (int i = 0; i < n; i++){ in += grp[ids[i] - 1] == 0; for (int j = 0; j < i; j++) if (dot(base + (size_t)(ids[i] - 1) * DIM, base + (size_t)(ids[j] - 1) * DIM) > 0.99999f) distinct = 0; }
      CHECK(n == 10 && in == 1 && distinct, "unique: one id of the group, 10 different vectors (%d of the group)", in);
      kci_vec_searcher_set_unique(s1, 0); }

    /* a filter sees every id */
    { uint64_t ids[10]; float out[10];
      int n = kci_vec_search_filtered(s1, gvec, 10, 200, 200, even_only, NULL, ids, out), ok = n == 10, in = 0;
      for (int i = 0; i < n; i++){ ok &= (ids[i] & 1) == 0; in += grp[ids[i] - 1] == 0; }
      CHECK(ok && in == 10, "filter (even ids): 10 even ids of the largest group (%d of %d)", in, n);
      /* group 10 has 2 ids: filter out the one that owns the row; the other must still come first */
      uint64_t a = 0, b = 0; for (int i = 0; i < total; i++) if (grp[i] == 10){ if (!a) a = (uint64_t)i + 1; else b = (uint64_t)i + 1; }
      n = kci_vec_search_filtered(s1, gvec + (size_t)10 * DIM, 3, 200, 200, not_id, &a, ids, out);
      CHECK(n == 3 && ids[0] == b, "the row's own id filtered out: the id beside it is returned (%llu, want %llu)", (unsigned long long)ids[0], (unsigned long long)b);
      n = kci_vec_search_filtered(s1, gvec + (size_t)10 * DIM, 3, 200, 200, not_id, &b, ids, out);
      CHECK(n == 3 && ids[0] == a && ids[1] != b && ids[2] != b, "the id beside the row filtered out: only the row's own id");
      kci_vec_searcher_set_unique(s1, 1);
      n = kci_vec_search_filtered(s1, gvec + (size_t)10 * DIM, 3, 200, 200, not_id, &a, ids, out);
      CHECK(n == 3 && ids[0] == b, "unique + filter: the one id of the group that passes");
      kci_vec_searcher_set_unique(s1, 0); }

    /* insert */
    { uint32_t rows = (uint32_t)(ND + ngroups); uint64_t ids[10]; float out[10], x[DIM];
      CHECK(kci_vec_insert(on, 900001, gvec + (size_t)3 * DIM) > 0, "insert an identical vector");
      kci_vec_dup_stats(on, st);
      CHECK(st[0] == rows && st[6] == (uint64_t)(dups - ngroups + 1) && kci_vec_count(on) == (uint32_t)total + 1, "it joined its row: no new row (%llu rows)", (unsigned long long)st[0]);
      mk(x, 0.15f);
      CHECK(kci_vec_insert(on, 900002, x) > 0, "insert a new vector");
      CHECK(kci_vec_insert(on, 900003, x) > 0, "and the same one again");
      kci_vec_dup_stats(on, st);
      CHECK(st[0] == rows + 1 && st[6] == (uint64_t)(dups - ngroups + 2) && st[7] == (uint64_t)ngroups + 1, "one new row, the second id beside it");
      kci_vec_searcher_free(s1); s1 = kci_vec_searcher_new(on);
      int n = kci_vec_search_rerank(s1, x, 3, 100, 100, ids, out);
      CHECK(n == 3 && ((ids[0] == 900002 && ids[1] == 900003) || (ids[0] == 900003 && ids[1] == 900002)) && out[1] > 0.9999f, "both are found by their vector");
      /* 900001 gets a vector of its own: it leaves group 3 */
      mk(x, 0.15f);
      CHECK(kci_vec_insert(on, 900001, x) > 0, "insert an id that shared a row, with another vector");
      n = kci_vec_search_rerank(s1, x, 1, 100, 100, ids, out);
      CHECK(n == 1 && ids[0] == 900001, "it is found by the new vector");
      int seen = 0; uint64_t big[80]; float bo[80];
      n = kci_vec_search_rerank(s1, gvec + (size_t)3 * DIM, 80, 200, 200, big, bo);
      for (int i = 0; i < n; i++) seen += big[i] == 900001 && bo[i] > 0.9999f;
      CHECK(seen == 0, "and no longer by the group's vector"); }

    /* delete */
    { uint64_t ids[40]; float out[40]; uint64_t m[64]; int nm = 0;
      for (int i = 0; i < total; i++) if (grp[i] == 6) m[nm++] = (uint64_t)i + 1;                 /* group 6: 20 ids */
      int n = kci_vec_search_rerank(s1, gvec + (size_t)6 * DIM, 40, 200, 200, ids, out), in = 0;
      for (int i = 0; i < n; i++) in += out[i] > 0.9999f;
      CHECK(nm == 20 && in == 20, "group 6: all 20 ids (%d)", in);
      uint64_t owner = ids[0];                                                                   /* the row's own id comes first */
      uint64_t other = m[0] == owner ? m[1] : m[0];
      CHECK(kci_vec_delete(on, other) == 0 && kci_vec_delete(on, other) == -1, "delete an id beside the row, once");
      CHECK(kci_vec_delete(on, owner) == 0, "delete the row's own id");
      n = kci_vec_search_rerank(s1, gvec + (size_t)6 * DIM, 40, 200, 200, ids, out); in = 0;
      int bad = 0; for (int i = 0; i < n; i++){ in += out[i] > 0.9999f; bad += ids[i] == owner || ids[i] == other; }
      CHECK(in == 18 && bad == 0, "18 ids left, the two deleted ones gone (%d, %d)", in, bad);
      for (int i = 0; i < nm; i++) if (m[i] != owner && m[i] != other) CHECK(kci_vec_delete(on, m[i]) == 0, "delete the rest");
      n = kci_vec_search_rerank(s1, gvec + (size_t)6 * DIM, 10, 200, 200, ids, out); in = 0;
      for (int i = 0; i < n; i++) in += out[i] > 0.9999f;
      CHECK(in == 0, "the whole group deleted: its vector finds none of it (%d)", in);
      kci_vec_dup_stats(on, st);
      CHECK(kci_vec_count(on) == (uint32_t)total + 3 - 19 - 0 && st[7] == (uint64_t)ngroups + 1 - 1, "counts after the deletes (%u ids, %llu rows with company)", kci_vec_count(on), (unsigned long long)st[7]); }

    { const char *paths[2] = { "test_dedup.kv", "test_dedup_enc.kv" }; const uint8_t key[] = "a passphrase for the test";
      CHECK(kci_vec_image_save(on, "test_dedup.kvimg") != 0, "image_save still refuses an index with ids beside rows");
      for (int enc = 0; enc < 2; enc++){
          CHECK(kci_vec_save(on, paths[enc], enc ? key : NULL, enc ? (int)sizeof key - 1 : 0) == 0, "save (enc=%d)", enc);
          FILE *fp = fopen(paths[enc], "rb"); char m[8] = {0}; if (fp){ if (fread(m, 8, 1, fp) != 1) m[0] = 0; fclose(fp); }
          CHECK(memcmp(m, enc ? "KCIVTW05" : "KCIVTW04", 8) == 0, "the file says it holds such ids (%.8s)", m);
          KciVec *ld = kci_vec_load(paths[enc], enc ? key : NULL, enc ? (int)sizeof key - 1 : 0);
          CHECK(ld != NULL, "load (enc=%d)", enc);
          if (!ld) continue;
          uint64_t a8[8], b8[8]; kci_vec_dup_stats(on, a8); kci_vec_dup_stats(ld, b8);
          CHECK(memcmp(a8, b8, sizeof a8) == 0 && kci_vec_count(ld) == kci_vec_count(on), "the loaded index holds the same rows and ids (%u, %llu beside rows)", kci_vec_count(ld), (unsigned long long)b8[6]);
          KciVecSearcher *sl = kci_vec_searcher_new(ld); int same = 1;
          for (int g = 0; g < 12 && same; g++){ uint64_t x[40], y[40]; float sx[40], sy[40];
              int n0 = kci_vec_search_rerank(s1, gvec + (size_t)g * DIM, 40, 200, 200, x, sx), n1 = kci_vec_search_rerank(sl, gvec + (size_t)g * DIM, 40, 200, 200, y, sy);
              if (n0 != n1) same = 0; for (int i = 0; i < n0 && same; i++) if (x[i] != y[i] || sx[i] != sy[i]) same = 0; }
          for (int qi = 0; qi < 100 && same; qi++){ uint64_t x[10], y[10]; float sx[10], sy[10];
              int n0 = kci_vec_search_rerank(s1, q + (size_t)qi * DIM, 10, 100, 100, x, sx), n1 = kci_vec_search_rerank(sl, q + (size_t)qi * DIM, 10, 100, 100, y, sy);
              if (n0 != n1) same = 0; for (int i = 0; i < n0 && same; i++) if (x[i] != y[i] || sx[i] != sy[i]) same = 0; }
          CHECK(same, "the loaded index answers the same, ids and scores (enc=%d)", enc);
          /* it is a dedup index still: an identical vector joins, the row's own id can go and another keeps the row */
          uint64_t before = b8[0];
          CHECK(kci_vec_insert(ld, 910001, gvec + (size_t)1 * DIM) > 0, "insert into the loaded index");
          kci_vec_dup_stats(ld, b8);
          CHECK(b8[0] == before && b8[6] == a8[6] + 1, "the identical vector joined its row after the load");
          uint64_t x[400]; float sx[400]; int n = kci_vec_search_rerank(sl, gvec + (size_t)1 * DIM, 400, 400, 400, x, sx), in = 0;
          for (int i = 0; i < n; i++) in += sx[i] > 0.9999f;
          CHECK(in == 301 && kci_vec_delete(ld, x[0]) == 0, "group 1 has 301 ids now; delete the first");
          n = kci_vec_search_rerank(sl, gvec + (size_t)1 * DIM, 400, 400, 400, x, sx); in = 0;
          for (int i = 0; i < n; i++) in += sx[i] > 0.9999f;
          CHECK(in == 300, "300 left (%d)", in);
          kci_vec_searcher_free(sl); kci_vec_free(ld);
          if (enc){ const uint8_t bad[] = "another passphrase";
              CHECK(kci_vec_load(paths[enc], bad, (int)sizeof bad - 1) == NULL && kci_vec_load(paths[enc], NULL, 0) == NULL, "wrong key / no key refused");
              /* one byte of the last 100 changed (inside the ids): the tag catches it */
              fp = fopen(paths[enc], "rb+"); if (fp){ fseek(fp, -60, SEEK_END); int c = fgetc(fp); fseek(fp, -60, SEEK_END); fputc(c ^ 1, fp); fclose(fp); }
              CHECK(kci_vec_load(paths[enc], key, (int)sizeof key - 1) == NULL, "a changed byte among the ids is refused (encrypted file)"); }
          else { /* cut short inside the ids: refused, not loaded with fewer */
              fp = fopen(paths[enc], "rb"); fseek(fp, 0, SEEK_END); long sz = ftell(fp); char *buf = malloc((size_t)sz); rewind(fp); if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) sz = 0; fclose(fp);
              fp = fopen("test_dedup_cut.kv", "wb"); fwrite(buf, 1, (size_t)(sz > 100 ? sz - 100 : 0), fp); fclose(fp); free(buf);
              CHECK(kci_vec_load("test_dedup_cut.kv", NULL, 0) == NULL, "a file cut short inside the ids is refused");
              remove("test_dedup_cut.kv"); } }
      /* without dedup: the file is as it always was, and a dedup index with no duplicates keeps its setting */
      CHECK(kci_vec_save(off, "test_dedup_off.kv", NULL, 0) == 0, "an index without dedup saves");
      { FILE *fp = fopen("test_dedup_off.kv", "rb"); char m[8] = {0}; if (fp){ if (fread(m, 8, 1, fp) != 1) m[0] = 0; fclose(fp); }
        CHECK(memcmp(m, "KCIVEC04", 8) == 0, "under the old magic (%.8s)", m); }
      { KciVec *e = kci_vec_new(DIM, 16, 200); kci_vec_keep_float(e, 1); kci_vec_set_dedup(e, 1);
        for (int i = 0; i < 500; i++) kci_vec_add(e, (uint64_t)i + 1, base + (size_t)i * DIM * 0 + (size_t)i * DIM);
        int nd = 0; for (int i = 0; i < 500; i++) nd += grp[i] >= 0;
        kci_vec_build(e, 1, NULL);
        CHECK(kci_vec_save(e, "test_dedup_e.kv", NULL, 0) == 0, "a small dedup index saves");
        KciVec *l = kci_vec_load("test_dedup_e.kv", NULL, 0); float x[DIM]; mk(x, 0.15f);
        CHECK(l && kci_vec_insert(l, 7001, x) > 0 && kci_vec_insert(l, 7002, x) > 0, "load, insert the same vector twice");
        uint64_t s8[8]; if (l){ kci_vec_dup_stats(l, s8); CHECK(s8[1] == 0, "no two rows alike after the load: dedup stayed on (%llu groups among rows)", (unsigned long long)s8[1]); kci_vec_free(l); }
        (void)nd; kci_vec_free(e); remove("test_dedup_e.kv"); }
      remove("test_dedup_off.kv"); remove("test_dedup.kvimg"); remove(paths[0]); remove(paths[1]); }

    /* compact: the groups stay */
    { uint64_t ids[10]; float out[10]; uint32_t before = kci_vec_count(on);
      for (int i = 0; i < 200; i++) if (grp[i * 31 % total] < 0) kci_vec_delete(on, (uint64_t)(i * 31 % total) + 1);
      uint32_t expect = kci_vec_count(on) ; (void)before;
      kci_vec_dup_stats(on, st); uint64_t beside = st[6];
      uint32_t c = kci_vec_compact(on);
      kci_vec_dup_stats(on, st);
      kci_vec_searcher_free(s1); s1 = kci_vec_searcher_new(on);
      int n = kci_vec_search_rerank(s1, gvec, 10, 100, 100, ids, out), in = 0;
      for (int i = 0; i < n; i++) in += grp[ids[i] - 1] == 0 && out[i] > 0.9999f;
      CHECK(st[6] == beside && in == 10, "compact keeps the ids beside rows (%llu) and finds the group (%d)", (unsigned long long)st[6], in);
      CHECK(c == kci_vec_count(on) && c <= expect, "compact returns the ids left (%u)", c);
      double f1 = recall(s1, base, total, q, 100, 100, sc);
      printf("after insert / delete / compact: re-ranked recall at ef 100 against the original rows %.4f\n", f1);
      CHECK(f1 >= 0.93, "the index still answers (%.4f)", f1); }

    kci_vec_searcher_free(s0); kci_vec_searcher_free(s1); kci_vec_free(off); kci_vec_free(on);
    free(base); free(q); free(sc); free(gvec); free(grp); free(gsize);
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
