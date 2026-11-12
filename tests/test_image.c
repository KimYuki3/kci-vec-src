/*
 test_image.c the read-only image (kci_vec_image_save / kci_vec_attach).
 *   gcc -O1 -g -fsanitize=address,undefined -march=native -I. tests/test_image.c kci_vec.c -lpthread -lm -o test_image
 * Builds lists over random vectors, saves the image, attaches it, and checks that the attached index answers
 * exactly like the original (ids and scores, lists search and exact search), that writes are refused, that a
 * truncated / altered file is refused, and that a second attach shares (same answers after the first is freed).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "kci_vec.h"

static int fails = 0;
#define CHECK(c, ...) do{ if(!(c)){ fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } }while(0)

static uint64_t rs = 12345;
static float frand(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f; }

static int same_answer(KciVec *a, KciVec *b, const float *q, int k, int nprobe, int rr, int exact){
    uint64_t ia[64], ib[64]; float sa[64], sb[64]; int na, nb;
    if(exact){ na = kci_vec_search_exact(a, q, k, 1, ia, sa); nb = kci_vec_search_exact(b, q, k, 1, ib, sb); }
    else {
        KciVecSearcher *x = kci_vec_searcher_new(a), *y = kci_vec_searcher_new(b);
        na = kci_vec_search_ivf(x, q, k, nprobe, rr, ia, sa); nb = kci_vec_search_ivf(y, q, k, nprobe, rr, ib, sb);
        kci_vec_searcher_free(x); kci_vec_searcher_free(y);
    }
    if(na != nb) return 0;
    for(int i = 0; i < na; i++) if(ia[i] != ib[i] || fabsf(sa[i] - sb[i]) > 1e-6f) return 0;
    return na > 0;
}

int main(void){
    const int N = 20000, D = 36, K = 10;
    float *X = malloc((size_t)N * D * sizeof(float));
    for(size_t i = 0; i < (size_t)N * D; i++) X[i] = frand();
    for(int t = 0; t < 2; t++){                    /* t=0: pairs of dims per code (dp 2); t=1: one dim per code */
        KciVec *v = kci_vec_new(D, 16, 100);
        kci_vec_keep_float(v, 1);
        for(int i = 0; i < N; i++) CHECK(kci_vec_add(v, 1000 + i, X + (size_t)i * D) > 0, "add %d", i);
        if(t) kci_vec_set_ivf_codes(v, 1, 0.55f);
        CHECK(kci_vec_build_ivf(v, 64, 4) == 0, "build_ivf");
        for(int i = 0; i < 50; i++) CHECK(kci_vec_delete(v, 1000 + 7 * i) == 0, "delete");   /* tombstones go into the image */
        const char *path = t ? "test_image_1.kvimg" : "test_image_0.kvimg";
        CHECK(kci_vec_image_save(v, path) == 0, "image_save");

        KciVec *a = kci_vec_attach(path);
        CHECK(a != NULL, "attach");
        if(!a){ kci_vec_free(v); continue; }
        CHECK(kci_vec_count(a) == (uint32_t)N && kci_vec_dim(a) == D, "count/dim");
        int ok_l = 0, ok_e = 0, nq = 200;
        for(int qi = 0; qi < nq; qi++){
            float q[D]; for(int d = 0; d < D; d++) q[d] = frand();
            ok_l += same_answer(v, a, q, K, 8, 40, 0);
            ok_e += same_answer(v, a, q, K, 0, 0, 1);
        }
        CHECK(ok_l == nq, "lists: %d/%d queries identical (t=%d)", ok_l, nq, t);
        CHECK(ok_e == nq, "exact: %d/%d queries identical (t=%d)", ok_e, nq, t);
        {   /* tombstoned ids never come back */
            KciVecSearcher *s = kci_vec_searcher_new(a); uint64_t ids[64]; float sc[64];
            int bad = 0;
            for(int qi = 0; qi < 50; qi++){ int n = kci_vec_search_ivf(s, X + (size_t)(7 * qi) * D, K, 16, 80, ids, sc);
                for(int i = 0; i < n; i++) if(ids[i] == (uint64_t)(1000 + 7 * qi)) bad++; }
            CHECK(bad == 0, "tombstoned ids returned: %d", bad);
            kci_vec_searcher_free(s);
        }
        {   /* writes are refused */
            float q[D]; for(int d = 0; d < D; d++) q[d] = frand();
            uint64_t ids[16]; float sc[16];
            CHECK(kci_vec_add(a, 1, q) < 0, "add on attached");
            CHECK(kci_vec_insert(a, 1, q) < 0, "insert on attached");
            CHECK(kci_vec_delete(a, 1000) < 0, "delete on attached");
            CHECK(kci_vec_forget(a, 1000) < 0, "forget on attached");
            CHECK(kci_vec_compact(a) == (uint32_t)N, "compact on attached");
            CHECK(kci_vec_build(a, 1, NULL) < 0, "build on attached");
            CHECK(kci_vec_build_ivf(a, 32, 1) < 0, "build_ivf on attached");
            CHECK(kci_vec_protect(a) < 0, "protect on attached");
            CHECK(kci_vec_set_float(a, 1000, q) < 0, "set_float on attached");
            CHECK(kci_vec_search(a, q, K, 50, ids, sc) < 0, "graph search on attached (no graph)");
        }
        {   /* a saved copy of the attached index loads and answers the same */
            CHECK(kci_vec_save(a, "test_image_copy.kvec", (const uint8_t*)"pw", 2) == 0, "save attached");
            KciVec *c = kci_vec_load("test_image_copy.kvec", (const uint8_t*)"pw", 2);
            CHECK(c != NULL, "load the copy");
            if(c){ int ok = 0; for(int qi = 0; qi < 20; qi++){ float q[D]; for(int d = 0; d < D; d++) q[d] = frand(); ok += same_answer(a, c, q, K, 8, 40, 0); }
                CHECK(ok == 20, "copy answers: %d/20", ok); kci_vec_free(c); }
            remove("test_image_copy.kvec");
        }
        {   /* two attachments at once; the first freed, the second still right */
            KciVec *b = kci_vec_attach(path);
            CHECK(b != NULL, "second attach");
            kci_vec_free(a); a = NULL;
            if(b){ int ok = 0; for(int qi = 0; qi < 20; qi++){ float q[D]; for(int d = 0; d < D; d++) q[d] = frand(); ok += same_answer(v, b, q, K, 8, 40, 0); }
                CHECK(ok == 20, "second attach answers: %d/20", ok); kci_vec_free(b); }
        }
        {
            KciVec *r = kci_vec_attach(path), *w = r ? kci_vec_thaw(r) : NULL;
            CHECK(w != NULL, "thaw");
            CHECK(kci_vec_thaw(v) == NULL, "thaw of a non-attached index refused");
            if(w){
                int ok = 0; for(int qi = 0; qi < 20; qi++){ float q[D]; for(int d = 0; d < D; d++) q[d] = frand(); ok += same_answer(v, w, q, K, 8, 40, 0) && same_answer(v, w, q, K, 0, 0, 1); }
                CHECK(ok == 20, "thawed copy answers: %d/20", ok);
                CHECK(kci_vec_count(w) == (uint32_t)N, "thawed count");
                const int NY = 500; float *Y = malloc((size_t)NY * D * sizeof(float));
                for(size_t i = 0; i < (size_t)NY * D; i++) Y[i] = frand();
                for(int i = 50; i < 80; i++) CHECK(kci_vec_delete(w, 1000 + 7 * i) == 0, "delete on thawed %d", i);
                for(int i = 0; i < NY; i++) CHECK(kci_vec_insert(w, 100000 + i, Y + (size_t)i * D) > 0, "insert on thawed %d", i);
                CHECK(kci_vec_insert(w, 1000, Y) > 0, "re-insert a tombstoned id");
                KciVecSearcher *s1 = kci_vec_searcher_new(w); uint64_t before[20][K]; int nb[20]; float Q[20][D];
                for(int qi = 0; qi < 20; qi++){ for(int d = 0; d < D; d++) Q[qi][d] = frand(); float sc[K]; nb[qi] = kci_vec_search_ivf(s1, Q[qi], K, 8, 40, before[qi], sc); }
                kci_vec_searcher_free(s1);
                int64_t rm = kci_vec_squeeze(w);
                CHECK(rm == 80, "squeeze removed %lld rows (80 expected)", (long long)rm);
                CHECK(kci_vec_count(w) == (uint32_t)(N - 80 + NY + 1), "count after squeeze %u", kci_vec_count(w));
                CHECK(kci_vec_squeeze(w) == 0, "second squeeze is a no-op");
                {   KciVecSearcher *s2 = kci_vec_searcher_new(w); int same = 0, found = 0, gone = 0; uint64_t ids[K]; float sc[K];
                    for(int qi = 0; qi < 20; qi++){ int n = kci_vec_search_ivf(s2, Q[qi], K, 8, 40, ids, sc); int eq = n == nb[qi];
                        for(int j = 0; eq && j < n; j++) eq = ids[j] == before[qi][j];
                        same += eq; }
                    CHECK(same == 20, "answers before / after the squeeze identical: %d/20", same);
                    /* the 500 new vectors are found by their own vector; the 80 deleted ones never come back */
                    for(int i = 0; i < NY; i += 5){ int n = kci_vec_search_ivf(s2, Y + (size_t)i * D, 1, 8, 40, ids, sc); found += n == 1 && (ids[0] == (uint64_t)(100000 + i) || (i == 0 && ids[0] == 1000)); }   /* Y[0] is in twice (id 1000 too) */
                    CHECK(found == NY / 5, "new vectors found: %d/%d", found, NY / 5);
                    for(int i = 0; i < 80; i++){ int n = kci_vec_search_ivf(s2, X + (size_t)(7 * i) * D, K, 16, 80, ids, sc); for(int j = 0; j < n; j++) if(ids[j] == (uint64_t)(1000 + 7 * i) && i) gone++; }
                    CHECK(gone == 0, "deleted ids returned after squeeze: %d", gone);
                    { int n = kci_vec_search_ivf(s2, Y, 1, 8, 40, ids, sc); CHECK(n == 1 && (ids[0] == 1000 || ids[0] == 100000), "re-inserted id 1000 found (got %llu)", (unsigned long long)(n ? ids[0] : 0)); }
                    kci_vec_searcher_free(s2); }
                /* the next image: attached, it answers like the thawed index */
                const char *path2 = "test_image_next.kvimg";
                CHECK(kci_vec_image_save(w, path2) == 0, "image_save of the thawed index");
                KciVec *a2 = kci_vec_attach(path2);
                CHECK(a2 != NULL, "attach the next image");
                if(a2){ int ok2 = 0; for(int qi = 0; qi < 20; qi++){ float q[D]; for(int d = 0; d < D; d++) q[d] = frand(); ok2 += same_answer(w, a2, q, K, 8, 40, 0) && same_answer(w, a2, q, K, 0, 0, 1); }
                    CHECK(ok2 == 20, "next image answers: %d/20", ok2);
                    CHECK(kci_vec_count(a2) == kci_vec_count(w), "next image count");
                    /* and a thaw of the next image again, with a tombstone-free start: squeeze 0 */
                    KciVec *w2 = kci_vec_thaw(a2); CHECK(w2 && kci_vec_squeeze(w2) == 0, "thaw of the next image, nothing dead"); if(w2) kci_vec_free(w2);
                    kci_vec_free(a2); }
                remove(path2);
                free(Y); kci_vec_free(w);
            }
            if(r) kci_vec_free(r);
        }
        {   /* truncated and altered files are refused */
            FILE *f = fopen(path, "rb"); fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
            uint8_t *buf = malloc(len); CHECK(fread(buf, 1, len, f) == (size_t)len, "read the image back"); fclose(f);
            FILE *g = fopen("test_image_bad.kvimg", "wb"); fwrite(buf, 1, len / 2, g); fclose(g);
            CHECK(kci_vec_attach("test_image_bad.kvimg") == NULL, "truncated file attached");
            uint8_t *alt = malloc(len); memcpy(alt, buf, len); alt[3] ^= 0xFF;
            CHECK(kci_vec_attach_mem(alt, len) == NULL, "wrong magic attached");
            memcpy(alt, buf, len); uint32_t *hdr = (uint32_t*)(alt + 8); hdr[3] += 1;   /* count + 1: lengths no longer match */
            CHECK(kci_vec_attach_mem(alt, len) == NULL, "wrong count attached");
            CHECK(kci_vec_attach_mem(buf, 100) == NULL, "short buffer attached");
            CHECK(kci_vec_attach("no_such_file.kvimg") == NULL, "missing file attached");
            free(alt); free(buf); remove("test_image_bad.kvimg");
        }
        if(a) kci_vec_free(a);
        kci_vec_free(v);
        remove(path);
    }
    free(X);
    printf(fails ? "test_image: %d FAILED\n" : "test_image: OK\n", fails);
    return fails ? 1 : 0;
}
