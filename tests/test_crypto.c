/*
 test_crypto.c — snapshot encryption (KCIVEC05: AES-256-CTR + AES-256-CMAC).
 *
 * Snapshots used ChaCha20 without a tag; a damaged or altered file
 * loaded as wrong data. Checks: AES-256 and CMAC known answers (FIPS-197,
 * SP 800-38B) on every AES path; the fast path equals the portable one; save /
 * load round trip; wrong or missing key; any flipped, cut or added byte is
 * refused; plaintext still writes KCIVEC04; an old ChaCha20 KCIVEC04 file
 * (tests/data/kcivec04_chacha.kv, written by the previous kci_vec.c) still loads.
 *
 * Includes kci_vec.c to reach the static AES / CMAC functions:
 *   gcc -O2 -march=native -I. tests/test_crypto.c -lpthread -lm -o test_crypto   (run from the repo root)
 * The portable AES is checked on every machine (path 0); the fast path the
 * machine has is checked against the same answers and against the portable one.
 */
#include "kci_vec.c"
#include <stdio.h>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail = 1; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void unhex(const char *h, uint8_t *out, size_t *n) {
    size_t k = 0;
    for (; h[0] && h[1]; h += 2) { unsigned b; sscanf(h, "%2x", &b); out[k++] = (uint8_t)b; }
    if (n) *n = k;
}

/* ── known answers ── */
static void test_kat(int path) {
    KvAes a; uint8_t k[32], pt[16], ct[16], o[16];
    unhex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", k, NULL);
    unhex("00112233445566778899aabbccddeeff", pt, NULL);
    unhex("8ea2b7ca516745bfeafc49904b496089", ct, NULL);
    kv_aes_init(&a, k); a.path = path;
    kv_aes_block(&a, pt, o);
    CHECK(memcmp(o, ct, 16) == 0, "FIPS-197 C.3 AES-256 (path %d)", path);

    /* SP 800-38B D.3 (AES-256 CMAC), messages of 0, 16, 40 and 64 bytes */
    static const char *msg = "6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
                             "30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710";
    static const struct { size_t len; const char *tag; } v[4] = {
        { 0, "028962f61b7bf89efc6b551f4667d983" }, { 16, "28a7023f452e8f82bd4bf28d8c37c35c" },
        { 40, "aaf3d8f1de5640c232f5b169b9c911e6" }, { 64, "e1992190549f6ed5696a2c056c315410" } };
    uint8_t m[64], ck[32], want[16], tag[16];
    unhex(msg, m, NULL);
    unhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", ck, NULL);
    for (int i = 0; i < 4; i++) {
        unhex(v[i].tag, want, NULL);
        KvCmac c; kv_cmac_init(&c, ck); c.a.path = path;
        kv_cmac_update(&c, m, v[i].len); kv_cmac_final(&c, tag);
        CHECK(memcmp(tag, want, 16) == 0, "SP 800-38B AES-256 CMAC %zu bytes (path %d)", v[i].len, path);
        /* the same message fed in 1..7-byte pieces */
        kv_cmac_init(&c, ck); c.a.path = path;
        for (size_t off = 0, step = 1; off < v[i].len; off += step, step = step % 7 + 1)
            kv_cmac_update(&c, m + off, off + step <= v[i].len ? step : v[i].len - off);
        kv_cmac_final(&c, tag);
        CHECK(memcmp(tag, want, 16) == 0, "CMAC %zu bytes in pieces (path %d)", v[i].len, path);
    }
}

/* the fast path must give exactly the portable keystream and tags */
static void test_paths_agree(int fast) {
    uint8_t key[32]; for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i * 37 + 1);
    KvAes a, s; kv_aes_init(&a, key); s = a; a.path = fast; s.path = 0;
    size_t n = 1 << 20; uint8_t *x = malloc(n), *y = malloc(n);
    int bad = 0;
    for (size_t len = 0; len < 300; len++) {
        memset(x, 0x3C, len); memset(y, 0x3C, len);
        kv_ctr(&a, 0x1234567890ABCDEFULL + len, x, len); kv_ctr(&s, 0x1234567890ABCDEFULL + len, y, len);
        bad += memcmp(x, y, len) != 0;
    }
    for (size_t i = 0; i < n; i++) x[i] = y[i] = (uint8_t)(i * 131);
    kv_ctr(&a, 7, x, n); kv_ctr(&s, 7, y, n);
    bad += memcmp(x, y, n) != 0;
    KvCmac c1, c2; kv_cmac_init(&c1, key); kv_cmac_init(&c2, key); c1.a.path = fast; c2.a.path = 0;
    kv_cmac_update(&c1, x, n - 5); kv_cmac_update(&c2, x, n - 5);
    uint8_t t1[16], t2[16]; kv_cmac_final(&c1, t1); kv_cmac_final(&c2, t2);
    bad += memcmp(t1, t2, 16) != 0;
    CHECK(bad == 0, "fast path %d == portable (CTR lengths 0..299, 1 MB, CMAC)", fast);
    free(x); free(y);
}

/* ── snapshots ── */
#define DIM 16
static void mkvec(uint64_t id, float *v) {
    uint64_t s = id * 0x9E3779B97F4A7C15ULL + 12345;
    for (int i = 0; i < DIM; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = (float)((int64_t)(s % 2001) - 1000) / 1000.0f; }
}
/* every live id 7*1..7*300 (except `dead`) is present with its vector and finds itself */
static int check_index(KciVec *v, uint64_t dead) {
    if (!v) return 0;
    KciVecSearcher *s = kci_vec_searcher_new(v);
    float f[DIM], u[DIM], g[DIM]; uint64_t ids[5]; float sc[5]; int ok = 1;
    for (uint64_t id = 7; id <= 2100; id += 7) {
        mkvec(id, f); l2norm(DIM, f, u);
        int n = kci_vec_search_rerank(s, f, 5, 50, 50, ids, sc), hit = 0;
        for (int i = 0; i < n; i++) hit |= ids[i] == id;
        if (id == dead) { ok &= !hit; continue; }
        ok &= hit;
        uint32_t idx = h_get(v, id);
        ok &= idx != KV_HEMPTY && kci_vec_get_float(v, idx, g) == 0 && memcmp(g, u, sizeof u) == 0;
    }
    kci_vec_searcher_free(s);
    return ok;
}
static long fsize(const char *p) { FILE *f = fopen(p, "rb"); if (!f) return -1; fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f); return n; }
static void flip(const char *p, long off) {
    FILE *f = fopen(p, "r+b"); fseek(f, off, SEEK_SET); int c = fgetc(f); fseek(f, off, SEEK_SET); fputc(c ^ 0x01, f); fclose(f);
}
static void copy(const char *a, const char *b) {
    FILE *x = fopen(a, "rb"), *y = fopen(b, "wb"); int c; while ((c = fgetc(x)) != EOF) fputc(c, y); fclose(x); fclose(y);
}

static void test_snapshots(void) {
    const uint8_t *key = (const uint8_t *)"test key"; int kl = 8;
    KciVec *v = kci_vec_new(DIM, 16, 100); kci_vec_keep_float(v, 1); float f[DIM];
    for (uint64_t id = 7; id <= 2100; id += 7) { mkvec(id, f); kci_vec_add(v, id, f); }
    kci_vec_build(v, 1, NULL);
    kci_vec_delete(v, 35);

    CHECK(kci_vec_save(v, "_c5.kv", key, kl) == 0, "encrypted save");
    char m[9] = {0}; FILE *fp = fopen("_c5.kv", "rb"); if (fread(m, 1, 8, fp) != 8) m[0] = 0; fclose(fp);
    CHECK(strcmp(m, "KCIVEC05") == 0, "encrypted save writes KCIVEC05 (got %s)", m);
    KciVec *w = kci_vec_load("_c5.kv", key, kl);
    CHECK(check_index(w, 35), "KCIVEC05 round trip: ids, vectors, search, tombstone");
    if (w) kci_vec_free(w);
    CHECK(kci_vec_load("_c5.kv", (const uint8_t *)"test kez", kl) == NULL, "wrong key refused");
    CHECK(kci_vec_load("_c5.kv", NULL, 0) == NULL, "missing key refused");

    /* every flipped byte is refused: header, salt, key check, payload, tag */
    long n = fsize("_c5.kv"), bad = 0, tried = 0;
    long offs[] = { 0, 7, 8, 12, 24, 39, 40, 55, 56, 71, 72, 100, 1000, n / 2, n - 17, n - 16, n - 1 };
    for (size_t i = 0; i < sizeof offs / sizeof offs[0]; i++) {
        copy("_c5.kv", "_c5x.kv"); flip("_c5x.kv", offs[i]); tried++;
        KciVec *x = kci_vec_load("_c5x.kv", key, kl);
        if (x) { bad++; printf("  flipped byte %ld loaded\n", offs[i]); kci_vec_free(x); }
    }
    for (long off = 3; off < n; off += n / 40) {                   /* and 40 spread over the file */
        copy("_c5.kv", "_c5x.kv"); flip("_c5x.kv", off); tried++;
        KciVec *x = kci_vec_load("_c5x.kv", key, kl);
        if (x) { bad++; kci_vec_free(x); }
    }
    CHECK(bad == 0, "flipped bytes refused (%ld of %ld loaded)", bad, tried);
    /* cut the last byte, add one byte */
    copy("_c5.kv", "_c5x.kv");
    { FILE *a = fopen("_c5.kv", "rb"), *b = fopen("_c5x.kv", "wb"); for (long i = 0; i < n - 1; i++) fputc(fgetc(a), b); fclose(a); fclose(b); }
    CHECK(kci_vec_load("_c5x.kv", key, kl) == NULL, "truncated file refused");
    copy("_c5.kv", "_c5x.kv");
    { FILE *b = fopen("_c5x.kv", "ab"); fputc(0, b); fclose(b); }
    CHECK(kci_vec_load("_c5x.kv", key, kl) == NULL, "file with an extra byte refused");

    /* plaintext: unchanged KCIVEC04 */
    CHECK(kci_vec_save(v, "_c4.kv", NULL, 0) == 0, "plaintext save");
    fp = fopen("_c4.kv", "rb"); if (fread(m, 1, 8, fp) != 8) m[0] = 0; fclose(fp);
    CHECK(strcmp(m, "KCIVEC04") == 0, "plaintext save writes KCIVEC04 (got %s)", m);
    w = kci_vec_load("_c4.kv", NULL, 0);
    CHECK(check_index(w, 35), "KCIVEC04 plaintext round trip");
    if (w) kci_vec_free(w);
    kci_vec_free(v);
    remove("_c5.kv"); remove("_c5x.kv"); remove("_c4.kv");

    /* an older ChaCha20 snapshot (written by the previous kci_vec.c) still loads */
    w = kci_vec_load("tests/data/kcivec04_chacha.kv", (const uint8_t *)"legacy key", 10);
    CHECK(w != NULL && kci_vec_count(w) == 300, "KCIVEC04 ChaCha20 file loads");
    CHECK(check_index(w, 35), "KCIVEC04 ChaCha20 file: ids, vectors, search, tombstone");
    if (w) kci_vec_free(w);
    CHECK(kci_vec_load("tests/data/kcivec04_chacha.kv", (const uint8_t *)"wrong", 5) == NULL, "KCIVEC04 wrong key refused");
}

static void test_damaged(void) {
    enum { N = 2000, D = 32 };
    srand(7); int r0 = rand(); srand(7);
    KciVec *v = kci_vec_new(D, 8, 60); float x[D]; uint32_t s = 99;
    for (int i = 0; i < N; i++) { for (int d = 0; d < D; d++) { s = s * 1664525u + 1013904223u; x[d] = (float)((s >> 8) & 0xFFFF) / 65536.0f - 0.5f; } kci_vec_add(v, (uint64_t)i + 1, x); }
    kci_vec_build(v, 1, NULL);
    CHECK(rand() == r0, "kci_vec_build changed the program's rand() sequence");
    CHECK(kv_graph_ok(v, v->count, v->ep, (uint32_t)v->maxlevel) == 1, "a built graph is sound");
    CHECK(kci_vec_save(v, "_d4.kv", NULL, 0) == 0, "plaintext save");
    FILE *f = fopen("_d4.kv", "rb"); fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(len), *alt = malloc(len); CHECK(fread(buf, 1, len, f) == (size_t)len, "read the file back"); fclose(f);
    size_t rowb = (size_t)(v->M0 + 1) * sizeof(uint32_t); long l0 = -1;
    for (long o = 0; o + (long)rowb <= len; o++) if (memcmp(buf + o, v->l0, rowb) == 0) { l0 = o; break; }
    CHECK(l0 > 0 && v->l0[0] > 0, "layer 0 not found in the file");
    #define TRY(name, edit) do { memcpy(alt, buf, len); edit; FILE *g = fopen("_d4x.kv", "wb"); fwrite(alt, 1, len, g); fclose(g); \
        KciVec *w = kci_vec_load("_d4x.kv", NULL, 0); CHECK(w == NULL, "%s loaded", name); if (w) kci_vec_free(w); } while (0)
    { KciVec *w = kci_vec_load("_d4.kv", NULL, 0); CHECK(w != NULL && kci_vec_count(w) == N, "the untouched file loads"); if (w) kci_vec_free(w); }
    TRY("count 0xF0000000", { uint32_t c = 0xF0000000u; memcpy(alt + 24, &c, 4); });
    TRY("count 2^31 + 5", { uint32_t c = 0x80000005u; memcpy(alt + 24, &c, 4); });
    TRY("count + 1", { uint32_t c = N + 1; memcpy(alt + 24, &c, 4); });
    TRY("dim 0", { uint32_t c = 0; memcpy(alt + 12, &c, 4); });
    if (l0 > 0) {
        TRY("link count 1000", { uint32_t c = 1000; memcpy(alt + l0, &c, 4); });
        TRY("neighbour past the rows", { uint32_t c = 0xFFFFFFF0u; memcpy(alt + l0 + 4, &c, 4); });
        TRY("neighbour = count", { uint32_t c = N; memcpy(alt + l0 + 4, &c, 4); });
    }
    TRY("entry point past the rows", { uint32_t c = N; memcpy(alt + 32, &c, 4); });
    CHECK(kci_vec_load_ex("_d4.kv", NULL, 0, KCI_VEC_LOAD_AUTH_ONLY) == NULL, "auth only: a plaintext file loaded");
    CHECK(kci_vec_load_ex("tests/data/kcivec04_chacha.kv", (const uint8_t *)"legacy key", 10, KCI_VEC_LOAD_AUTH_ONLY) == NULL, "auth only: a ChaCha20 file loaded");
    CHECK(kci_vec_save(v, "_d5.kv", (const uint8_t *)"k", 1) == 0, "encrypted save");
    { KciVec *w = kci_vec_load_ex("_d5.kv", (const uint8_t *)"k", 1, KCI_VEC_LOAD_AUTH_ONLY); CHECK(w != NULL && kci_vec_count(w) == N, "auth only: an AES + CMAC file refused"); if (w) kci_vec_free(w); }
    CHECK(kci_vec_load_ex("_d5.kv", (const uint8_t *)"x", 1, KCI_VEC_LOAD_AUTH_ONLY) == NULL, "auth only: wrong key loaded");
    remove("_d5.kv");
    #undef TRY
    free(buf); free(alt); kci_vec_free(v); remove("_d4.kv"); remove("_d4x.kv");
}

static int qg_answers(KciVec *v, const float *Q, int nq, int d, uint64_t *ids, float *sc) {
    KciVecSearcher *s = kci_vec_searcher_new(v); if (!s) return -1;
    if (kci_vec_searcher_set_qg(s, 1) != 0) { kci_vec_searcher_free(s); return -2; }
    kci_vec_searcher_set_qg_exact(s, 2);
    int bad = 0;
    for (int q = 0; q < nq; q++) if (kci_vec_search_rerank(s, Q + (size_t)q * d, 10, 80, 40, ids + (size_t)q * 10, sc + (size_t)q * 10) != 10) bad++;
    kci_vec_searcher_free(s); return bad;
}
static void test_qg_snapshot(void) {
    enum { N = 6000, D = 48, NQ = 30 };
    static float X[N * D], Q[NQ * D]; uint32_t s = 4242;
    for (int i = 0; i < N * D; i++) { s = s * 1664525u + 1013904223u; X[i] = (float)((s >> 8) & 0xFFFF) / 65536.0f - 0.5f; }
    for (int i = 0; i < NQ * D; i++) { s = s * 1664525u + 1013904223u; Q[i] = (float)((s >> 8) & 0xFFFF) / 65536.0f - 0.5f; }
    static uint64_t i0[NQ * 10], i1[NQ * 10]; static float s0[NQ * 10], s1[NQ * 10];
    for (int hv = 0; hv < 2; hv++) {           /* 0: floats, 1: halves */
        KciVec *v = kci_vec_new(D, 16, 100); kci_vec_keep_float(v, 1);
        for (int i = 0; i < N; i++) kci_vec_add(v, (uint64_t)i + 1, X + (size_t)i * D);
        kci_vec_build(v, 1, NULL);
        CHECK(kci_vec_build_qg_ex(v, 2, 16, 1) == 0, "build_qg_ex");
        if (hv) CHECK(kci_vec_half_floats(v) == 0, "half_floats");
        CHECK(qg_answers(v, Q, NQ, D, i0, s0) == 0, "answers of the built index");
        for (int enc = 0; enc < 2; enc++) {
            const uint8_t *key = enc ? (const uint8_t *)"qg key" : NULL; int kl = enc ? 6 : 0;
            CHECK(kci_vec_save(v, "_q5.kv", key, kl) == 0, "save with a quantized graph (halves %d, enc %d)", hv, enc);
            KciVec *w = kci_vec_load("_q5.kv", key, kl);
            CHECK(w != NULL && kci_vec_qg_bytes(w) == kci_vec_qg_bytes(v) && kci_vec_qg_bytes(w) > 0, "load keeps the quantized graph (halves %d, enc %d)", hv, enc);
            if (w) { CHECK(qg_answers(w, Q, NQ, D, i1, s1) == 0, "answers of the loaded index");
                CHECK(memcmp(i0, i1, sizeof i0) == 0 && memcmp(s0, s1, sizeof s0) == 0, "loaded index answers differently (halves %d, enc %d)", hv, enc);
                CHECK((w->vh != NULL) == (hv != 0), "halves in the loaded index: %d", w->vh != NULL);
                kci_vec_free(w); }
            FILE *f = fopen("_q5.kv", "rb"); fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
            uint8_t *buf = malloc(len), *alt = malloc(len); CHECK(fread(buf, 1, len, f) == (size_t)len, "read back"); fclose(f);
            #define TRYQ(name, edit, n2, want_null) do { memcpy(alt, buf, len); edit; FILE *g2 = fopen("_q5x.kv", "wb"); fwrite(alt, 1, (n2), g2); fclose(g2); \
                KciVec *t = kci_vec_load("_q5x.kv", key, kl); CHECK((t == NULL) == (want_null), "%s (halves %d, enc %d): %s", name, hv, enc, t ? "loaded" : "refused"); \
                if (t && !(want_null)) { CHECK(kci_vec_qg_bytes(t) == 0, "%s: a quantized graph came out", name); \
                    uint64_t ii[10]; float ss[10]; KciVecSearcher *z = kci_vec_searcher_new(t); \
                    CHECK(kci_vec_searcher_set_qg(z, 1) != 0 && kci_vec_search_r(z, Q, 10, 80, ii, ss) == 10, "%s: plain graph search", name); kci_vec_searcher_free(z); } \
                if (t) kci_vec_free(t); } while (0)
            if (enc) {
                CHECK(kci_vec_load("_q5.kv", (const uint8_t *)"wrong!", 6) == NULL, "wrong key loaded");
                TRYQ("a byte altered in the section", alt[len - 200] ^= 1, len, 1);
                TRYQ("cut short", (void)0, len - 64, 1);
            } else {
                long sec = -1; for (long o = len - 8; o >= 0; o--) if (memcmp(buf + o, "KVQG", 4) == 0 && buf[o + 5] == 0 && buf[o + 6] == 0 && buf[o + 7] == 0) { sec = o; break; }
                CHECK(sec > 0, "section not found");
                if (sec > 0) {
                    TRYQ("section magic damaged", alt[sec] ^= 0xFF, len, 1);
                    TRYQ("section M damaged", { uint32_t m; memcpy(&m, alt + sec + 8, 4); m += 2; memcpy(alt + sec + 8, &m, 4); }, len, 1);
                    TRYQ("unknown part bit", alt[sec + 4] |= 0x40, len, 1);
                }
                TRYQ("cut short", (void)0, len - 64, 1);
                TRYQ("flag cleared (a reader from before)", { uint32_t fl; memcpy(&fl, alt + 8, 4); fl &= ~64u; memcpy(alt + 8, &fl, 4); }, len, 0);
            }
            #undef TRYQ
            free(buf); free(alt);
        }
        kci_vec_free(v);
    }
    remove("_q5.kv"); remove("_q5x.kv");
}

int main(void) {
    KvAes probe; uint8_t z[32] = {0}; kv_aes_init(&probe, z);
    printf("AES path on this machine: %s\n", probe.path == 1 ? "AES-NI" : probe.path == 2 ? "ARMv8 AES" : "portable");
    test_kat(0);
    if (probe.path) { test_kat(probe.path); test_paths_agree(probe.path); }
    test_snapshots();
    test_damaged();
    test_qg_snapshot();
    printf("%s\n", g_fail ? "FAIL" : "ALL PASS");
    return g_fail;
}
