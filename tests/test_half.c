/*
 kci_vec_half_floats: the retained floats as 16-bit halves.
 *   gcc -O2 -march=native -I. tests/test_half.c -lpthread -lm -o test_half     (includes kci_vec.c for the converters)
 * Checks: the portable converters (exact values, round to nearest even, against F16C where the compiler has it);
 * the rerank pass over the halves returns the floats' answers (ids nearly all the same, scores within 1e-4, true cosines);
 * plain_bytes halves; a second call, an index without floats and a not-built index are refused; insert is refused
 * afterwards (it needs the floats); delete still hides a row; with the quantized graph too.
 */
#include "../kci_vec.c"
#include <stdio.h>

static int fails;
#define CHECK(c, ...) do{ if(!(c)){ fails++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } }while(0)
enum { N=20000, D=96, NQ=200, K=10 };
static float *base, *qry;
static uint32_t rs=12345u;
static float rnd(void){ rs=rs*1664525u+1013904223u; return (float)((rs>>8)&0xFFFF)/65536.0f-0.5f; }
static void unit(float *x){ double s=0; for(int d=0;d<D;d++) s+=(double)x[d]*x[d]; s=sqrt(s); for(int d=0;d<D;d++) x[d]=(float)(x[d]/s); }

int main(void){
    /* converters */
    CHECK(kv_f2h(0.0f)==0 && kv_f2h(-0.0f)==0x8000, "zero");
    CHECK(kv_f2h(1.0f)==0x3C00 && kv_f2h(-2.0f)==0xC000 && kv_f2h(0.5f)==0x3800, "1, -2, 0.5");
    CHECK(kv_f2h(65504.0f)==0x7BFF && kv_f2h(1e6f)==0x7C00, "largest / overflow");
    CHECK(kv_f2h(5.9604645e-8f)==0x0001 && kv_f2h(1e-9f)==0x0000, "smallest subnormal / underflow");
    CHECK(kv_f2h(1.0f+1.0f/2048)==0x3C00 && kv_f2h(1.0f+3.0f/2048)==0x3C02, "ties to even");
    CHECK(kv_h2f(0x3C00)==1.0f && kv_h2f(0xC000)==-2.0f && kv_h2f(0x0001)==5.9604645e-8f && kv_h2f(0x7BFF)==65504.0f, "h2f");
    { int bad=0; for(int i=0;i<200000;i++){ float x=rnd()*(i%3==0 ? 1e-3f : (i%3==1 ? 0.2f : 4.0f)); uint16_t h=kv_f2h(x); float y=kv_h2f(h);
          if(fabsf(y-x) > fabsf(x)/2048.0f+6e-8f) bad++;
#if defined(__AVX2__) && defined(__F16C__)
          { float in[8]={x,x,x,x,x,x,x,x}; uint16_t o[8]; _mm_storeu_si128((__m128i*)o, _mm256_cvtps_ph(_mm256_loadu_ps(in), _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
            float back[8]; _mm256_storeu_ps(back, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)o)));
            if(o[0]!=h || back[0]!=y) bad++; }
#endif
      }
      CHECK(bad==0, "converters: %d of 200000 off", bad); }

    base=malloc(sizeof(float)*N*D); qry=malloc(sizeof(float)*NQ*D);
    for(int i=0;i<N;i++){ for(int d=0;d<D;d++) base[(size_t)i*D+d]=rnd()+(d<8 ? 0.6f*rnd() : 0); unit(base+(size_t)i*D); }
    for(int i=0;i<NQ;i++){ for(int d=0;d<D;d++) qry[(size_t)i*D+d]=rnd(); unit(qry+(size_t)i*D); }
    KciVec *v=kci_vec_new(D, 16, 100);
    CHECK(kci_vec_half_floats(v)==-1, "not built: refused");
    kci_vec_keep_float(v,1);
    for(int i=0;i<N;i++) kci_vec_add(v, (uint64_t)i+1, base+(size_t)i*D);
    kci_vec_build(v, 0, NULL);
    KciVecSearcher *s=kci_vec_searcher_new(v);
    uint64_t *id0=malloc(sizeof(uint64_t)*NQ*K), ids[K]; float *sc0=malloc(sizeof(float)*NQ*K), sc[K];
    for(int q=0;q<NQ;q++) CHECK(kci_vec_search_rerank(s, qry+(size_t)q*D, K, 100, 100, id0+(size_t)q*K, sc0+(size_t)q*K)==K, "rerank before");
    uint64_t pb0[6], pb1[6]; kci_vec_plain_bytes(v, pb0);
    CHECK(kci_vec_half_floats(v)==0, "half_floats");
    CHECK(kci_vec_half_floats(v)==-1, "a second call is refused");
    kci_vec_plain_bytes(v, pb1);
    CHECK(pb1[1]*2<=pb0[1] && pb1[1]>0, "floats in RAM %llu -> %llu", (unsigned long long)pb0[1], (unsigned long long)pb1[1]);
    for(int pass=0;pass<2;pass++){          /* 0: plain graph, 1: quantized graph */
        if(pass){ CHECK(kci_vec_build_qg(v,1,0)==0, "build_qg"); CHECK(kci_vec_searcher_set_qg(s,1)==0, "set_qg"); }
        long same=0; int bad=0;
        for(int q=0;q<NQ;q++){
            int n=kci_vec_search_rerank(s, qry+(size_t)q*D, K, 100, 100, ids, sc); if(n!=K){ bad++; continue; }
            for(int a=0;a<n;a++){
                float c=0; for(int d=0;d<D;d++) c+=qry[(size_t)q*D+d]*base[(size_t)(ids[a]-1)*D+d];
                if(fabsf(c-sc[a])>1e-4f) bad++;                       /* a true cosine */
                if(a && sc[a]>sc[a-1]+1e-6f) bad++;
                if(!pass && fabsf(sc[a]-sc0[(size_t)q*K+a])>1e-4f) bad++;   /* the same score at the same rank as the floats gave */
                for(int b=0;b<K;b++) if(id0[(size_t)q*K+b]==ids[a]){ same++; break; } } }
        double fr=(double)same/((double)NQ*K);
        printf("%s: the halves' top %d against the floats': %.4f the same ids\n", pass ? "quantized graph" : "graph", K, fr);
        CHECK(bad==0, "pass %d: %d bad rows", pass, bad);
        CHECK(fr >= (pass ? 0.88 : 0.995), "pass %d: only %.4f of the floats' ids", pass, fr); }
    kci_vec_searcher_set_qg(s,0);
    CHECK(kci_vec_insert(v, 999999, base)<0, "insert needs the floats: refused");
    { uint64_t victim=id0[0]; CHECK(kci_vec_delete(v, victim)==0 || kci_vec_delete(v, victim)==1, "delete");
      int n=kci_vec_search_rerank(s, qry, K, 100, 100, ids, sc); int seen=0; for(int a=0;a<n;a++) if(ids[a]==victim) seen=1;
      CHECK(!seen, "a deleted id came back"); }
    kci_vec_searcher_free(s); kci_vec_free(v);
    { KciVec *w=kci_vec_new(D, 8, 50);                 /* built without keep_float: no floats to turn */
      for(int i=0;i<500;i++) kci_vec_add(w, (uint64_t)i+1, base+(size_t)i*D);
      kci_vec_build(w, 1, NULL); CHECK(kci_vec_half_floats(w)==-1, "no floats: refused"); kci_vec_free(w); }
    CHECK(kci_vec_half_floats(NULL)==-1, "NULL");
    free(base); free(qry); free(id0); free(sc0);
    if(fails){ printf("%d FAILED\n", fails); return 1; }
    printf("ALL PASS\n"); return 0;
}
