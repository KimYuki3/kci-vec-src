/*

 */
#include "../kci_vec.c"
#include <stdio.h>

static int fails;
#define CHECK(c, ...) do{ if(!(c)){ fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } }while(0)
enum { N=12000, D=96, NQ=100, K=10 };
static float *base, *qry;
static uint32_t rs=777u;
static float rnd(void){ rs=rs*1664525u+1013904223u; return (float)((rs>>8)&0xFFFF)/65536.0f-0.5f; }
static int flt_even(uint64_t id, void *ctx){ (void)ctx; return id%2==0; }

static double agree(KciVecSearcher *s, uint64_t (*ref)[K], int ef, int rr, int filtered, int *bad, int metric){
    long hit=0; *bad=0;
    for(int q=0;q<NQ;q++){ uint64_t ids[K]; float sc[K];
        int n = filtered ? kci_vec_search_filtered(s, qry+(size_t)q*D, K, ef, rr, flt_even, NULL, ids, sc)
                         : kci_vec_search_rerank(s, qry+(size_t)q*D, K, ef, rr, ids, sc);
        if(n!=K){ (*bad)++; continue; }
        for(int a=0;a<n;a++){
            if(filtered && ids[a]%2) (*bad)++;
            if(metric==0){ float c=0, nq=0, nb=0; const float *x=base+(size_t)(ids[a]-1)*D, *y=qry+(size_t)q*D;
                for(int d=0;d<D;d++){ c+=x[d]*y[d]; nq+=y[d]*y[d]; nb+=x[d]*x[d]; }
                c/=sqrtf(nq*nb); if(fabsf(c-sc[a])>2e-3f) (*bad)++;          /* halves: within 1e-4; allow for the query's own rounding */
                if(a && sc[a]>sc[a-1]+1e-6f) (*bad)++; }
            for(int b=0;b<K;b++) if(ref[q][b]==ids[a]){ hit++; break; } } }
    return (double)hit/((double)NQ*K);
}

int main(void){
    base=malloc(sizeof(float)*N*D); qry=malloc(sizeof(float)*NQ*D);
    for(int i=0;i<N*D;i++) base[i]=rnd()+((i%D)<6 ? 0.8f*rnd() : 0);
    for(int i=0;i<NQ*D;i++) qry[i]=rnd();
    for(int metric=0;metric<3;metric++) for(int shortrows=0;shortrows<2;shortrows++){
        KciVec *v = metric==0 ? kci_vec_new(D,16,100) : metric==1 ? kci_vec_new_l2(D,16,100,0.0f) : kci_vec_new_ip(D,16,100,0.0f);
        kci_vec_keep_float(v,1);
        for(int i=0;i<N;i++) kci_vec_add(v,(uint64_t)i+1,base+(size_t)i*D);
        CHECK(kci_vec_search_only(v)==-1, "not built: refused");
        kci_vec_build(v,0,NULL);
        CHECK(kci_vec_search_only(v)==-1, "no quantized graph: refused");
        CHECK((shortrows ? kci_vec_build_qg_ex(v,2,48,0) : kci_vec_build_qg(v,2,0))==0, "build_qg");
        for(int i=0;i<20;i++) CHECK(kci_vec_delete(v,(uint64_t)(50+13*i))==0, "delete before");
        KciVecSearcher *s=kci_vec_searcher_new(v);
        static uint64_t ref[NQ][K], reff[NQ][K]; int bad; uint64_t ids[K]; float sc[K];
        for(int q=0;q<NQ;q++){ CHECK(kci_vec_search_rerank(s,qry+(size_t)q*D,K,120,120,ref[q],sc)==K, "reference");
                               CHECK(kci_vec_search_filtered(s,qry+(size_t)q*D,K,120,120,flt_even,NULL,reff[q],sc)==K, "filtered reference"); }
        CHECK(kci_vec_half_floats(v)==0, "half_floats");
        double n0=agree(s,ref,120,2*K,0,&bad,metric);
        uint64_t pb0[6], pb1[6]; kci_vec_plain_bytes(v,pb0);
        CHECK(kci_vec_search_only(v)==0, "search_only (metric %d, short rows %d)", metric, shortrows);
        CHECK(kci_vec_search_only(v)==-1, "a second call is refused");
        kci_vec_plain_bytes(v,pb1);
        CHECK(pb1[0]+(uint64_t)N*D<=pb0[0] && pb1[3]<pb0[3], "plain bytes: codes %llu -> %llu, graph %llu -> %llu",
              (unsigned long long)pb0[0],(unsigned long long)pb1[0],(unsigned long long)pb0[3],(unsigned long long)pb1[3]);
        double a1=agree(s,ref,120,120,0,&bad,metric); CHECK(bad==0, "search-only rerank: %d bad", bad);
        double a2=agree(s,ref,120,2*K,0,&bad,metric); CHECK(bad==0, "search-only narrow rerank: %d bad", bad);
        double a3=agree(s,reff,120,120,1,&bad,metric); CHECK(bad==0, "search-only filtered: %d bad", bad);
        printf("metric %d, short rows %d: same ids as before the switch %.3f (narrow pass %.3f, filtered %.3f)\n", metric, shortrows, a1, a2, a3);
        CHECK(a1>=0.90 && a3>=0.90 && a2>=n0-0.05, "search-only answers: %.3f / %.3f (before the switch %.3f) / %.3f", a1, a2, n0, a3);
        for(int q=0;q<NQ;q++){ int n=kci_vec_search_rerank(s,qry+(size_t)q*D,K,120,120,ids,sc);
            for(int a=0;a<n;a++) for(int i=0;i<20;i++) if(ids[a]==(uint64_t)(50+13*i)) fails++, printf("FAIL: a deleted id came back\n"); }
        /* everything that would touch the rows */
        CHECK(kci_vec_search(v,qry,K,50,ids,sc)<0 && kci_vec_search_r(s,qry,K,50,ids,sc)<0 && kci_vec_search_exact(v,qry,K,1,ids,sc)<0, "plain / int8 / exact searches");
        CHECK(kci_vec_search_filtered(s,qry,K,50,0,flt_even,NULL,ids,sc)<0, "filtered without an exact pass");
        CHECK(kci_vec_searcher_set_qg(s,0)==0 && kci_vec_search_rerank(s,qry,K,50,50,ids,sc)<0 && kci_vec_search_filtered(s,qry,K,50,50,flt_even,NULL,ids,sc)<0, "searcher set to the plain graph");
        CHECK(kci_vec_searcher_set_qg(s,-1)==0 && kci_vec_search_rerank(s,qry,K,50,50,ids,sc)==K, "back to the quantized graph");
        CHECK(kci_vec_searcher_set_qg_exact(s,1)==0 && kci_vec_search_rerank(s,qry,K,50,50,ids,sc)==K, "qg_exact 1 falls back without the int8 rows");
        kci_vec_searcher_set_qg_exact(s,-1);
        CHECK(kci_vec_insert(v,999999,base)<0 && kci_vec_delete(v,3)<0 && kci_vec_add(v,999998,base)<0 && kci_vec_compact(v)==(uint32_t)N, "writes");
        CHECK(kci_vec_build(v,1,NULL)<0 && kci_vec_build_qg(v,1,0)<0 && kci_vec_build_qg_ex(v,2,32,0)<0 && kci_vec_build_ivf(v,16,1)<0 && kci_vec_half_floats(v)<0, "builds");
        CHECK(kci_vec_save(v,"_so.kv",NULL,0)<0 && kci_vec_image_save(v,"_so.kvimg")<0 && kci_vec_protect(v)<0 && kci_vec_reserve(v,N+10)<0, "save / image / protect / reserve");
        { float out[D]; CHECK(kci_vec_get_float(v,0,out)<0, "get_float"); }
        { KciVecSearcher *t=kci_vec_searcher_new(v); CHECK(t && kci_vec_search_rerank(t,qry,K,60,60,ids,sc)==K, "a new searcher"); kci_vec_searcher_free(t); }
        kci_vec_searcher_free(s); kci_vec_free(v);
    }
    { KciVec *w=kci_vec_new(D,8,60); kci_vec_keep_float(w,1);            /* floats instead of halves are enough */
      for(int i=0;i<3000;i++) kci_vec_add(w,(uint64_t)i+1,base+(size_t)i*D);
      kci_vec_build(w,1,NULL); CHECK(kci_vec_build_qg(w,2,1)==0 && kci_vec_search_only(w)==0, "with floats");
      KciVecSearcher *s=kci_vec_searcher_new(w); uint64_t ids[K]; float sc[K];
      CHECK(kci_vec_search_rerank(s,qry,K,60,60,ids,sc)==K, "search with floats"); kci_vec_searcher_free(s); kci_vec_free(w); }
    { KciVec *w=kci_vec_new(D,8,60);                                     /* no floats at all: refused */
      for(int i=0;i<3000;i++) kci_vec_add(w,(uint64_t)i+1,base+(size_t)i*D);
      kci_vec_build(w,1,NULL); CHECK(kci_vec_build_qg(w,2,1)==0 && kci_vec_search_only(w)==-1, "no exact pass: refused"); kci_vec_free(w); }
    CHECK(kci_vec_search_only(NULL)==-1, "NULL");
    free(base); free(qry); remove("_so.kv"); remove("_so.kvimg");
    if(fails){ printf("%d FAILED\n", fails); return 1; }
    printf("ALL PASS\n"); return 0;
}
