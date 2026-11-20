/*

 */
#include "../kci_vec.c"
#include <stdio.h>

static int fails;
#define CHECK(c, ...) do{ if(!(c)){ fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } }while(0)
enum { N=15000, D=128, NQ=100, K=10, LAT=40 };
static float *base, *qry;
static uint32_t rs=4242u;
static float rnd(void){ rs=rs*1664525u+1013904223u; return (float)((rs>>8)&0xFFFF)/65536.0f-0.5f; }
static int flt_even(uint64_t id, void *ctx){ (void)ctx; return id%2==0; }
/* vectors with most of their energy on LAT hidden directions, so that the principal directions mean something */
static void gen(float *out, int n, const float *mix){
    for(int i=0;i<n;i++){ float z[LAT]; for(int a=0;a<LAT;a++) z[a]=rnd()*(1.0f+3.0f/(1+a));
        for(int d=0;d<D;d++){ float x=0.15f*rnd(); for(int a=0;a<LAT;a++) x+=z[a]*mix[(size_t)a*D+d]; out[(size_t)i*D+d]=x; } }
}
static double recall(KciVecSearcher *s, uint64_t (*gt)[K], int ef, int filtered, uint64_t (*keep)[K]){
    long hit=0;
    for(int q=0;q<NQ;q++){ uint64_t ids[K]; float sc[K];
        int n = filtered ? kci_vec_search_filtered(s,qry+(size_t)q*D,K,ef,ef,flt_even,NULL,ids,sc) : kci_vec_search_rerank(s,qry+(size_t)q*D,K,ef,ef,ids,sc);
        if(n!=K) fails++, printf("FAIL: %d results\n", n);
        for(int a=0;a<n;a++){ if(filtered && ids[a]%2) fails++, printf("FAIL: filter\n"); for(int b=0;b<K;b++) if(gt[q][b]==ids[a]){ hit++; break; } }
        if(keep) memcpy(keep[q], ids, sizeof ids); }
    return (double)hit/((double)NQ*K);
}

int main(void){
    static float mix[LAT*D]; for(int i=0;i<LAT*D;i++) mix[i]=rnd();
    base=malloc(sizeof(float)*N*D); qry=malloc(sizeof(float)*NQ*D); gen(base,N,mix); gen(qry,NQ,mix);
    uint8_t key[32]; for(int i=0;i<32;i++) key[i]=(uint8_t)(i*7+1);
    for(int metric=0;metric<3;metric++) for(int Mg=8;Mg<=16;Mg+=8){
        KciVec *v = metric==0 ? kci_vec_new(D,Mg,100) : metric==1 ? kci_vec_new_l2(D,Mg,100,0.0f) : kci_vec_new_ip(D,Mg,100,0.0f);
        kci_vec_keep_float(v,1);
        for(int i=0;i<N;i++) kci_vec_add(v,(uint64_t)i+1,base+(size_t)i*D);
        CHECK(kci_vec_build_qg_res(v,64,32,0)==-1, "no graph yet: refused");
        kci_vec_build(v,0,NULL);
        static uint64_t gt[NQ][K], gtf[NQ][K], a0[NQ][K], a1[NQ][K]; float sc[K];
        for(int q=0;q<NQ;q++){ CHECK(kci_vec_search_exact(v,qry+(size_t)q*D,K,1,gt[q],sc)==K, "exact");
            /* the exact answers among the even ids: a wide filtered search over the plain graph */
            KciVecSearcher *t=kci_vec_searcher_new(v); kci_vec_searcher_set_qg(t,0);
            CHECK(kci_vec_search_filtered(t,qry+(size_t)q*D,K,2000,2000,flt_even,NULL,gtf[q],sc)==K, "filtered reference"); kci_vec_searcher_free(t); }
        CHECK(kci_vec_build_qg_res(v,0,0,0)==-1 && kci_vec_build_qg_res(v,64,7,0)==-1 && kci_vec_build_qg_res(v,64,66,0)==-1 && kci_vec_build_qg_res(v,D+8,0,0)==-1
              && kci_vec_build_qg_res(NULL,64,32,0)==-1, "bad arguments");
        CHECK(kci_vec_build_qg_ex(v,2,64,0)==0, "plain codes");
        int64_t b0=kci_vec_qg_bytes(v);
        KciVecSearcher *s=kci_vec_searcher_new(v); double r0=recall(s,gt,60,0,NULL), f0=recall(s,gtf,60,1,NULL); kci_vec_searcher_free(s);
        CHECK(kci_vec_build_qg_res(v,64,32,0)==0, "build_qg_res");
        int64_t b1=kci_vec_qg_bytes(v);
        CHECK(b1>0 && b1<b0, "blocks %lld -> %lld bytes", (long long)b0, (long long)b1);
        s=kci_vec_searcher_new(v); double r1=recall(s,gt,60,0,a0), f1=recall(s,gtf,60,1,NULL);
        CHECK(kci_vec_searcher_set_qg_exact(s,0)==0 && recall(s,gt,60,0,a1)==r1 && memcmp(a0,a1,sizeof a0)==0, "qg_exact 0 is taken as 2");
        kci_vec_searcher_free(s);
        printf("metric %d M %d: blocks %lld -> %lld KB, recall@10 ef 60 %.3f -> %.3f, filtered %.3f -> %.3f\n", metric, Mg, (long long)(b0>>10), (long long)(b1>>10), r0, r1, f0, f1);
        CHECK(r1>=r0-0.03 && r1>=0.80 && f1>=f0-0.03, "recall %.3f -> %.3f, filtered %.3f -> %.3f", r0, r1, f0, f1);
        /* the encrypted save and the image */
        CHECK(kci_vec_half_floats(v)==0, "half_floats");
        s=kci_vec_searcher_new(v); recall(s,gt,60,0,a0); kci_vec_searcher_free(s);
        CHECK(kci_vec_save(v,"_qr.kv",key,32)==0, "save");
        KciVec *w=kci_vec_load("_qr.kv",key,32); CHECK(w!=NULL, "load");
        if(w){ CHECK(kci_vec_qg_bytes(w)==b1, "loaded blocks"); s=kci_vec_searcher_new(w); recall(s,gt,60,0,a1); kci_vec_searcher_free(s);
            CHECK(memcmp(a0,a1,sizeof a0)==0, "the loaded index answers the same"); kci_vec_free(w); }
        CHECK(kci_vec_image_save(v,"_qr.kvimg")==0, "image_save");
        w=kci_vec_attach("_qr.kvimg"); CHECK(w!=NULL, "attach");
        if(w){ s=kci_vec_searcher_new(w); recall(s,gt,60,0,a1); kci_vec_searcher_free(s);
            CHECK(memcmp(a0,a1,sizeof a0)==0, "the attached image answers the same"); kci_vec_free(w); }
        /* search-only on top */
        CHECK(kci_vec_search_only(v)==0, "search_only");
        s=kci_vec_searcher_new(v); double r2=recall(s,gt,60,0,NULL), f2=recall(s,gtf,60,1,NULL); kci_vec_searcher_free(s);
        CHECK(r2>=r1-0.03 && f2>=f1-0.03, "search-only recall %.3f -> %.3f, filtered %.3f -> %.3f", r1, r2, f1, f2);
        kci_vec_free(v);
    }
    /* an image whose residual flag is flipped must not attach (the block header no longer fits) */
    { FILE *f=fopen("_qr.kvimg","rb"); CHECK(f!=NULL, "open image");
      if(f){ fseek(f,0,SEEK_END); long len=ftell(f); fseek(f,0,SEEK_SET); uint8_t *img=malloc((size_t)len); CHECK(fread(img,1,(size_t)len,f)==(size_t)len, "read"); fclose(f);
        KciVec *w=kci_vec_attach_mem(img,(size_t)len); CHECK(w!=NULL, "attach_mem"); kci_vec_free(w);
        int found=0;
        for(long o=0;o+(long)sizeof(KvImgG)<=len && !found;o+=8) if(!memcmp(img+o,"KCIVIGR1",8)){ KvImgG x; memcpy(&x,img+o,sizeof x);
            CHECK(x.pad==1, "the image carries the residual flag");
            x.pad=0; memcpy(img+o,&x,sizeof x); w=kci_vec_attach_mem(img,(size_t)len); CHECK(w==NULL, "flag 0: refused"); kci_vec_free(w);
            x.pad=2; memcpy(img+o,&x,sizeof x); w=kci_vec_attach_mem(img,(size_t)len); CHECK(w==NULL, "flag 2: refused"); kci_vec_free(w); found=1; }
        CHECK(found, "graph part found"); free(img); } }
    free(base); free(qry); remove("_qr.kv"); remove("_qr.kvimg");
    if(fails){ printf("%d FAILED\n", fails); return 1; }
    printf("ALL PASS\n"); return 0;
}
