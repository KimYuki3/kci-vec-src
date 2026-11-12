/*
 * kci_vec.c — KCI-Vec, embeddable approximate nearest-neighbour search (implementation).
 *
 * Dimension-agnostic, multithreaded int8 HNSW with optional AES-256-encrypted,
 * authenticated snapshots. De-globalized so multiple indexes coexist in one process.
 */
#define _GNU_SOURCE   /* expose clock_gettime/CLOCK_MONOTONIC under -std=c11 on glibc (Linux) */
#include "kci_vec.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include "kci_win_compat.h"
#else
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif


/* L2-normalize src into dst (for cosine: SQ then operates on unit vectors). */
static void l2norm(int dim, const float *src, float *dst) {
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)src[i] * src[i];
    float inv = (ss > 0.0) ? (float)(1.0 / sqrt(ss)) : 0.0f;
    for (int i = 0; i < dim; i++) dst[i] = src[i] * inv;
}

/* ── binary heap of (float key, id) ─────────────────────────────────────── */
typedef struct { float d; uint32_t id; } He;
typedef struct { He *a; int n, cap, ismax; } Heap;
static void hp_init(Heap *h, int cap, int ismax){ h->a=malloc(sizeof(He)*cap); h->n=0; h->cap=cap; h->ismax=ismax; }
static inline int hp_less(Heap *h, He x, He y){ return h->ismax ? x.d>y.d : x.d<y.d; }
static void hp_push(Heap *h, float d, uint32_t id){
    if(h->n>=h->cap){ h->cap*=2; h->a=realloc(h->a,sizeof(He)*h->cap); }
    int i=h->n++; h->a[i].d=d; h->a[i].id=id;
    while(i){ int p=(i-1)/2; if(hp_less(h,h->a[i],h->a[p])){He t=h->a[i];h->a[i]=h->a[p];h->a[p]=t;i=p;}else break; }
}
static inline He hp_top(Heap *h){ return h->a[0]; }
static void hp_pop(Heap *h){ h->a[0]=h->a[--h->n]; int i=0;
    for(;;){ int l=2*i+1,r=2*i+2,m=i; if(l<h->n&&hp_less(h,h->a[l],h->a[m]))m=l; if(r<h->n&&hp_less(h,h->a[r],h->a[m]))m=r; if(m==i)break; He t=h->a[i];h->a[i]=h->a[m];h->a[m]=t;i=m; } }

/* ── index ──────────────────────────────────────────────────────────────── */
struct KciVec {
    int dim, M, M0, efc; double mL;
    int8_t   *vq;       /* [cap*dim] signed int8 codes (x*gs), row-major */
    float     gs;       /* global quantization scale: code = round(x*gs), x unit-norm */
    float    *vf;       /* [cap*dim] retained unit floats (re-rank); malloc'd or mmap'd */
    int       keepf;    /* user asked to retain floats (rerank); else freed at build */
    int       protect;
    int       vf_mmap;  /* 1 = vf is a read-only mmap of a disk file (RAM stays int8) */
    size_t    vf_maplen;
    int       vf_fd;
    int       building; /* 1 = accepting adds (reserve allocates training vf); 0 = loaded */
    uint64_t *vid;      /* [cap] caller ids */
    uint32_t count, cap;
    /* graph */
    uint8_t  *level;
    uint32_t *l0;
    uint32_t **lu;
    uint32_t  ep; int maxlevel; uint32_t gN;
    uint32_t  gcap;     /* allocated graph capacity (>= gN); grows on incremental insert */
    int built;
    uint8_t  *lock;
    uint32_t  bnext;
    pthread_mutex_t epmtx;
    /* incremental update (tombstone delete + live insert) */
    uint8_t  *dead;     /* [gcap] 1 = tombstoned (excluded from results, kept as a graph hop) */
    uint32_t  ndead;
    uint64_t *hkey;     /* id → index open-addressing hash (for delete-by-id) */
    uint32_t *hval;     /* parallel slot: index, or KV_HEMPTY */
    uint32_t  hbits;    /* hash size = 1<<hbits */
    struct KvIvf *ivf;
    struct KvQg  *qg;
    KciVecFloatFn ffn; void *fctx; void (*ffree)(void *);
    int ivf_dp; float ivf_avqt;
    int ivf_km_iters; uint32_t ivf_km_sample;
    const uint8_t *img; size_t imglen; int img_mmap;
};
struct KvIvf;
static void kv_ivf_free(struct KvIvf *f);
static void kv_ivf_scr_free(void *z);
static int kv_ivf_add(KciVec *v, uint32_t i);
static int kv_ensure_dead(KciVec *v);
struct KvQg;
static void kv_qg_free(struct KvQg *g);
struct KvBq;
static void kv_bq_free(struct KvBq *q);
#define KV_HEMPTY 0xFFFFFFFFu

static inline const int8_t* vrow(const KciVec *v, uint32_t i){ return v->vq + (size_t)i*v->dim; }

static void kv_wipe(void *p, size_t n){
    if(!p || !n) return;
    memset(p, 0, n);
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" : : "r"(p) : "memory");
#else
    { volatile unsigned char *q = (volatile unsigned char *)p; (void)q[0]; }
#endif
}
static size_t kv_pagesz(void){
#ifdef _WIN32
    return 4096;
#else
    long ps = sysconf(_SC_PAGESIZE); return ps > 0 ? (size_t)ps : 4096;
#endif
}
static int kv_protect_region(void *p, size_t n){
    if(!p || !n) return 3;                 /* nothing to protect: does not lower the result */
    int r = 0;
#ifdef _WIN32
    if(VirtualLock(p, n)) r |= 2;
#else
    uintptr_t ps = kv_pagesz(), a = ((uintptr_t)p + ps - 1) & ~(ps - 1), e = ((uintptr_t)p + n) & ~(ps - 1);
    if(e <= a) return 3;
    if(e > a){
# ifdef MADV_DONTDUMP
        if(madvise((void *)a, e - a, MADV_DONTDUMP) == 0) r |= 1;
# endif
        if(mlock((void *)a, e - a) == 0) r |= 2;
    }
#endif
    return r;
}
static void kv_unlock_region(void *p, size_t n){
    if(!p || !n) return;
#ifdef _WIN32
    VirtualUnlock(p, n);
#else
    uintptr_t ps = kv_pagesz(), a = ((uintptr_t)p + ps - 1) & ~(ps - 1), e = ((uintptr_t)p + n) & ~(ps - 1);
    if(e > a) munlock((void *)a, e - a);
#endif
}
/* allocate a vector buffer: page-aligned and rounded up to whole pages when the index is protected */
static void *kv_valloc(const KciVec *v, size_t n){
#ifndef _WIN32
    if(v->protect){ size_t ps = kv_pagesz(); void *p = NULL; n = (n + ps - 1) / ps * ps;
        return posix_memalign(&p, ps, n ? n : ps) == 0 ? p : NULL; }
#endif
    (void)v; return malloc(n ? n : 1);
}
/* wipe, unlock and free a vector buffer of n bytes */
static void kv_vfree(const KciVec *v, void *p, size_t n){
    if(!p) return;
    kv_wipe(p, n);
    if(v->protect) kv_unlock_region(p, n);
    free(p);
}

/* ── id → index hash (open addressing, for delete-by-id) ─────────────────── */
static inline uint64_t hmix(uint64_t x){ x^=x>>33; x*=0xff51afd7ed558ccdULL; x^=x>>33; return x; }
static int h_rebuild(KciVec *v, uint32_t bits){
    uint32_t sz=1u<<bits; uint64_t *k=malloc((size_t)sz*8); uint32_t *val=malloc((size_t)sz*4);
    if(!k||!val){ free(k); free(val); return -1; }
    for(uint32_t i=0;i<sz;i++) val[i]=KV_HEMPTY;
    uint32_t m=sz-1;
    for(uint32_t i=0;i<v->count;i++){ uint32_t s=(uint32_t)hmix(v->vid[i])&m;
        int dead_i = v->dead && v->dead[i];
        while(val[s]!=KV_HEMPTY && k[s]!=v->vid[i]) s=(s+1)&m;
        if(val[s]!=KV_HEMPTY && dead_i) continue;     /* keep the row already there over a dead later one */
        k[s]=v->vid[i]; val[s]=i; }
    free(v->hkey); free(v->hval); v->hkey=k; v->hval=val; v->hbits=bits;
    return 0;
}
static void h_put(KciVec *v, uint64_t id, uint32_t idx){
    if(!v->hval) return;
    uint32_t sz=1u<<v->hbits;
    if(((uint64_t)v->count)*4 >= (uint64_t)sz*3){ if(h_rebuild(v, v->hbits+1)!=0) return; sz=1u<<v->hbits; }
    uint32_t m=sz-1, s=(uint32_t)hmix(id)&m;
    while(v->hval[s]!=KV_HEMPTY){ if(v->hkey[s]==id){ v->hval[s]=idx; return; } s=(s+1)&m; }
    v->hkey[s]=id; v->hval[s]=idx;
}
static uint32_t h_get(KciVec *v, uint64_t id){
    if(!v->hval) return KV_HEMPTY;
    uint32_t sz=1u<<v->hbits, m=sz-1, s=(uint32_t)hmix(id)&m;
    while(v->hval[s]!=KV_HEMPTY){ if(v->hkey[s]==id) return v->hval[s]; s=(s+1)&m; }
    return KV_HEMPTY;
}
static void kv_move(void **dst, void *nb, size_t used, size_t newsz, size_t oldsz){
    if(*dst && used) memcpy(nb, *dst, used);
    memset((char*)nb+used, 0, newsz-used);
    if(*dst){ kv_wipe(*dst, oldsz); free(*dst); }
    *dst=nb;
}

/* grow the graph arrays (level/l0/lu/lock/dead) to hold >= need nodes; new region zeroed. All five are
 * allocated first (a failure changes nothing), then filled, and the old ones wiped and freed (kv_move).
 * The arrays move → caller must be single-writer (no live search). */
static int ensure_graph_cap(KciVec *v, uint32_t need){
    if(need <= v->gcap) return 0;
    uint32_t nc = v->gcap ? v->gcap : need; while(nc<need) nc*=2;
    size_t old=v->gcap, r0=(size_t)(v->M0+1)*sizeof(uint32_t);
    uint8_t  *nlv=malloc(nc);
    uint32_t *nl0=malloc((size_t)nc*r0);
    uint32_t **nlu=malloc((size_t)nc*sizeof(uint32_t*));
    uint8_t  *nlk=malloc(nc);
    uint8_t  *ndd=malloc(nc);
    if(!nlv||!nl0||!nlu||!nlk||!ndd){ free(nlv); free(nl0); free(nlu); free(nlk); free(ndd); return -1; }
    kv_move((void**)&v->level, nlv, old, nc, old);
    kv_move((void**)&v->l0, nl0, old*r0, (size_t)nc*r0, old*r0);
    kv_move((void**)&v->lu, nlu, old*sizeof(uint32_t*), (size_t)nc*sizeof(uint32_t*), old*sizeof(uint32_t*));
    kv_move((void**)&v->lock, nlk, old, nc, old);
    kv_move((void**)&v->dead, ndd, v->dead? old : 0, nc, old);
    v->gcap=nc;
    return 0;
}

static inline int32_t sdot(const KciVec *v, const int8_t *a, const int8_t *b){
    int dim=v->dim; int i=0; int32_t s=0;
#if defined(__ARM_NEON)
  #if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc=vdupq_n_s32(0);
    for(; i+16<=dim; i+=16) acc=vdotq_s32(acc, vld1q_s8(a+i), vld1q_s8(b+i));
    s=vaddvq_s32(acc);
  #else
    int32x4_t acc=vdupq_n_s32(0);
    for(; i+8<=dim; i+=8){ int16x8_t p=vmull_s8(vld1_s8(a+i), vld1_s8(b+i)); acc=vpadalq_s16(acc,p); }
    s=vaddvq_s32(acc);
  #endif
    for(; i<dim; i++) s += (int32_t)a[i]*b[i];
    return s;
#elif defined(__AVX2__)
    __m256i acc=_mm256_setzero_si256();
    const __m256i ones=_mm256_set1_epi16(1);
    for(; i+32<=dim; i+=32){
        __m256i av=_mm256_loadu_si256((const __m256i*)(a+i)), bv=_mm256_loadu_si256((const __m256i*)(b+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(av,av), _mm256_sign_epi8(bv,av));
        acc=_mm256_add_epi32(acc, _mm256_madd_epi16(p,ones));
    }
    for(; i+16<=dim; i+=16){
        __m256i av=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+i)));
        __m256i bv=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b+i)));
        acc=_mm256_add_epi32(acc, _mm256_madd_epi16(av,bv));      /* 16×(s16·s16) → 8×s32 */
    }
    __m128i lo=_mm256_castsi256_si128(acc), hi=_mm256_extracti128_si256(acc,1);
    __m128i q=_mm_add_epi32(lo,hi);
    q=_mm_add_epi32(q,_mm_srli_si128(q,8)); q=_mm_add_epi32(q,_mm_srli_si128(q,4));
    s=_mm_cvtsi128_si32(q);
    for(; i<dim; i++) s += (int32_t)a[i]*b[i];
    return s;
#else
    for(; i<dim; i++) s += (int32_t)a[i]*b[i];
    return s;
#endif
}
/* encode a unit-norm float vector to signed int8 codes (caller normalizes first). */
static inline void sq_encode(const KciVec *v, const float *x, int8_t *code){
    float gs=v->gs;
    for(int d=0; d<v->dim; d++){
        int r=(int)lrintf(x[d]*gs);
        code[d]=(int8_t)(r < -127 ? -127 : (r > 127 ? 127 : r));
    }
}
/* navigation distance: smaller = closer. gs² is constant so ranking == cosine. */
static inline float vdist(const KciVec *v, const int8_t *a, const int8_t *b){ return -(float)sdot(v, a, b); }
static inline uint32_t* g_l0(KciVec *v, uint32_t i){ return v->l0 + (size_t)i*(v->M0+1); }
static inline uint32_t* g_lu(KciVec *v, uint32_t i, int l){ return v->lu[i] + (size_t)(l-1)*(v->M+1); }
static inline uint32_t* g_links(KciVec *v, uint32_t i, int l){ return l==0? g_l0(v,i): g_lu(v,i,l); }

static inline void nlock(KciVec *v, uint32_t i){ while(__atomic_test_and_set(&v->lock[i], __ATOMIC_ACQUIRE)){} }
static inline void nunlock(KciVec *v, uint32_t i){ __atomic_clear(&v->lock[i], __ATOMIC_RELEASE); }

/* per-thread search scratch */
typedef struct { Heap C, W; uint16_t *vis; uint32_t visn; uint16_t epoch; He *tmp; int tmpcap; float eps;   /* eps: see kci_vec_searcher_set_eps */
                 const float *qf; int use_qg; uint8_t *lut; float *lutf; int lutM; struct KvBq *bq; } Scr;   /* bq: the quantized graph's candidate buckets */
static void scr_init(Scr *s, uint32_t N, int wcap){ hp_init(&s->C,4096,0); hp_init(&s->W,wcap,1); s->vis=calloc(N,sizeof(uint16_t)); s->visn=N; s->epoch=0; s->tmpcap=wcap+16; s->tmp=malloc(sizeof(He)*s->tmpcap); s->eps=0;
                                                    s->qf=NULL; s->use_qg=0; s->lut=NULL; s->lutf=NULL; s->lutM=0; s->bq=NULL; }
static void scr_free(Scr *s){ free(s->C.a); free(s->W.a); free(s->vis); free(s->tmp);
                              if(s->lut) kv_wipe(s->lut, (size_t)(s->lutM+4)*16); if(s->lutf) kv_wipe(s->lutf, (size_t)s->lutM*16*sizeof(float)); free(s->lut); free(s->lutf); kv_bq_free(s->bq); }
static inline void scr_next(Scr *s){ if(++s->epoch==0){ memset(s->vis,0,(size_t)s->visn*sizeof(uint16_t)); s->epoch=1; } }
static inline int visited(Scr *s, uint32_t id){ if(s->vis[id]==s->epoch)return 1; s->vis[id]=s->epoch; return 0; }
static int kv_qg_collect(KciVec *v, Scr *s, const int8_t *q, int ef, int rescore8);
static int scr_fit(Scr *s, uint32_t N){
    if(s->visn>=N) return 0;
    uint16_t *nv=realloc(s->vis,(size_t)N*sizeof(uint16_t)); if(!nv) return -1;
    memset(nv,0,(size_t)N*sizeof(uint16_t)); s->vis=nv; s->visn=N; s->epoch=0; return 0;
}

static int default_threads(int t){
    if (t > 0) return t > 64 ? 64 : t;
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 4; if (n > 64) n = 64;
    return (int)n;
}

/* ── core graph traversal ───────────────────────────────────────────────── */
static inline float sl_bound(Heap *W, int ef, float k1, float gs2){
    if(W->n<ef) return FLT_MAX;
    float t=hp_top(W).d; return k1>0 ? k1*(gs2+t)-gs2 : t;
}
static void search_layer(KciVec *v, const int8_t *q, uint32_t ep, int ef, int layer, Scr *s){
    Heap *C=&s->C, *W=&s->W; scr_next(s); C->n=0; W->n=0;
    const float gs2=v->gs*v->gs, k1=(layer==0 && s->eps>0) ? 1.0f+s->eps : 0.0f;
    float d0=vdist(v,q,vrow(v,ep)); visited(s,ep);
    hp_push(C,d0,ep); hp_push(W,d0,ep);
    while(C->n){
        He c=hp_top(C); hp_pop(C);
        if(c.d>sl_bound(W,ef,k1,gs2)) break;
        uint32_t *lk=g_links(v,c.id,layer);
        uint32_t cnt=__atomic_load_n(&lk[0], __ATOMIC_ACQUIRE);
        for(uint32_t e=1;e<=cnt;e++){ uint32_t nb=lk[e]; if(nb<v->gN) __builtin_prefetch(&s->vis[nb]); }
        uint32_t cand[256]; int nc=0;
        for(uint32_t e=1;e<=cnt && nc<256;e++){
            uint32_t nb=lk[e];
            if(nb>=v->gN) continue;                  /* guard a torn read */
            if(visited(s,nb)) continue;
            const int8_t *r=vrow(v,nb);
            __builtin_prefetch(r); __builtin_prefetch(r+v->dim-1);
            cand[nc++]=nb;
        }
        float bound=sl_bound(W,ef,k1,gs2);              /* recomputed only when W changes */
        for(int i=0;i<nc;i++){
            uint32_t nb=cand[i];
            float d=vdist(v,q,vrow(v,nb));
            if(d<bound){                                 /* within the bound: explore from it ... */
                hp_push(C,d,nb);
                if(W->n<ef || d<hp_top(W).d){ hp_push(W,d,nb); if(W->n>ef) hp_pop(W); bound=sl_bound(W,ef,k1,gs2); }   /* ... and keep it if among the ef best */
            }
        }
    }
}

/* HNSW neighbour-selection heuristic: keep a candidate only if no already-
 * chosen node is closer to it than the query is (diversity pruning). */
static int select_neighbors(KciVec *v, const int8_t *q, Heap *W, int M, uint32_t *out){
    int m=W->n; He *cand=malloc(sizeof(He)*m);
    for(int i=0;i<m;i++) cand[i]=W->a[i];
    for(int i=0;i<m;i++){ int mi=i; for(int j=i+1;j<m;j++) if(cand[j].d<cand[mi].d) mi=j; He t=cand[i];cand[i]=cand[mi];cand[mi]=t; }
    int no=0;
    for(int i=0;i<m&&no<M;i++){ uint32_t e=cand[i].id; int good=1;
        for(int j=0;j<no;j++){ if(vdist(v,vrow(v,e),vrow(v,out[j]))<cand[i].d){ good=0; break; } }
        if(good) out[no++]=e; }
    for(int i=0;i<m&&no<M;i++){ uint32_t e=cand[i].id; int dup=0; for(int j=0;j<no;j++) if(out[j]==e){dup=1;break;} if(!dup) out[no++]=e; }
    free(cand); return no;
}

/* publish a neighbour list: entries first, then release-store the count, so a
 * concurrent lock-free reader never sees a count larger than valid entries. */
static inline void publish(uint32_t *lk, const uint32_t *nbs, int n){
    for(int i=0;i<n;i++) lk[i+1]=nbs[i];
    __atomic_store_n(&lk[0], (uint32_t)n, __ATOMIC_RELEASE);
}

static void connect(KciVec *v, uint32_t cur, uint32_t *nbs, int n, int layer){
    int Mx = layer==0? v->M0 : v->M;
    nlock(v,cur);
    uint32_t *cl=g_links(v,cur,layer); uint32_t have=cl[0];
    if(have==0){
        publish(cl, nbs, n);
    } else {                                   /* concurrent back-links present → merge, don't clobber */
        Heap W; hp_init(&W, have+n+4, 1);
        for(uint32_t e=1;e<=have;e++) hp_push(&W, vdist(v,vrow(v,cur),vrow(v,cl[e])), cl[e]);
        for(int k=0;k<n;k++){
            int dup=0; for(uint32_t e=1;e<=have;e++) if(cl[e]==nbs[k]){ dup=1; break; }
            if(!dup) hp_push(&W, vdist(v,vrow(v,cur),vrow(v,nbs[k])), nbs[k]);
        }
        uint32_t sel[512]; int ns=select_neighbors(v,vrow(v,cur),&W,Mx,sel);
        publish(cl, sel, ns);
        free(W.a);
    }
    nunlock(v,cur);
    for(int i=0;i<n;i++){
        uint32_t nb=nbs[i];
        nlock(v,nb);
        uint32_t *nl=g_links(v,nb,layer); uint32_t cc=nl[0];
        int has=0; for(uint32_t e=1;e<=cc;e++) if(nl[e]==cur){has=1;break;}
        if(!has){
            if(cc<(uint32_t)Mx){ nl[cc+1]=cur; __atomic_store_n(&nl[0], cc+1, __ATOMIC_RELEASE); }
            else {
                Heap W; hp_init(&W,cc+2,1);
                for(uint32_t e=1;e<=cc;e++) hp_push(&W, vdist(v,vrow(v,nb),vrow(v,nl[e])), nl[e]);
                hp_push(&W, vdist(v,vrow(v,nb),vrow(v,cur)), cur);
                uint32_t sel[512]; int ns=select_neighbors(v,vrow(v,nb),&W,Mx,sel);
                publish(nl, sel, ns);
                free(W.a);
            }
        }
        nunlock(v,nb);
    }
}

static void insert_node(KciVec *v, uint32_t i, Scr *s){
    const int8_t *q=vrow(v,i);
    pthread_mutex_lock(&v->epmtx);
    uint32_t ep=v->ep; int L=v->maxlevel;
    pthread_mutex_unlock(&v->epmtx);
    int lvl=v->level[i];
    for(int l=L;l>lvl;l--){ search_layer(v,q,ep,1,l,s); ep=hp_top(&s->W).id; }
    int top=lvl<L?lvl:L;
    uint32_t stride=(uint32_t)(v->M0>v->M?v->M0:v->M);
    uint32_t *sel=malloc((size_t)(top+1)*stride*sizeof(uint32_t));   /* NULL: connect as we go */
    int nsel[256];
    for(int l=top;l>=0;l--){
        search_layer(v,q,ep,v->efc,l,s);
        int Mx=l==0?v->M0:v->M;
        if(sel) nsel[l]=select_neighbors(v,q,&s->W,Mx,sel+(size_t)l*stride);
        else { uint32_t one[512]; int ns=select_neighbors(v,q,&s->W,Mx,one); connect(v,i,one,ns,l); }
        { const He *wa=s->W.a; int bi=0; for(int z=1;z<s->W.n;z++) if(wa[z].d<wa[bi].d) bi=z; if(s->W.n>0) ep=wa[bi].id; }
    }
    if(sel){ for(int l=0;l<=top;l++) connect(v,i,sel+(size_t)l*stride,nsel[l],l); free(sel); }
    if(lvl>L){
        pthread_mutex_lock(&v->epmtx);
        if(lvl>v->maxlevel){ v->maxlevel=lvl; v->ep=i; }
        pthread_mutex_unlock(&v->epmtx);
    }
}

typedef struct { KciVec *v; } BuildArg;
static void* build_worker(void *arg){
    KciVec *v=((BuildArg*)arg)->v;
    Scr s; scr_init(&s, v->gN, v->efc+16);
    for(;;){
        uint32_t i=__atomic_fetch_add(&v->bnext,1,__ATOMIC_RELAXED);
        if(i>=v->gN) break;
        if(i==0) continue;            /* node 0 seeds the graph as initial ep */
        insert_node(v,i,&s);
    }
    scr_free(&s);
    return NULL;
}

/* mark (seen=1) every node reachable on layer 0 from `from`, using queue q;
 * returns how many were newly marked */
static uint32_t reach_mark(KciVec *v, uint32_t from, uint8_t *seen, uint32_t *q){
    if(seen[from]) return 0;
    uint32_t h=0,t=0; q[t++]=from; seen[from]=1;
    while(h<t){ uint32_t *lk=g_l0(v,q[h++]);
        for(uint32_t e=1;e<=lk[0];e++) if(!seen[lk[e]]){ seen[lk[e]]=1; q[t++]=lk[e]; } }
    return t;
}
static uint32_t repair_reach(KciVec *v){
    uint32_t n=v->gN; if(n<2) return 0;
    uint8_t *seen=malloc(n); uint32_t *q=malloc((size_t)n*sizeof(uint32_t));
    if(!seen||!q){ free(seen); free(q); return n; }
    Scr s; scr_init(&s, n, v->efc+16);
    uint32_t left=0;
    for(int round=0; round<8; round++){
        memset(seen,0,n);
        left = n - reach_mark(v, v->ep, seen, q);
        if(left==0 || round==7) break;
        for(uint32_t u=0;u<n && left;u++){
            if(seen[u]) continue;
            const int8_t *x=vrow(v,u); uint32_t ep=v->ep;
            for(int l=v->maxlevel;l>0;l--){ search_layer(v,x,ep,1,l,&s); ep=hp_top(&s.W).id; }
            search_layer(v,x,ep,v->efc,0,&s);
            uint32_t best=UINT32_MAX; float bd=0;
            for(int i=0;i<s.W.n;i++){ He c=s.W.a[i];
                if(c.id!=u && seen[c.id] && (best==UINT32_MAX || c.d<bd)){ best=c.id; bd=c.d; } }
            if(best==UINT32_MAX) continue;
            uint32_t *lk=g_l0(v,best);
            if(lk[0]<(uint32_t)v->M0){ lk[lk[0]+1]=u; lk[0]++; }
            else { uint32_t farthest=1; float fd=vdist(v,vrow(v,best),vrow(v,lk[1]));
                for(uint32_t e=2;e<=lk[0];e++){ float d=vdist(v,vrow(v,best),vrow(v,lk[e])); if(d>fd){ fd=d; farthest=e; } }
                lk[farthest]=u; }
            left -= reach_mark(v, u, seen, q);              /* u and all it reaches */
        }
        /* next round re-counts: a replaced edge may have cut another node off */
    }
    scr_free(&s); free(seen); free(q);
    return left;
}

static int reorder_bfs(KciVec *v){
    uint32_t n=v->gN;
    if(n<2 || v->count!=n || !v->built || v->vf_mmap) return -1;
    uint32_t *order=malloc((size_t)n*4), *pos=malloc((size_t)n*4); uint8_t *seen=calloc(n,1);
    int8_t *nq=malloc((size_t)v->cap*v->dim);
    float *nf=v->vf ? malloc((size_t)v->cap*v->dim*sizeof(float)) : NULL;
    uint64_t *nid=malloc((size_t)v->cap*sizeof(uint64_t));
    uint8_t *nlv=malloc(v->gcap), *ndd=malloc(v->gcap);
    uint32_t *nl0=malloc((size_t)v->gcap*(v->M0+1)*sizeof(uint32_t));
    uint32_t **nlu=calloc(v->gcap,sizeof(uint32_t*));
    if(!order||!pos||!seen||!nq||(v->vf&&!nf)||!nid||!nlv||!ndd||!nl0||!nlu){
        free(order);free(pos);free(seen);free(nq);free(nf);free(nid);free(nlv);free(ndd);free(nl0);free(nlu); return -1; }
    uint32_t h=0,t=0; order[t++]=v->ep; seen[v->ep]=1;
    while(h<t){ uint32_t *lk=g_l0(v,order[h++]);
        for(uint32_t e=1;e<=lk[0];e++){ uint32_t nb=lk[e]; if(nb<n && !seen[nb]){ seen[nb]=1; order[t++]=nb; } } }
    for(uint32_t i=0;i<n;i++) if(!seen[i]) order[t++]=i;   /* unreachable (should be none) */
    for(uint32_t i=0;i<n;i++) pos[order[i]]=i;
    memset(nl0+(size_t)n*(v->M0+1),0,(size_t)(v->gcap-n)*(v->M0+1)*sizeof(uint32_t));   /* room for inserts */
    memset(nlv+n,0,v->gcap-n); memset(ndd+n,0,v->gcap-n);
    size_t D=(size_t)v->dim;
    for(uint32_t i=0;i<n;i++){ uint32_t o=order[i];
        memcpy(nq+i*D, v->vq+o*D, D);
        if(nf) memcpy(nf+i*D, v->vf+o*D, D*sizeof(float));
        nid[i]=v->vid[o]; nlv[i]=v->level[o]; ndd[i]=v->dead[o]; nlu[i]=v->lu[o];
        const uint32_t *src=g_l0(v,o); uint32_t *dst=nl0+(size_t)i*(v->M0+1);
        dst[0]=src[0]; for(uint32_t e=1;e<=src[0];e++) dst[e]=pos[src[e]];
        for(int l=1;l<=nlv[i];l++){ uint32_t *u=nlu[i]+(size_t)(l-1)*(v->M+1);
            for(uint32_t e=1;e<=u[0];e++) u[e]=pos[u[e]]; }
    }
    kv_vfree(v, v->vq, (size_t)v->cap*v->dim); v->vq=nq;
    if(nf){ kv_vfree(v, v->vf, (size_t)v->cap*v->dim*sizeof(float)); v->vf=nf; }
    free(v->vid); v->vid=nid;
    kv_wipe(v->level, v->gcap); free(v->level); v->level=nlv; kv_wipe(v->dead, v->gcap); free(v->dead); v->dead=ndd;
    kv_wipe(v->l0, (size_t)v->gcap*(v->M0+1)*sizeof(uint32_t)); free(v->l0); v->l0=nl0;
    kv_wipe(v->lu, (size_t)v->gcap*sizeof(uint32_t*)); free(v->lu); v->lu=nlu;
    v->ep=pos[v->ep];
    h_rebuild(v, v->hbits);
    free(order); free(pos); free(seen);
    return 0;
}

/* ── public: lifecycle / add ────────────────────────────────────────────── */
static void graph_free(KciVec *v){
    kv_qg_free(v->qg); v->qg=NULL;
    if(v->lu){ for(uint32_t i=0;i<v->gN;i++) if(v->lu[i]){
                   kv_wipe(v->lu[i], (size_t)(v->level? v->level[i] : 0)*(v->M+1)*sizeof(uint32_t)); free(v->lu[i]); }
               kv_wipe(v->lu, (size_t)v->gcap*sizeof(uint32_t*)); free(v->lu); v->lu=NULL; }
    kv_wipe(v->level, v->gcap); free(v->level); v->level=NULL;
    kv_wipe(v->l0, (size_t)v->gcap*(v->M0+1)*sizeof(uint32_t)); free(v->l0); v->l0=NULL;
    kv_wipe(v->lock, v->gcap); free(v->lock); v->lock=NULL;
    kv_wipe(v->dead, v->gcap); free(v->dead); v->dead=NULL; v->ndead=0;
    free(v->hkey); v->hkey=NULL; free(v->hval); v->hval=NULL; v->hbits=0;
    v->built=0; v->gN=0; v->gcap=0;
}

KciVec *kci_vec_new(int dim, int M, int efc){
    if (dim <= 0) return NULL;   /* any dim: the SDOT kernels handle the remainder with a scalar tail */
    if (M <= 0) M = 16;
    if (efc <= 0) efc = 200;
    if (M < 4) M = 4; if (M > 64) M = 64;
    if (efc < M) efc = M;
    KciVec *v = calloc(1, sizeof(KciVec));
    if (!v) return NULL;
    v->dim=dim; v->M=M; v->M0=2*M; v->efc=efc; v->mL=1.0/log((double)M);
    v->building=1;
    pthread_mutex_init(&v->epmtx, NULL);
    return v;
}

/* release the float store, whether malloc'd or mmap-backed */
static void vf_free(KciVec *v){
    if(v->vf_mmap){
#ifndef _WIN32
        if(v->vf) munmap(v->vf, v->vf_maplen);
        if(v->vf_fd>=0) close(v->vf_fd);
#else
        if(v->vf) munmap(v->vf, v->vf_maplen);
        if(v->vf_fd>=0) close(v->vf_fd);
#endif
        v->vf_mmap=0; v->vf_maplen=0; v->vf_fd=-1;
    } else {
        free(v->vf);
    }
    v->vf=NULL;
}

void kci_vec_free(KciVec *v){
    if(!v) return;
    if(v->img){
        kv_ivf_free(v->ivf); kv_qg_free(v->qg);
        if(v->img_mmap) munmap((void*)v->img, v->imglen);
        pthread_mutex_destroy(&v->epmtx);
        free(v); return; }
    graph_free(v);
    kv_ivf_free(v->ivf);
    kv_vfree(v, v->vq, (size_t)v->cap * v->dim); v->vq = NULL;
    if(v->vf && !v->vf_mmap){ kv_wipe(v->vf, (size_t)v->cap * v->dim * sizeof(float)); if(v->protect) kv_unlock_region(v->vf, (size_t)v->cap * v->dim * sizeof(float)); }
    vf_free(v); free(v->vid);
    if(v->ffree) v->ffree(v->fctx);
    pthread_mutex_destroy(&v->epmtx);
    free(v);
}

/* Move the retained float store to a disk file and mmap it read-only, so the
 * resident set stays int8 (codes + graph) and only the float rows actually touched
 * by re-rank fault in. Call once after build (with kci_vec_keep_float). The float
 * store then becomes read-only: no insert/compact afterwards. Returns 0 on success. */
int kci_vec_float_mmap(KciVec *v, const char *path){
    if(!v || !v->built || !v->keepf || !v->vf || v->vf_mmap || v->img) return -1;
    size_t n = (size_t)v->count * v->dim * sizeof(float);
    FILE *f=fopen(path,"wb"); if(!f) return -1;
    if(fwrite(v->vf,1,n,f)!=n){ fclose(f); remove(path); return -1; }
    fclose(f);
    int fd=open(path, O_RDONLY); if(fd<0) return -1;
    void *m=mmap(NULL, n, PROT_READ, MAP_SHARED, fd, 0);
    if(!m || m==(void*)-1){ close(fd); return -1; }
    free(v->vf);
    v->vf=(float*)m; v->vf_mmap=1; v->vf_maplen=n; v->vf_fd=fd;
    return 0;
}

int kci_vec_set_params(KciVec *v, int M, int efc){
    if(!v) return -1;
    if(M>0){ if(M<4)M=4; if(M>64)M=64; v->M=M; v->M0=2*M; v->mL=1.0/log((double)M); }
    if(efc>0) v->efc=efc;
    if(v->efc<v->M) v->efc=v->M;
    return 0;
}

int kci_vec_reserve(KciVec *v, uint32_t need){
    if(!v || v->img) return -1;
    if(need <= v->cap) return 0;
    uint32_t nc = v->cap ? v->cap : 1024;
    while(nc < need) nc *= 2;
    int wantf = (v->building || v->keepf) && !v->vf_mmap && !v->ffn;
    size_t nq_b = (size_t)nc * v->dim, nf_b = nq_b * sizeof(float);
    int8_t   *nq = kv_valloc(v, nq_b);
    uint64_t *ni = malloc((size_t)nc * sizeof(uint64_t));
    float    *nf = wantf ? kv_valloc(v, nf_b) : NULL;
    if(!nq || !ni || (wantf && !nf)){ free(nq); free(ni); free(nf); return -1; }   /* old buffers untouched */
    size_t oq_b = (size_t)v->cap * v->dim;
    if(v->vq){ memcpy(nq, v->vq, (size_t)v->count * v->dim); kv_vfree(v, v->vq, oq_b); }
    if(v->vid){ memcpy(ni, v->vid, (size_t)v->count * sizeof(uint64_t)); free(v->vid); }
    if(wantf){
        if(v->vf){ memcpy(nf, v->vf, (size_t)v->count * v->dim * sizeof(float)); kv_vfree(v, v->vf, oq_b * sizeof(float)); }
        v->vf = nf;
        if(v->protect) kv_protect_region(nf, nf_b);
    }
    v->vq = nq; v->vid = ni;
    if(v->protect) kv_protect_region(nq, nq_b);
    v->cap = nc;
    return 0;
}


/* Retain the unit-norm float vectors after build to enable full-precision
 * re-ranking. Otherwise they are freed at build (pure SQ8 codes remain). */
int kci_vec_keep_float(KciVec *v, int on){
    if(!v || v->count>0 || v->img) return -1;
    v->keepf = on?1:0;
    return 0;
}


/* store a unit-normalized copy of `src` at row `i` of v->vf (training buffer) */
static void store_float_norm(KciVec *v, uint32_t i, const float *src){
    float *r = v->vf + (size_t)i*v->dim;
    l2norm(v->dim, src, r);
}

int64_t kci_vec_add_i8(KciVec *v, uint64_t id, const int8_t *vec){
    if(!v) return -1;
    if(kci_vec_reserve(v, v->count+1) != 0) return -1;
    /* legacy symmetric int8 input → reconstruct float into the training buffer */
    float *dst = v->vf + (size_t)v->count*v->dim;
    for(int d=0;d<v->dim;d++) dst[d]=(float)vec[d]/127.0f;
    v->vid[v->count] = id;
    if(v->built) graph_free(v);
    return ++v->count;
}

int64_t kci_vec_add(KciVec *v, uint64_t id, const float *vec){
    if(!v) return -1;
    if(kci_vec_reserve(v, v->count+1) != 0) return -1;
    store_float_norm(v, v->count, vec);   /* unit-norm float → training buffer */
    v->vid[v->count] = id;
    if(v->built) graph_free(v);
    return ++v->count;
}

/* global scalar-quantization training: a single scale gs = 127/max|x| over all
 * unit-norm training vectors, so codes use the full signed-int8 range. One scale
 * (not per-dim) is what lets the distance be a pure integer dot. */
static int cmpf(const void *a, const void *b){ float x=*(const float*)a,y=*(const float*)b; return x<y?-1:(x>y?1:0); }
static int sq_train(KciVec *v){
    int dim=v->dim; uint32_t N=v->count;
    float *mx = malloc((size_t)N*sizeof(float));
    if(!mx) return -1;
    for(uint32_t i=0;i<N;i++){ const float *r=v->vf+(size_t)i*dim; float m=0.0f;
        for(int d=0;d<dim;d++){ float a=fabsf(r[d]); if(a>m)m=a; } mx[i]=m; }
    qsort(mx, N, sizeof(float), cmpf);
    float p = mx[(uint32_t)(0.99*(double)(N-1))];
    free(mx);
    v->gs = (p>1e-12f) ? 127.0f/p : 127.0f;
    return 0;
}

/* ── public: build ──────────────────────────────────────────────────────── */
int kci_vec_build(KciVec *v, int threads, double *out_secs){
    if(!v || v->count==0 || !v->vf || v->img) return -1;
    graph_free(v);
    kv_ivf_free(v->ivf); v->ivf=NULL;
    uint32_t N = v->count;
    v->gN = N;
    /* global SQ: pick one scale, encode every vector to signed int8 codes (vq). */
    if(sq_train(v)!=0) return -1;
    for(uint32_t i=0;i<N;i++) sq_encode(v, v->vf+(size_t)i*v->dim, v->vq+(size_t)i*v->dim);
    v->level = calloc(N, 1);
    v->l0    = calloc((size_t)N*(v->M0+1), sizeof(uint32_t));
    v->lu    = calloc(N, sizeof(uint32_t*));
    v->lock  = calloc(N, 1);
    v->dead  = calloc(N, 1);
    if(!v->level || !v->l0 || !v->lu || !v->lock || !v->dead){ graph_free(v); return -1; }
    v->gcap=N; v->ndead=0;
    srand(42);
    for(uint32_t i=0;i<N;i++){
        double r=(double)(rand()+1)/((double)RAND_MAX+2.0);
        int lvl=(int)(-log(r)*v->mL); if(lvl>255)lvl=255; v->level[i]=(uint8_t)lvl;
        if(lvl>0){ v->lu[i]=calloc((size_t)lvl*(v->M+1),sizeof(uint32_t)); if(!v->lu[i]){ graph_free(v); return -1; } }
    }
    v->ep=0; v->maxlevel=v->level[0]; v->bnext=1;
    int T = default_threads(threads);
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    pthread_t th[64]; BuildArg ba = { v };
    for(int t=0;t<T-1;t++) pthread_create(&th[t],NULL,build_worker,&ba);
    build_worker(&ba);
    for(int t=0;t<T-1;t++) pthread_join(th[t],NULL);
    repair_reach(v);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    v->built=1;
    uint32_t bits=1; while((1u<<bits) < N*2u) bits++;
    h_rebuild(v, bits);
    /* drop training floats unless retained for full-precision re-ranking */
    if(!v->keepf){ kv_vfree(v, v->vf, (size_t)v->cap*v->dim*sizeof(float)); v->vf=NULL; }
    reorder_bfs(v);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    if(out_secs) *out_secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    return 0;
}

/* ── public: search ─────────────────────────────────────────────────────── */
static int search_collect(KciVec *v, const int8_t *q, int ef, Scr *s){
    if(scr_fit(s, v->gN)!=0) return 0;
    if(s->use_qg && v->qg && s->qf) return kv_qg_collect(v, s, q, ef, 1);
    Heap *W=&s->W;
    uint32_t ep=v->ep; int L=v->maxlevel;
    for(int l=L;l>0;l--){ search_layer(v,q,ep,1,l,s); ep=hp_top(W).id; }
    search_layer(v,q,ep,ef,0,s);
    int m=W->n;
    if(m>s->tmpcap){ s->tmpcap=m; s->tmp=realloc(s->tmp,sizeof(He)*s->tmpcap); }
    memcpy(s->tmp, W->a, sizeof(He)*(size_t)m);
    return m;
}
/* Move the `n` smallest of t[0..m) to t[0..n) (any order): quickselect, O(m). */
static void select_smallest(He *t, int m, int n){
    int lo=0, hi=m-1;
    if(n<=0 || n>=m) return;
    while(lo<hi){
        float p=t[lo+(hi-lo)/2].d; int i=lo, j=hi;
        while(i<=j){ while(t[i].d<p) i++; while(t[j].d>p) j--;
            if(i<=j){ He x=t[i]; t[i]=t[j]; t[j]=x; i++; j--; } }
        if(n-1<=j) hi=j; else if(n-1>=i) lo=i; else break;
    }
}
static void select_smallest_f(float *t, int m, int n){
    int lo=0, hi=m-1;
    if(n<=0 || n>=m) return;
    while(lo<hi){
        float p=t[lo+(hi-lo)/2]; int i=lo, j=hi;
        while(i<=j){ while(t[i]<p) i++; while(t[j]>p) j--;
            if(i<=j){ float x=t[i]; t[i]=t[j]; t[j]=x; i++; j--; } }
        if(n-1<=j) hi=j; else if(n-1>=i) lo=i; else break;
    }
}
static int emit_topk(KciVec *v, He *t, int m, int k, float scale, uint64_t *out_ids, float *out_scores){
    int live=m;
    if(v->dead){ live=0; for(int i=0;i<m;i++) if(!v->dead[t[i].id]) t[live++]=t[i]; }
    if(live>k){ select_smallest(t, live, k); live=k; }
    for(int i=live/2-1;i>=0;i--){ int j=i;                                        /* heapify (max on d) ... */
        for(;;){ int l=2*j+1,r=l+1,mx=j; if(l<live&&t[l].d>t[mx].d)mx=l; if(r<live&&t[r].d>t[mx].d)mx=r; if(mx==j)break; He y=t[j];t[j]=t[mx];t[mx]=y; j=mx; } }
    for(int i=live-1;i>0;i--){ He x=t[0]; t[0]=t[i]; t[i]=x; int j=0;             /* ... then heap sort: ascending d */
        for(;;){ int l=2*j+1,r=l+1,mx=j; if(l<i&&t[l].d>t[mx].d)mx=l; if(r<i&&t[r].d>t[mx].d)mx=r; if(mx==j)break; He y=t[j];t[j]=t[mx];t[mx]=y; j=mx; } }
    for(int i=0;i<live;i++){ out_ids[i]=v->vid[t[i].id]; if(out_scores) out_scores[i] = -t[i].d * scale; }
    return live;
}

static int search_with_scr(KciVec *v, const int8_t *q, int k, int ef, Scr *s,
                           uint64_t *out_ids, float *out_scores){
    int m = search_collect(v, q, ef, s);
    /* heap distance is -(integer dot); cosine ≈ dot/gs² (both unit-norm). */
    float invgs2 = (v->gs>0) ? 1.0f/(v->gs*v->gs) : 1.0f;
    return emit_topk(v, s->tmp, m, k, invgs2, out_ids, out_scores);
}

static inline float fdot(int dim, const float *a, const float *b){
    int i=0; float s;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 s0=_mm256_setzero_ps(), s1=_mm256_setzero_ps();
    for(; i+16<=dim; i+=16){
        s0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),   _mm256_loadu_ps(b+i),   s0);
        s1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8), _mm256_loadu_ps(b+i+8), s1);
    }
    if(i+8<=dim){ s0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), s0); i+=8; }
    s0=_mm256_add_ps(s0,s1);
    __m128 h=_mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0,1));
    h=_mm_add_ps(h,_mm_movehl_ps(h,h)); h=_mm_add_ss(h,_mm_shuffle_ps(h,h,1));
    s=_mm_cvtss_f32(h);
#elif defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t s0=vdupq_n_f32(0), s1=vdupq_n_f32(0);
    for(; i+8<=dim; i+=8){ s0=vfmaq_f32(s0,vld1q_f32(a+i),vld1q_f32(b+i)); s1=vfmaq_f32(s1,vld1q_f32(a+i+4),vld1q_f32(b+i+4)); }
    s=vaddvq_f32(vaddq_f32(s0,s1));
#else
    float p0=0,p1=0,p2=0,p3=0;
    for(; i+4<=dim; i+=4){ p0+=a[i]*b[i]; p1+=a[i+1]*b[i+1]; p2+=a[i+2]*b[i+2]; p3+=a[i+3]*b[i+3]; }
    s=(p0+p1)+(p2+p3);
#endif
    for(; i<dim; i++) s+=a[i]*b[i];
    return s;
}
/* encode a query: L2-normalize into tmp, then global SQ into signed int8 code. */
static void enc_query(const KciVec *v, const float *query, float *tmp, int8_t *code){
    l2norm(v->dim, query, tmp);
    sq_encode(v, tmp, code);
}

static int kv_search_(KciVec *v, const float *query, int k, int ef,
                      uint64_t *out_ids, float *out_scores){
    if(!v || !v->built) return -1;
    if(k<1) k=1; if(ef<k) ef=k;
    int8_t *q = malloc(v->dim);
    float *tmp = malloc((size_t)v->dim*sizeof(float));
    enc_query(v, query, tmp, q);
    Scr s; scr_init(&s, v->gN, ef+16);
    int no = search_with_scr(v, q, k, ef, &s, out_ids, out_scores);
    scr_free(&s); free(q); free(tmp);
    return no;
}
int kci_vec_search(KciVec *v, const float *query, int k, int ef,
                   uint64_t *out_ids, float *out_scores){
    return kv_search_(v, query, k, ef, out_ids, out_scores);
}

/* reusable per-thread searcher: amortizes the visited-set allocation */
struct KciVecSearcher { KciVec *v; Scr s; int8_t *q; float *qf; void *ivf; int rerank_i8; KciVecFilter filt; void *fctx; int dim; float gap; int minp; };
int kci_vec_search_ivf_filtered(KciVecSearcher *s, const float *query, int k, int nprobe, int rerank_n,
                                KciVecFilter filt, void *ctx, uint64_t *out_ids, float *out_scores){
    if(!s) return -1;
    s->filt=filt; s->fctx=ctx;
    int r=kci_vec_search_ivf(s, query, k, nprobe, rerank_n, out_ids, out_scores);
    s->filt=NULL; s->fctx=NULL;
    return r;
}
int kci_vec_searcher_set_rerank_int8(KciVecSearcher *s, int on){ if(!s) return -1; s->rerank_i8 = on?1:0; return 0; }
int kci_vec_searcher_set_eps(KciVecSearcher *s, float eps){ if(!s || !(eps==eps)) return -1; s->s.eps = eps>0 ? eps : 0; return 0; }
int kci_vec_searcher_set_qg(KciVecSearcher *s, int on){ if(!s) return -1; if(on && !s->v->qg) return -2; s->s.use_qg = on?1:0; return 0; }
int kci_vec_searcher_set_probe_gap(KciVecSearcher *s, float gap, int min_probe){
    if(!s || !(gap==gap)) return -1;
    s->gap = gap>0 ? gap : 0; s->minp = min_probe>1 ? min_probe : 1; return 0; }
KciVecSearcher *kci_vec_searcher_new(KciVec *v){
    if(!v) return NULL;
    KciVecSearcher *s = calloc(1, sizeof(KciVecSearcher));
    if(!s) return NULL;
    s->v = v;
    scr_init(&s->s, v->gN ? v->gN : v->count, v->efc+16);
    s->dim = v->dim;
    s->q = malloc(v->dim);
    s->qf = malloc((size_t)v->dim*sizeof(float));
    if(!s->s.vis || !s->q || !s->qf){ kci_vec_searcher_free(s); return NULL; }
    s->s.qf = s->qf;
    return s;
}
void kci_vec_searcher_free(KciVecSearcher *s){
    if(!s) return;
    kv_wipe(s->q, (size_t)s->dim); kv_wipe(s->qf, (size_t)s->dim*sizeof(float));
    scr_free(&s->s); free(s->q); free(s->qf);
    if(s->ivf) kv_ivf_scr_free(s->ivf);
    free(s);
}
static int kv_search_r_(KciVecSearcher *s, const float *query, int k, int ef,
                        uint64_t *out_ids, float *out_scores){
    if(!s || !s->v->built) return -1;
    if(k<1) k=1; if(ef<k) ef=k;
    enc_query(s->v, query, s->qf, s->q);
    return search_with_scr(s->v, s->q, k, ef, &s->s, out_ids, out_scores);
}

static int kv_search_rerank_(KciVecSearcher *s, const float *query, int k, int ef,
                             int rerank_n, uint64_t *out_ids, float *out_scores){
    if(!s || !s->v->built) return -1;
    KciVec *v=s->v;
    if(!v->keepf || !v->vf)                 /* no retained floats → plain SQ8 search */
        return kci_vec_search_r(s, query, k, ef, out_ids, out_scores);
    if(k<1) k=1;
    if(rerank_n<k) rerank_n=k;
    if(ef<rerank_n) ef=rerank_n;
    enc_query(v, query, s->qf, s->q);       /* s->qf = unit query; s->q = code */
    int m = (s->s.use_qg && v->qg) ? kv_qg_collect(v, &s->s, s->q, ef, 0) : search_collect(v, s->q, ef, &s->s);
    He *t=s->s.tmp;
    if(m>rerank_n){ select_smallest(t, m, rerank_n); m=rerank_n; }
    { size_t rb=(size_t)v->dim*sizeof(float);
      for(int i=0;i<m;i++){ const char *p=(const char*)(v->vf + (size_t)t[i].id*v->dim); for(size_t o=0;o<rb;o+=64) __builtin_prefetch(p+o); } }
    for(int i=0;i<m;i++) t[i].d = -fdot(v->dim, s->qf, v->vf + (size_t)t[i].id*v->dim);
    return emit_topk(v, t, m, k, 1.0f, out_ids, out_scores);   /* top-k live by float score */
}

/* ── filtered search: traverse through all nodes, collect only matching ──── */
typedef int (*KciVecFilter)(uint64_t id, void *ctx);
static inline int node_ok(KciVec *v, uint32_t id, KciVecFilter filt, void *ctx){
    if(v->dead && v->dead[id]) return 0;
    return filt ? filt(v->vid[id], ctx) : 1;
}
/* Route through the upper layers unfiltered, then explore the base layer pushing
 * every reachable node onto the frontier C but only filter-passing nodes into the
 * result heap W. `budget` caps base-layer distance evals (selective-filter guard).
 * Returns the matches (unsorted) in s->tmp; the caller selects the top-k. */
static int search_collect_filtered(KciVec *v, const int8_t *q, int ef, Scr *s,
                                   KciVecFilter filt, void *ctx, int budget){
    if(scr_fit(s, v->gN)!=0) return 0;
    Heap *C=&s->C, *W=&s->W; uint32_t ep=v->ep; int L=v->maxlevel;
    for(int l=L;l>0;l--){ search_layer(v,q,ep,1,l,s); ep=hp_top(W).id; }
    scr_next(s); C->n=0; W->n=0; int vis=0;
    const float gs2=v->gs*v->gs, k1=(s->eps>0) ? 1.0f+s->eps : 0.0f;
    float d0=vdist(v,q,vrow(v,ep)); visited(s,ep); vis++;
    hp_push(C,d0,ep);
    if(node_ok(v,ep,filt,ctx)) hp_push(W,d0,ep);
    while(C->n){
        He c=hp_top(C); hp_pop(C);
        if(c.d>sl_bound(W,ef,k1,gs2)) break;
        if(vis>=budget) break;
        uint32_t *lk=g_links(v,c.id,0);
        uint32_t cnt=__atomic_load_n(&lk[0], __ATOMIC_ACQUIRE);
        for(uint32_t e=1;e<=cnt;e++){ uint32_t nb=lk[e]; if(nb<v->gN) __builtin_prefetch(&s->vis[nb]); }
        float bound=sl_bound(W,ef,k1,gs2);
        for(uint32_t e=1;e<=cnt;e++){
            uint32_t nb=lk[e]; if(nb>=v->gN) continue;
            if(visited(s,nb)) continue;
            float d=vdist(v,q,vrow(v,nb)); vis++;
            if(d<bound){
                hp_push(C,d,nb);
                if((W->n<ef || d<hp_top(W).d) && node_ok(v,nb,filt,ctx)){ hp_push(W,d,nb); if(W->n>ef) hp_pop(W); bound=sl_bound(W,ef,k1,gs2); }
            }
            if(vis>=budget) break;
        }
    }
    int m=W->n;
    if(m>s->tmpcap){ s->tmpcap=m; s->tmp=realloc(s->tmp,sizeof(He)*s->tmpcap); }
    memcpy(s->tmp, W->a, sizeof(He)*(size_t)m);   /* unsorted: the caller selects top-k */
    return m;
}

static int kv_search_filtered_(KciVecSearcher *s, const float *query, int k, int ef,
                               int rerank_n, KciVecFilter filt, void *ctx,
                               uint64_t *out_ids, float *out_scores){
    if(!s || !s->v->built) return -1;
    if(!filt) return kci_vec_search_rerank(s, query, k, ef, rerank_n, out_ids, out_scores);
    KciVec *v=s->v; if(k<1)k=1; if(ef<k)ef=k;
    enc_query(v, query, s->qf, s->q);
    int budget = ef*64; if(budget<4096) budget=4096;
    int m = search_collect_filtered(v, s->q, ef, &s->s, filt, ctx, budget);
    He *t=s->s.tmp;
    int reranked = (v->keepf && v->vf && rerank_n>0);
    if(reranked) for(int i=0;i<m;i++) t[i].d = -fdot(v->dim, s->qf, v->vf + (size_t)t[i].id*v->dim);
    float invgs2 = (v->gs>0) ? 1.0f/(v->gs*v->gs) : 1.0f;
    int no=0;
    for(int slot=0; slot<m && no<k; slot++){
        int mi=slot; for(int j=slot+1;j<m;j++) if(t[j].d<t[mi].d) mi=j;
        He x=t[slot]; t[slot]=t[mi]; t[mi]=x;            /* W already filter+live only */
        out_ids[no]=v->vid[t[slot].id];
        if(out_scores) out_scores[no]= reranked ? -t[slot].d : -t[slot].d*invgs2;
        no++;
    }
    return no;
}

#define KV_IVF_T 0.2
#define KV_LOWDIM 32
#define KV_LOWDIM_T (35.0f/64.0f)
#define KV_LUT_TOP 63
#define KV_PF_NEXT 1024
#define KV_SEL_B 256
#define KV_SOAR_DEFAULT 0.0f

typedef struct { uint32_t n, capb; uint8_t *codes; uint32_t *ids; int own; int8_t *U; int32_t *usum; float uscale, cscale; } KvList;
struct KvIvf {
    int nlist, M, nlistp, D4;
    uint32_t nmax;
    int D;
    double eta;
    int dp; float avqt;
    float soar;
    int d1; float *R, *cent1;
    int lr; size_t bb;
    int ld, LD4;       /* dims the low-rank bases live in (d1 when the residuals are projected first, else dim) and ld/4 rounded up */
    uint8_t *arena; uint32_t *arena_ids;   /* all lists side by side as built / loaded (one walk per list) */
    float *cent;       /* nlist x dim, unit */
    float *pq;
    int8_t *c8;
    int32_t *c8sum;    /* 128 * sum of codes, per centroid */
    float c8scale;
    KvList *lists;
    int ro;
};

static void kv_ivf_free(struct KvIvf *f){
    if(!f) return;
    if(f->ro){ free(f->cent1); free(f->lists); free(f->c8); free(f->c8sum); free(f); return; }
    if(f->lists) for(int c=0;c<f->nlist;c++){ KvList *L=&f->lists[c];
        if(L->codes && L->capb) kv_wipe(L->codes, (size_t)L->capb * f->bb);
        if(L->own){ free(L->codes); free(L->ids); }
        if(L->U) kv_wipe(L->U, (size_t)(f->lr/16)*f->LD4*64);
        if(L->usum) kv_wipe(L->usum, (size_t)f->lr*sizeof(int32_t));
        free(L->U); free(L->usum); }
    free(f->arena); free(f->arena_ids);
    if(f->R) kv_wipe(f->R, (size_t)f->D*f->d1*sizeof(float));
    if(f->cent1) kv_wipe(f->cent1, (size_t)f->nlist*f->d1*sizeof(float));
    if(f->cent) kv_wipe(f->cent, (size_t)f->nlist*f->D*sizeof(float));
    if(f->pq) kv_wipe(f->pq, (size_t)(f->M+1)*32*sizeof(float));
    if(f->c8) kv_wipe(f->c8, (size_t)(f->nlistp/16)*f->D4*64);
    if(f->c8sum) kv_wipe(f->c8sum, (size_t)f->nlistp*sizeof(int32_t));
    free(f->R); free(f->cent1);
    free(f->lists); free(f->cent); free(f->pq); free(f->c8); free(f->c8sum); free(f);
}

/* tiny deterministic generator (no global rand state) */
static uint64_t kv_rng(uint64_t *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

/* parallel for over [0,n) with the library's thread count */
typedef struct { void (*fn)(void*,size_t,size_t); void *ctx; size_t n; int T, t; } KvPar;
static void *kv_par_run(void *a){ KvPar *p=a; p->fn(p->ctx, p->n*(size_t)p->t/p->T, p->n*(size_t)(p->t+1)/p->T); return NULL; }
static void kv_par(size_t n, int threads, void (*fn)(void*,size_t,size_t), void *ctx){
    int T=default_threads(threads); if((size_t)T>n) T=n?(int)n:1;
    pthread_t th[64]; KvPar p[64];
    for(int t=0;t<T;t++){ p[t]=(KvPar){fn,ctx,n,T,t}; if(t) pthread_create(&th[t],NULL,kv_par_run,&p[t]); }
    kv_par_run(&p[0]);
    for(int t=1;t<T;t++) pthread_join(th[t],NULL);
}

/* derived tables for scoring the centroids */
static int kv_ivf_derive(struct KvIvf *f, int D){
    f->nlistp=(f->nlist+15)/16*16; f->D4=(D+3)/4;
    f->c8=calloc((size_t)(f->nlistp/16)*f->D4*64,1);
    f->c8sum=calloc(f->nlistp,sizeof(int32_t));
    if(!f->c8||!f->c8sum) return -1;
    float amax=0; for(size_t i=0;i<(size_t)f->nlist*D;i++) if(fabsf(f->cent[i])>amax) amax=fabsf(f->cent[i]);
    f->c8scale = amax>0 ? 127.0f/amax : 1.0f;
    for(int c=0;c<f->nlist;c++){ int g=c/16, l=c%16;
        for(int d=0;d<D;d++){ float x=f->cent[(size_t)c*D+d];
            int r=(int)lrintf(x*f->c8scale); int8_t v8=(int8_t)(r<-127?-127:(r>127?127:r));
            f->c8[(((size_t)g*f->D4+d/4)*16+l)*4+(d&3)]=v8; f->c8sum[c]+=128*v8; } }
    return 0;
}

static void kv_encode_res(const float *pq, int M, int dp, double eta, const float *x, const float *ce, int D, uint8_t *cd){
#define KV_PQ(m,k,j) pq[(m)*32+2*(k)+(j)]
    float r[2*M+2]; double p=0;
    for(int m=0;m<M;m++){
        int ia=dp*m, ib=(dp==2 && ia+1<D) ? ia+1 : -1;
        float x0=x[ia]-ce[ia], x1= ib>=0 ? x[ib]-ce[ib] : 0;
        float best=1e30f; int bi=0;
        for(int k=0;k<16;k++){ float a=x0-KV_PQ(m,k,0), b=x1-KV_PQ(m,k,1), e=a*a+b*b; if(e<best){ best=e; bi=k; } }
        cd[m]=(uint8_t)bi; r[2*m]=x0-KV_PQ(m,bi,0); r[2*m+1]=x1-KV_PQ(m,bi,1);
        p+=r[2*m]*x[ia]+(ib>=0 ? r[2*m+1]*x[ib] : 0);
    }
    if(eta<=1.0) return;
    for(int round=0; round<3; round++) for(int m=0;m<M;m++){
        int ia=dp*m, ib=(dp==2 && ia+1<D) ? ia+1 : -1;
        float u0=x[ia], u1= ib>=0 ? x[ib] : 0;
        int cur=cd[m]; float o0=r[2*m]+KV_PQ(m,cur,0), o1=r[2*m+1]+KV_PQ(m,cur,1);
        double pb=p-(r[2*m]*u0+r[2*m+1]*u1), best=1e300; int bi=cur;
        for(int k=0;k<16;k++){ double e0=o0-KV_PQ(m,k,0), e1=o1-KV_PQ(m,k,1), pp=pb+e0*u0+e1*u1;
            double L=e0*e0+e1*e1+(eta-1)*pp*pp; if(L<best){ best=L; bi=k; } }
        cd[m]=(uint8_t)bi; r[2*m]=o0-KV_PQ(m,bi,0); r[2*m+1]=o1-KV_PQ(m,bi,1);
        p=pb+r[2*m]*u0+r[2*m+1]*u1;
    }
#undef KV_PQ
}
static void kv_proj(const struct KvIvf *f, int D, const float *x, float *restrict y){
    const int d1=f->d1; const float *R=f->R; int j0=0;
#if defined(__AVX2__) && defined(__FMA__)
    for(; j0+64<=d1; j0+=64){ __m256 a0=_mm256_setzero_ps(), a1=a0, a2=a0, a3=a0, a4=a0, a5=a0, a6=a0, a7=a0;
        for(int d=0;d<D;d++){ __m256 xd=_mm256_set1_ps(x[d]); const float *r=R+(size_t)d*d1+j0;
            a0=_mm256_fmadd_ps(xd,_mm256_loadu_ps(r),a0);    a1=_mm256_fmadd_ps(xd,_mm256_loadu_ps(r+8),a1);
            a2=_mm256_fmadd_ps(xd,_mm256_loadu_ps(r+16),a2); a3=_mm256_fmadd_ps(xd,_mm256_loadu_ps(r+24),a3);
            a4=_mm256_fmadd_ps(xd,_mm256_loadu_ps(r+32),a4); a5=_mm256_fmadd_ps(xd,_mm256_loadu_ps(r+40),a5);
            a6=_mm256_fmadd_ps(xd,_mm256_loadu_ps(r+48),a6); a7=_mm256_fmadd_ps(xd,_mm256_loadu_ps(r+56),a7); }
        _mm256_storeu_ps(y+j0,a0); _mm256_storeu_ps(y+j0+8,a1); _mm256_storeu_ps(y+j0+16,a2); _mm256_storeu_ps(y+j0+24,a3);
        _mm256_storeu_ps(y+j0+32,a4); _mm256_storeu_ps(y+j0+40,a5); _mm256_storeu_ps(y+j0+48,a6); _mm256_storeu_ps(y+j0+56,a7); }
    for(; j0+8<=d1; j0+=8){ __m256 a=_mm256_setzero_ps();
        for(int d=0;d<D;d++) a=_mm256_fmadd_ps(_mm256_set1_ps(x[d]),_mm256_loadu_ps(R+(size_t)d*d1+j0),a);
        _mm256_storeu_ps(y+j0,a); }
#endif
    for(int j=j0;j<d1;j++) y[j]=0;
    if(j0<d1) for(int d=0;d<D;d++){ float xd=x[d]; const float *r=R+(size_t)d*d1; for(int j=j0;j<d1;j++) y[j]+=xd*r[j]; }
}
/* codes of x in list c: on all dims, or (d1 > 0) on the projected residual */
static void kv_ivf_encode(const struct KvIvf *f, const float *x, int D, int c, uint8_t *cd){
    if(!f->d1){ kv_encode_res(f->pq, f->M, f->dp, f->eta, x, f->cent+(size_t)c*D, D, cd); return; }
    float y[f->d1]; kv_proj(f, D, x, y);
    kv_encode_res(f->pq, f->M, f->dp, f->eta, y, f->cent1+(size_t)c*f->d1, f->d1, cd);
}
/* write slot j of block blk: 4-bit codes cd (M pieces), or (lr > 0) the lr int8 coefficients co and their sum */
static void kv_block_put(const struct KvIvf *f, uint8_t *blk, uint32_t j, const uint8_t *cd, const int8_t *co){
    if(!f->lr){
        for(int m=0;m<f->M;m++){ uint8_t *byte=blk+(size_t)m*16+(j&15);
            *byte = j<16 ? (uint8_t)((*byte&0xF0)|cd[m]) : (uint8_t)((*byte&0x0F)|(cd[m]<<4)); }
    } else {
        int32_t s=0; for(int k=0;k<f->lr;k++){ ((int8_t*)blk)[((size_t)(k>>2)*32+j)*4+(k&3)]=co[k]; s+=co[k]; }
        ((int32_t*)(blk+(size_t)f->lr*32))[j]=s;
    }
}
static void kv_block_clear(const struct KvIvf *f, uint8_t *blk, uint32_t j){
    if(!f->lr){
        for(int m=0;m<f->M;m++){ uint8_t *byte=blk+(size_t)m*16+(j&15); *byte &= j<16 ? 0xF0 : 0x0F; }
    } else {
        for(int k=0;k<f->lr;k++) ((int8_t*)blk)[((size_t)(k>>2)*32+j)*4+(k&3)]=0;
        ((int32_t*)(blk+(size_t)f->lr*32))[j]=0;
    }
}
/* append row `row` with codes cd (or coefficients co) to list L */
static int kv_list_push(KvList *L, struct KvIvf *f, uint32_t row, const uint8_t *cd, const int8_t *co){
    size_t bb=f->bb; uint32_t slot=L->n, b=slot/32, j=slot%32;
    if(slot+1>f->nmax) f->nmax=slot+1;
    if(b>=L->capb){ uint32_t nc=L->capb?L->capb*2:1;
        if(!L->own){                             /* leaving the arena: own copy from now on */
            uint8_t *c8=malloc((size_t)nc*bb); uint32_t *ii=malloc((size_t)nc*32*sizeof(uint32_t));
            if(!c8||!ii){ free(c8); free(ii); return -1; }
            if(L->capb){ memcpy(c8,L->codes,(size_t)L->capb*bb); memcpy(ii,L->ids,(size_t)L->capb*32*sizeof(uint32_t)); }
            L->codes=c8; L->ids=ii; L->own=1;
        } else {
        uint8_t *nc8=malloc((size_t)nc*bb); uint32_t *ni=malloc((size_t)nc*32*sizeof(uint32_t));
        if(!nc8||!ni){ free(nc8); free(ni); return -1; }
        memcpy(nc8,L->codes,(size_t)L->capb*bb); memcpy(ni,L->ids,(size_t)L->capb*32*sizeof(uint32_t));
        kv_wipe(L->codes,(size_t)L->capb*bb); free(L->codes); free(L->ids);
        L->codes=nc8; L->ids=ni;
        }
        memset(L->codes+(size_t)L->capb*bb,0,(size_t)(nc-L->capb)*bb);
        memset(L->ids+(size_t)L->capb*32,0xFF,(size_t)(nc-L->capb)*32*sizeof(uint32_t));
        L->capb=nc; }
    L->ids[(size_t)b*32+j]=row;
    kv_block_put(f, L->codes+(size_t)b*bb, j, cd, co);
    L->n++;
    return 0;
}

/* lay the lists out side by side: counts[c] vectors in list c */
static int kv_ivf_arena(struct KvIvf *f, const uint32_t *counts){
    size_t nb=0; for(int c=0;c<f->nlist;c++) nb+=(counts[c]+31)/32;
    size_t abytes=(nb?nb:1)*f->bb; f->arena=NULL;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    { void *p=NULL; size_t al=(size_t)2<<20, sz=(abytes+al-1)/al*al;
      if(abytes>=al && posix_memalign(&p,al,sz)==0){ madvise(p,sz,MADV_HUGEPAGE); memset(p,0,abytes); f->arena=p; } }
#endif
    if(!f->arena) f->arena=calloc(nb?nb:1,f->bb);
    f->arena_ids=malloc((nb?nb:1)*32*sizeof(uint32_t));
    if(!f->arena||!f->arena_ids) return -1;
    memset(f->arena_ids,0xFF,(nb?nb:1)*32*sizeof(uint32_t));
    size_t off=0;
    for(int c=0;c<f->nlist;c++){ KvList *L=&f->lists[c]; uint32_t b=(counts[c]+31)/32;
        L->n=0; L->capb=b; L->own=0; L->codes=f->arena+off*f->bb; L->ids=f->arena_ids+off*32; off+=b; }
    return 0;
}

/* move every list into one fresh arena (after a load, or after many inserts) */
static int kv_ivf_pack(struct KvIvf *f){
    uint32_t *cnt=malloc(f->nlist*sizeof(uint32_t)); if(!cnt) return -1;
    KvList *old=malloc(f->nlist*sizeof(KvList)); if(!old){ free(cnt); return -1; }
    memcpy(old,f->lists,f->nlist*sizeof(KvList));
    for(int c=0;c<f->nlist;c++) cnt[c]=old[c].n;
    uint8_t *oa=f->arena; uint32_t *oi=f->arena_ids; f->arena=NULL; f->arena_ids=NULL;
    if(kv_ivf_arena(f,cnt)!=0){ free(f->arena); free(f->arena_ids); f->arena=oa; f->arena_ids=oi;
        memcpy(f->lists,old,f->nlist*sizeof(KvList)); free(old); free(cnt); return -1; }
    for(int c=0;c<f->nlist;c++){ uint32_t b=(cnt[c]+31)/32; KvList *L=&f->lists[c];
        if(b){ memcpy(L->codes,old[c].codes,(size_t)b*f->bb); memcpy(L->ids,old[c].ids,(size_t)b*32*sizeof(uint32_t)); }
        L->n=cnt[c];
        if(old[c].own){ free(old[c].codes); free(old[c].ids); } }
    free(oa); free(oi); free(old); free(cnt);
    return 0;
}

/* build helpers (parallel bodies) */
typedef struct { const float *X; size_t n; int D, k; const float *C; int *as; int *as2; float lam; } KvKm;
#define KV_SOAR_TOP 16
static int kv_soar_second(const float *x, int D, const float *C, const float *top_s, const int *top_c, int nt, int c1, float lam){
    float r1[D]; double n1=0; const float *ce=C+(size_t)c1*D;
    for(int d=0;d<D;d++){ r1[d]=x[d]-ce[d]; n1+=r1[d]*r1[d]; }
    if(n1<=0) return top_c[0]==c1 && nt>1 ? top_c[1] : top_c[0];
    double best=1e300; int bi=-1;
    for(int t=0;t<nt;t++){ int c=top_c[t]; if(c==c1) continue; const float *cc=C+(size_t)c*D;
        double pr=0; for(int d=0;d<D;d++) pr+=r1[d]*(x[d]-cc[d]);        /* r1 . r2 */
        double n2=2.0-2.0*top_s[t];                                        /* |x - c2|^2 for unit vectors */
        double L=n2+lam*pr*pr/n1; if(L<best){ best=L; bi=c; } }
    return bi<0 ? c1 : bi;
}
static void kv_km_assign(void *a, size_t lo, size_t hi){ KvKm *k=a;
    for(size_t i=lo;i<hi;i++){ const float *x=k->X+i*k->D;
        if(!k->as2){ float best=-1e30f; int bi=0;
            for(int c=0;c<k->k;c++){ float s=fdot(k->D,x,k->C+(size_t)c*k->D); if(s>best){best=s;bi=c;} } k->as[i]=bi; continue; }
        float ts[KV_SOAR_TOP]; int tc[KV_SOAR_TOP]; int nt=0; tc[0]=0;     /* the KV_SOAR_TOP best, sorted, best first */
        for(int c=0;c<k->k;c++){ float s=fdot(k->D,x,k->C+(size_t)c*k->D);
            if(nt<KV_SOAR_TOP || s>ts[nt-1]){ int j = nt<KV_SOAR_TOP ? nt++ : nt-1;
                while(j>0 && ts[j-1]<s){ ts[j]=ts[j-1]; tc[j]=tc[j-1]; j--; } ts[j]=s; tc[j]=c; } }
        k->as[i]=tc[0]; k->as2[i]= nt>1 ? kv_soar_second(x,k->D,k->C,ts,tc,nt,tc[0],k->lam) : tc[0]; } }
typedef struct { const struct KvIvf *f; const float *X; int D; const int *as; uint8_t *cv; } KvEnc;
static void kv_enc_run(void *a, size_t lo, size_t hi){ KvEnc *e=a;
    for(size_t i=lo;i<hi;i++) kv_ivf_encode(e->f, e->X+i*e->D, e->D, e->as[i], e->cv+i*e->f->M); }

/* spherical k-means on n unit rows, k centres */
static int kv_kmeans_sph(const float *X, size_t n, int D, int k, int iters, int threads, float *C){
    uint64_t s=0x9E3779B97F4A7C15ULL;
    for(int c=0;c<k;c++) memcpy(C+(size_t)c*D, X+(size_t)(kv_rng(&s)%n)*D, (size_t)D*sizeof(float));
    int *as=malloc(n*sizeof(int)); double *acc=malloc((size_t)k*D*sizeof(double)); int *cnt=malloc(k*sizeof(int));
    if(!as||!acc||!cnt){ free(as); free(acc); free(cnt); return -1; }
    KvKm km={X,n,D,k,C,as,NULL,0};
    for(int it=0; it<iters; it++){
        kv_par(n,threads,kv_km_assign,&km);
        memset(acc,0,(size_t)k*D*sizeof(double)); memset(cnt,0,k*sizeof(int));
        for(size_t i=0;i<n;i++){ int c=as[i]; cnt[c]++; for(int j=0;j<D;j++) acc[(size_t)c*D+j]+=X[i*D+j]; }
        for(int c=0;c<k;c++){
            if(!cnt[c]){ memcpy(C+(size_t)c*D, X+(size_t)(kv_rng(&s)%n)*D, (size_t)D*sizeof(float)); continue; }
            double t=0; for(int j=0;j<D;j++) t+=acc[(size_t)c*D+j]*acc[(size_t)c*D+j]; t=t>0?1.0/sqrt(t):0;
            for(int j=0;j<D;j++) C[(size_t)c*D+j]=(float)(acc[(size_t)c*D+j]*t); }
    }
    free(as); free(acc); free(cnt);
    return 0;
}
/* plain k-means on 2-d points, 16 centres */
static void kv_kmeans2(const float *P, size_t n, float *C, uint64_t seed){
    uint64_t s=seed|1; for(int c=0;c<16;c++){ size_t r=kv_rng(&s)%n; C[2*c]=P[2*r]; C[2*c+1]=P[2*r+1]; }
    for(int it=0; it<25; it++){ double a[32]={0}; int cnt[16]={0};
        for(size_t i=0;i<n;i++){ float x=P[2*i], y=P[2*i+1]; float best=1e30f; int bi=0;
            for(int c=0;c<16;c++){ float dx=x-C[2*c], dy=y-C[2*c+1], e=dx*dx+dy*dy; if(e<best){best=e;bi=c;} }
            cnt[bi]++; a[2*bi]+=x; a[2*bi+1]+=y; }
        for(int c=0;c<16;c++) if(cnt[c]){ C[2*c]=(float)(a[2*c]/cnt[c]); C[2*c+1]=(float)(a[2*c+1]/cnt[c]); } }
}

#define KV_PQ_SAMPLE 16384
typedef struct { const float *RS; size_t np, step; int RD, M; float *pq; int fail; int dp; } KvPqJob;
static void kv_pq_run(void *a, size_t lo, size_t hi){ KvPqJob *j=a;
    float *P2=malloc(j->np*2*sizeof(float)); if(!P2){ j->fail=1; return; }
    for(size_t m=lo;m<hi;m++){
        for(size_t i=0;i<j->np;i++){ const float *r=j->RS+(i*j->step)*(size_t)j->RD;
            if(j->dp==1){ P2[2*i]=r[m]; P2[2*i+1]=0; }
            else { P2[2*i]=r[2*m]; P2[2*i+1]=(2*(int)m+1<j->RD)? r[2*m+1] : 0; } }
        kv_kmeans2(P2, j->np, j->pq+m*32, 0xC0FFEEULL+m);
    }
    free(P2);
}

typedef struct { const float *X; int D; double *cov; } KvCov;
static void kv_cov_run(void *a, size_t lo, size_t hi){ KvCov *c=a; int D=c->D;
    double *acc=calloc((size_t)D*D,sizeof(double)); if(!acc) return;
    for(size_t i=lo;i<hi;i++){ const float *x=c->X+i*D;
        for(int p=0;p<D;p++){ double xp=x[p]; if(xp==0) continue; double *row=acc+(size_t)p*D; for(int q=p;q<D;q++) row[q]+=xp*x[q]; } }
    static pthread_mutex_t mu=PTHREAD_MUTEX_INITIALIZER; pthread_mutex_lock(&mu);
    for(size_t i=0;i<(size_t)D*D;i++) c->cov[i]+=acc[i];
    pthread_mutex_unlock(&mu); free(acc); }
static int kv_pca(const float *X, size_t n, int D, int d1, int threads, float *R){
    double *cov=calloc((size_t)D*D,sizeof(double)), *Q=malloc((size_t)D*d1*sizeof(double)), *Z=malloc((size_t)D*d1*sizeof(double));
    if(!cov||!Q||!Z){ free(cov); free(Q); free(Z); return -1; }
    KvCov kc={X,D,cov}; kv_par(n,threads,kv_cov_run,&kc);
    for(int p=0;p<D;p++) for(int q=0;q<p;q++) cov[(size_t)p*D+q]=cov[(size_t)q*D+p];   /* the upper half was accumulated */
    uint64_t s=0x5DEECE66DULL; for(size_t i=0;i<(size_t)D*d1;i++) Q[i]=((double)(kv_rng(&s)>>11)/9007199254740992.0)-0.5;
    for(int it=0; it<40; it++){
        for(int p=0;p<D;p++){ const double *row=cov+(size_t)p*D; double *z=Z+(size_t)p*d1;   /* Z = cov Q */
            for(int j=0;j<d1;j++) z[j]=0;
            for(int q=0;q<D;q++){ double cq=row[q]; const double *qq=Q+(size_t)q*d1; for(int j=0;j<d1;j++) z[j]+=cq*qq[j]; } }
        for(int j=0;j<d1;j++){                                                             /* Gram-Schmidt on the columns */
            for(int k=0;k<j;k++){ double dot=0; for(int p=0;p<D;p++) dot+=Z[(size_t)p*d1+j]*Z[(size_t)p*d1+k];
                for(int p=0;p<D;p++) Z[(size_t)p*d1+j]-=dot*Z[(size_t)p*d1+k]; }
            double nn=0; for(int p=0;p<D;p++) nn+=Z[(size_t)p*d1+j]*Z[(size_t)p*d1+j]; nn = nn>0 ? 1.0/sqrt(nn) : 0;
            for(int p=0;p<D;p++) Z[(size_t)p*d1+j]*=nn; }
        memcpy(Q,Z,(size_t)D*d1*sizeof(double));
    }
    for(size_t i=0;i<(size_t)D*d1;i++) R[i]=(float)Q[i];
    free(cov); free(Q); free(Z);
    return 0;
}

static void kv_lr_coef(const struct KvIvf *f, int D, const KvList *L, const float *r, int8_t *co){
    int lr=f->lr, LD4=f->LD4, ld=f->ld; double inv=1.0/((double)L->uscale*L->cscale);
    float rp[ld]; const float *rr=r;
    if(f->d1){ kv_proj(f, D, r, rp); rr=rp; }                  /* bases live in the projected space */
    for(int k=0;k<lr;k++){ int g=k/16, l=k%16; double s=0;
        for(int d=0;d<ld;d++) s+=(double)rr[d]*L->U[(((size_t)g*LD4+d/4)*16+l)*4+(d&3)];
        int q=(int)lrint(s*inv); co[k]=(int8_t)(q<-127?-127:(q>127?127:q)); }
}
/* build one low-rank list: residuals of its members -> top-lr directions (U, int8) -> int8 coefficients, written
 * straight into the list's blocks (the list was sized by kv_ivf_arena) */
typedef struct { struct KvIvf *f; const KciVec *v; const uint32_t *order, *start; } KvLrB;
static void kv_lr_build_run(void *a, size_t lo, size_t hi){ KvLrB *B=a; struct KvIvf *f=B->f; const KciVec *v=B->v;
    int D=v->dim, lr=f->lr, ld=f->ld, LD4=f->LD4;
    for(size_t c=lo;c<hi;c++){ KvList *L=&f->lists[c]; uint32_t nc=B->start[c+1]-B->start[c]; const uint32_t *rows=B->order+B->start[c];
        L->U=calloc((size_t)(lr/16)*LD4*64,1); L->usum=calloc(lr,sizeof(int32_t)); L->uscale=1; L->cscale=1; L->n=0;
        if(!L->U||!L->usum){ f->lr=-1; return; }                     /* flagged: the build fails */
        if(!nc) continue;
        float *Rm=malloc((size_t)nc*ld*sizeof(float)), *Uf=malloc((size_t)ld*lr*sizeof(float)), *W=malloc((size_t)nc*lr*sizeof(float));
        int8_t co[lr]; float r[D];
        if(!Rm||!Uf||!W){ free(Rm); free(Uf); free(W); f->lr=-1; return; }
        const float *ce=f->cent+(size_t)c*D;
        for(uint32_t i=0;i<nc;i++){ const float *x=v->vf+(size_t)rows[i]*D;
            for(int d=0;d<D;d++) r[d]=x[d]-ce[d];
            if(f->d1) kv_proj(f, D, r, Rm+(size_t)i*ld); else memcpy(Rm+(size_t)i*ld, r, (size_t)D*sizeof(float)); }
        if(kv_pca(Rm, nc, ld, lr, 1, Uf)!=0){ free(Rm); free(Uf); free(W); f->lr=-1; return; }
        float umax=0; for(size_t i=0;i<(size_t)ld*lr;i++) if(fabsf(Uf[i])>umax) umax=fabsf(Uf[i]);
        L->uscale = umax>0 ? 127.0f/umax : 1.0f;
        for(int k=0;k<lr;k++){ int g=k/16, l=k%16;
            for(int d=0;d<ld;d++){ int q=(int)lrintf(Uf[(size_t)d*lr+k]*L->uscale); int8_t u8=(int8_t)(q<-127?-127:(q>127?127:q));
                L->U[(((size_t)g*LD4+d/4)*16+l)*4+(d&3)]=u8; L->usum[k]+=128*u8; } }
        float wmax=0;
        for(uint32_t i=0;i<nc;i++){ const float *rr=Rm+(size_t)i*ld; float *w=W+(size_t)i*lr;
            for(int k=0;k<lr;k++){ double s=0; for(int d=0;d<ld;d++) s+=(double)rr[d]*Uf[(size_t)d*lr+k]; w[k]=(float)s; if(fabsf(w[k])>wmax) wmax=fabsf(w[k]); } }
        L->cscale = wmax>0 ? wmax/127.0f : 1.0f;
        for(uint32_t i=0;i<nc;i++){ const float *w=W+(size_t)i*lr;
            for(int k=0;k<lr;k++){ int q=(int)lrintf(w[k]/L->cscale); co[k]=(int8_t)(q<-127?-127:(q>127?127:q)); }
            kv_block_put(f, L->codes+(size_t)(i/32)*f->bb, i%32, NULL, co); L->ids[i]=rows[i]; }
        L->n=nc;
        free(Rm); free(Uf); free(W);
    }
}

static int kv_ensure_dead(KciVec *v){            /* tombstones for an index without the graph */
    if(v->built) return 0;                       /* the graph keeps them (gcap) */
    if(v->dead && v->gcap>=v->cap) return 0;
    uint32_t old=v->dead? v->gcap : 0;
    uint8_t *nd=malloc(v->cap); if(!nd) return -1;
    kv_move((void**)&v->dead, nd, old, v->cap, old); v->gcap=v->cap;
    return 0;
}

int kci_vec_build_ivf_p(KciVec *v, const KciVecIvfParams *P){
    if(!v || !v->vf || v->count==0 || v->vf_mmap || v->img) return -1;
    int nlist = P ? P->nlist : 0, threads = P ? P->threads : 0, d1 = P ? P->code_dims : 0, lr = P ? P->lowrank : 0; float soar = P ? P->soar : 0;
    int D=v->dim; size_t n=v->count;
    if(nlist<=0){ nlist=(int)sqrt((double)n)*2; if(nlist<1) nlist=1; }
    if((size_t)nlist>n) nlist=(int)n;
    if(nlist<2) soar=0;                                      /* no second list to spill to */
    if(lr<0) lr=0; if(lr){ lr=(lr+15)/16*16; if(lr>256) lr=256; if(lr>=D) lr=0; }
    if(d1<0 || d1>=D) d1=0; d1&=~1; if(d1 && d1<8) d1=8;
    if(lr && d1 && lr>=d1) lr=(d1-1)/16*16;                  /* low rank inside the projected space must be below it */
    if(lr && d1 && d1%4) d1&=~3;                             /* the int8 bases are read 4 dims at a time */
    struct KvIvf *f=calloc(1,sizeof *f); if(!f) return -1;
    f->D=D;
    int Dc = d1 ? d1 : D;                                    /* dims the codes describe */
    f->dp = v->ivf_dp ? v->ivf_dp : (Dc<=KV_LOWDIM ? 1 : 2);
    f->avqt = v->ivf_avqt>0 ? v->ivf_avqt : (Dc<=KV_LOWDIM ? KV_LOWDIM_T : (float)KV_IVF_T);
    f->nlist=nlist; f->M = lr ? 0 : (f->dp==2 ? (Dc+1)/2 : Dc); f->eta=(Dc-1)*(double)f->avqt*f->avqt/(1-(double)f->avqt*f->avqt);
    f->soar = soar>0 ? soar : 0; f->d1=d1;
    f->lr=lr; f->D4=(D+3)/4; f->bb = lr ? (size_t)lr*32+128 : (size_t)f->M*16;
    f->ld = d1 ? d1 : D; f->LD4=(f->ld+3)/4;                 /* low-rank bases live in the projected space when there is one (smaller U) */
    f->cent=malloc((size_t)nlist*D*sizeof(float)); f->pq=malloc((size_t)(f->M+1)*32*sizeof(float));
    f->lists=calloc(nlist,sizeof(KvList));
    size_t nsmax = v->ivf_km_sample ? v->ivf_km_sample : 250000; int km_iters = v->ivf_km_iters ? v->ivf_km_iters : 10;
    size_t ns = n<nsmax ? n : nsmax, step=n/ns;              /* training sample: every step-th vector */
    float *S=malloc(ns*D*sizeof(float)); int *as=malloc(n*sizeof(int));
    uint8_t *cv=malloc(n*(size_t)f->M+1);                    /* +1: M = 0 for low-rank lists */
    int *as2 = f->soar>0 ? malloc(n*sizeof(int)) : NULL; uint8_t *cv2 = f->soar>0 ? malloc(n*(size_t)f->M+1) : NULL;
    float *S1=NULL;                                          /* projected residuals of the sample (d1 > 0) */
    if(d1){ f->R=malloc((size_t)D*d1*sizeof(float)); f->cent1=malloc((size_t)nlist*d1*sizeof(float)); S1=malloc(ns*(size_t)d1*sizeof(float)); }
    int ok = f->cent && f->pq && f->lists && S && as && cv && (f->soar<=0 || (as2 && cv2)) && (!d1 || (f->R && f->cent1 && S1));
    if(ok){
        for(size_t i=0;i<ns;i++) memcpy(S+i*D, v->vf+(i*step)*D, (size_t)D*sizeof(float));
        ok = kv_kmeans_sph(S, ns, D, nlist, km_iters, threads, f->cent)==0;
    }
    if(ok){
        KvKm km={v->vf,n,D,nlist,f->cent,as,as2,f->soar}; kv_par(n,threads,kv_km_assign,&km);
        for(size_t i=0;i<ns;i++){ const float *ce=f->cent+(size_t)as[i*step]*D; float *s2=S+i*D; for(int d=0;d<D;d++) s2[d]-=ce[d]; }  /* S = residuals */
        if(d1){                                              /* basis from the residuals, projected centroids and residuals */
            ok = kv_pca(S, ns, D, d1, threads, f->R)==0;
            if(ok){ for(int c=0;c<nlist;c++) kv_proj(f, D, f->cent+(size_t)c*D, f->cent1+(size_t)c*d1);
                    for(size_t i=0;i<ns;i++) kv_proj(f, D, S+i*D, S1+i*d1); }
        }
        const float *RS = d1 ? S1 : S; int RD = Dc;        /* residual sample the codebooks are trained on */
        if(ok && f->M>0){
            size_t np = ns<KV_PQ_SAMPLE ? ns : KV_PQ_SAMPLE;
            KvPqJob pj={RS, np, ns/np, RD, f->M, f->pq, 0, f->dp};
            kv_par((size_t)f->M, threads, kv_pq_run, &pj);
            ok = !pj.fail;
        }
        if(ok && !lr){ KvEnc ke={f,v->vf,D,as,cv}; kv_par(n,threads,kv_enc_run,&ke); }
        if(ok && !lr && as2){ KvEnc ke2={f,v->vf,D,as2,cv2}; kv_par(n,threads,kv_enc_run,&ke2); }
        uint32_t *cnt=calloc(nlist,sizeof(uint32_t)); ok = ok && cnt!=NULL;
        if(ok){ for(size_t i=0;i<n;i++){ cnt[as[i]]++; if(as2) cnt[as2[i]]++; } ok = kv_ivf_arena(f,cnt)==0; }
        if(ok && lr){                                        /* members of each list, then one low-rank basis per list in parallel */
            size_t ne = as2 ? 2*n : n; uint32_t *start=calloc(nlist+1,sizeof(uint32_t)), *order=malloc(ne*sizeof(uint32_t)), *fill=calloc(nlist,sizeof(uint32_t));
            ok = start && order && fill;
            if(ok){ for(int c=0;c<nlist;c++) start[c+1]=start[c]+cnt[c];
                for(size_t i=0;i<n;i++){ order[start[as[i]]+fill[as[i]]++]=(uint32_t)i; if(as2) order[start[as2[i]]+fill[as2[i]]++]=(uint32_t)i; }
                KvLrB B={f,v,order,start}; kv_par(nlist,threads,kv_lr_build_run,&B); ok = f->lr>0; }
            free(start); free(order); free(fill); }
        free(cnt);
        if(!lr) for(size_t i=0;i<n && ok;i++){ ok = kv_list_push(&f->lists[as[i]], f, (uint32_t)i, cv+i*f->M, NULL)==0;
            if(ok && as2) ok = kv_list_push(&f->lists[as2[i]], f, (uint32_t)i, cv2+i*f->M, NULL)==0; }
        if(ok){ f->nmax=0; for(int c=0;c<nlist;c++) if(f->lists[c].n>f->nmax) f->nmax=f->lists[c].n; }
        if(ok && !v->built){ ok = sq_train(v)==0; if(ok) for(size_t i=0;i<n;i++) sq_encode(v, v->vf+i*D, v->vq+i*D); }
        ok = ok && kv_ivf_derive(f, D)==0 && kv_ensure_dead(v)==0;
        if(ok && !v->hkey){ uint32_t bits=1; while((1u<<bits) < v->count*2u) bits++; ok = h_rebuild(v,bits)==0; }
    }
    kv_wipe(S, ns*D*sizeof(float)); if(S1) kv_wipe(S1, ns*(size_t)d1*sizeof(float));
    kv_wipe(cv, n*(size_t)f->M+1); if(cv2) kv_wipe(cv2, n*(size_t)f->M+1);
    free(S); free(S1); free(as); free(cv); free(as2); free(cv2);
    if(!ok){ kv_ivf_free(f); return -1; }
    kv_ivf_free(v->ivf); v->ivf=f;
    return 0;
}
int kci_vec_set_ivf_codes(KciVec *v, int dims_per_code, float avq_threshold){
    if(!v || (dims_per_code!=0 && dims_per_code!=1 && dims_per_code!=2) || !(avq_threshold>=0) || !(avq_threshold<0.985f)) return -1;
    v->ivf_dp = dims_per_code;
    float t = avq_threshold>0 ? (float)lrintf(avq_threshold*64.0f)/64.0f : 0.0f;   /* the file keeps it in 1/64 steps */
    v->ivf_avqt = (avq_threshold>0 && t<=0) ? 1.0f/64.0f : t;
    return 0;
}
int kci_vec_set_ivf_training(KciVec *v, int iterations, uint32_t sample){
    if(!v || iterations<0 || iterations>1000) return -1;
    v->ivf_km_iters=iterations; v->ivf_km_sample=sample; return 0;
}
int kci_vec_build_ivf_ex(KciVec *v, int nlist, int threads, float soar){
    KciVecIvfParams p={nlist, threads, soar, 0, 0}; return kci_vec_build_ivf_p(v, &p); }
int kci_vec_build_ivf(KciVec *v, int nlist, int threads){ return kci_vec_build_ivf_ex(v, nlist, threads, KV_SOAR_DEFAULT); }

/* add row i (already stored in vf) to the lists (both lists when the index spills) */
static int kv_ivf_add_x(KciVec *v, uint32_t i, const float *x);
static int kv_ivf_add(KciVec *v, uint32_t i){ return kv_ivf_add_x(v, i, v->vf+(size_t)i*v->dim); }
static int kv_ivf_add_x(KciVec *v, uint32_t i, const float *x){
    struct KvIvf *f=v->ivf;
    int D=v->dim; uint8_t cd[f->M+1]; int8_t co[f->lr+1]; float r[D]; int c, c2;
    { int as, as2=-1; KvKm km={x,1,D,f->nlist,f->cent,&as, f->soar>0 ? &as2 : NULL, f->soar};
      kv_km_assign(&km,0,1); c=as; c2=as2; }
    for(int pass=0; pass<2; pass++){ int cc = pass ? c2 : c; if(pass && (c2<0 || c2==c)) break;
        if(f->lr){ const float *ce=f->cent+(size_t)cc*D; for(int d=0;d<D;d++) r[d]=x[d]-ce[d]; kv_lr_coef(f, D, &f->lists[cc], r, co); }
        else kv_ivf_encode(f,x,D,cc,cd);
        if(kv_list_push(&f->lists[cc], f, i, cd, co)!=0) return -1; }
    return 0;
}

/* searcher scratch for the lists */
typedef struct { int n; float *qc; He *lh, *lh2; uint8_t *lut; float *lutf; float *q1; size_t qcb, lutb, lutfb, q1b; He *sl; int slcap; } KvIvfScr;   /* q1: projected query (d1 > 0); *b: bytes, for the wipe */

static void kv_ivf_scr_free(void *p){ KvIvfScr *z=p; if(!z) return;
    kv_wipe(z->qc, z->qcb); kv_wipe(z->lut, z->lutb); kv_wipe(z->lutf, z->lutfb); kv_wipe(z->q1, z->q1b);
    if(z->sl) kv_wipe(z->sl, (size_t)z->slcap*sizeof(He));
    free(z->qc); free(z->lh); free(z->lh2); free(z->lut); free(z->lutf); free(z->q1); free(z->sl); free(z); }

typedef struct { He *a; int n, cap, R; float thr; } KvSl;
static void kv_sl_compact(KvSl *S){
    int n=S->n, R=S->R; He *a=S->a;
    if(n<=R) return;
    float lo, hi;
    { float l4[4], h4[4]; for(int u=0;u<4;u++){ l4[u]=h4[u]=a[0].d; }   /* 4 chains: min/max latency, not a long chain */
      int i=0; for(; i+4<=n; i+=4) for(int u=0;u<4;u++){ float d=a[i+u].d; l4[u] = d<l4[u] ? d : l4[u]; h4[u] = d>h4[u] ? d : h4[u]; }
      for(; i<n; i++){ float d=a[i].d; l4[0] = d<l4[0] ? d : l4[0]; h4[0] = d>h4[0] ? d : h4[0]; }
      lo=l4[0]; hi=h4[0]; for(int u=1;u<4;u++){ lo = l4[u]<lo ? l4[u] : lo; hi = h4[u]>hi ? h4[u] : hi; } }
    int keep;
    if(!(hi>lo)) keep=R;                                  /* all equal: any R */
    else {
        /* one level of 256 buckets, counted in 4 histograms (a run of equal buckets would chain the increments
         * through memory); everything up to the bucket holding the R-th is kept, a few more than R */
        float ks=255.0f/(hi-lo); uint32_t h[4][256]; uint8_t key[n];
        memset(h,0,sizeof h);
        for(int i=0;i<n;i++){ float t=(a[i].d-lo)*ks; uint32_t k = t>=255.0f ? 255u : (uint32_t)t; key[i]=(uint8_t)k; h[i&3][k]++; }
        uint32_t c=0, T=0; for(; T<256; T++){ c+=h[0][T]+h[1][T]+h[2][T]+h[3][T]; if(c>=(uint32_t)R) break; }
        /* thr = the top edge of bucket T (a hair above, for rounding), at most hi: not below the R-th kept, so a new
         * entry that would make the best R still enters; only a little looser than the worst kept. Was the max of
         * the kept, one compare chain over n */
        int w=0; float t = lo+(float)(T+1)/ks+(hi-lo)*1e-5f;
        t = t<hi ? t : hi;
        for(int i=0;i<n;i++){ He x=a[i]; a[w]=x; w += key[i]<=T; }
        if(w<=R+(S->cap-R)/2){ S->n=w; S->thr=t; return; }
        keep=R;                                          /* too many ties at the cut: exact, on the w kept (they hold the R smallest) */
        select_smallest(a, w, R);
    }
    S->n=keep; float t=a[0].d; for(int i=1;i<keep;i++) t = a[i].d>t ? a[i].d : t; S->thr=t;
}

#if defined(__AVX2__)
static const uint32_t kv_cmp8[256]={
    0x00000000u, 0x00000000u, 0x00000001u, 0x00000010u, 0x00000002u, 0x00000020u, 0x00000021u, 0x00000210u,
    0x00000003u, 0x00000030u, 0x00000031u, 0x00000310u, 0x00000032u, 0x00000320u, 0x00000321u, 0x00003210u,
    0x00000004u, 0x00000040u, 0x00000041u, 0x00000410u, 0x00000042u, 0x00000420u, 0x00000421u, 0x00004210u,
    0x00000043u, 0x00000430u, 0x00000431u, 0x00004310u, 0x00000432u, 0x00004320u, 0x00004321u, 0x00043210u,
    0x00000005u, 0x00000050u, 0x00000051u, 0x00000510u, 0x00000052u, 0x00000520u, 0x00000521u, 0x00005210u,
    0x00000053u, 0x00000530u, 0x00000531u, 0x00005310u, 0x00000532u, 0x00005320u, 0x00005321u, 0x00053210u,
    0x00000054u, 0x00000540u, 0x00000541u, 0x00005410u, 0x00000542u, 0x00005420u, 0x00005421u, 0x00054210u,
    0x00000543u, 0x00005430u, 0x00005431u, 0x00054310u, 0x00005432u, 0x00054320u, 0x00054321u, 0x00543210u,
    0x00000006u, 0x00000060u, 0x00000061u, 0x00000610u, 0x00000062u, 0x00000620u, 0x00000621u, 0x00006210u,
    0x00000063u, 0x00000630u, 0x00000631u, 0x00006310u, 0x00000632u, 0x00006320u, 0x00006321u, 0x00063210u,
    0x00000064u, 0x00000640u, 0x00000641u, 0x00006410u, 0x00000642u, 0x00006420u, 0x00006421u, 0x00064210u,
    0x00000643u, 0x00006430u, 0x00006431u, 0x00064310u, 0x00006432u, 0x00064320u, 0x00064321u, 0x00643210u,
    0x00000065u, 0x00000650u, 0x00000651u, 0x00006510u, 0x00000652u, 0x00006520u, 0x00006521u, 0x00065210u,
    0x00000653u, 0x00006530u, 0x00006531u, 0x00065310u, 0x00006532u, 0x00065320u, 0x00065321u, 0x00653210u,
    0x00000654u, 0x00006540u, 0x00006541u, 0x00065410u, 0x00006542u, 0x00065420u, 0x00065421u, 0x00654210u,
    0x00006543u, 0x00065430u, 0x00065431u, 0x00654310u, 0x00065432u, 0x00654320u, 0x00654321u, 0x06543210u,
    0x00000007u, 0x00000070u, 0x00000071u, 0x00000710u, 0x00000072u, 0x00000720u, 0x00000721u, 0x00007210u,
    0x00000073u, 0x00000730u, 0x00000731u, 0x00007310u, 0x00000732u, 0x00007320u, 0x00007321u, 0x00073210u,
    0x00000074u, 0x00000740u, 0x00000741u, 0x00007410u, 0x00000742u, 0x00007420u, 0x00007421u, 0x00074210u,
    0x00000743u, 0x00007430u, 0x00007431u, 0x00074310u, 0x00007432u, 0x00074320u, 0x00074321u, 0x00743210u,
    0x00000075u, 0x00000750u, 0x00000751u, 0x00007510u, 0x00000752u, 0x00007520u, 0x00007521u, 0x00075210u,
    0x00000753u, 0x00007530u, 0x00007531u, 0x00075310u, 0x00007532u, 0x00075320u, 0x00075321u, 0x00753210u,
    0x00000754u, 0x00007540u, 0x00007541u, 0x00075410u, 0x00007542u, 0x00075420u, 0x00075421u, 0x00754210u,
    0x00007543u, 0x00075430u, 0x00075431u, 0x00754310u, 0x00075432u, 0x00754320u, 0x00754321u, 0x07543210u,
    0x00000076u, 0x00000760u, 0x00000761u, 0x00007610u, 0x00000762u, 0x00007620u, 0x00007621u, 0x00076210u,
    0x00000763u, 0x00007630u, 0x00007631u, 0x00076310u, 0x00007632u, 0x00076320u, 0x00076321u, 0x00763210u,
    0x00000764u, 0x00007640u, 0x00007641u, 0x00076410u, 0x00007642u, 0x00076420u, 0x00076421u, 0x00764210u,
    0x00007643u, 0x00076430u, 0x00076431u, 0x00764310u, 0x00076432u, 0x00764320u, 0x00764321u, 0x07643210u,
    0x00000765u, 0x00007650u, 0x00007651u, 0x00076510u, 0x00007652u, 0x00076520u, 0x00076521u, 0x00765210u,
    0x00007653u, 0x00076530u, 0x00076531u, 0x00765310u, 0x00076532u, 0x00765320u, 0x00765321u, 0x07653210u,
    0x00007654u, 0x00076540u, 0x00076541u, 0x00765410u, 0x00076542u, 0x00765420u, 0x00765421u, 0x07654210u,
    0x00076543u, 0x00765430u, 0x00765431u, 0x07654310u, 0x00765432u, 0x07654320u, 0x07654321u, 0x76543210u,
};
#endif

/* scan one list into the shortlist S (entries d = -score, smallest R kept) */
static void kv_scan_list(const struct KvIvf *f, const KvList *Lst, const uint8_t *lut, float base, float invs, KvSl *S, uint32_t tag,
                         uint16_t *vis, uint16_t ep, KciVecFilter filt, void *fctx, const uint64_t *vid){
    int M=f->M; const float sc=1.0f/invs; (void)sc;   /* sc: block skip (SIMD paths) */
    uint32_t nb=(Lst->n+31)/32;
    for(uint32_t b=0;b<nb;b++){
        const uint8_t *blk=Lst->codes+(size_t)b*M*16;
#if defined(__AVX2__)
        __m256i aEl=_mm256_setzero_si256(), aOl=aEl, aEh=aEl, aOh=aEl;
        const __m256i m4=_mm256_set1_epi8(0x0F);
        int m=0;
        if(b+1<nb){ const char *nx=(const char*)(blk+M*16); for(int o=0;o<M*16;o+=64) __builtin_prefetch(nx+o); }
  #if defined(__AVX512BW__)
        { const __m512i z4=_mm512_set1_epi8(0x0F);
          __m512i bEl=_mm512_setzero_si512(), bOl=bEl, bEh=bEl, bOh=bEl;
          for(; m+16<=M; m+=16){
            __m512i vl=_mm512_setzero_si512(), vh=vl;
            for(int u=0;u<4;u++){ __m512i bb=_mm512_loadu_si512((const void*)(blk+m*16+64*u)), L=_mm512_loadu_si512((const void*)(lut+m*16+64*u));
                vl=_mm512_add_epi8(vl,_mm512_shuffle_epi8(L,_mm512_and_si512(bb,z4)));
                vh=_mm512_add_epi8(vh,_mm512_shuffle_epi8(L,_mm512_and_si512(_mm512_srli_epi16(bb,4),z4))); }
            bEl=_mm512_add_epi16(bEl,vl); bOl=_mm512_add_epi16(bOl,_mm512_srli_epi16(vl,8));
            bEh=_mm512_add_epi16(bEh,vh); bOh=_mm512_add_epi16(bOh,_mm512_srli_epi16(vh,8)); }
          for(; m+4<=M; m+=4){
            __m512i bb=_mm512_loadu_si512((const void*)(blk+m*16)), L=_mm512_loadu_si512((const void*)(lut+m*16));
            __m512i vl=_mm512_shuffle_epi8(L,_mm512_and_si512(bb,z4)), vh=_mm512_shuffle_epi8(L,_mm512_and_si512(_mm512_srli_epi16(bb,4),z4));
            bEl=_mm512_add_epi16(bEl,vl); bOl=_mm512_add_epi16(bOl,_mm512_srli_epi16(vl,8));
            bEh=_mm512_add_epi16(bEh,vh); bOh=_mm512_add_epi16(bOh,_mm512_srli_epi16(vh,8)); }
          aEl=_mm256_add_epi16(_mm512_castsi512_si256(bEl),_mm512_extracti64x4_epi64(bEl,1));
          aOl=_mm256_add_epi16(_mm512_castsi512_si256(bOl),_mm512_extracti64x4_epi64(bOl,1));
          aEh=_mm256_add_epi16(_mm512_castsi512_si256(bEh),_mm512_extracti64x4_epi64(bEh,1));
          aOh=_mm256_add_epi16(_mm512_castsi512_si256(bOh),_mm512_extracti64x4_epi64(bOh,1)); }
  #endif
        for(; m+8<=M; m+=8){       /* 4 piece pairs summed as bytes, then widened */
            __m256i vl=_mm256_setzero_si256(), vh=vl;
            for(int u=0;u<4;u++){ __m256i bb=_mm256_loadu_si256((const __m256i*)(blk+m*16+32*u)), L=_mm256_loadu_si256((const __m256i*)(lut+m*16+32*u));
                vl=_mm256_add_epi8(vl,_mm256_shuffle_epi8(L,_mm256_and_si256(bb,m4)));
                vh=_mm256_add_epi8(vh,_mm256_shuffle_epi8(L,_mm256_and_si256(_mm256_srli_epi16(bb,4),m4))); }
            aEl=_mm256_add_epi16(aEl,vl); aOl=_mm256_add_epi16(aOl,_mm256_srli_epi16(vl,8));
            aEh=_mm256_add_epi16(aEh,vh); aOh=_mm256_add_epi16(aOh,_mm256_srli_epi16(vh,8)); }
        for(; m+2<=M; m+=2){       /* raw u16 sums (even byte + 256 * odd byte) and the odd bytes */
            __m256i bb=_mm256_loadu_si256((const __m256i*)(blk+m*16)), L=_mm256_loadu_si256((const __m256i*)(lut+m*16));
            __m256i vl=_mm256_shuffle_epi8(L,_mm256_and_si256(bb,m4)), vh=_mm256_shuffle_epi8(L,_mm256_and_si256(_mm256_srli_epi16(bb,4),m4));
            aEl=_mm256_add_epi16(aEl,vl); aOl=_mm256_add_epi16(aOl,_mm256_srli_epi16(vl,8));
            aEh=_mm256_add_epi16(aEh,vh); aOh=_mm256_add_epi16(aOh,_mm256_srli_epi16(vh,8)); }
        __m128i El=_mm_add_epi16(_mm256_castsi256_si128(aEl),_mm256_extracti128_si256(aEl,1));
        __m128i Ol=_mm_add_epi16(_mm256_castsi256_si128(aOl),_mm256_extracti128_si256(aOl,1));
        __m128i Eh=_mm_add_epi16(_mm256_castsi256_si128(aEh),_mm256_extracti128_si256(aEh,1));
        __m128i Oh=_mm_add_epi16(_mm256_castsi256_si128(aOh),_mm256_extracti128_si256(aOh,1));
        if(m<M){ const __m128i m4s=_mm_set1_epi8(0x0F);
            __m128i bb=_mm_loadu_si128((const __m128i*)(blk+m*16)), L=_mm_loadu_si128((const __m128i*)(lut+m*16));
            __m128i vl=_mm_shuffle_epi8(L,_mm_and_si128(bb,m4s)), vh=_mm_shuffle_epi8(L,_mm_and_si128(_mm_srli_epi16(bb,4),m4s));
            El=_mm_add_epi16(El,vl); Ol=_mm_add_epi16(Ol,_mm_srli_epi16(vl,8));
            Eh=_mm_add_epi16(Eh,vh); Oh=_mm_add_epi16(Oh,_mm_srli_epi16(vh,8)); }
        El=_mm_sub_epi16(El,_mm_slli_epi16(Ol,8)); Eh=_mm_sub_epi16(Eh,_mm_slli_epi16(Oh,8));
        uint32_t cand=0xFFFFFFFFu;             /* bit b: lane b of (El,Ol,Eh,Oh) packed as e0[8] o0[8] e1[8] o1[8] */
        if(S->thr<1e29f){
            /* sums are unsigned 16-bit (up to 65,000): flip the top bit so the signed compare orders them */
            float need=(-S->thr-base)*sc;            /* need < 0: every sum qualifies */
            if(need>=0){ int t = need>65000 ? 65000 : (int)need;
                const __m128i X=_mm_set1_epi16((short)0x8000);
                __m128i T=_mm_xor_si128(_mm_set1_epi16((short)(uint16_t)t),X);
                __m128i cl=_mm_packs_epi16(_mm_cmpgt_epi16(_mm_xor_si128(El,X),T),_mm_cmpgt_epi16(_mm_xor_si128(Ol,X),T));
                __m128i ch=_mm_packs_epi16(_mm_cmpgt_epi16(_mm_xor_si128(Eh,X),T),_mm_cmpgt_epi16(_mm_xor_si128(Oh,X),T));
                cand=(uint32_t)_mm_movemask_epi8(cl) | ((uint32_t)_mm_movemask_epi8(ch)<<16); }
        }
        if(b+1==nb && (Lst->n&31)){            /* last block: drop the unused lanes (codes 0, no row) */
            uint32_t rem=Lst->n&31, ok=0;
            for(int bt=0;bt<32;bt++){ uint32_t j=(uint32_t)((bt&16) + 2*(bt&7) + ((bt>>3)&1)); ok |= (uint32_t)(j<rem)<<bt; }
            cand&=ok; }
        if(!cand) continue;
        { uint16_t sv[32];                      /* sv[0..7]=El, [8..15]=Ol, [16..23]=Eh, [24..31]=Oh */
            _mm_storeu_si128((__m128i*)(sv),El); _mm_storeu_si128((__m128i*)(sv+8),Ol);
            _mm_storeu_si128((__m128i*)(sv+16),Eh); _mm_storeu_si128((__m128i*)(sv+24),Oh);
            if(tag!=UINT32_MAX){                /* positions only; rows are looked up for the survivors */
                const __m256 vb=_mm256_set1_ps(-base), vi=_mm256_set1_ps(-invs);
                const __m256i sh=_mm256_setr_epi32(0,4,8,12,16,20,24,28), s7=_mm256_set1_epi32(7);
                const __m256i pb0=_mm256_set1_epi32((int)(tag|(b*32))), j2=_mm256_setr_epi32(0,2,4,6,8,10,12,14);
                const __m128i grp[4]={El,Ol,Eh,Oh};
                for(int g=0;g<4;g++){ uint32_t m8=(cand>>(8*g))&0xFFu;
                    __m256 dv=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(grp[g])),vi,vb);
                    __m256i pv=_mm256_add_epi32(pb0,_mm256_add_epi32(j2,_mm256_set1_epi32((g&1)+16*(g>>1))));
                    __m256i ix=_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)kv_cmp8[m8]),sh),s7);
                    dv=_mm256_permutevar8x32_ps(dv,ix); __m256 pf=_mm256_castsi256_ps(_mm256_permutevar8x32_epi32(pv,ix));
                    __m256 lo=_mm256_unpacklo_ps(dv,pf), hi=_mm256_unpackhi_ps(dv,pf);   /* d0 p0 d1 p1 | d4 p4 d5 p5 ... */
                    float *o=(float*)(S->a+S->n);
                    _mm256_storeu_ps(o,_mm256_permute2f128_ps(lo,hi,0x20)); _mm256_storeu_ps(o+8,_mm256_permute2f128_ps(lo,hi,0x31));
                    S->n += __builtin_popcount(m8); }
                if(S->n>=S->cap) kv_sl_compact(S);
                continue; }
            const uint32_t *ids=Lst->ids+(size_t)b*32;
            while(cand){ int bt=__builtin_ctz(cand); cand&=cand-1;
                int j=((bt&16)) + 2*(bt&7) + ((bt>>3)&1);       /* lane -> vector index in the block */
                if(ids[j]==UINT32_MAX) continue;
                float d=-(base+(float)sv[bt]*invs);
                if(d>=S->thr) continue;
                if(filt && !filt(vid[ids[j]], fctx)) continue;
                if(vis){ if(vis[ids[j]]==ep) continue; vis[ids[j]]=ep; }
                S->a[S->n].d=d; S->a[S->n].id=ids[j];
                if(++S->n==S->cap) kv_sl_compact(S); } }
#else
        { uint32_t sum[32]; memset(sum,0,sizeof sum);
          for(int m=0;m<M;m++){ const uint8_t *bb=blk+m*16, *L=lut+m*16;
              for(int j=0;j<16;j++){ sum[j]+=L[bb[j]&15]; sum[16+j]+=L[bb[j]>>4]; } }
          const uint32_t *ids=Lst->ids+(size_t)b*32;
          uint32_t nj = b+1==nb && (Lst->n&31) ? (Lst->n&31) : 32;
          for(uint32_t j=0;j<nj;j++){
              float d=-(base+(float)sum[j]*invs);
              if(d>=S->thr) continue;
              uint32_t r;
              if(tag!=UINT32_MAX) r=tag|(b*32+j);
              else {
                  if(ids[j]==UINT32_MAX) continue;
                  if(filt && !filt(vid[ids[j]], fctx)) continue;
                  if(vis){ if(vis[ids[j]]==ep) continue; vis[ids[j]]=ep; }
                  r=ids[j]; }
              S->a[S->n].d=d; S->a[S->n].id=r;
              if(++S->n==S->cap) kv_sl_compact(S); } }
#endif
    }
}

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#define KV_LR_QOFF 128
#else
#define KV_LR_QOFF 64
#endif
/* qu = U^T q for one list (lr values) from the u8 query q8 (D4*4 bytes); qinv = 1 / (query scale x U scale) */
static void kv_lr_project(const struct KvIvf *f, const KvList *L, const uint8_t *q8, float qinv, float *qu){
    int lr=f->lr, D4=f->LD4;                                   /* q8 has ld = LD4*4 entries (the projected query when d1 > 0) */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    for(int g=0; g<lr/16; g++){ __m512i acc=_mm512_setzero_si512(); const int8_t *ug=L->U+(size_t)g*D4*64;
        for(int d4=0; d4<D4; d4++){ uint32_t w; memcpy(&w,q8+4*d4,4);
            acc=_mm512_dpbusd_epi32(acc,_mm512_set1_epi32((int)w),_mm512_loadu_si512((const void*)(ug+d4*64))); }
        acc=_mm512_sub_epi32(acc,_mm512_loadu_si512((const void*)(L->usum+g*16)));
        _mm512_storeu_ps(qu+g*16,_mm512_mul_ps(_mm512_cvtepi32_ps(acc),_mm512_set1_ps(qinv))); }
#elif defined(__AVX2__)
    const __m256i ones=_mm256_set1_epi16(1);
    for(int g=0; g<lr/16; g++){ __m256i a0=_mm256_setzero_si256(), a1=a0; const int8_t *ug=L->U+(size_t)g*D4*64;
        for(int d4=0; d4<D4; d4++){ uint32_t w; memcpy(&w,q8+4*d4,4); __m256i qv=_mm256_set1_epi32((int)w);
            a0=_mm256_add_epi32(a0,_mm256_madd_epi16(_mm256_maddubs_epi16(qv,_mm256_loadu_si256((const __m256i*)(ug+d4*64))),ones));
            a1=_mm256_add_epi32(a1,_mm256_madd_epi16(_mm256_maddubs_epi16(qv,_mm256_loadu_si256((const __m256i*)(ug+d4*64+32))),ones)); }
        a0=_mm256_sub_epi32(a0,_mm256_srai_epi32(_mm256_loadu_si256((const __m256i*)(L->usum+g*16)),1));
        a1=_mm256_sub_epi32(a1,_mm256_srai_epi32(_mm256_loadu_si256((const __m256i*)(L->usum+g*16+8)),1));
        _mm256_storeu_ps(qu+g*16,_mm256_mul_ps(_mm256_cvtepi32_ps(a0),_mm256_set1_ps(qinv)));
        _mm256_storeu_ps(qu+g*16+8,_mm256_mul_ps(_mm256_cvtepi32_ps(a1),_mm256_set1_ps(qinv))); }
#else
    for(int k=0;k<lr;k++){ int g=k/16, l=k%16; int32_t s=0;
        for(int d4=0; d4<D4; d4++) for(int t=0;t<4;t++) s+=(int32_t)q8[4*d4+t]*L->U[(((size_t)g*D4+d4)*16+l)*4+t];
        qu[k]=(float)(s-L->usum[k]*KV_LR_QOFF/128)*qinv; }
#endif
}
/* scan one low-rank list: raw = sum_k qu8[k] coef[k] per vector (32 at a time), score = base + (raw - qoff csum) sc */
static void kv_scan_lr(const struct KvIvf *f, const KvList *L, const uint8_t *qu8, float sc, float base, Heap *H, int R,
                       uint16_t *vis, uint16_t ep, KciVecFilter filt, void *fctx, const uint64_t *vid){
    int lr=f->lr; size_t bb=f->bb; uint32_t nb=(L->n+31)/32;
    for(uint32_t b=0;b<nb;b++){
        const uint8_t *blk=L->codes+(size_t)b*bb; const int32_t *csum=(const int32_t*)(blk+(size_t)lr*32);
        float sco[32] __attribute__((aligned(64)));   /* explicit: gcc -O1 + ASan turned the 512-bit store into an aligned one */
        if(b+1<nb) for(size_t o=0;o<bb;o+=64) __builtin_prefetch(blk+bb+o);
        /* scores of the 32 vectors in SIMD, then a candidate mask against the shortlist's worst (as the table scan does);
         * a scalar pass over all 32 cost more than the dots themselves */
        float thr = H->n>=R ? -hp_top(H).d : -1e30f; uint32_t cand;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i a0=_mm512_setzero_si512(), a1=a0, b0=a0, b1=a0;                    /* two chains per half: the dpbusd latency was the limit */
        for(int kq=0; kq+2<=lr/4; kq+=2){ uint32_t w0, w1; memcpy(&w0,qu8+4*kq,4); memcpy(&w1,qu8+4*kq+4,4);
            __m512i q0=_mm512_set1_epi32((int)w0), q1=_mm512_set1_epi32((int)w1); const uint8_t *p=blk+(size_t)kq*128;
            a0=_mm512_dpbusd_epi32(a0,q0,_mm512_loadu_si512((const void*)p));     a1=_mm512_dpbusd_epi32(a1,q0,_mm512_loadu_si512((const void*)(p+64)));
            b0=_mm512_dpbusd_epi32(b0,q1,_mm512_loadu_si512((const void*)(p+128))); b1=_mm512_dpbusd_epi32(b1,q1,_mm512_loadu_si512((const void*)(p+192))); }
        a0=_mm512_add_epi32(a0,b0); a1=_mm512_add_epi32(a1,b1);
        a0=_mm512_sub_epi32(a0,_mm512_slli_epi32(_mm512_loadu_si512((const void*)csum),7));         /* qoff = 128 */
        a1=_mm512_sub_epi32(a1,_mm512_slli_epi32(_mm512_loadu_si512((const void*)(csum+16)),7));
        __m512 s0=_mm512_fmadd_ps(_mm512_cvtepi32_ps(a0),_mm512_set1_ps(sc),_mm512_set1_ps(base));
        __m512 s1=_mm512_fmadd_ps(_mm512_cvtepi32_ps(a1),_mm512_set1_ps(sc),_mm512_set1_ps(base));
        _mm512_storeu_ps(sco,s0); _mm512_storeu_ps(sco+16,s1);
        cand=(uint32_t)_mm512_cmp_ps_mask(s0,_mm512_set1_ps(thr),_CMP_GT_OQ) | ((uint32_t)_mm512_cmp_ps_mask(s1,_mm512_set1_ps(thr),_CMP_GT_OQ)<<16);
#elif defined(__AVX2__)
        const __m256i ones=_mm256_set1_epi16(1); __m256i a[4]; for(int t=0;t<4;t++) a[t]=_mm256_setzero_si256();
        for(int kq=0; kq<lr/4; kq++){ uint32_t w; memcpy(&w,qu8+4*kq,4); __m256i qv=_mm256_set1_epi32((int)w); const uint8_t *p=blk+(size_t)kq*128;
            for(int t=0;t<4;t++) a[t]=_mm256_add_epi32(a[t],_mm256_madd_epi16(_mm256_maddubs_epi16(qv,_mm256_loadu_si256((const __m256i*)(p+32*t))),ones)); }
        cand=0;
        for(int t=0;t<4;t++){ __m256i r=_mm256_sub_epi32(a[t],_mm256_slli_epi32(_mm256_loadu_si256((const __m256i*)(csum+8*t)),6));   /* qoff = 64 */
            __m256 s=_mm256_fmadd_ps(_mm256_cvtepi32_ps(r),_mm256_set1_ps(sc),_mm256_set1_ps(base)); _mm256_storeu_ps(sco+8*t,s);
            cand|=(uint32_t)_mm256_movemask_ps(_mm256_cmp_ps(s,_mm256_set1_ps(thr),_CMP_GT_OQ))<<(8*t); }
#else
        cand=0;
        for(int j=0;j<32;j++){ int32_t s=0; for(int k=0;k<lr;k++) s+=(int32_t)qu8[k]*(int8_t)blk[((size_t)(k>>2)*32+j)*4+(k&3)];
            sco[j]=base+(float)(s-csum[j]*KV_LR_QOFF)*sc; if(sco[j]>thr) cand|=1u<<j; }
#endif
        const uint32_t *ids=L->ids+(size_t)b*32;
        while(cand){ int j=__builtin_ctz(cand); cand&=cand-1;
            if(ids[j]==UINT32_MAX) continue;
            float d=-sco[j];
            if(H->n>=R && d>=hp_top(H).d) continue;
            if(filt && !filt(vid[ids[j]], fctx)) continue;
            if(vis){ if(vis[ids[j]]==ep) continue; vis[ids[j]]=ep; }
            if(H->n>=R) hp_pop(H);
            hp_push(H,d,ids[j]); }
    }
}

static void kv_lut_build(const float *pqs, int M, int dp, const float *qp, int Dq, float *lutf, uint8_t *lut, float *base, float *invs, int topmax){
    float summin=0, maxrange=0;
    for(int m=0;m<M;m++){ float lo=1e30f, hi=-1e30f, q0=qp[dp*m], q1=(dp==2 && 2*m+1<Dq)? qp[2*m+1] : 0; float *lf=lutf+m*16; const float *pq=pqs+m*32;
#if defined(__AVX2__)
        const __m256 qq=_mm256_setr_ps(q0,q1,q0,q1,q0,q1,q0,q1); const __m256i ix=_mm256_setr_epi32(0,1,4,5,2,3,6,7);
        __m256 t0=_mm256_permutevar8x32_ps(_mm256_hadd_ps(_mm256_mul_ps(qq,_mm256_loadu_ps(pq)),_mm256_mul_ps(qq,_mm256_loadu_ps(pq+8))),ix);
        __m256 t1=_mm256_permutevar8x32_ps(_mm256_hadd_ps(_mm256_mul_ps(qq,_mm256_loadu_ps(pq+16)),_mm256_mul_ps(qq,_mm256_loadu_ps(pq+24))),ix);
        __m256 mn=_mm256_min_ps(t0,t1), mx=_mm256_max_ps(t0,t1);
        __m128 mn4=_mm_min_ps(_mm256_castps256_ps128(mn),_mm256_extractf128_ps(mn,1)), mx4=_mm_max_ps(_mm256_castps256_ps128(mx),_mm256_extractf128_ps(mx,1));
        mn4=_mm_min_ps(mn4,_mm_movehl_ps(mn4,mn4)); mn4=_mm_min_ps(mn4,_mm_shuffle_ps(mn4,mn4,1));
        mx4=_mm_max_ps(mx4,_mm_movehl_ps(mx4,mx4)); mx4=_mm_max_ps(mx4,_mm_shuffle_ps(mx4,mx4,1));
        lo=_mm_cvtss_f32(mn4); hi=_mm_cvtss_f32(mx4);
        _mm256_storeu_ps(lf,_mm256_sub_ps(t0,_mm256_set1_ps(lo))); _mm256_storeu_ps(lf+8,_mm256_sub_ps(t1,_mm256_set1_ps(lo)));
#else
        for(int j=0;j<16;j++){ float t=q0*pq[2*j]+q1*pq[2*j+1]; lf[j]=t; if(t<lo)lo=t; if(t>hi)hi=t; }
        for(int j=0;j<16;j++) lf[j]-=lo;
#endif
        summin+=lo; if(hi-lo>maxrange) maxrange=hi-lo; }
    float top = 65000.0f/M < (float)topmax ? 65000.0f/M : (float)topmax;
    float scale = maxrange>0 ? top/maxrange : 1.0f;
#if defined(__AVX2__)
    { const __m256 vs=_mm256_set1_ps(scale), v255=_mm256_set1_ps(255.0f);
      for(int i=0;i<M*16;i+=16){                       /* 16 entries per step: round, clamp to 255, pack to u8 */
          __m256i a=_mm256_cvtps_epi32(_mm256_min_ps(_mm256_mul_ps(_mm256_loadu_ps(lutf+i),vs),v255));
          __m256i b=_mm256_cvtps_epi32(_mm256_min_ps(_mm256_mul_ps(_mm256_loadu_ps(lutf+i+8),vs),v255));
          __m256i w=_mm256_permute4x64_epi64(_mm256_packus_epi32(a,b),0xD8);   /* u16 lanes back in order 0..15 */
          __m256i u=_mm256_packus_epi16(w,w);                                    /* u8: lanes 0-7 in the low half, 8-15 in the high */
          _mm_storel_epi64((__m128i*)(lut+i),_mm256_castsi256_si128(u)); _mm_storel_epi64((__m128i*)(lut+i+8),_mm256_extracti128_si256(u,1)); } }
#else
    for(int i=0;i<M*16;i++){ int r=(int)lrintf(lutf[i]*scale); lut[i]=(uint8_t)(r>255?255:r); }
#endif
    memset(lut+M*16,0,64);
    *base=summin; *invs=1.0f/scale;
}

static int kv_count_ge(const float *qc, int n, float T){
    int c=0, i=0;
#if defined(__AVX2__)
    const __m256 vT=_mm256_set1_ps(T); __m256i A=_mm256_setzero_si256(), B=A;
    for(; i+16<=n; i+=16){
        A=_mm256_sub_epi32(A,_mm256_castps_si256(_mm256_cmp_ps(_mm256_loadu_ps(qc+i),vT,_CMP_GE_OQ)));
        B=_mm256_sub_epi32(B,_mm256_castps_si256(_mm256_cmp_ps(_mm256_loadu_ps(qc+i+8),vT,_CMP_GE_OQ))); }
    A=_mm256_add_epi32(A,B);
    __m128i a4=_mm_add_epi32(_mm256_castsi256_si128(A),_mm256_extracti128_si256(A,1));
    a4=_mm_add_epi32(a4,_mm_shuffle_epi32(a4,0x4E)); a4=_mm_add_epi32(a4,_mm_shuffle_epi32(a4,0xB1));
    c=_mm_cvtsi128_si32(a4);
#endif
    for(; i<n; i++) c += qc[i]>=T;
    return c;
}

static void kv_minmax(const float *qc, int n, float *mn, float *mx){
    float a=qc[0], b=qc[0]; int i=0;
#if defined(__AVX2__)
    if(n>=8){ __m256 A=_mm256_loadu_ps(qc), B=A;
        for(i=8; i+8<=n; i+=8){ __m256 y=_mm256_loadu_ps(qc+i); A=_mm256_min_ps(A,y); B=_mm256_max_ps(B,y); }
        float ta[8], tb[8]; _mm256_storeu_ps(ta,A); _mm256_storeu_ps(tb,B);
        a=ta[0]; b=tb[0]; for(int u=1;u<8;u++){ a = ta[u]<a ? ta[u] : a; b = tb[u]>b ? tb[u] : b; } }
#endif
    for(; i<n; i++){ a = qc[i]<a ? qc[i] : a; b = qc[i]>b ? qc[i] : b; }
    *mn=a; *mx=b;
}

static int kv_search_ivf_(KciVecSearcher *ks, const float *query, int k, int nprobe, int rerank_n,
                          uint64_t *out_ids, float *out_scores){
    if(!ks || !ks->v->ivf) return -1;
    KciVec *v=ks->v; const struct KvIvf *f=v->ivf; int D=v->dim, M=f->M;
    if(k<1) k=1;
    if(rerank_n<k) rerank_n=k;
    if(nprobe<1) nprobe=1;
    if(nprobe>f->nlist) nprobe=f->nlist;
    KvIvfScr *z=(KvIvfScr*)ks->ivf;
    if(!z || z->n!=f->nlistp){
        if(z) kv_ivf_scr_free(z);
        z=calloc(1,sizeof *z); ks->ivf=z; if(!z) return -1;
        z->n=f->nlistp; z->qcb=f->nlistp*sizeof(float); z->qc=malloc(z->qcb); z->lh=malloc(f->nlistp*sizeof(He)); z->lh2=malloc(f->nlistp*sizeof(He));
        z->lutb=(size_t)(M+4)*16; z->lut=malloc(z->lutb); z->lutfb=(size_t)M*16*sizeof(float); z->lutf=malloc(z->lutfb);
        z->q1b = f->d1 ? (size_t)f->d1*sizeof(float) : 0; z->q1 = f->d1 ? malloc(z->q1b) : NULL;
        if(!z->qc||!z->lh||!z->lh2||!z->lut||!z->lutf||(f->d1&&!z->q1)) return -1;
    }
    l2norm(D, query, ks->qf); const float *q=ks->qf; float *qc=z->qc;
    /* 1. centroid scores */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    { uint8_t q8[f->D4*4+4]; float qa=0; for(int d=0;d<D;d++) if(fabsf(q[d])>qa) qa=fabsf(q[d]);
      float qs = qa>0 ? 127.0f/qa : 1.0f; memset(q8,128,sizeof q8);
      for(int d=0;d<D;d++){ int r=(int)lrintf(q[d]*qs); q8[d]=(uint8_t)((r<-127?-127:(r>127?127:r))+128); }
      float inv=1.0f/(qs*f->c8scale);
      for(int g=0; g<f->nlistp/16; g++){ __m512i acc=_mm512_setzero_si512(); const int8_t *cg=f->c8+(size_t)g*f->D4*64;
          for(int d4=0; d4<f->D4; d4++){ uint32_t w; memcpy(&w,q8+4*d4,4);
              acc=_mm512_dpbusd_epi32(acc,_mm512_set1_epi32((int)w),_mm512_loadu_si512((const void*)(cg+d4*64))); }
          acc=_mm512_sub_epi32(acc,_mm512_loadu_si512((const void*)(f->c8sum+g*16)));
          _mm512_storeu_ps(qc+g*16,_mm512_mul_ps(_mm512_cvtepi32_ps(acc),_mm512_set1_ps(inv))); } }
#elif defined(__AVX2__)
    { uint8_t q8[f->D4*4+4]; float qa=0; for(int d=0;d<D;d++) if(fabsf(q[d])>qa) qa=fabsf(q[d]);
      float qs = qa>0 ? 63.0f/qa : 1.0f; memset(q8,64,sizeof q8);
      for(int d=0;d<D;d++){ int r=(int)lrintf(q[d]*qs); q8[d]=(uint8_t)((r<-63?-63:(r>63?63:r))+64); }
      float inv=1.0f/(qs*f->c8scale); const __m256i ones=_mm256_set1_epi16(1); const __m256 vinv=_mm256_set1_ps(inv);
      for(int g=0; g<f->nlistp/16; g++){ __m256i a0=_mm256_setzero_si256(), a1=a0; const int8_t *cg=f->c8+(size_t)g*f->D4*64;
          for(int d4=0; d4<f->D4; d4++){ uint32_t w; memcpy(&w,q8+4*d4,4); __m256i qv=_mm256_set1_epi32((int)w);
              a0=_mm256_add_epi32(a0,_mm256_madd_epi16(_mm256_maddubs_epi16(qv,_mm256_loadu_si256((const __m256i*)(cg+d4*64))),ones));
              a1=_mm256_add_epi32(a1,_mm256_madd_epi16(_mm256_maddubs_epi16(qv,_mm256_loadu_si256((const __m256i*)(cg+d4*64+32))),ones)); }
          a0=_mm256_sub_epi32(a0,_mm256_srai_epi32(_mm256_loadu_si256((const __m256i*)(f->c8sum+g*16)),1));
          a1=_mm256_sub_epi32(a1,_mm256_srai_epi32(_mm256_loadu_si256((const __m256i*)(f->c8sum+g*16+8)),1));
          _mm256_storeu_ps(qc+g*16,_mm256_mul_ps(_mm256_cvtepi32_ps(a0),vinv));
          _mm256_storeu_ps(qc+g*16+8,_mm256_mul_ps(_mm256_cvtepi32_ps(a1),vinv)); } }
#else
    for(int c=0;c<f->nlist;c++) qc[c]=fdot(D,q,f->cent+(size_t)c*D);
#endif
    /* 2. the best nprobe lists, nearest first */
    He *lh=z->lh; int cnt=0; float qmax;                     /* qmax: the nearest centroid's score (probe gap) */
    { float T, a, b; kv_minmax(qc, f->nlist, &a, &b); qmax=b;
      if(nprobe>=f->nlist) T=-1e30f;
      else { int goal=nprobe+nprobe/4+8; T=a;
          for(int it=0; it<16 && b>a; it++){ float mid=0.5f*(a+b); int nc=kv_count_ge(qc, f->nlist, mid);
              if(nc>=nprobe){ a=mid; T=mid; if(nc<=goal) break; } else b=mid; } }
#if defined(__AVX512F__)
      { const __m512 vT=_mm512_set1_ps(T);
        for(int c=0;c<f->nlist;c+=16){
            __mmask16 lim = (f->nlist-c>=16) ? (__mmask16)0xFFFF : (__mmask16)((1u<<(f->nlist-c))-1);
            uint32_t m=_mm512_mask_cmp_ps_mask(lim,_mm512_loadu_ps(qc+c),vT,_CMP_GE_OQ);
            while(m){ int b=__builtin_ctz(m); m&=m-1; lh[cnt].d=-qc[c+b]; lh[cnt].id=(uint32_t)(c+b); cnt++; } } }
#elif defined(__AVX2__)
      { const __m256 vT=_mm256_set1_ps(T); int c=0;
        for(; c+8<=f->nlist; c+=8){
            uint32_t m=(uint32_t)_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(qc+c),vT,_CMP_GE_OQ));
            while(m){ int b=__builtin_ctz(m); m&=m-1; lh[cnt].d=-qc[c+b]; lh[cnt].id=(uint32_t)(c+b); cnt++; } }
        for(; c<f->nlist; c++) if(qc[c]>=T){ lh[cnt].d=-qc[c]; lh[cnt].id=(uint32_t)c; cnt++; } }
#else
      for(int c=0;c<f->nlist;c++) if(qc[c]>=T){ lh[cnt].d=-qc[c]; lh[cnt].id=(uint32_t)c; cnt++; }
#endif
    }
    if(cnt<nprobe){ cnt=f->nlist; for(int c=0;c<f->nlist;c++){ lh[c].d=-qc[c]; lh[c].id=(uint32_t)c; } }
    { He *o=z->lh2; float lo=1e30f, hi=-1e30f;
      for(int i=0;i<cnt;i++){ float x=lh[i].d; lo = x<lo ? x : lo; hi = x>hi ? x : hi; }
      float inv = hi>lo ? (float)(KV_SEL_B-1)/(hi-lo) : 0.0f;
      int cntb[KV_SEL_B+1]; memset(cntb,0,sizeof cntb);
      for(int i=0;i<cnt;i++){ int kk=(int)((lh[i].d-lo)*inv); kk = kk<0 ? 0 : (kk>=KV_SEL_B ? KV_SEL_B-1 : kk); cntb[kk+1]++; }
      for(int kk=0;kk<KV_SEL_B;kk++) cntb[kk+1]+=cntb[kk];                  /* cntb[kk] = where bucket kk starts */
      int fill[KV_SEL_B]; memcpy(fill,cntb,sizeof fill);
      for(int i=0;i<cnt;i++){ int kk=(int)((lh[i].d-lo)*inv); kk = kk<0 ? 0 : (kk>=KV_SEL_B ? KV_SEL_B-1 : kk); o[fill[kk]++]=lh[i]; }
      if(nprobe<cnt){ int kb=0; while(cntb[kb+1]<=nprobe) kb++;              /* the bucket the cut falls in */
          if(cntb[kb]<nprobe) select_smallest(o+cntb[kb], cntb[kb+1]-cntb[kb], nprobe-cntb[kb]); }
      z->lh2=lh; z->lh=o; lh=o; }
    if(ks->gap>0){ float lim=-qmax+ks->gap; int mp = ks->minp<nprobe ? ks->minp : nprobe;
        int np2=0; for(int i=0;i<nprobe;i++) np2 += lh[i].d<=lim;
        if(np2<nprobe){
            if(np2<mp){ select_smallest(lh, nprobe, mp); np2=mp; }
            else { He *o=z->lh2; int w=0;
                for(int i=0;i<nprobe;i++) if(lh[i].d<=lim) o[w++]=lh[i];
                for(int i=0;i<nprobe;i++) if(!(lh[i].d<=lim)) o[w++]=lh[i];
                z->lh2=lh; z->lh=o; lh=o; }
            nprobe=np2; } }
    Heap *H=&ks->s.W; H->n=0;
    He *sa=NULL; int sn=0;                          /* the shortlist handed to the re-rank (heap or table-scan buffer) */
    uint16_t *vis=NULL, ep=0;
    if(f->soar>0){
        if(ks->s.visn<v->count){ uint16_t *nv=realloc(ks->s.vis,(size_t)v->count*sizeof(uint16_t)); if(!nv) return -1;
            memset(nv,0,(size_t)v->count*sizeof(uint16_t)); ks->s.vis=nv; ks->s.visn=v->count; ks->s.epoch=0; }
        scr_next(&ks->s); vis=ks->s.vis; ep=ks->s.epoch; }
    if(f->lr){
        int lr=f->lr, ld=f->ld; float qu[256] __attribute__((aligned(64))); uint8_t qu8[256]; uint8_t q8[f->LD4*4+4];   /* lr <= 256 */
        const float qtop = KV_LR_QOFF==128 ? 127.0f : 63.0f;
        const float *qq=q; if(f->d1){ kv_proj(f, D, q, z->q1); qq=z->q1; }   /* bases live in the projected space */
        float qa=0; for(int d=0;d<ld;d++) if(fabsf(qq[d])>qa) qa=fabsf(qq[d]);
        float qs = qa>0 ? qtop/qa : 1.0f; memset(q8,KV_LR_QOFF,sizeof q8);
        for(int d=0;d<ld;d++){ int r=(int)lrintf(qq[d]*qs); int t=(int)qtop; q8[d]=(uint8_t)((r<-t?-t:(r>t?t:r))+KV_LR_QOFF); }
        for(int p=0;p<nprobe;p++){ int c=lh[p].id; const KvList *L=&f->lists[c]; if(!L->n) continue;
            if(p+1<nprobe){ const KvList *nx=&f->lists[lh[p+1].id]; if(nx->codes) for(size_t o=0;o<4096 && o<nx->capb*f->bb;o+=64) __builtin_prefetch((const char*)nx->codes+o); }
            kv_lr_project(f, L, q8, 1.0f/(qs*L->uscale), qu);
            float qm=0; for(int t=0;t<lr;t++) if(fabsf(qu[t])>qm) qm=fabsf(qu[t]);
            float qus = qm>0 ? qtop/qm : 1.0f; int tt=(int)qtop;
            for(int t=0;t<lr;t++){ int r=(int)lrintf(qu[t]*qus); qu8[t]=(uint8_t)((r<-tt?-tt:(r>tt?tt:r))+KV_LR_QOFF); }
            kv_scan_lr(f, L, qu8, L->cscale/qus, qc[c], H, rerank_n, vis, ep, ks->filt, ks->fctx, v->vid); }
        sa=H->a; sn=H->n;
    } else {
    const float *qp=q; int Dq=D;
    if(f->d1){ kv_proj(f, D, q, z->q1); qp=z->q1; Dq=f->d1;
        for(int p=0;p<nprobe;p++){ int c=lh[p].id; qc[c]=fdot(f->d1, z->q1, f->cent1+(size_t)c*f->d1); } }   /* qc reused: lists are chosen */
    float summin, invs; kv_lut_build(f->pq, M, f->dp, qp, Dq, z->lutf, z->lut, &summin, &invs, KV_LUT_TOP);
    /* 4. scan the lists into a shortlist of rerank_n */
    int cap = rerank_n<16 ? rerank_n+48 : 3*rerank_n;
    if(z->slcap<cap+32){ He *nb=realloc(z->sl,(size_t)(cap+32)*sizeof(He)); if(!nb) return -1; z->sl=nb; z->slcap=cap+32; }   /* +32: a block's stores past cap */
    KvSl S={z->sl, 0, cap, rerank_n, 1e30f};
    int pb=1; while((1<<pb)<nprobe) pb++;
    uint32_t pmax = pb>=32 ? 0 : (uint32_t)(1ull<<(32-pb));
    int defer = !vis && !ks->filt && pb<24 && f->nmax<=pmax;
    for(int p=0;p<nprobe;p++){ int c=lh[p].id;
        if(p+2<nprobe) __builtin_prefetch(&f->lists[lh[p+2].id]);
        if(p+1<nprobe){ const KvList *nx=&f->lists[lh[p+1].id]; if(nx->codes) for(int o=0;o<KV_PF_NEXT && o<(int)(nx->capb*M*16);o+=64) __builtin_prefetch((const char*)nx->codes+o); }
        kv_scan_list(f, &f->lists[c], z->lut, qc[c]+summin, invs, &S, defer ? (uint32_t)p<<(32-pb) : UINT32_MAX,
                     vis, ep, ks->filt, ks->fctx, v->vid); }
    kv_sl_compact(&S);
    if(defer){                                       /* rows of the survivors; cleared entries (forget) drop out */
        uint32_t lo=pmax-1; int w=0;
        for(int i=0;i<S.n;i++) __builtin_prefetch(f->lists[lh[S.a[i].id>>(32-pb)].id].ids+(S.a[i].id&lo));
        for(int i=0;i<S.n;i++){ uint32_t r=f->lists[lh[S.a[i].id>>(32-pb)].id].ids[S.a[i].id&lo];
            S.a[w].d=S.a[i].d; S.a[w].id=r; w += r!=UINT32_MAX; }
        S.n=w; }
    sa=S.a; sn=S.n;
    }
    /* 5. float re-rank, top-k live */
    int m=sn; He *t=ks->s.tmp;
    if(m>ks->s.tmpcap){ ks->s.tmpcap=m; ks->s.tmp=t=realloc(ks->s.tmp,sizeof(He)*m); if(!t) return -1; }
    memcpy(t,sa,sizeof(He)*m);
    if(ks->rerank_i8 || (!v->vf && !v->ffn)){
        sq_encode(v, q, ks->q);
        for(int i=0;i<m;i++){ const char *p=(const char*)(v->vq+(size_t)t[i].id*D); for(int o=0;o<D;o+=64) __builtin_prefetch(p+o); }
        for(int i=0;i<m;i++) t[i].d=-(float)sdot(v, ks->q, v->vq+(size_t)t[i].id*D);
        return emit_topk(v, t, m, k, 1.0f/(v->gs*v->gs), out_ids, out_scores);
    }
    if(!v->vf && v->ffn){
        float x[D], u[D];
        for(int i=0;i<m;i++){
            uint32_t r=t[i].id;
            if(v->dead && v->dead[r]){ t[i].d=1e30f; continue; }
            if(v->ffn(v->vid[r], x, v->fctx)!=0){ t[i].d=1e30f; continue; }   /* gone from the source */
            l2norm(D, x, u);
            t[i].d=-fdot(D, q, u);
        }
        kv_wipe(x, sizeof x); kv_wipe(u, sizeof u);
        int w=0; for(int i=0;i<m;i++) if(t[i].d<1e30f) t[w++]=t[i];
        return emit_topk(v, t, w, k, 1.0f, out_ids, out_scores);
    }
    { size_t rb=(size_t)D*sizeof(float);
      for(int i=0;i<m;i++){ const char *p=(const char*)(v->vf+(size_t)t[i].id*D); for(size_t o=0;o<rb;o+=64) __builtin_prefetch(p+o); } }
    for(int i=0;i<m;i++) t[i].d=-fdot(D, q, v->vf+(size_t)t[i].id*D);
    return emit_topk(v, t, m, k, 1.0f, out_ids, out_scores);
}

#ifndef KV_QG_LUT_TOP
#define KV_QG_LUT_TOP 255
#endif
struct KvQg { int M, dp, nblk, nfull, half; size_t stride, hdr; float *pq; double eta; uint8_t *blocks; uint32_t n; };
static void kv_qg_free(struct KvQg *g){
    if(!g) return;
    if(g->blocks && g->n) kv_wipe(g->blocks, (size_t)g->n*g->stride);   /* the vectors' codes, and who is next to whom */
    if(g->pq) kv_wipe(g->pq, (size_t)(g->M+1)*32*sizeof(float));
    free(g->blocks); free(g->pq); free(g);
}
/* a node's unit vector: the retained float row, or the int8 row scaled back */
static void kv_qg_row(const KciVec *v, uint32_t i, float *x){
    int D=v->dim;
    if(v->vf){ memcpy(x, v->vf+(size_t)i*D, (size_t)D*sizeof(float)); return; }
    const int8_t *r=vrow(v,i); float inv = v->gs>0 ? 1.0f/v->gs : 1.0f;
    for(int d=0;d<D;d++) x[d]=(float)r[d]*inv;
}
typedef struct { const KciVec *v; const struct KvQg *g; uint8_t *cv; } KvQgEnc;
static void kv_qg_enc_run(void *a, size_t lo, size_t hi){ KvQgEnc *e=a; int D=e->v->dim;
    float x[D], zero[D]; memset(zero,0,sizeof zero);
    for(size_t i=lo;i<hi;i++){ kv_qg_row(e->v, (uint32_t)i, x); kv_encode_res(e->g->pq, e->g->M, e->g->dp, e->g->eta, x, zero, D, e->cv+i*(size_t)e->g->M); }
    kv_wipe(x, sizeof x);
}
typedef struct { KciVec *v; struct KvQg *g; const uint8_t *cv; } KvQgFill;
static void kv_qg_fill_run(void *a, size_t lo, size_t hi){ KvQgFill *f=a; KciVec *v=f->v; struct KvQg *g=f->g; int M=g->M;
    for(size_t i=lo;i<hi;i++){
        uint8_t *b=g->blocks+i*g->stride; memset(b,0,g->stride);
        const uint32_t *lk=g_l0(v,(uint32_t)i); uint32_t cnt=lk[0]; if(cnt>(uint32_t)v->M0) cnt=(uint32_t)v->M0;
        uint32_t *ids=(uint32_t*)(b+4), w=0;
        for(uint32_t j=0;j<cnt;j++){ uint32_t nb=lk[1+j]; if(nb>=v->gN) continue;
            ids[w]=nb; const uint8_t *cd=f->cv+(size_t)nb*M;
            if(w < (uint32_t)g->nfull*32){
                uint8_t *codes=b+g->hdr+(size_t)(w>>5)*M*16; int lo16=(w&31)<16;
                for(int m=0;m<M;m++){ uint8_t *byte=codes+(size_t)m*16+(w&15); *byte = lo16 ? (uint8_t)((*byte&0xF0)|cd[m]) : (uint8_t)((*byte&0x0F)|(cd[m]<<4)); }
            } else {
                uint8_t *codes=b+g->hdr+(size_t)g->nfull*M*16; uint32_t j=w-(uint32_t)g->nfull*32;
                for(int m=0;m<M;m++){ uint8_t *byte=codes+(size_t)(m>>1)*16+j; *byte = (m&1) ? (uint8_t)((*byte&0x0F)|(cd[m]<<4)) : (uint8_t)((*byte&0xF0)|cd[m]); }
            }
            w++; }
        *(uint32_t*)b=w;
    }
}
int kci_vec_build_qg(KciVec *v, int dims_per_code, int threads){
    if(!v || !v->built || !v->l0 || v->gN==0 || v->gN!=v->count) return -1;
    if(dims_per_code!=0 && dims_per_code!=1 && dims_per_code!=2) return -1;
    int D=v->dim, dp = dims_per_code ? dims_per_code : 1;
    struct KvQg *g=calloc(1,sizeof *g); if(!g) return -1;
    g->dp=dp; g->M = dp==2 ? (D+1)/2 : D; g->nblk=(v->M0+31)/32;
    g->nfull=v->M0/32; g->half = v->M0%32>0 && v->M0%32<=16;
    if(!g->half) g->nfull=g->nblk;
    { const char *e=getenv("KCI_VEC_QG_NO_HALF"); if(e && *e=='1'){ g->half=0; g->nfull=g->nblk; } }
    g->hdr=((size_t)4+4*(size_t)v->M0+63)/64*64;
    g->stride=(g->hdr+(size_t)g->nfull*g->M*16+(g->half ? (size_t)((g->M+1)/2)*16 : 0)+63)/64*64;
    float t = v->ivf_avqt>0 ? v->ivf_avqt : (float)KV_IVF_T;
    g->eta=(D-1)*(double)t*t/(1-(double)t*t);
    size_t N=v->gN, ns = N<65536 ? N : 65536, step=N/ns, bytes=N*g->stride;
    float *S=malloc(ns*(size_t)D*sizeof(float)); uint8_t *cv=malloc(N*(size_t)g->M); g->pq=malloc((size_t)(g->M+1)*32*sizeof(float));
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    { void *p=NULL; size_t al=(size_t)2<<20, sz=(bytes+al-1)/al*al;
      if(bytes>=al && posix_memalign(&p,al,sz)==0){ madvise(p,sz,MADV_HUGEPAGE); g->blocks=p; } }
#endif
    if(!g->blocks) g->blocks=malloc(bytes);
    int ok = S && cv && g->pq && g->blocks;
    if(ok){
        for(size_t i=0;i<ns;i++) kv_qg_row(v, (uint32_t)(i*step), S+i*(size_t)D);          /* 1. codebooks from a sample */
        size_t np = ns<KV_PQ_SAMPLE ? ns : KV_PQ_SAMPLE;
        KvPqJob pj={S, np, ns/np, D, g->M, g->pq, 0, dp}; kv_par((size_t)g->M, threads, kv_pq_run, &pj); ok=!pj.fail;
    }
    if(ok){ KvQgEnc e={v,g,cv}; kv_par(N, threads, kv_qg_enc_run, &e); }                   /* 2. every node's codes */
    if(ok){ KvQgFill fj={v,g,cv}; kv_par(N, threads, kv_qg_fill_run, &fj); g->n=(uint32_t)N; } /* 3. the blocks */
    if(S) kv_wipe(S, ns*(size_t)D*sizeof(float)); if(cv) kv_wipe(cv, N*(size_t)g->M);
    free(S); free(cv);
    if(!ok){ kv_qg_free(g); return -1; }
    kv_qg_free(v->qg); v->qg=g;
    return 0;
}
int64_t kci_vec_qg_bytes(const KciVec *v){ return v && v->qg ? (int64_t)((size_t)v->qg->n*v->qg->stride + (size_t)(v->qg->M+1)*32*sizeof(float)) : 0; }

static inline void kv_qg_sum32(const uint8_t *blk, const uint8_t *lut, int M, uint16_t *out){
#if defined(__AVX2__)
    __m256i aEl=_mm256_setzero_si256(), aOl=aEl, aEh=aEl, aOh=aEl;
    const __m256i m4=_mm256_set1_epi8(0x0F);
    int m=0;
  #if defined(__AVX512BW__)
    { const __m512i z4=_mm512_set1_epi8(0x0F);
      __m512i bEl=_mm512_setzero_si512(), bOl=bEl, bEh=bEl, bOh=bEl;
#if KV_QG_LUT_TOP <= 63
      for(; m+16<=M; m+=16){
        __m512i vl=_mm512_setzero_si512(), vh=vl;
        for(int u=0;u<4;u++){ __m512i bb=_mm512_loadu_si512((const void*)(blk+m*16+64*u)), L=_mm512_loadu_si512((const void*)(lut+m*16+64*u));
            vl=_mm512_add_epi8(vl,_mm512_shuffle_epi8(L,_mm512_and_si512(bb,z4)));
            vh=_mm512_add_epi8(vh,_mm512_shuffle_epi8(L,_mm512_and_si512(_mm512_srli_epi16(bb,4),z4))); }
        bEl=_mm512_add_epi16(bEl,vl); bOl=_mm512_add_epi16(bOl,_mm512_srli_epi16(vl,8));
        bEh=_mm512_add_epi16(bEh,vh); bOh=_mm512_add_epi16(bOh,_mm512_srli_epi16(vh,8)); }
#endif
      for(; m+4<=M; m+=4){
        __m512i bb=_mm512_loadu_si512((const void*)(blk+m*16)), L=_mm512_loadu_si512((const void*)(lut+m*16));
        __m512i vl=_mm512_shuffle_epi8(L,_mm512_and_si512(bb,z4)), vh=_mm512_shuffle_epi8(L,_mm512_and_si512(_mm512_srli_epi16(bb,4),z4));
        bEl=_mm512_add_epi16(bEl,vl); bOl=_mm512_add_epi16(bOl,_mm512_srli_epi16(vl,8));
        bEh=_mm512_add_epi16(bEh,vh); bOh=_mm512_add_epi16(bOh,_mm512_srli_epi16(vh,8)); }
      aEl=_mm256_add_epi16(_mm512_castsi512_si256(bEl),_mm512_extracti64x4_epi64(bEl,1));
      aOl=_mm256_add_epi16(_mm512_castsi512_si256(bOl),_mm512_extracti64x4_epi64(bOl,1));
      aEh=_mm256_add_epi16(_mm512_castsi512_si256(bEh),_mm512_extracti64x4_epi64(bEh,1));
      aOh=_mm256_add_epi16(_mm512_castsi512_si256(bOh),_mm512_extracti64x4_epi64(bOh,1)); }
  #endif
#if KV_QG_LUT_TOP <= 63
    for(; m+8<=M; m+=8){
        __m256i vl=_mm256_setzero_si256(), vh=vl;
        for(int u=0;u<4;u++){ __m256i bb=_mm256_loadu_si256((const __m256i*)(blk+m*16+32*u)), L=_mm256_loadu_si256((const __m256i*)(lut+m*16+32*u));
            vl=_mm256_add_epi8(vl,_mm256_shuffle_epi8(L,_mm256_and_si256(bb,m4)));
            vh=_mm256_add_epi8(vh,_mm256_shuffle_epi8(L,_mm256_and_si256(_mm256_srli_epi16(bb,4),m4))); }
        aEl=_mm256_add_epi16(aEl,vl); aOl=_mm256_add_epi16(aOl,_mm256_srli_epi16(vl,8));
        aEh=_mm256_add_epi16(aEh,vh); aOh=_mm256_add_epi16(aOh,_mm256_srli_epi16(vh,8)); }
#endif
    for(; m+2<=M; m+=2){
        __m256i bb=_mm256_loadu_si256((const __m256i*)(blk+m*16)), L=_mm256_loadu_si256((const __m256i*)(lut+m*16));
        __m256i vl=_mm256_shuffle_epi8(L,_mm256_and_si256(bb,m4)), vh=_mm256_shuffle_epi8(L,_mm256_and_si256(_mm256_srli_epi16(bb,4),m4));
        aEl=_mm256_add_epi16(aEl,vl); aOl=_mm256_add_epi16(aOl,_mm256_srli_epi16(vl,8));
        aEh=_mm256_add_epi16(aEh,vh); aOh=_mm256_add_epi16(aOh,_mm256_srli_epi16(vh,8)); }
    __m128i El=_mm_add_epi16(_mm256_castsi256_si128(aEl),_mm256_extracti128_si256(aEl,1));
    __m128i Ol=_mm_add_epi16(_mm256_castsi256_si128(aOl),_mm256_extracti128_si256(aOl,1));
    __m128i Eh=_mm_add_epi16(_mm256_castsi256_si128(aEh),_mm256_extracti128_si256(aEh,1));
    __m128i Oh=_mm_add_epi16(_mm256_castsi256_si128(aOh),_mm256_extracti128_si256(aOh,1));
    if(m<M){ const __m128i m4s=_mm_set1_epi8(0x0F);
        __m128i bb=_mm_loadu_si128((const __m128i*)(blk+m*16)), L=_mm_loadu_si128((const __m128i*)(lut+m*16));
        __m128i vl=_mm_shuffle_epi8(L,_mm_and_si128(bb,m4s)), vh=_mm_shuffle_epi8(L,_mm_and_si128(_mm_srli_epi16(bb,4),m4s));
        El=_mm_add_epi16(El,vl); Ol=_mm_add_epi16(Ol,_mm_srli_epi16(vl,8));
        Eh=_mm_add_epi16(Eh,vh); Oh=_mm_add_epi16(Oh,_mm_srli_epi16(vh,8)); }
    El=_mm_sub_epi16(El,_mm_slli_epi16(Ol,8)); Eh=_mm_sub_epi16(Eh,_mm_slli_epi16(Oh,8));   /* even slots alone (mod 2^16: exact, sums <= 65,000) */
    _mm_storeu_si128((__m128i*)(out),    _mm_unpacklo_epi16(El,Ol)); _mm_storeu_si128((__m128i*)(out+8),  _mm_unpackhi_epi16(El,Ol));
    _mm_storeu_si128((__m128i*)(out+16), _mm_unpacklo_epi16(Eh,Oh)); _mm_storeu_si128((__m128i*)(out+24), _mm_unpackhi_epi16(Eh,Oh));
#else
    uint32_t sum[32]; memset(sum,0,sizeof sum);
    for(int m=0;m<M;m++){ const uint8_t *bb=blk+m*16, *L=lut+m*16; for(int j=0;j<16;j++){ sum[j]+=L[bb[j]&15]; sum[16+j]+=L[bb[j]>>4]; } }
    for(int j=0;j<32;j++) out[j]=(uint16_t)sum[j];
#endif
}
static inline void kv_qg_sum16(const uint8_t *blk, const uint8_t *lutE, const uint8_t *lutO, int P, uint16_t *out){
#if defined(__AVX2__)
    __m256i aE=_mm256_setzero_si256(), aO=aE; const __m256i m4=_mm256_set1_epi8(0x0F);
    int p=0;
    for(; p+2<=P; p+=2){
        __m256i bb=_mm256_loadu_si256((const __m256i*)(blk+p*16));
        __m256i vl=_mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i*)(lutE+p*16)),_mm256_and_si256(bb,m4));
        __m256i vh=_mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i*)(lutO+p*16)),_mm256_and_si256(_mm256_srli_epi16(bb,4),m4));
        aE=_mm256_add_epi16(aE,vl); aO=_mm256_add_epi16(aO,_mm256_srli_epi16(vl,8));
        aE=_mm256_add_epi16(aE,vh); aO=_mm256_add_epi16(aO,_mm256_srli_epi16(vh,8)); }
    __m128i E=_mm_add_epi16(_mm256_castsi256_si128(aE),_mm256_extracti128_si256(aE,1));
    __m128i O=_mm_add_epi16(_mm256_castsi256_si128(aO),_mm256_extracti128_si256(aO,1));
    if(p<P){ const __m128i m4s=_mm_set1_epi8(0x0F); __m128i bb=_mm_loadu_si128((const __m128i*)(blk+p*16));
        __m128i vl=_mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(lutE+p*16)),_mm_and_si128(bb,m4s));
        __m128i vh=_mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(lutO+p*16)),_mm_and_si128(_mm_srli_epi16(bb,4),m4s));
        E=_mm_add_epi16(E,vl); O=_mm_add_epi16(O,_mm_srli_epi16(vl,8));
        E=_mm_add_epi16(E,vh); O=_mm_add_epi16(O,_mm_srli_epi16(vh,8)); }
    E=_mm_sub_epi16(E,_mm_slli_epi16(O,8));                       /* even slots alone, as in sum32 */
    _mm_storeu_si128((__m128i*)(out),   _mm_unpacklo_epi16(E,O)); _mm_storeu_si128((__m128i*)(out+8), _mm_unpackhi_epi16(E,O));
#else
    uint32_t sum[16]; memset(sum,0,sizeof sum);
    for(int p=0;p<P;p++){ const uint8_t *bb=blk+p*16, *LE=lutE+p*16, *LO=lutO+p*16; for(int j=0;j<16;j++) sum[j]+=LE[bb[j]&15]+LO[bb[j]>>4]; }
    for(int j=0;j<16;j++) out[j]=(uint16_t)sum[j];
#endif
}
#define KV_BQ_N 4096
typedef struct { float d; uint32_t id; int32_t next; } KvBqE;
struct KvBq { KvBqE *pool; int n, cap; int32_t head[KV_BQ_N]; uint64_t bits[KV_BQ_N/64]; uint64_t top; };
static void kv_bq_free(struct KvBq *q){ if(q){ free(q->pool); free(q); } }
static struct KvBq *kv_bq_new(void){ struct KvBq *q=malloc(sizeof *q); if(!q) return NULL; q->cap=4096; q->pool=malloc(sizeof(KvBqE)*q->cap); if(!q->pool){ free(q); return NULL; } return q; }
static inline void kv_bq_reset(struct KvBq *q){ memset(q->head,0xFF,sizeof q->head); memset(q->bits,0,sizeof q->bits); q->top=0; q->n=0; }
static inline int kv_bq_idx(float d){ int b=(int)((d+1.0f)*(KV_BQ_N/2.0f)); return b<0 ? 0 : (b>=KV_BQ_N ? KV_BQ_N-1 : b); }
static inline int kv_bq_lowest(const struct KvBq *q){ if(!q->top) return KV_BQ_N; int w=__builtin_ctzll(q->top); return (w<<6)+__builtin_ctzll(q->bits[w]); }
static inline int kv_bq_push(struct KvBq *q, float d, uint32_t id){
    if(q->n>=q->cap){ KvBqE *np=realloc(q->pool,sizeof(KvBqE)*(size_t)q->cap*2); if(!np) return -1; q->pool=np; q->cap*=2; }
    int b=kv_bq_idx(d); KvBqE *e=&q->pool[q->n]; e->d=d; e->id=id; e->next=q->head[b]; q->head[b]=q->n++;
    q->bits[b>>6] |= 1ull<<(b&63); q->top |= 1ull<<(b>>6); return 0;
}
static inline int kv_bq_pop(struct KvBq *q, float *d, uint32_t *id){
    if(!q->top) return 0;
    int w=__builtin_ctzll(q->top), b=(w<<6)+__builtin_ctzll(q->bits[w]);
    KvBqE *e=&q->pool[q->head[b]]; *d=e->d; *id=e->id; q->head[b]=e->next;
    if(e->next<0){ q->bits[w] &= ~(1ull<<(b&63)); if(!q->bits[w]) q->top &= ~(1ull<<w); }
    return 1;
}
/* Layer 0 over the quantized graph. A popped node's block gives the estimated cosine distance -(q.x) of every
 * neighbour at once; only those within the bound look at the visited marks (NGT's order: the marks are random
 * accesses, the estimates are not). W keeps ef estimates; eps (kci_vec_searcher_set_eps) is the radius as in
 * search_layer (gs2 = 1: cosine units). Returns the m entries of W in s->tmp, re-scored with the int8 rows
 * (d = -(int dot), search_layer's units) when rescore8, else left as estimates for the caller's float pass. */
#ifndef KV_QG_PFN_INTEL
#define KV_QG_PFN_INTEL 8
#endif
static size_t kv_qg_pfn_bytes(size_t stride){
    static int lines=-1;
    if(lines<0){ int n=0;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        unsigned a,b,c,d; if(__get_cpuid(0,&a,&b,&c,&d) && b==0x756e6547u && d==0x49656e69u && c==0x6c65746eu) n=KV_QG_PFN_INTEL;   /* "GenuineIntel" */
#endif
        lines=n; }
    return lines>0 && (size_t)lines*64<stride ? (size_t)lines*64 : stride;
}
static int kv_qg_collect(KciVec *v, Scr *s, const int8_t *q, int ef, int rescore8){
    const struct KvQg *g=v->qg; const int M=g->M;
    if(scr_fit(s, v->gN)!=0) return 0;
    if(s->lutM!=M){
        if(s->lut) kv_wipe(s->lut,(size_t)(s->lutM+4)*16); if(s->lutf) kv_wipe(s->lutf,(size_t)s->lutM*16*sizeof(float));
        free(s->lut); free(s->lutf); s->lut=malloc((size_t)(M+4)*16); s->lutf=malloc((size_t)M*16*sizeof(float)); s->lutM=M;
        if(!s->lut||!s->lutf){ s->lutM=0; return 0; } }
    float base, invs; kv_lut_build(g->pq, M, g->dp, s->qf, v->dim, s->lutf, s->lut, &base, &invs, KV_QG_LUT_TOP);
    uint32_t ep=v->ep;
    for(int l=v->maxlevel;l>0;l--){ search_layer(v,q,ep,1,l,s); ep=hp_top(&s->W).id; }   /* upper layers: int8 rows */
    Heap *W=&s->W; scr_next(s); W->n=0;
    const float k1 = s->eps>0 ? 1.0f+s->eps : 0.0f;
    float d0 = -(float)sdot(v,q,vrow(v,ep)) / (v->gs*v->gs);          /* the entry point exactly, in cosine units */
    if(!s->bq && !(s->bq=kv_bq_new())) return 0; struct KvBq *BQ=s->bq; kv_bq_reset(BQ);
    visited(s,ep); kv_bq_push(BQ,d0,ep); hp_push(W,d0,ep);
    uint16_t sums[g->nblk*32];
    const size_t pfn=kv_qg_pfn_bytes(g->stride);
    const int P=(M+1)/2; uint8_t lutE[g->half ? P*16 : 1], lutO[g->half ? P*16 : 1];
    if(g->half) for(int p=0;p<P;p++){ memcpy(lutE+p*16, s->lut+(size_t)(2*p)*16, 16);
        if(2*p+1<M) memcpy(lutO+p*16, s->lut+(size_t)(2*p+1)*16, 16); else memset(lutO+p*16, 0, 16); }
    const uint32_t full32=(uint32_t)g->nfull*32;
    for(He c; kv_bq_pop(BQ,&c.d,&c.id); ){
        if(c.d>sl_bound(W,ef,k1,1.0f)) break;
        const uint8_t *b=g->blocks+(size_t)c.id*g->stride;
        for(size_t o=0;o<g->stride;o+=64) __builtin_prefetch(b+o);   /* every line of the block at once */
#ifndef KV_QG_NO_PFNEXT
        { int lb=kv_bq_lowest(BQ); if(lb<KV_BQ_N){ const uint8_t *nx=g->blocks+(size_t)BQ->pool[BQ->head[lb]].id*g->stride; for(size_t o=0;o<pfn;o+=64) __builtin_prefetch(nx+o); } }
#endif
        uint32_t cnt=*(const uint32_t*)b; const uint32_t *ids=(const uint32_t*)(b+4);
        if(cnt>(uint32_t)v->M0) cnt=(uint32_t)v->M0;
        for(uint32_t k=0;k<(uint32_t)g->nfull && k*32<cnt;k++) kv_qg_sum32(b+g->hdr+(size_t)k*M*16, s->lut, M, sums+k*32);
        if(g->half && cnt>full32) kv_qg_sum16(b+g->hdr+(size_t)g->nfull*M*16, lutE, lutO, P, sums+full32);
        float bound=sl_bound(W,ef,k1,1.0f);
#if defined(__AVX2__)
        { const __m256 vb=_mm256_set1_ps(-base), vi=_mm256_set1_ps(-invs), vbound=_mm256_set1_ps(bound);
          for(uint32_t k0=0;k0<cnt;k0+=32){ uint32_t mask=0, lim = cnt-k0<32 ? cnt-k0 : 32;
            for(uint32_t u=0;u<lim;u+=8){
                __m256 dv=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(sums+k0+u)))),vi,vb);
                mask |= (uint32_t)_mm256_movemask_ps(_mm256_cmp_ps(dv,vbound,_CMP_LT_OQ))<<u; }
            if(lim<32) mask &= (1u<<lim)-1;
            while(mask){ uint32_t j=k0+(uint32_t)__builtin_ctz(mask); mask&=mask-1;
                float d=-(base+(float)sums[j]*invs);
                if(!(d<bound)) continue;
                uint32_t nb=ids[j];
                if(nb>=v->gN || visited(s,nb)) continue;
                if(kv_bq_idx(d)<=kv_bq_lowest(BQ)){ const uint8_t *nx=g->blocks+(size_t)nb*g->stride; for(size_t o=0;o<pfn;o+=64) __builtin_prefetch(nx+o); }
                if(kv_bq_push(BQ,d,nb)<0) return 0;
                if(W->n<ef || d<hp_top(W).d){ hp_push(W,d,nb); if(W->n>ef) hp_pop(W); bound=sl_bound(W,ef,k1,1.0f); }
            } } }
#else
        for(uint32_t j=0;j<cnt;j++){
            float d=-(base+(float)sums[j]*invs);
            if(!(d<bound)) continue;
            uint32_t nb=ids[j];
            if(nb>=v->gN || visited(s,nb)) continue;
            if(kv_bq_push(BQ,d,nb)<0) return 0;
            if(W->n<ef || d<hp_top(W).d){ hp_push(W,d,nb); if(W->n>ef) hp_pop(W); bound=sl_bound(W,ef,k1,1.0f); }
        }
#endif
    }
    int m=W->n;
    if(m>s->tmpcap){ s->tmpcap=m; s->tmp=realloc(s->tmp,sizeof(He)*s->tmpcap); }
    memcpy(s->tmp, W->a, sizeof(He)*(size_t)m);
    if(rescore8){ He *t=s->tmp;
        for(int i=0;i<m;i++){ const int8_t *r=vrow(v,t[i].id); __builtin_prefetch(r); __builtin_prefetch(r+v->dim-1); }
        for(int i=0;i<m;i++) t[i].d=vdist(v,q,vrow(v,t[i].id)); }
    return m;
}

int kci_vec_set_float_source(KciVec *v, KciVecFloatFn fn, void *ctx, void (*ctx_free)(void *)){
    if(!v || !fn || !v->ivf || v->built || v->vf_mmap || v->img) return -1;
    if(v->ffree) v->ffree(v->fctx);
    v->ffn=fn; v->fctx=ctx; v->ffree=ctx_free;
    if(v->vf){ kv_wipe(v->vf, (size_t)v->cap*v->dim*sizeof(float));
        if(v->protect) kv_unlock_region(v->vf, (size_t)v->cap*v->dim*sizeof(float));
        vf_free(v); v->vf=NULL; }
    v->keepf=0; v->building=0;
    return 0;
}

/* ── public: exact brute-force (ground truth), multithreaded ─────────────── */
typedef struct { KciVec *v; const int8_t *q; const float *qf; uint32_t lo, hi; int k; Heap h; } ExactJob;
static void* exact_worker(void *arg){
    ExactJob *j=arg; KciVec *v=j->v; int k=j->k;
    int usef = j->qf != NULL;
    hp_init(&j->h, k, 0);                       /* min-heap of top-k by score */
    for(uint32_t i=j->lo;i<j->hi;i++){
        if(v->dead && v->dead[i]) continue;
        float sc = usef ? fdot(v->dim, j->qf, v->vf+(size_t)i*v->dim) : (float)sdot(v, j->q, vrow(v,i));
        if(j->h.n<k) hp_push(&j->h, sc, i);
        else if(sc>hp_top(&j->h).d){ hp_pop(&j->h); hp_push(&j->h, sc, i); }
    }
    return NULL;
}
static int kv_search_exact_(KciVec *v, const float *query, int k, int threads,
                            uint64_t *out_ids, float *out_scores){
    if(!v || v->count==0) return -1;
    int usef = v->keepf && v->vf;               /* floats are unit vectors: fdot = cosine */
    if(!usef && !v->built && !v->ivf) return -1; /* no floats and no codes yet */
    if(k<1) k=1;
    int8_t *q = malloc(v->dim);
    float *tmp = malloc((size_t)v->dim*sizeof(float));
    if(usef) l2norm(v->dim, query, tmp); else enc_query(v, query, tmp, q);
    int T = default_threads(threads);
    if((uint32_t)T > v->count) T = 1;
    pthread_t th[64]; ExactJob jobs[64];
    uint32_t per = v->count / T;
    for(int t=0;t<T;t++){
        jobs[t].v=v; jobs[t].q=q; jobs[t].qf=usef?tmp:NULL; jobs[t].k=k;
        jobs[t].lo = (uint32_t)t*per;
        jobs[t].hi = (t==T-1)? v->count : (uint32_t)(t+1)*per;
        if(t<T-1) pthread_create(&th[t],NULL,exact_worker,&jobs[t]);
    }
    exact_worker(&jobs[T-1]);
    for(int t=0;t<T-1;t++) pthread_join(th[t],NULL);
    /* merge per-thread heaps into a global top-k */
    Heap g; hp_init(&g, k, 0);
    for(int t=0;t<T;t++) for(int i=0;i<jobs[t].h.n;i++){
        He e=jobs[t].h.a[i];
        if(g.n<k) hp_push(&g, e.d, e.id);
        else if(e.d>hp_top(&g).d){ hp_pop(&g); hp_push(&g, e.d, e.id); }
    }
    int total=g.n;
    /* extract ascending then reverse → best (largest score) first */
    He *arr=malloc(sizeof(He)*total);
    for(int i=total-1;i>=0;i--){ arr[i]=hp_top(&g); hp_pop(&g); }
    for(int i=0;i<total;i++){ out_ids[i]=v->vid[arr[i].id]; if(out_scores) out_scores[i]=arr[i].d; }
    free(arr); free(g.a);
    for(int t=0;t<T;t++) free(jobs[t].h.a);
    free(q); free(tmp);
    return total;
}

/* ── public: incremental update (single-writer; no concurrent search) ──────
 * Insert one vector into the already-built graph, delete-by-id via tombstone, and
 * compact (rebuild) to reclaim tombstoned slots. These mutate the graph in place
 * (realloc may move it), so they must not run concurrently with searches — pause
 * serving, batch the updates, resume. Bulk loads should still use add()+build(). */
int64_t kci_vec_insert(KciVec *v, uint64_t id, const float *vec){
    if(!v || (!v->built && !v->ivf) || v->img) return -1;
    if(v->ffn && !v->built){
        uint32_t i = v->count;
        if(kci_vec_reserve(v, i+1) != 0 || kv_ensure_dead(v) != 0) return -1;
        float x[v->dim]; l2norm(v->dim, vec, x);
        sq_encode(v, x, v->vq+(size_t)i*v->dim);
        v->vid[i]=id; v->dead[i]=0; v->count=i+1;
        int rc = kv_ivf_add_x(v, i, x);
        kv_wipe(x, sizeof x);
        if(rc != 0) return -1;
        h_put(v, id, i);
        return v->count;
    }
    if(!v->keepf || !v->vf || v->vf_mmap) return -2;   /* need retained, writable floats */
    uint32_t i = v->count;
    if(kci_vec_reserve(v, i+1) != 0) return -1;   /* grows vq/vid/vf (keepf) */
    if(!v->built){
        if(kv_ensure_dead(v) != 0) return -1;
        store_float_norm(v, i, vec);
        sq_encode(v, v->vf+(size_t)i*v->dim, v->vq+(size_t)i*v->dim);
        v->vid[i]=id; v->dead[i]=0; v->count=i+1;
        if(kv_ivf_add(v, i) != 0) return -1;
        h_put(v, id, i);
        return v->count;
    }
    if(ensure_graph_cap(v, i+1) != 0) return -1;  /* grows level/l0/lu/lock/dead */
    store_float_norm(v, i, vec);                  /* vf[i] = unit(vec) */
    sq_encode(v, v->vf+(size_t)i*v->dim, v->vq+(size_t)i*v->dim);
    v->vid[i] = id;
    v->dead[i] = 0;
    double r=(double)(rand()+1)/((double)RAND_MAX+2.0);
    int lvl=(int)(-log(r)*v->mL); if(lvl>255)lvl=255; v->level[i]=(uint8_t)lvl;
    v->lu[i]=NULL;
    if(lvl>0){ v->lu[i]=calloc((size_t)lvl*(v->M+1),sizeof(uint32_t)); if(!v->lu[i]) return -1; }
    v->count = i+1; v->gN = v->count;             /* node is now addressable */
    Scr s; scr_init(&s, v->gN, v->efc+16);
    if(!s.vis){ scr_free(&s); return -1; }
    insert_node(v, i, &s);                        /* greedy connect into the live graph */
    scr_free(&s);
    if(v->qg){ kv_qg_free(v->qg); v->qg=NULL; }
    h_put(v, id, i);
    if(v->ivf && kv_ivf_add(v, i) != 0) return -1;
    return v->count;
}

/* Tombstone the vector with caller id. Returns 0 on success, -1 if not found.
 * The node stays in the graph as a routing hop but is excluded from results. */
int kci_vec_delete(KciVec *v, uint64_t id){
    if(!v || (!v->built && !v->ivf) || v->img) return -1;
    uint32_t idx = h_get(v, id);
    if(idx==KV_HEMPTY || idx>=v->count) return -1;
    if(v->dead && !v->dead[idx]){ v->dead[idx]=1; v->ndead++; }
    return 0;
}

/* a row's unit vector: from vf, or from the float source (normalized). 0 / -1 */
static int kv_row_float(KciVec *v, uint32_t r, float *out){
    int D=v->dim;
    if(v->vf){ memcpy(out, v->vf+(size_t)r*D, (size_t)D*sizeof(float)); return 0; }
    if(!v->ffn) return -1;
    float x[D];
    int rc = v->ffn(v->vid[r], x, v->fctx);
    if(rc==0) l2norm(D, x, out);
    kv_wipe(x, sizeof x);
    return rc==0 ? 0 : -1;
}

static int kv_ivf_recenter(KciVec *v, int c){
    struct KvIvf *f=v->ivf; int D=v->dim; KvList *L=&f->lists[c];
    double acc[D]; float x[D]; uint8_t cd[f->M+1]; int nlive=0;
    for(int d=0;d<D;d++) acc[d]=0;
    for(uint32_t j=0;j<L->n;j++){ uint32_t r=L->ids[j];
        if(r==UINT32_MAX || r>=v->count || (v->dead && v->dead[r])) continue;
        if(kv_row_float(v, r, x)!=0) continue;
        for(int d=0;d<D;d++) acc[d]+=x[d];
        nlive++; }
    float *ce=f->cent+(size_t)c*D; double n2=0;
    for(int d=0;d<D;d++) n2+=acc[d]*acc[d];
    for(int d=0;d<D;d++) ce[d] = nlive && n2>0 ? (float)(acc[d]/sqrt(n2)) : 0.0f;
    if(f->d1) kv_proj(f, D, ce, f->cent1+(size_t)c*f->d1);
    for(uint32_t j=0;j<L->n;j++){ uint32_t r=L->ids[j];
        if(r==UINT32_MAX || r>=v->count) continue;
        if(v->dead && v->dead[r]){ kv_block_clear(f, L->codes+(size_t)(j/32)*f->bb, j%32); L->ids[j]=UINT32_MAX; continue; }
        if(kv_row_float(v, r, x)!=0) continue;
        kv_ivf_encode(f, x, D, c, cd);
        kv_block_clear(f, L->codes+(size_t)(j/32)*f->bb, j%32);
        kv_block_put(f, L->codes+(size_t)(j/32)*f->bb, j%32, cd, NULL); }
    kv_wipe(acc, sizeof acc); kv_wipe(x, sizeof x);
    return 0;
}

int kci_vec_forget(KciVec *v, uint64_t id){
    if(!v || v->img) return -1;
    if(v->built || v->vf_mmap) return -2;
    if(v->ivf && (v->ivf->lr || (!v->vf && !v->ffn))) return -2;
    uint8_t *aff = v->ivf ? calloc((size_t)v->ivf->nlist, 1) : NULL;
    if(v->ivf && !aff) return -1;
    int removed=0; size_t D=(size_t)v->dim;
    for(uint32_t i=0;i<v->count;){
        if(v->vid[i]!=id){ i++; continue; }
        uint32_t last=v->count-1;
        if(v->ivf){ struct KvIvf *f=v->ivf;
            for(int c=0;c<f->nlist;c++){ KvList *L=&f->lists[c];
                for(uint32_t j=0;j<L->n;j++){
                    if(L->ids[j]==i){ L->ids[j]=UINT32_MAX; kv_block_clear(f, L->codes+(size_t)(j/32)*f->bb, j%32); aff[c]=1; }
                    else if(L->ids[j]==last && last!=i) L->ids[j]=i; } } }
        int was_dead = v->dead && v->dead[i];
        if(last!=i){
            memcpy(v->vq+(size_t)i*D, v->vq+(size_t)last*D, D);
            if(v->vf) memcpy(v->vf+(size_t)i*D, v->vf+(size_t)last*D, D*sizeof(float));
            v->vid[i]=v->vid[last];
            if(v->dead) v->dead[i]=v->dead[last];
        }
        kv_wipe(v->vq+(size_t)last*D, D);
        if(v->vf) kv_wipe(v->vf+(size_t)last*D, D*sizeof(float));
        v->vid[last]=0;
        if(v->dead) v->dead[last]=0;
        if(was_dead && v->ndead) v->ndead--;
        v->count--; removed++;                       /* row i now holds the former last row: test it again */
    }
    if(removed && v->hval && h_rebuild(v, v->hbits)!=0){ free(aff); return -1; }
    if(removed && v->ivf){
        struct KvIvf *f=v->ivf;
        for(int c=0;c<f->nlist;c++) if(aff[c]) kv_ivf_recenter(v, c);
        kv_wipe(f->c8, (size_t)(f->nlistp/16)*f->D4*64); free(f->c8); free(f->c8sum); f->c8=NULL; f->c8sum=NULL;
        if(kv_ivf_derive(f, v->dim)!=0){ free(aff); return -1; }
    }
    free(aff);
    return removed;
}

/* Rebuild the graph over only the live (non-tombstoned) vectors, reclaiming
 * tombstoned slots. Returns the new live count. Requires retained floats. */
uint32_t kci_vec_compact(KciVec *v){
    if(!v || (!v->built && !v->ivf)) return 0;
    if(v->img) return v->count;
    if(v->ndead==0) return v->count;
    if(!v->keepf || !v->vf) return v->count;      /* cannot rebuild without floats */
    uint32_t live=0; for(uint32_t i=0;i<v->count;i++) if(!v->dead[i]) live++;
    if(live==0) return 0;
    float    *nf = malloc((size_t)live*v->dim*sizeof(float));
    uint64_t *nid= malloc((size_t)live*sizeof(uint64_t));
    if(!nf||!nid){ free(nf); free(nid); return v->count; }
    uint32_t j=0;
    for(uint32_t i=0;i<v->count;i++) if(!v->dead[i]){
        memcpy(nf+(size_t)j*v->dim, v->vf+(size_t)i*v->dim, (size_t)v->dim*sizeof(float));
        nid[j]=v->vid[i]; j++; }
    int had_graph=v->built, nl=v->ivf? v->ivf->nlist : 0;
    kv_ivf_free(v->ivf); v->ivf=NULL;
    graph_free(v);                                 /* frees level/l0/lu/lock/dead/hash */
    v->built=0; v->gcap=0;
    kv_vfree(v, v->vq, (size_t)v->cap*v->dim); v->vq=NULL;
    if(v->vf && !v->vf_mmap){ kv_vfree(v, v->vf, (size_t)v->cap*v->dim*sizeof(float)); v->vf=NULL; }
    vf_free(v); free(v->vid); v->vid=NULL;
    v->count=0; v->cap=0; v->building=1;
    if(kci_vec_reserve(v, live)!=0){ free(nf); free(nid); return 0; }
    for(uint32_t i=0;i<live;i++){
        store_float_norm(v,i,nf+(size_t)i*v->dim);
        v->vid[i]=nid[i]; }
    v->count=live;
    kv_wipe(nf, (size_t)live*v->dim*sizeof(float)); free(nf); free(nid);
    double secs;
    if(had_graph) kci_vec_build(v, 0, &secs);
    if(nl) kci_vec_build_ivf(v, nl, 0);
    return v->count;
}

#define CC_ROT(x,n) (((x)<<(n))|((x)>>(32-(n))))
static void cc20_block(const uint8_t key[32], uint32_t counter, const uint8_t nonce[12], uint8_t out[64]){
    static const uint32_t C[4]={0x61707865,0x3320646e,0x79622d32,0x6b206574};
    uint32_t s[16],x[16];
    s[0]=C[0];s[1]=C[1];s[2]=C[2];s[3]=C[3];
    for(int i=0;i<8;i++) s[4+i]=(uint32_t)key[i*4]|(uint32_t)key[i*4+1]<<8|(uint32_t)key[i*4+2]<<16|(uint32_t)key[i*4+3]<<24;
    s[12]=counter;
    for(int i=0;i<3;i++) s[13+i]=(uint32_t)nonce[i*4]|(uint32_t)nonce[i*4+1]<<8|(uint32_t)nonce[i*4+2]<<16|(uint32_t)nonce[i*4+3]<<24;
    memcpy(x,s,sizeof x);
    for(int r=0;r<10;r++){
        #define QR(a,b,c,d) x[a]+=x[b];x[d]^=x[a];x[d]=CC_ROT(x[d],16); x[c]+=x[d];x[b]^=x[c];x[b]=CC_ROT(x[b],12); x[a]+=x[b];x[d]^=x[a];x[d]=CC_ROT(x[d],8); x[c]+=x[d];x[b]^=x[c];x[b]=CC_ROT(x[b],7);
        QR(0,4,8,12);QR(1,5,9,13);QR(2,6,10,14);QR(3,7,11,15);
        QR(0,5,10,15);QR(1,6,11,12);QR(2,7,8,13);QR(3,4,9,14);
        #undef QR
    }
    for(int i=0;i<16;i++){ uint32_t v=x[i]+s[i]; out[i*4]=v;out[i*4+1]=v>>8;out[i*4+2]=v>>16;out[i*4+3]=v>>24; }
}
/* XOR ChaCha20 keystream over buf. `chunk_id` becomes the nonce so each snapshot
 * chunk has a unique keystream; the block counter runs within the chunk. */
static void cc20_crypt(uint8_t *buf, uint32_t len, const uint8_t key[32], uint32_t chunk_id){
    uint8_t nonce[12]={0}; nonce[0]=(uint8_t)chunk_id; nonce[1]=(uint8_t)(chunk_id>>8);
    nonce[2]=(uint8_t)(chunk_id>>16); nonce[3]=(uint8_t)(chunk_id>>24);
    uint8_t ks[64]; uint32_t ctr=0, done=0;
    while(done<len){ cc20_block(key,ctr++,nonce,ks);
        uint32_t c=len-done<64?len-done:64;
        for(uint32_t i=0;i<c;i++) buf[done+i]^=ks[i];
        done+=c; }
}
/* derive a 32-byte key from an arbitrary caller key (PBKDF2-HMAC-SHA256, below). */
/* ── SHA-256 (FIPS 180-4) → HMAC → PBKDF2: proper key derivation ──────────── */
typedef struct { uint32_t h[8]; uint64_t bits; uint8_t buf[64]; size_t n; } SHA256;
static const uint32_t SHA256_K[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define SHR(x,n) ((x)>>(n))
#define ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void sha256_blk(SHA256*c,const uint8_t*p){
    uint32_t w[64],a,b,cc,d,e,f,g,h,t1,t2;
    for(int i=0;i<16;i++) w[i]=(uint32_t)p[i*4]<<24|p[i*4+1]<<16|p[i*4+2]<<8|p[i*4+3];
    for(int i=16;i<64;i++){ uint32_t s0=ROTR(w[i-15],7)^ROTR(w[i-15],18)^SHR(w[i-15],3);
        uint32_t s1=ROTR(w[i-2],17)^ROTR(w[i-2],19)^SHR(w[i-2],10); w[i]=w[i-16]+s0+w[i-7]+s1; }
    a=c->h[0];b=c->h[1];cc=c->h[2];d=c->h[3];e=c->h[4];f=c->h[5];g=c->h[6];h=c->h[7];
    for(int i=0;i<64;i++){ uint32_t S1=ROTR(e,6)^ROTR(e,11)^ROTR(e,25), ch=(e&f)^(~e&g);
        t1=h+S1+ch+SHA256_K[i]+w[i]; uint32_t S0=ROTR(a,2)^ROTR(a,13)^ROTR(a,22), maj=(a&b)^(a&cc)^(b&cc);
        t2=S0+maj; h=g;g=f;f=e;e=d+t1;d=cc;cc=b;b=a;a=t1+t2; }
    c->h[0]+=a;c->h[1]+=b;c->h[2]+=cc;c->h[3]+=d;c->h[4]+=e;c->h[5]+=f;c->h[6]+=g;c->h[7]+=h;
}
static void sha256_init(SHA256*c){ static const uint32_t H0[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(c->h,H0,sizeof H0); c->bits=0; c->n=0; }
static void sha256_upd(SHA256*c,const uint8_t*d,size_t n){ c->bits+=(uint64_t)n*8;
    while(n){ size_t k=64-c->n; if(k>n)k=n; memcpy(c->buf+c->n,d,k); c->n+=k; d+=k; n-=k;
        if(c->n==64){ sha256_blk(c,c->buf); c->n=0; } } }
static void sha256_fin(SHA256*c,uint8_t out[32]){ uint64_t bits=c->bits; uint8_t pad=0x80;
    sha256_upd(c,&pad,1); uint8_t z=0; while(c->n!=56) sha256_upd(c,&z,1);
    uint8_t lb[8]; for(int i=0;i<8;i++) lb[i]=(uint8_t)(bits>>(56-i*8)); sha256_upd(c,lb,8);
    for(int i=0;i<8;i++){ out[i*4]=c->h[i]>>24;out[i*4+1]=c->h[i]>>16;out[i*4+2]=c->h[i]>>8;out[i*4+3]=c->h[i]; } }
static void hmac_sha256(const uint8_t*key,size_t kl,const uint8_t*msg,size_t ml,uint8_t out[32]){
    uint8_t k[64]={0}, ip[64], op[64], ih[32]; SHA256 c;
    if(kl>64){ sha256_init(&c); sha256_upd(&c,key,kl); sha256_fin(&c,k); } else memcpy(k,key,kl);
    for(int i=0;i<64;i++){ ip[i]=k[i]^0x36; op[i]=k[i]^0x5c; }
    sha256_init(&c); sha256_upd(&c,ip,64); sha256_upd(&c,msg,ml); sha256_fin(&c,ih);
    sha256_init(&c); sha256_upd(&c,op,64); sha256_upd(&c,ih,32); sha256_fin(&c,out);
}
/* PBKDF2-HMAC-SHA256, dkLen=32 (one block) */
static void pbkdf2(const uint8_t*pw,size_t pl,const uint8_t*salt,size_t sl,uint32_t iters,uint8_t out32[32]){
    uint8_t s[64],u[32],t[32]; size_t n=sl<60?sl:60; memcpy(s,salt,n);
    s[n]=0;s[n+1]=0;s[n+2]=0;s[n+3]=1;                      /* INT(1) big-endian */
    hmac_sha256(pw,pl,s,n+4,u); memcpy(t,u,32);
    for(uint32_t i=1;i<iters;i++){ hmac_sha256(pw,pl,u,32,u); for(int j=0;j<32;j++) t[j]^=u[j]; }
    memcpy(out32,t,32);
}
#define KV_KDF_ITERS 100000u
static void kdf_derive(const uint8_t *key, int keylen, const uint8_t *salt, int saltlen, uint8_t out32[32]){
    pbkdf2(key,(size_t)keylen,salt,(size_t)saltlen,KV_KDF_ITERS,out32);
}
/* cryptographic salt: /dev/urandom, with a hashed-entropy fallback. */
static void kv_randbytes(uint8_t *out, int n){
    FILE *f=fopen("/dev/urandom","rb");
    if(f){ size_t r=fread(out,1,n,f); fclose(f); if(r==(size_t)n) return; }
    static uint64_t ctr=0; struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    uint8_t seed[40]; uintptr_t a1=(uintptr_t)out, a2=(uintptr_t)&ts; ctr++;
    memcpy(seed,&ts,sizeof ts); memcpy(seed+16,&a1,8); memcpy(seed+24,&a2,8); memcpy(seed+32,&ctr,8);
    uint8_t h[32]; SHA256 c; sha256_init(&c); sha256_upd(&c,seed,sizeof seed); sha256_fin(&c,h);
    memcpy(out,h,n<32?n:32);
}

static const uint8_t kv_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16 };
static inline uint8_t kv_xt(uint8_t x){ return (uint8_t)((x<<1)^((x>>7)*0x1b)); }

/* FIPS-197 key expansion: 15 round keys in standard byte order */
static void kv_aes_expand(const uint8_t key[32], uint8_t w[240]){
    uint8_t rcon=1;
    memcpy(w,key,32);
    for(int i=8;i<60;i++){
        uint8_t t[4]; memcpy(t,w+4*(i-1),4);
        if(i%8==0){ uint8_t t0=t[0];
            t[0]=kv_sbox[t[1]]^rcon; t[1]=kv_sbox[t[2]]; t[2]=kv_sbox[t[3]]; t[3]=kv_sbox[t0];
            rcon=kv_xt(rcon); }
        else if(i%8==4){ for(int k=0;k<4;k++) t[k]=kv_sbox[t[k]]; }
        for(int k=0;k<4;k++) w[4*i+k]=w[4*(i-8)+k]^t[k];
    }
}
/* portable AES-256 encryption of one block */
static void kv_aes_block_soft(const uint8_t w[240], const uint8_t in[16], uint8_t out[16]){
    uint8_t s[16], t[16];
    for(int i=0;i<16;i++) s[i]=in[i]^w[i];
    for(int r=1;r<=14;r++){
        for(int i=0;i<16;i++) t[i]=kv_sbox[s[i]];                                   /* SubBytes  */
        for(int c=0;c<4;c++) for(int q=0;q<4;q++) s[q+4*c]=t[q+4*((c+q)&3)];          /* ShiftRows */
        if(r!=14) for(int c=0;c<4;c++){                                              /* MixColumns */
            uint8_t *col=s+4*c, a0=col[0], a1=col[1], a2=col[2], a3=col[3], x=a0^a1^a2^a3;
            col[0]=a0^x^kv_xt(a0^a1); col[1]=a1^x^kv_xt(a1^a2);
            col[2]=a2^x^kv_xt(a2^a3); col[3]=a3^x^kv_xt(a3^a0); }
        for(int i=0;i<16;i++) s[i]^=w[16*r+i];
    }
    memcpy(out,s,16);
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#define KV_AES_X86 1
#include <wmmintrin.h>
#include <emmintrin.h>
#define KV_AESNI_FN __attribute__((target("aes,sse2")))
KV_AESNI_FN static void kv_aes_block_ni(const uint8_t w[240], const uint8_t in[16], uint8_t out[16]){
    const __m128i *k=(const __m128i*)w;
    __m128i b=_mm_xor_si128(_mm_loadu_si128((const __m128i*)in), _mm_loadu_si128(k));
    for(int r=1;r<14;r++) b=_mm_aesenc_si128(b,_mm_loadu_si128(k+r));
    _mm_storeu_si128((__m128i*)out, _mm_aesenclast_si128(b,_mm_loadu_si128(k+14)));
}
/* CTR keystream XOR, 4 blocks in flight. Counter block = [nonce LE 8 | block LE 8]. */
KV_AESNI_FN static void kv_ctr_ni(const uint8_t w[240], uint64_t nonce, uint8_t *buf, size_t len){
    __m128i rk[15]; for(int i=0;i<15;i++) rk[i]=_mm_loadu_si128((const __m128i*)(w+16*i));
    __m128i ctr=_mm_set_epi64x(0,(long long)nonce); const __m128i one=_mm_set_epi64x(1,0);
    size_t pos=0;
    for(; pos+64<=len; pos+=64){
        __m128i c0=ctr, c1=_mm_add_epi64(c0,one), c2=_mm_add_epi64(c1,one), c3=_mm_add_epi64(c2,one);
        ctr=_mm_add_epi64(c3,one);
        c0=_mm_xor_si128(c0,rk[0]); c1=_mm_xor_si128(c1,rk[0]); c2=_mm_xor_si128(c2,rk[0]); c3=_mm_xor_si128(c3,rk[0]);
        for(int r=1;r<14;r++){ c0=_mm_aesenc_si128(c0,rk[r]); c1=_mm_aesenc_si128(c1,rk[r]);
                               c2=_mm_aesenc_si128(c2,rk[r]); c3=_mm_aesenc_si128(c3,rk[r]); }
        c0=_mm_aesenclast_si128(c0,rk[14]); c1=_mm_aesenclast_si128(c1,rk[14]);
        c2=_mm_aesenclast_si128(c2,rk[14]); c3=_mm_aesenclast_si128(c3,rk[14]);
        __m128i *p=(__m128i*)(buf+pos);
        _mm_storeu_si128(p+0,_mm_xor_si128(_mm_loadu_si128(p+0),c0));
        _mm_storeu_si128(p+1,_mm_xor_si128(_mm_loadu_si128(p+1),c1));
        _mm_storeu_si128(p+2,_mm_xor_si128(_mm_loadu_si128(p+2),c2));
        _mm_storeu_si128(p+3,_mm_xor_si128(_mm_loadu_si128(p+3),c3));
    }
    for(; pos<len; pos+=16){
        __m128i b=_mm_xor_si128(ctr,rk[0]); ctr=_mm_add_epi64(ctr,one);
        for(int r=1;r<14;r++) b=_mm_aesenc_si128(b,rk[r]);
        b=_mm_aesenclast_si128(b,rk[14]);
        uint8_t ks[16]; _mm_storeu_si128((__m128i*)ks,b);
        size_t c=len-pos<16?len-pos:16;
        for(size_t i=0;i<c;i++) buf[pos+i]^=ks[i];
    }
}
/* CBC-MAC chain over n whole blocks (CMAC body): x = E(x ^ block) */
KV_AESNI_FN static void kv_cbcmac_ni(const uint8_t w[240], uint8_t x[16], const uint8_t *p, size_t n){
    __m128i rk[15]; for(int i=0;i<15;i++) rk[i]=_mm_loadu_si128((const __m128i*)(w+16*i));
    __m128i s=_mm_loadu_si128((const __m128i*)x);
    for(size_t j=0;j<n;j++){
        s=_mm_xor_si128(_mm_xor_si128(s,_mm_loadu_si128((const __m128i*)(p+16*j))),rk[0]);
        for(int r=1;r<14;r++) s=_mm_aesenc_si128(s,rk[r]);
        s=_mm_aesenclast_si128(s,rk[14]);
    }
    _mm_storeu_si128((__m128i*)x,s);
}
#elif defined(__aarch64__) && (defined(__ARM_FEATURE_AES) || defined(__ARM_FEATURE_CRYPTO))
#define KV_AES_ARM 1
/* AESE = AddRoundKey + SubBytes + ShiftRows; AESMC = MixColumns */
static void kv_aes_block_arm(const uint8_t w[240], const uint8_t in[16], uint8_t out[16]){
    uint8x16_t b=vld1q_u8(in);
    for(int r=0;r<13;r++) b=vaesmcq_u8(vaeseq_u8(b,vld1q_u8(w+16*r)));
    b=veorq_u8(vaeseq_u8(b,vld1q_u8(w+16*13)),vld1q_u8(w+16*14));
    vst1q_u8(out,b);
}
#endif

typedef struct { uint8_t rk[240]; int path; } KvAes;   /* path: 0 portable, 1 AES-NI, 2 ARMv8 */
static void kv_aes_block(const KvAes *a, const uint8_t in[16], uint8_t out[16]){
#if defined(KV_AES_X86)
    if(a->path==1){ kv_aes_block_ni(a->rk,in,out); return; }
#elif defined(KV_AES_ARM)
    if(a->path==2){ kv_aes_block_arm(a->rk,in,out); return; }
#endif
    kv_aes_block_soft(a->rk,in,out);
}
/* expand the key and pick the fastest path that gives the FIPS-197 C.3 answer */
static void kv_aes_init(KvAes *a, const uint8_t key[32]){
    kv_aes_expand(key,a->rk); a->path=0;
#if defined(KV_AES_X86) || defined(KV_AES_ARM)
    static const uint8_t pt[16]={0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff};
    static const uint8_t ct[16]={0x8e,0xa2,0xb7,0xca,0x51,0x67,0x45,0xbf,0xea,0xfc,0x49,0x90,0x4b,0x49,0x60,0x89};
    uint8_t k[32], w[240], o[16];
    for(int i=0;i<32;i++) k[i]=(uint8_t)i;
    kv_aes_expand(k,w);
#if defined(KV_AES_X86)
    if(__builtin_cpu_supports("aes")){ kv_aes_block_ni(w,pt,o); if(memcmp(o,ct,16)==0) a->path=1; }
#else
    kv_aes_block_arm(w,pt,o); if(memcmp(o,ct,16)==0) a->path=2;
#endif
#endif
}
/* XOR the AES-256-CTR keystream for `nonce` over buf (block counter from 0) */
static void kv_ctr(const KvAes *a, uint64_t nonce, uint8_t *buf, size_t len){
#if defined(KV_AES_X86)
    if(a->path==1){ kv_ctr_ni(a->rk,nonce,buf,len); return; }
#endif
    uint8_t cb[16], ks[16]; uint64_t blk=0;
    for(size_t pos=0; pos<len; pos+=16, blk++){
        for(int k=0;k<8;k++){ cb[k]=(uint8_t)(nonce>>(8*k)); cb[8+k]=(uint8_t)(blk>>(8*k)); }
        kv_aes_block(a,cb,ks);
        size_t c=len-pos<16?len-pos:16;
        for(size_t i=0;i<c;i++) buf[pos+i]^=ks[i];
    }
}

/* AES-256-CMAC (NIST SP 800-38B), streaming: the last full block is held back
 * until final, because it is padded with K1 instead of K2 */
typedef struct { KvAes a; uint8_t k1[16], k2[16], x[16], buf[16]; uint32_t fill; } KvCmac;
static void kv_cmac_dbl(const uint8_t in[16], uint8_t out[16]){
    uint8_t carry=(uint8_t)(in[0]>>7);
    for(int i=0;i<15;i++) out[i]=(uint8_t)((in[i]<<1)|(in[i+1]>>7));
    out[15]=(uint8_t)((in[15]<<1)^(carry*0x87));
}
static void kv_cmac_init(KvCmac *m, const uint8_t key[32]){
    memset(m,0,sizeof *m);
    kv_aes_init(&m->a,key);
    uint8_t l[16]={0}; kv_aes_block(&m->a,l,l);
    kv_cmac_dbl(l,m->k1); kv_cmac_dbl(m->k1,m->k2);
}
static void kv_cmac_absorb(KvCmac *m, const uint8_t *p, size_t nblk){
#if defined(KV_AES_X86)
    if(m->a.path==1){ kv_cbcmac_ni(m->a.rk,m->x,p,nblk); return; }
#endif
    for(size_t j=0;j<nblk;j++){ for(int i=0;i<16;i++) m->x[i]^=p[16*j+i]; kv_aes_block(&m->a,m->x,m->x); }
}
static void kv_cmac_update(KvCmac *m, const void *data, size_t len){
    const uint8_t *p=(const uint8_t*)data;
    while(len){
        if(m->fill==16){ kv_cmac_absorb(m,m->buf,1); m->fill=0; }   /* more data follows */
        if(m->fill==0 && len>16){                   /* whole blocks straight from the input */
            size_t n=(len-1)/16;                    /* keep at least 1 byte for the held block */
            kv_cmac_absorb(m,p,n); p+=16*n; len-=16*n;
            continue;
        }
        size_t take=16-m->fill; if(take>len) take=len;
        memcpy(m->buf+m->fill,p,take); m->fill+=(uint32_t)take; p+=take; len-=take;
    }
}
static void kv_cmac_final(KvCmac *m, uint8_t tag[16]){
    const uint8_t *k=m->k1;
    if(m->fill<16){ m->buf[m->fill]=0x80; memset(m->buf+m->fill+1,0,15-m->fill); k=m->k2; }
    for(int i=0;i<16;i++) m->x[i]^=m->buf[i]^k[i];
    kv_aes_block(&m->a,m->x,tag);
}
/* one key, one use: HMAC-SHA256(master, label) */
static void kv_subkey(const uint8_t master[32], const char *label, uint8_t out[32]){
    hmac_sha256(master,32,(const uint8_t*)label,strlen(label),out);
}

#define KV_ENC_CHUNK (256u*1024*1024)
typedef struct { int kind; const uint8_t *cc; KvAes aes; KvCmac mac; uint32_t chunk; } KvIO;
static int kv_wraw(FILE *f, const void *p, size_t n, KvIO *io){   /* header bytes: written as is, MAC'd */
    if(io->kind==2) kv_cmac_update(&io->mac,p,n);
    return fwrite(p,1,n,f)==n ? 0 : -1;
}
static int enc_fwrite(FILE *f, const void *src, size_t n, KvIO *io){
    if(n==0) return 0;
    if(io->kind==0) return fwrite(src,1,n,f)==n ? 0 : -1;
    uint8_t *tmp=malloc(n<KV_ENC_CHUNK?n:KV_ENC_CHUNK); if(!tmp) return -1;
    const uint8_t *p=src; size_t off=0; int ok=1;
    while(off<n){ size_t c=n-off<KV_ENC_CHUNK?n-off:KV_ENC_CHUNK;
        memcpy(tmp,p+off,c); kv_ctr(&io->aes,io->chunk++,tmp,c); kv_cmac_update(&io->mac,tmp,c);
        if(fwrite(tmp,1,c,f)!=c){ ok=0; break; } off+=c; }
    free(tmp); return ok?0:-1;
}
static int enc_fread(FILE *f, void *dst, size_t n, KvIO *io){
    if(n==0) return 0;
    if(fread(dst,1,n,f)!=n) return -1;
    if(io->kind==0) return 0;
    uint8_t *p=dst; size_t off=0;
    while(off<n){ size_t c=n-off<KV_ENC_CHUNK?n-off:KV_ENC_CHUNK;
        if(io->kind==1) cc20_crypt(p+off,(uint32_t)c,io->cc,io->chunk++);
        else kv_ctr(&io->aes,io->chunk++,p+off,c);
        off+=c; }
    return 0;
}
#ifdef _WIN32
#define kv_fseek _fseeki64
#define kv_ftell _ftelli64
#else
#define kv_fseek fseeko
#define kv_ftell ftello
#endif
#ifdef _WIN32
static int kv_replace_file(const char *from, const char *to){
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
}
#else
static int kv_replace_file(const char *from, const char *to){ return rename(from, to); }
#endif
static int kv_verify_file(FILE *f, KvCmac *m){
    int64_t here=(int64_t)kv_ftell(f);
    if(here<0 || kv_fseek(f,0,SEEK_END)!=0) return -1;
    int64_t size=(int64_t)kv_ftell(f);
    if(size<16+8+32 || kv_fseek(f,0,SEEK_SET)!=0) return -1;
    uint8_t *buf=malloc(1u<<20); if(!buf) return -1;
    int64_t left=size-16; int ok=1;
    while(left>0){ size_t c=left<(1<<20)?(size_t)left:(1u<<20);
        if(fread(buf,1,c,f)!=c){ ok=0; break; }
        kv_cmac_update(m,buf,c); left-=(int64_t)c; }
    uint8_t want[16], tag[16];
    if(ok && fread(want,1,16,f)!=16) ok=0;
    free(buf);
    if(!ok) return -1;
    kv_cmac_final(m,tag);
    uint8_t d=0; for(int i=0;i<16;i++) d|=(uint8_t)(tag[i]^want[i]);   /* constant time */
    if(kv_fseek(f,here,SEEK_SET)!=0) return -1;
    return d==0 ? 0 : -1;
}

#define KCIVEC_MAGIC  "KCIVEC04"   /* 04: signed int8 + global scale gs + tombstone bitmap */
#define KCIVEC_MAGIC5 "KCIVEC05"   /* 05: same layout, AES-256-CTR + CMAC */
#define KCIVEC_KEYCHK "KCIVEC-KEYCHECK!"   /* 16 bytes, encrypted as a key probe */
#define KV_FLAG_ENC   1u
#define KV_FLAG_GRAPH 2u
#define KV_FLAG_FLOAT 4u
#define KV_FLAG_IVF   8u
#define KV_FLAG_FLOAT_EXT 16u
#define KV_FLAG_RAWKEY 32u
#define KV_KC_TWEAK   0xFFFFFFFEu           /* keycheck CTR nonce (≠ payload chunks) */
#define KV_LABEL_ENC  "KCIVEC05 encryption key"
#define KV_LABEL_MAC  "KCIVEC05 mac key"

int kci_vec_save(KciVec *v, const char *path, const uint8_t *key, int keylen){
    return kci_vec_save_ex(v, path, key, keylen, 0);
}
int kci_vec_save_ex(KciVec *v, const char *path, const uint8_t *key, int keylen, int sflags){
    if(!v) return -1;
    int nofloat = (sflags & KCI_VEC_SAVE_NO_FLOAT) && v->vf && !v->vf_mmap;
    int rawkey = (sflags & KCI_VEC_SAVE_RAW_KEY) != 0;
    if(rawkey && (!key || keylen != 32)) return -1;
    KvIO io; memset(&io,0,sizeof io);
    uint8_t salt[16];
    if(key && keylen>0){
        uint8_t mk[32], ek[32], ak[32];
        kv_randbytes(salt,16);
        if(rawkey) hmac_sha256(key,32,salt,16,mk);
        else kdf_derive(key,keylen,salt,16,mk);
        kv_subkey(mk,KV_LABEL_ENC,ek); kv_subkey(mk,KV_LABEL_MAC,ak);
        kv_aes_init(&io.aes,ek); kv_cmac_init(&io.mac,ak); io.kind=2;
        memset(mk,0,32); memset(ek,0,32); memset(ak,0,32);
    }
    /* atomic save: write to <path>.tmp then rename, so a crash mid-save never
     * corrupts an existing good snapshot. */
    char tmp[1100];
    if(snprintf(tmp,sizeof(tmp),"%s.tmp",path) >= (int)sizeof(tmp)) return -1;
    FILE *f=fopen(tmp,"wb");
    if(!f) return -1;
    uint32_t flags = (io.kind?KV_FLAG_ENC:0) | (v->built?KV_FLAG_GRAPH:0) | (v->vf && !nofloat?KV_FLAG_FLOAT:0) | (v->ivf?KV_FLAG_IVF:0)
                   | (nofloat?KV_FLAG_FLOAT_EXT:0) | (rawkey && io.kind?KV_FLAG_RAWKEY:0);
    uint32_t hdr[8] = { flags, (uint32_t)v->dim, (uint32_t)v->M, (uint32_t)v->efc,
                        v->count, v->built?(uint32_t)v->maxlevel:0, v->ep, 0 };
    memcpy(&hdr[7], &v->gs, sizeof(float));            /* global SQ scale */
    int ok=1;
    ok &= kv_wraw(f, io.kind?KCIVEC_MAGIC5:KCIVEC_MAGIC, 8, &io)==0;
    ok &= kv_wraw(f, hdr, sizeof(hdr), &io)==0;
    if(io.kind) ok &= kv_wraw(f, salt, 16, &io)==0;   /* PBKDF2 salt (plaintext, not secret) */
    { uint8_t kc[16]; memcpy(kc,KCIVEC_KEYCHK,16);    /* key-check probe */
      if(io.kind) kv_ctr(&io.aes,KV_KC_TWEAK,kc,16);
      ok &= kv_wraw(f, kc, 16, &io)==0; }
    ok &= enc_fwrite(f, v->vid, (size_t)v->count*sizeof(uint64_t), &io)==0;
    ok &= enc_fwrite(f, v->vq,  (size_t)v->count*v->dim, &io)==0;
    if(v->vf && !nofloat) ok &= enc_fwrite(f, v->vf, (size_t)v->count*v->dim*sizeof(float), &io)==0;
    if(v->built){
        ok &= enc_fwrite(f, v->level, v->count, &io)==0;
        ok &= enc_fwrite(f, v->l0, (size_t)v->count*(v->M0+1)*sizeof(uint32_t), &io)==0;
        for(uint32_t i=0;i<v->count && ok;i++){ int lvl=v->level[i];
            if(lvl>0) ok &= enc_fwrite(f, v->lu[i], (size_t)lvl*(v->M+1)*sizeof(uint32_t), &io)==0; }
        ok &= enc_fwrite(f, v->dead, v->count, &io)==0;   /* tombstone bitmap */
    }
    if(v->ivf){                                     /* lists: centroids, codewords, per list n + blocks */
        struct KvIvf *iv=v->ivf;
        uint32_t tc = fabsf(iv->avqt-(float)KV_IVF_T)<1e-6f ? 0u : (uint32_t)lrintf(iv->avqt*64.0f);
        uint32_t ih[4]={ (uint32_t)iv->nlist, (uint32_t)iv->M, (uint32_t)lrintf(iv->soar*1000.0f),
                         (uint32_t)iv->d1 | ((uint32_t)iv->lr<<16) | ((uint32_t)(iv->dp==1)<<25) | (tc<<26) };
        ok &= enc_fwrite(f, ih, sizeof ih, &io)==0;
        ok &= enc_fwrite(f, iv->cent, (size_t)iv->nlist*v->dim*sizeof(float), &io)==0;
        ok &= enc_fwrite(f, iv->pq, (size_t)iv->M*32*sizeof(float), &io)==0;
        if(iv->d1) ok &= enc_fwrite(f, iv->R, (size_t)v->dim*iv->d1*sizeof(float), &io)==0;
        for(int c=0;c<iv->nlist && ok;c++){ KvList *L=&iv->lists[c]; uint32_t nb=(L->n+31)/32;
            ok &= enc_fwrite(f, &L->n, 4, &io)==0;
            ok &= enc_fwrite(f, L->codes, (size_t)nb*iv->bb, &io)==0;
            ok &= enc_fwrite(f, L->ids, (size_t)nb*32*sizeof(uint32_t), &io)==0;
            if(iv->lr){ ok &= enc_fwrite(f, L->U, (size_t)(iv->lr/16)*iv->LD4*64, &io)==0;     /* the list's basis, sums, scales */
                ok &= enc_fwrite(f, L->usum, (size_t)iv->lr*sizeof(int32_t), &io)==0;
                float sc2[2]={L->uscale,L->cscale}; ok &= enc_fwrite(f, sc2, sizeof sc2, &io)==0; } }
        if(!v->built) ok &= enc_fwrite(f, v->dead, v->count, &io)==0;   /* no graph: tombstones here */
    }
    if(io.kind){ uint8_t tag[16]; kv_cmac_final(&io.mac,tag); ok &= fwrite(tag,16,1,f)==1; }
    memset(&io,0,sizeof io);
    if(fflush(f)!=0) ok=0;
    fclose(f);
    if(ok && kv_replace_file(tmp, path)==0) return 0;   /* atomic publish */
    remove(tmp);                               /* failed: leave existing snapshot intact */
    return -1;
}

static KciVec *kv_load(FILE *f, const uint8_t *key, int keylen, KvIO *io, int lflags){
    char magic[8]; uint32_t hdr[8];
    if(fread(magic,8,1,f)!=1) return NULL;
    int v5 = memcmp(magic,KCIVEC_MAGIC5,8)==0;
    if(!v5 && memcmp(magic,KCIVEC_MAGIC,8)!=0) return NULL;
    if(fread(hdr,sizeof(hdr),1,f)!=1) return NULL;
    uint32_t flags=hdr[0]; int dim=hdr[1], M=hdr[2], efc=hdr[3];
    uint32_t count=hdr[4], ml=hdr[5], ep=hdr[6];
    float gs; memcpy(&gs,&hdr[7],sizeof(float));
    uint8_t mk[32];
    if(flags & KV_FLAG_ENC){
        if(!key || keylen<=0){ fprintf(stderr,"kci_vec_load: snapshot is encrypted, key required\n"); return NULL; }
        uint8_t salt[16];
        if(fread(salt,16,1,f)!=1) return NULL;       /* PBKDF2 salt */
        int raw_file = (flags & KV_FLAG_RAWKEY) != 0, raw_call = (lflags & KCI_VEC_LOAD_RAW_KEY) != 0;
        if(raw_file != raw_call || (raw_call && keylen != 32)){ fprintf(stderr,"kci_vec_load: key kind does not match the file\n"); return NULL; }
        if(raw_file) hmac_sha256(key,32,salt,16,mk); else kdf_derive(key,keylen,salt,16,mk);
        if(v5){ uint8_t ek[32], ak[32];
            kv_subkey(mk,KV_LABEL_ENC,ek); kv_subkey(mk,KV_LABEL_MAC,ak);
            kv_aes_init(&io->aes,ek); kv_cmac_init(&io->mac,ak); io->kind=2;
            memset(ek,0,32); memset(ak,0,32); }
        else { io->cc=mk; io->kind=1; }              /* KCIVEC04: ChaCha20 keyed by the master */
    } else if(v5) return NULL;                       /* 05 is always encrypted */
    uint8_t kc[16];
    if(fread(kc,16,1,f)!=1) return NULL;
    if(io->kind){
        if(io->kind==1) cc20_crypt(kc,16,mk,KV_KC_TWEAK); else kv_ctr(&io->aes,KV_KC_TWEAK,kc,16);
        if(memcmp(kc,KCIVEC_KEYCHK,16)!=0){ fprintf(stderr,"kci_vec_load: wrong key\n"); return NULL; }
    }
    if(io->kind==2 && kv_verify_file(f,&io->mac)!=0){
        fprintf(stderr,"kci_vec_load: snapshot failed authentication (damaged or altered)\n"); return NULL; }
    KciVec *v=kci_vec_new(dim,M,efc);
    if(!v) return NULL;
    v->building=0;                                  /* loaded: no training vf */
    if(kci_vec_reserve(v,count)!=0){ kci_vec_free(v); return NULL; }
    int ok=1;
    ok &= enc_fread(f, v->vid, (size_t)count*sizeof(uint64_t), io)==0;
    ok &= enc_fread(f, v->vq,  (size_t)count*dim, io)==0;
    if(flags & KV_FLAG_FLOAT){
        v->keepf=1; v->vf=malloc((size_t)v->cap*dim*sizeof(float));
        if(!v->vf){ kci_vec_free(v); return NULL; }
        ok &= enc_fread(f, v->vf, (size_t)count*dim*sizeof(float), io)==0;
    } else if(flags & KV_FLAG_FLOAT_EXT){                /* the caller fills the floats (kci_vec_set_float) */
        v->keepf=1; v->vf=calloc((size_t)v->cap*dim, sizeof(float));
        if(!v->vf){ kci_vec_free(v); return NULL; }
    }
    v->count=count; v->gs=gs;
    if(flags & KV_FLAG_GRAPH){
        v->gN=count;
        v->level=calloc(count,1);
        v->l0=calloc((size_t)count*(v->M0+1),sizeof(uint32_t));
        v->lu=calloc(count,sizeof(uint32_t*));
        v->lock=calloc(count,1);
        v->dead=calloc(count,1);
        if(!v->level||!v->l0||!v->lu||!v->lock||!v->dead){ kci_vec_free(v); return NULL; }
        v->gcap=count;
        ok &= enc_fread(f, v->level, count, io)==0;
        ok &= enc_fread(f, v->l0, (size_t)count*(v->M0+1)*sizeof(uint32_t), io)==0;
        for(uint32_t i=0;i<count && ok;i++){ int lvl=v->level[i];
            if(lvl>0){ v->lu[i]=calloc((size_t)lvl*(v->M+1),sizeof(uint32_t));
                ok &= enc_fread(f, v->lu[i], (size_t)lvl*(v->M+1)*sizeof(uint32_t), io)==0; } }
        ok &= enc_fread(f, v->dead, count, io)==0;       /* tombstone bitmap */
        if(ok){ for(uint32_t i=0;i<count;i++) if(v->dead[i]) v->ndead++;
            uint32_t bits=1; while((1u<<bits) < count*2u) bits++; h_rebuild(v, bits); }
        v->ep=ep; v->maxlevel=(int)ml; v->built=1;
    }
    if(ok && (flags & KV_FLAG_IVF)){
        uint32_t ih[4]; struct KvIvf *iv=calloc(1,sizeof *iv);
        if(iv) iv->D=dim;
        ok = iv && enc_fread(f, ih, sizeof ih, io)==0 && ih[0]>0 && ih[0]<=(1u<<20);
        if(ok){ int d1=(int)(ih[3]&0xFFFF), lr=(int)((ih[3]>>16)&0x1FF), Dc = d1 ? d1 : dim;
            int dp = (ih[3]>>25)&1 ? 1 : 2; uint32_t tc = ih[3]>>26;
            iv->dp=dp; iv->avqt = tc ? (float)tc/64.0f : (float)KV_IVF_T;
            ok = (d1==0 || (d1%2==0 && d1>=8 && d1<dim)) && (lr==0 || (lr%16==0 && lr<=256 && lr<Dc)) && iv->avqt<1.0f
              && ih[1]==(uint32_t)(lr ? 0 : (dp==2 ? (Dc+1)/2 : Dc));
            v->ivf_dp = dp; v->ivf_avqt = iv->avqt;   /* compact rebuilds with the same codes */
            iv->nlist=(int)ih[0]; iv->M=(int)ih[1]; iv->d1=d1; iv->lr=lr; iv->D4=(dim+3)/4; iv->eta=(Dc-1)*(double)iv->avqt*iv->avqt/(1-(double)iv->avqt*iv->avqt);
            iv->bb = lr ? (size_t)lr*32+128 : (size_t)iv->M*16; iv->ld=Dc; iv->LD4=(Dc+3)/4;
            iv->soar=(float)ih[2]/1000.0f;
            iv->cent=malloc((size_t)iv->nlist*dim*sizeof(float)); iv->pq=malloc((size_t)(iv->M+1)*32*sizeof(float));
            iv->lists=calloc(iv->nlist,sizeof(KvList));
            ok = ok && iv->cent && iv->pq && iv->lists
              && enc_fread(f, iv->cent, (size_t)iv->nlist*dim*sizeof(float), io)==0
              && enc_fread(f, iv->pq, (size_t)iv->M*32*sizeof(float), io)==0;
            if(ok && d1){ iv->R=malloc((size_t)dim*d1*sizeof(float)); iv->cent1=malloc((size_t)iv->nlist*d1*sizeof(float));
                ok = iv->R && iv->cent1 && enc_fread(f, iv->R, (size_t)dim*d1*sizeof(float), io)==0;
                if(ok) for(int c=0;c<iv->nlist;c++) kv_proj(iv, dim, iv->cent+(size_t)c*dim, iv->cent1+(size_t)c*d1); } }
        for(int c=0; ok && c<iv->nlist; c++){ KvList *L=&iv->lists[c];
            ok = enc_fread(f, &L->n, 4, io)==0 && L->n<=count;
            if(!ok) break;
            uint32_t nb=(L->n+31)/32; L->capb=nb; L->own=1;
            if(nb){ L->codes=malloc((size_t)nb*iv->bb); L->ids=malloc((size_t)nb*32*sizeof(uint32_t));
                ok = L->codes && L->ids && enc_fread(f, L->codes, (size_t)nb*iv->bb, io)==0
                  && enc_fread(f, L->ids, (size_t)nb*32*sizeof(uint32_t), io)==0;
                for(uint32_t j=0; ok && j<nb*32; j++) if(L->ids[j]!=UINT32_MAX && L->ids[j]>=count) ok=0; }
            if(ok && iv->lr){ L->U=malloc((size_t)(iv->lr/16)*iv->LD4*64); L->usum=malloc((size_t)iv->lr*sizeof(int32_t)); float sc2[2];
                ok = L->U && L->usum && enc_fread(f, L->U, (size_t)(iv->lr/16)*iv->LD4*64, io)==0
                  && enc_fread(f, L->usum, (size_t)iv->lr*sizeof(int32_t), io)==0 && enc_fread(f, sc2, sizeof sc2, io)==0;
                L->uscale=sc2[0]; L->cscale=sc2[1]; if(!(L->uscale>0) || !(L->cscale>0)) ok=0; } }
        if(ok){ iv->nmax=0; for(int c=0;c<iv->nlist;c++) if(iv->lists[c].n>iv->nmax) iv->nmax=iv->lists[c].n; }
        ok = ok && kv_ivf_derive(iv, dim)==0;
        if(ok) kv_ivf_pack(iv);                     /* side by side, as after a build (failure: lists stay as read) */
        if(ok && !(flags & KV_FLAG_GRAPH)){
            v->dead=calloc(v->cap,1); v->gcap=v->cap;
            ok = v->dead && enc_fread(f, v->dead, count, io)==0;
            if(ok){ for(uint32_t i=0;i<count;i++) if(v->dead[i]) v->ndead++;
                uint32_t bits=1; while((1u<<bits) < count*2u) bits++; ok = h_rebuild(v, bits)==0; } }
        if(ok) v->ivf=iv; else kv_ivf_free(iv);
    }
    memset(mk,0,32);
    if(!ok){ kci_vec_free(v); return NULL; }
    return v;
}

KciVec *kci_vec_load(const char *path, const uint8_t *key, int keylen){
    return kci_vec_load_ex(path, key, keylen, 0);
}

#define KV_IMG_MAGIC "KCIVIMG1"
#define KV_IMG_AL 64
enum { KVI_VID, KVI_VQ, KVI_VF, KVI_DEAD, KVI_CENT, KVI_PQ, KVI_R, KVI_LN, KVI_CODES, KVI_IDS, KVI_U, KVI_USUM, KVI_SC, KVI_N };
typedef struct {
    char magic[8];
    uint32_t dim, M, efc, count, flags;           /* flags: 1 floats kept, 2 lists, 4 reserved */
    uint32_t nlist, ivfM, d1, lr, dp, tc, soar1000, nmax;
    float gs, avqt;
    uint32_t rsv[6];                              /* reserved (flags & 4) */
    uint64_t off[KVI_N], len[KVI_N], total;
} KvImg;
static uint64_t kv_img_al(uint64_t x){ return (x + KV_IMG_AL - 1) / KV_IMG_AL * KV_IMG_AL; }

int kci_vec_image_save(const KciVec *v, const char *path){
    if(!v || !path || v->vf_mmap) return -1;
    const struct KvIvf *f = v->ivf;
    KvImg h; memset(&h, 0, sizeof h);
    memcpy(h.magic, KV_IMG_MAGIC, 8);
    h.dim=(uint32_t)v->dim; h.M=(uint32_t)v->M; h.efc=(uint32_t)v->efc; h.count=v->count; h.gs=v->gs;
    h.flags = (v->vf ? 1u : 0u) | (f ? 2u : 0u);
    size_t D=(size_t)v->dim, n=v->count;
    h.len[KVI_VID]=n*sizeof(uint64_t); h.len[KVI_VQ]=n*D; h.len[KVI_VF]= v->vf ? n*D*sizeof(float) : 0; h.len[KVI_DEAD]=n;
    uint64_t nbsum=0;
    if(f){
        h.nlist=(uint32_t)f->nlist; h.ivfM=(uint32_t)f->M; h.d1=(uint32_t)f->d1; h.lr=(uint32_t)f->lr; h.dp=(uint32_t)f->dp;
        h.avqt=f->avqt; h.tc=(uint32_t)lrintf(f->avqt*64.0f); h.soar1000=(uint32_t)lrintf(f->soar*1000.0f); h.nmax=f->nmax;
        for(int c=0;c<f->nlist;c++) nbsum += (f->lists[c].n+31)/32;
        h.len[KVI_CENT]=(size_t)f->nlist*D*sizeof(float); h.len[KVI_PQ]=(size_t)f->M*32*sizeof(float);
        h.len[KVI_R]= f->d1 ? D*(size_t)f->d1*sizeof(float) : 0; h.len[KVI_LN]=(size_t)f->nlist*sizeof(uint32_t);
        h.len[KVI_CODES]=nbsum*f->bb; h.len[KVI_IDS]=nbsum*32*sizeof(uint32_t);
        if(f->lr){ h.len[KVI_U]=(size_t)f->nlist*(size_t)(f->lr/16)*f->LD4*64; h.len[KVI_USUM]=(size_t)f->nlist*f->lr*sizeof(int32_t);
                   h.len[KVI_SC]=(size_t)f->nlist*2*sizeof(float); }
    }
    uint64_t off=kv_img_al(sizeof h);
    for(int i=0;i<KVI_N;i++){ h.off[i]=off; off=kv_img_al(off+h.len[i]); }
    h.total=off;
    char tmp[1100]; if(snprintf(tmp,sizeof tmp,"%s.tmp",path) >= (int)sizeof tmp) return -1;
    FILE *fp=fopen(tmp,"wb"); if(!fp) return -1;
    int ok=1; uint64_t pos=0;
    static const uint8_t zero[KV_IMG_AL]={0};
#define KVI_PUT(p, nbytes) do{ size_t _n=(size_t)(nbytes); if(_n && ok && fwrite((p),1,_n,fp)!=_n) ok=0; pos+=_n; }while(0)
#define KVI_PAD(to) do{ while(ok && pos<(to)){ size_t _pad=(size_t)((to)-pos); if(_pad>KV_IMG_AL) _pad=KV_IMG_AL; KVI_PUT(zero,_pad); } }while(0)
    KVI_PUT(&h, sizeof h);
    KVI_PAD(h.off[KVI_VID]); KVI_PUT(v->vid, h.len[KVI_VID]);
    KVI_PAD(h.off[KVI_VQ]); KVI_PUT(v->vq, h.len[KVI_VQ]);
    if(v->vf){ KVI_PAD(h.off[KVI_VF]); KVI_PUT(v->vf, h.len[KVI_VF]); }
    KVI_PAD(h.off[KVI_DEAD]);
    if(v->dead) KVI_PUT(v->dead, n); else KVI_PAD(h.off[KVI_DEAD]+n);       /* no tombstones yet: all live */
    if(f){
        KVI_PAD(h.off[KVI_CENT]); KVI_PUT(f->cent, h.len[KVI_CENT]);
        KVI_PAD(h.off[KVI_PQ]); KVI_PUT(f->pq, h.len[KVI_PQ]);
        if(f->d1){ KVI_PAD(h.off[KVI_R]); KVI_PUT(f->R, h.len[KVI_R]); }
        KVI_PAD(h.off[KVI_LN]); for(int c=0;c<f->nlist;c++) KVI_PUT(&f->lists[c].n, sizeof(uint32_t));
        KVI_PAD(h.off[KVI_CODES]); for(int c=0;c<f->nlist;c++){ uint32_t nb=(f->lists[c].n+31)/32; if(nb) KVI_PUT(f->lists[c].codes, (size_t)nb*f->bb); }
        KVI_PAD(h.off[KVI_IDS]); for(int c=0;c<f->nlist;c++){ uint32_t nb=(f->lists[c].n+31)/32; if(nb) KVI_PUT(f->lists[c].ids, (size_t)nb*32*sizeof(uint32_t)); }
        if(f->lr){
            KVI_PAD(h.off[KVI_U]); for(int c=0;c<f->nlist;c++) KVI_PUT(f->lists[c].U, (size_t)(f->lr/16)*f->LD4*64);
            KVI_PAD(h.off[KVI_USUM]); for(int c=0;c<f->nlist;c++) KVI_PUT(f->lists[c].usum, (size_t)f->lr*sizeof(int32_t));
            KVI_PAD(h.off[KVI_SC]); for(int c=0;c<f->nlist;c++){ float sc2[2]={f->lists[c].uscale,f->lists[c].cscale}; KVI_PUT(sc2, sizeof sc2); } }
    }
    KVI_PAD(h.total);
#undef KVI_PUT
#undef KVI_PAD
    if(fflush(fp)!=0) ok=0;
    fclose(fp);
    if(ok && pos==h.total && kv_replace_file(tmp, path)==0) return 0;
    remove(tmp); return -1;
}

KciVec *kci_vec_attach_mem(const void *image, size_t len){
    if(!image || len < sizeof(KvImg)) return NULL;
    const uint8_t *p=image; KvImg h; memcpy(&h, p, sizeof h);
    if(memcmp(h.magic, KV_IMG_MAGIC, 8)!=0 || h.total>len || h.dim==0 || h.dim>(1u<<16)) return NULL;
    for(int i=0;i<KVI_N;i++) if(h.off[i]%KV_IMG_AL || h.off[i]+h.len[i]>h.total) return NULL;
    size_t D=h.dim, n=h.count;
    if(h.len[KVI_VID]!=n*sizeof(uint64_t) || h.len[KVI_VQ]!=n*D || h.len[KVI_DEAD]!=n
       || ((h.flags&1) && h.len[KVI_VF]!=n*D*sizeof(float))) return NULL;
    if(h.flags&4) return NULL;                   /* an index kind this release does not have */
    KciVec *v=kci_vec_new((int)h.dim, (int)h.M, (int)h.efc); if(!v) return NULL;
    v->img=p; v->imglen=len; v->building=0;
    v->vid=(uint64_t*)(p+h.off[KVI_VID]); v->vq=(int8_t*)(p+h.off[KVI_VQ]);
    if(h.flags&1){ v->vf=(float*)(p+h.off[KVI_VF]); v->keepf=1; }
    v->dead=(uint8_t*)(p+h.off[KVI_DEAD]); for(size_t i=0;i<n;i++) v->ndead += v->dead[i]!=0;
    v->count=(uint32_t)n; v->cap=(uint32_t)n; v->gcap=(uint32_t)n; v->gs=h.gs;
    if(h.flags&2){
        struct KvIvf *f=calloc(1,sizeof *f); if(!f){ kci_vec_free(v); return NULL; }
        f->ro=1; f->D=(int)D; f->nlist=(int)h.nlist; f->M=(int)h.ivfM; f->d1=(int)h.d1; f->lr=(int)h.lr; f->dp = h.dp==1 ? 1 : 2;
        f->avqt = h.tc ? (float)h.tc/64.0f : (float)KV_IVF_T;
        int Dc = f->d1 ? f->d1 : (int)D;
        int ok = h.nlist>0 && h.nlist<=(1u<<20) && (f->d1==0 || (f->d1%2==0 && f->d1>=8 && f->d1<(int)D))
              && (f->lr==0 || (f->lr%16==0 && f->lr<=256 && f->lr<Dc)) && f->avqt<1.0f
              && h.ivfM==(uint32_t)(f->lr ? 0 : (f->dp==2 ? (Dc+1)/2 : Dc));
        f->eta=(Dc-1)*(double)f->avqt*f->avqt/(1-(double)f->avqt*f->avqt);
        f->bb = f->lr ? (size_t)f->lr*32+128 : (size_t)f->M*16; f->ld=Dc; f->LD4=(Dc+3)/4; f->D4=(int)((D+3)/4);
        f->soar=(float)h.soar1000/1000.0f;
        ok = ok && h.len[KVI_CENT]==(size_t)f->nlist*D*sizeof(float) && h.len[KVI_PQ]==(size_t)f->M*32*sizeof(float)
                && h.len[KVI_R]==(f->d1 ? D*(size_t)f->d1*sizeof(float) : 0) && h.len[KVI_LN]==(size_t)f->nlist*sizeof(uint32_t);
        if(!ok){ free(f); kci_vec_free(v); return NULL; }
        f->cent=(float*)(p+h.off[KVI_CENT]); f->pq=(float*)(p+h.off[KVI_PQ]); if(f->d1) f->R=(float*)(p+h.off[KVI_R]);
        f->lists=calloc((size_t)f->nlist, sizeof(KvList));
        if(f->d1) f->cent1=malloc((size_t)f->nlist*f->d1*sizeof(float));
        if(!f->lists || (f->d1 && !f->cent1)){ kv_ivf_free(f); kci_vec_free(v); return NULL; }
        const uint32_t *ln=(const uint32_t*)(p+h.off[KVI_LN]);
        uint64_t nb=0; uint32_t nmax=0;
        uint64_t ubytes=(size_t)(f->lr/16)*f->LD4*64;
        for(int c=0; ok && c<f->nlist; c++){ KvList *L=&f->lists[c]; uint32_t b=(ln[c]+31)/32;
            if(ln[c]>n || (nb+b)*f->bb>h.len[KVI_CODES] || (nb+b)*32*sizeof(uint32_t)>h.len[KVI_IDS]){ ok=0; break; }
            L->n=ln[c]; L->capb=b; L->own=0;
            L->codes=(uint8_t*)(p+h.off[KVI_CODES]+nb*f->bb); L->ids=(uint32_t*)(p+h.off[KVI_IDS]+nb*32*sizeof(uint32_t));
            for(uint32_t j=0;j<b*32;j++) if(L->ids[j]!=UINT32_MAX && L->ids[j]>=n){ ok=0; break; }   /* a damaged file must not crash */
            if(f->lr){
                if((size_t)(c+1)*ubytes>h.len[KVI_U] || (size_t)(c+1)*f->lr*sizeof(int32_t)>h.len[KVI_USUM] || (size_t)(c+1)*2*sizeof(float)>h.len[KVI_SC]){ ok=0; break; }
                L->U=(int8_t*)(p+h.off[KVI_U]+(size_t)c*ubytes); L->usum=(int32_t*)(p+h.off[KVI_USUM]+(size_t)c*f->lr*sizeof(int32_t));
                const float *sc=(const float*)(p+h.off[KVI_SC]+(size_t)c*2*sizeof(float)); L->uscale=sc[0]; L->cscale=sc[1];
                if(!(L->uscale>0) || !(L->cscale>0)) ok=0; }
            if(L->n>nmax) nmax=L->n; nb+=b; }
        if(ok){ f->nmax=nmax;
            if(f->d1) for(int c=0;c<f->nlist;c++) kv_proj(f, (int)D, f->cent+(size_t)c*D, f->cent1+(size_t)c*f->d1);
            ok = kv_ivf_derive(f,(int)D)==0; }
        if(!ok){ kv_ivf_free(f); kci_vec_free(v); return NULL; }
        v->ivf=f;
    }
    return v;
}

KciVec *kci_vec_attach(const char *path){
    if(!path) return NULL;
    int fd=open(path, O_RDONLY); if(fd<0) return NULL;
    struct stat st;
    if(fstat(fd,&st)!=0 || st.st_size<(off_t)sizeof(KvImg)){ close(fd); return NULL; }
    size_t len=(size_t)st.st_size;
    void *p=mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0); close(fd);
    if(!p || p==(void*)-1) return NULL;
    KciVec *v=kci_vec_attach_mem(p, len);
    if(!v){ munmap(p,len); return NULL; }
    v->img_mmap=1;
    return v;
}
/* lane sj of block src into lane dj of block dst (table codes or low-rank coefficients) */
static void kv_block_copy(const struct KvIvf *f, uint8_t *dst, uint32_t dj, const uint8_t *src, uint32_t sj){
    if(!f->lr){
        for(int m=0;m<f->M;m++){ uint8_t s=src[(size_t)m*16+(sj&15)]; uint8_t cd = sj<16 ? (s&0x0F) : (s>>4);
            uint8_t *byte=dst+(size_t)m*16+(dj&15);
            *byte = dj<16 ? (uint8_t)((*byte&0xF0)|cd) : (uint8_t)((*byte&0x0F)|(cd<<4)); }
    } else {
        for(int k=0;k<f->lr;k++) ((int8_t*)dst)[((size_t)(k>>2)*32+dj)*4+(k&3)] = ((const int8_t*)src)[((size_t)(k>>2)*32+sj)*4+(k&3)];
        ((int32_t*)(dst+(size_t)f->lr*32))[dj] = ((const int32_t*)(src+(size_t)f->lr*32))[sj];
    }
}

KciVec *kci_vec_thaw(const KciVec *a){
    if(!a || !a->img) return NULL;
    size_t D=(size_t)a->dim, n=a->count;
    KciVec *v=kci_vec_new(a->dim, a->M, a->efc); if(!v) return NULL;
    v->building=0; v->keepf = a->vf ? 1 : 0;
    int ok = kci_vec_reserve(v, n ? (uint32_t)n : 1)==0;
    if(ok){
        memcpy(v->vid, a->vid, n*sizeof(uint64_t)); memcpy(v->vq, a->vq, n*D);
        if(a->vf) memcpy(v->vf, a->vf, n*D*sizeof(float));
        v->count=(uint32_t)n; v->gs=a->gs;
        v->dead=calloc(v->cap,1); ok = v->dead!=NULL;
        if(ok){ memcpy(v->dead, a->dead, n); v->ndead=a->ndead; v->gcap=v->cap; }
    }
    const struct KvIvf *s=a->ivf;
    if(ok && s){
        struct KvIvf *f=calloc(1,sizeof *f); ok = f!=NULL;
        if(ok){
            *f=*s; f->ro=0; f->R=NULL; f->cent1=NULL; f->cent=NULL; f->pq=NULL; f->c8=NULL; f->c8sum=NULL; f->lists=NULL; f->arena=NULL; f->arena_ids=NULL;
            f->cent=malloc((size_t)f->nlist*D*sizeof(float)); f->pq=malloc((size_t)(f->M+1)*32*sizeof(float));
            f->lists=calloc((size_t)f->nlist, sizeof(KvList));
            uint32_t *cnt=malloc((size_t)f->nlist*sizeof(uint32_t));
            ok = f->cent && f->pq && f->lists && cnt;
            if(ok){ memcpy(f->cent, s->cent, (size_t)f->nlist*D*sizeof(float)); memcpy(f->pq, s->pq, (size_t)f->M*32*sizeof(float)); }
            if(ok && f->d1){ f->R=malloc(D*(size_t)f->d1*sizeof(float)); f->cent1=malloc((size_t)f->nlist*f->d1*sizeof(float)); ok = f->R && f->cent1;
                if(ok){ memcpy(f->R, s->R, D*(size_t)f->d1*sizeof(float)); memcpy(f->cent1, s->cent1, (size_t)f->nlist*f->d1*sizeof(float)); } }
            if(ok){ for(int c=0;c<f->nlist;c++) cnt[c]=s->lists[c].n; ok = kv_ivf_arena(f, cnt)==0; }
            size_t ub=(size_t)(f->lr/16)*f->LD4*64;
            for(int c=0; ok && c<f->nlist; c++){ KvList *L=&f->lists[c]; const KvList *S=&s->lists[c]; uint32_t b=(S->n+31)/32;
                if(b){ memcpy(L->codes, S->codes, (size_t)b*f->bb); memcpy(L->ids, S->ids, (size_t)b*32*sizeof(uint32_t)); }
                L->n=S->n;
                if(f->lr){ L->U=malloc(ub); L->usum=malloc((size_t)f->lr*sizeof(int32_t)); ok = L->U && L->usum;
                    if(ok){ memcpy(L->U, S->U, ub); memcpy(L->usum, S->usum, (size_t)f->lr*sizeof(int32_t)); L->uscale=S->uscale; L->cscale=S->cscale; } } }
            free(cnt);
            if(ok){ f->nmax=s->nmax; ok = kv_ivf_derive(f, (int)D)==0; }
            if(ok){ v->ivf=f; v->ivf_dp=f->dp; v->ivf_avqt=f->avqt; } else kv_ivf_free(f);
        }
    }
    if(ok){ uint32_t bits=1; while((1u<<bits) < v->count*2u) bits++; ok = h_rebuild(v, bits)==0; }
    if(!ok){ kci_vec_free(v); return NULL; }
    return v;
}

int64_t kci_vec_squeeze(KciVec *v){
    if(!v || v->img) return -1;
    if(v->built || !v->ivf) return -2;
    if(v->ndead==0 || !v->dead) return 0;
    struct KvIvf *f=v->ivf; size_t D=(size_t)v->dim; uint32_t n=v->count, live=0;
    uint32_t *map=malloc((size_t)n*sizeof(uint32_t)); uint32_t *cnt=calloc((size_t)f->nlist, sizeof(uint32_t));
    KvList *old=malloc((size_t)f->nlist*sizeof(KvList));
    if(!map || !cnt || !old){ free(map); free(cnt); free(old); return -1; }
    for(uint32_t i=0;i<n;i++) map[i] = v->dead[i] ? UINT32_MAX : live++;
    /* live entries per list, then the lists again in a fresh arena (the old one, and own blocks, wiped) */
    for(int c=0;c<f->nlist;c++){ const KvList *L=&f->lists[c];
        for(uint32_t j=0;j<L->n;j++){ uint32_t r=L->ids[j]; if(r!=UINT32_MAX && r<n && map[r]!=UINT32_MAX) cnt[c]++; } }
    memcpy(old, f->lists, (size_t)f->nlist*sizeof(KvList));
    uint8_t *oa=f->arena; uint32_t *oi=f->arena_ids; f->arena=NULL; f->arena_ids=NULL;
    if(kv_ivf_arena(f, cnt)!=0){ free(f->arena); free(f->arena_ids); f->arena=oa; f->arena_ids=oi;
        memcpy(f->lists, old, (size_t)f->nlist*sizeof(KvList)); free(map); free(cnt); free(old); return -1; }
    f->nmax=0;
    for(int c=0;c<f->nlist;c++){ KvList *L=&f->lists[c]; const KvList *O=&old[c];
        L->U=O->U; L->usum=O->usum; L->uscale=O->uscale; L->cscale=O->cscale;   /* the list's basis stays */
        for(uint32_t j=0;j<O->n;j++){ uint32_t r=O->ids[j]; if(r==UINT32_MAX || r>=n || map[r]==UINT32_MAX) continue;
            uint32_t k=L->n; L->ids[(size_t)k]=map[r];
            kv_block_copy(f, L->codes+(size_t)(k/32)*f->bb, k%32, O->codes+(size_t)(j/32)*f->bb, j%32); L->n++; }
        if(L->n>f->nmax) f->nmax=L->n;
        if(O->capb) kv_wipe(O->codes, (size_t)O->capb*f->bb);
        if(O->own){ free(O->codes); free(O->ids); } }
    free(oa); free(oi); free(old); free(cnt);
    /* rows down into the holes (a row never moves up, so one forward pass), the tail wiped */
    for(uint32_t i=0;i<n;i++){ uint32_t t=map[i]; if(t==UINT32_MAX || t==i) continue;
        memcpy(v->vq+(size_t)t*D, v->vq+(size_t)i*D, D);
        if(v->vf) memcpy(v->vf+(size_t)t*D, v->vf+(size_t)i*D, D*sizeof(float));
        v->vid[t]=v->vid[i]; }
    kv_wipe(v->vq+(size_t)live*D, (size_t)(n-live)*D);
    if(v->vf) kv_wipe(v->vf+(size_t)live*D, (size_t)(n-live)*D*sizeof(float));
    memset(v->dead, 0, n); v->count=live; v->ndead=0;
    free(map);
    { uint32_t bits=1; while((1u<<bits) < v->count*2u) bits++; if(h_rebuild(v, bits)!=0) return -1; }
    return (int64_t)(n-live);
}

KciVec *kci_vec_load_ex(const char *path, const uint8_t *key, int keylen, int lflags){
    FILE *f=fopen(path,"rb");
    if(!f) return NULL;
    KvIO io; memset(&io,0,sizeof io);
    KciVec *v=kv_load(f,key,keylen,&io,lflags);
    memset(&io,0,sizeof io);
    fclose(f);
    return v;
}

/* Reconstruct a float (unit-norm) view of stored vector `i`: the exact float if
 * keep_float is on, else the int8 row dequantized (/127). For queries/probes. */
int kci_vec_plain_bytes(const KciVec *v, uint64_t out[6]){
    if(!v || !out) return -1;
    memset(out, 0, 6*sizeof(uint64_t));
    size_t D=(size_t)v->dim;
    if(v->vq) out[0] += (uint64_t)v->cap*D;
    if(v->vf && !v->vf_mmap) out[1] += (uint64_t)v->cap*D*sizeof(float);
    const struct KvIvf *f=v->ivf;
    if(f){
        if(f->lists) for(int c=0;c<f->nlist;c++){ const KvList *L=&f->lists[c];
            out[0] += (uint64_t)L->capb*f->bb;
            if(L->U) out[2] += (uint64_t)(f->lr/16)*f->LD4*64 + (uint64_t)f->lr*sizeof(int32_t); }
        if(f->cent) out[2] += (uint64_t)f->nlist*f->D*sizeof(float);
        if(f->cent1) out[2] += (uint64_t)f->nlist*f->d1*sizeof(float);
        if(f->R) out[2] += (uint64_t)f->D*f->d1*sizeof(float);
        if(f->pq) out[2] += (uint64_t)(f->M+1)*32*sizeof(float);
        if(f->c8) out[2] += (uint64_t)(f->nlistp/16)*f->D4*64 + (uint64_t)f->nlistp*sizeof(int32_t);
    }
    if(v->l0) out[3] += (uint64_t)v->gcap*(v->M0+1)*sizeof(uint32_t) + v->gcap*2u;   /* + level, lock */
    if(v->lu) for(uint32_t i=0;i<v->gN;i++) if(v->lu[i]) out[3] += (uint64_t)v->level[i]*(v->M+1)*sizeof(uint32_t);
    if(v->dead) out[3] += v->gcap;
    if(v->vid) out[4] += (uint64_t)v->cap*sizeof(uint64_t);
    if(v->hkey) out[4] += ((uint64_t)1<<v->hbits)*(sizeof(uint64_t)+sizeof(uint32_t));
    out[5] = out[0]+out[1]+out[2]+out[3]+out[4];
    return 0;
}

int kci_vec_protect(KciVec *v){
    if(!v || v->img) return -1;
    int r = 3;                      /* bits that held for every buffer */
    v->protect = 1;
    size_t q_b = (size_t)v->cap * v->dim, ps = kv_pagesz();
#ifndef _WIN32
    /* move vector buffers that are not page-aligned (malloc'd by a build, a reorder or a load) to aligned memory */
    if(v->vq && ((uintptr_t)v->vq % ps)){ int8_t *n = kv_valloc(v, q_b); if(!n) return -1;
        memcpy(n, v->vq, q_b); kv_wipe(v->vq, q_b); free(v->vq); v->vq = n; }
    if(v->vf && !v->vf_mmap && ((uintptr_t)v->vf % ps)){ float *n = kv_valloc(v, q_b * sizeof(float)); if(!n) return -1;
        memcpy(n, v->vf, q_b * sizeof(float)); kv_wipe(v->vf, q_b * sizeof(float)); free(v->vf); v->vf = n; }
#else
    (void)ps;
#endif
    if(v->vq) r &= kv_protect_region(v->vq, q_b);
    if(v->vf && !v->vf_mmap) r &= kv_protect_region(v->vf, q_b * sizeof(float));
    if(v->ivf && v->ivf->arena){ struct KvIvf *f = v->ivf; size_t ab = 0;   /* the arena: all lists side by side */
        for(int c = 0; c < f->nlist; c++) if(!f->lists[c].own) ab += (size_t)f->lists[c].capb * f->bb;
#ifndef _WIN32
        if(ab && ((uintptr_t)f->arena % ps)){
            uint8_t *na = kv_valloc(v, ab);
            if(na){ memcpy(na, f->arena, ab);
                for(int c = 0; c < f->nlist; c++){ KvList *L = &f->lists[c];
                    if(!L->own && L->codes) L->codes = na + (L->codes - f->arena); }
                kv_wipe(f->arena, ab); free(f->arena); f->arena = na; }
        }
#endif
        if(ab) r &= kv_protect_region(f->arena, ab);
        for(int c = 0; c < f->nlist; c++){ KvList *L = &f->lists[c];   /* lists grown out of the arena: best effort */
            if(L->own && L->codes) kv_protect_region(L->codes, (size_t)L->capb * f->bb); } }
    return r;
}

int kci_vec_set_float(KciVec *v, uint64_t id, const float *vec){
    if(!v || !vec || !v->keepf || !v->vf || v->vf_mmap || v->img) return -1;
    uint32_t i = h_get(v, id);
    if(i==KV_HEMPTY || i>=v->count) return -1;
    store_float_norm(v, i, vec);
    return 0;
}

int kci_vec_get_float(const KciVec *v, uint32_t i, float *out){
    if(!v || i>=v->count) return -1;
    if(v->keepf && v->vf){ memcpy(out, v->vf+(size_t)i*v->dim, (size_t)v->dim*sizeof(float)); return 0; }
    if(v->built && v->gs>0){ const int8_t *c=vrow(v,i); float inv=1.0f/v->gs; for(int d=0;d<v->dim;d++) out[d]=c[d]*inv; return 0; }
    return -1;
}

/* the public searches (the bodies above) */
int kci_vec_search_r(KciVecSearcher *s, const float *query, int k, int ef, uint64_t *out_ids, float *out_scores){
    return kv_search_r_(s, query, k, ef, out_ids, out_scores);
}
int kci_vec_search_rerank(KciVecSearcher *s, const float *query, int k, int ef, int rerank_n,
                          uint64_t *out_ids, float *out_scores){
    return kv_search_rerank_(s, query, k, ef, rerank_n, out_ids, out_scores);
}
int kci_vec_search_filtered(KciVecSearcher *s, const float *query, int k, int ef, int rerank_n,
                            KciVecFilter filt, void *ctx, uint64_t *out_ids, float *out_scores){
    return kv_search_filtered_(s, query, k, ef, rerank_n, filt, ctx, out_ids, out_scores);
}
int kci_vec_search_ivf(KciVecSearcher *s, const float *query, int k, int nprobe, int rerank_n,
                       uint64_t *out_ids, float *out_scores){
    return kv_search_ivf_(s, query, k, nprobe, rerank_n, out_ids, out_scores);
}
int kci_vec_search_exact(KciVec *v, const float *query, int k, int threads, uint64_t *out_ids, float *out_scores){
    return kv_search_exact_(v, query, k, threads, out_ids, out_scores);
}

uint32_t kci_vec_count(const KciVec *v){ return v?v->count:0; }
int kci_vec_dim(const KciVec *v){ return v ? v->dim : 0; }
int kci_vec_is_built(const KciVec *v){ return v?v->built:0; }
