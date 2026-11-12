/*
 * kci_vec_bench.c — public-data sanity bench for the kci_vec library.
 *
 * build: clang -O3 -mcpu=apple-m1 -o kci_vec_bench kci_vec_bench.c kci_vec.c -lpthread -lm
 * run:   ./kci_vec_bench <dir-with-sift_base.fvecs> [M] [efc] [nq] [threads]
 */
#include "kci_vec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include "kci_win_compat.h"
#include <psapi.h>
static double peak_ws_mb(void){ PROCESS_MEMORY_COUNTERS pmc;
    if(GetProcessMemoryInfo(GetCurrentProcess(),&pmc,sizeof(pmc))) return pmc.PeakWorkingSetSize/1e6;
    return 0; }
static double cur_ws_mb(void){ PROCESS_MEMORY_COUNTERS pmc;
    if(GetProcessMemoryInfo(GetCurrentProcess(),&pmc,sizeof(pmc))) return pmc.WorkingSetSize/1e6;
    return 0; }
#else
#include <sys/resource.h>
#include <pthread.h>
#endif

static float *read_fvecs(const char *path, int *n, int *d){
    FILE *f=fopen(path,"rb"); if(!f){ perror(path); exit(1); }
    int dim; if(fread(&dim,4,1,f)!=1){ fclose(f); exit(1); }
#ifdef _WIN32
    _fseeki64(f,0,SEEK_END); long long bytes=_ftelli64(f); _fseeki64(f,0,SEEK_SET);
#else
    fseeko(f,0,SEEK_END); long long bytes=(long long)ftello(f); fseeko(f,0,SEEK_SET);
#endif
    long long rec=4+(long long)dim*4; long long cnt=bytes/rec;
    float *out=malloc((size_t)cnt*(size_t)dim*sizeof(float));
    if(!out){ fprintf(stderr,"read_fvecs: malloc %.1fGB failed\n",(double)cnt*dim*4/1e9); exit(1); }
    for(long long i=0;i<cnt;i++){ int dd; if(fread(&dd,4,1,f)!=1)break; if(fread(out+(size_t)i*dim,4,dim,f)!=(size_t)dim)break; }
    fclose(f); *n=(int)cnt; *d=dim; return out;
}
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

#define K 10
static KciVec *g_v; static const float *g_q; static int g_dim, g_ef, g_rn;
static uint64_t *g_hres; static float *g_hsc;
typedef struct { int lo, hi; } QJob;
static void *qworker(void *arg){
    QJob *j=arg; KciVecSearcher *s=kci_vec_searcher_new(g_v);
    for(int i=j->lo;i<j->hi;i++){
        if(g_rn>0)
            kci_vec_search_rerank(s, g_q+(size_t)i*g_dim, K, g_ef, g_ef,
                                  g_hres+(size_t)i*K, g_hsc+(size_t)i*K);
        else
            kci_vec_search_r(s, g_q+(size_t)i*g_dim, K, g_ef,
                             g_hres+(size_t)i*K, g_hsc+(size_t)i*K);
    }
    kci_vec_searcher_free(s); return NULL;
}
static double run_threads(int nq, int T){
    pthread_t th[64]; QJob jb[64]; if(T>64)T=64;
    int per=(nq+T-1)/T; double t=now();
    for(int t_i=0;t_i<T;t_i++){ int lo=t_i*per,hi=lo+per; if(hi>nq)hi=nq; if(lo>nq)lo=nq;
        jb[t_i]=(QJob){lo,hi}; pthread_create(&th[t_i],NULL,qworker,&jb[t_i]); }
    for(int t_i=0;t_i<T;t_i++) pthread_join(th[t_i],NULL);
    return now()-t;
}

int main(int argc,char**argv){
    const char *dir = argc>1?argv[1]:".";
    int M = argc>2?atoi(argv[2]):16;
    int efc = argc>3?atoi(argv[3]):200;
    int nq = argc>4?atoi(argv[4]):2000;
    int T = argc>5?atoi(argv[5]):8;
    const char *gtfile = argc>6?argv[6]:NULL;   /* float-exact gt ids (nq*K u64) */
    int rerank_n = argc>7?atoi(argv[7]):0;       /* >0 = int8 shortlist + float rerank */
    char fb[1024]; snprintf(fb,sizeof(fb),"%s/sift_base.fvecs",dir);
    char fq[1024]; snprintf(fq,sizeof(fq),"%s/sift_query.fvecs",dir);
    int N,dim,nqf,dq;
    float *base=read_fvecs(fb,&N,&dim);
    float *query=read_fvecs(fq,&nqf,&dq);
    if(nq>nqf) nq=nqf;
    printf("loaded base N=%d dim=%d query=%d / M=%d efc=%d nq=%d T=%d\n",N,dim,nqf,M,efc,nq,T);
    fflush(stdout);

    KciVec *v=kci_vec_new(dim,M,efc);
    if(rerank_n>0){ kci_vec_keep_float(v,1); printf("rerank ON (shortlist=%d, floats retained)\n",rerank_n); }
    g_rn=rerank_n;
    kci_vec_reserve(v,N);
    for(int i=0;i<N;i++) kci_vec_add(v,(uint64_t)i, base+(size_t)i*dim);
    double bs; kci_vec_build(v,T,&bs);
    printf("build %.1fs (count=%u dim=%d)\n", bs, kci_vec_count(v), kci_vec_dim(v));
    if(rerank_n>0 && getenv("KVF_DISK")){          /* move float store to disk (mmap) → RAM stays int8 */
        if(kci_vec_float_mmap(v, "_kvf_mmap.bin")==0) printf("float store mmap'd to disk (RAM = int8 graph only)\n");
        else printf("WARN float mmap failed; floats stay in RAM\n");
        fflush(stdout);
    }
    free(base); base=NULL;   /* corpus copied into the index; free it so RSS reflects the index, not the loader */
#ifndef _WIN32
    { struct rusage ru; getrusage(RUSAGE_SELF,&ru);
      printf("RSS after build (pinned: int8 graph%s): %.0fMB(mac)/%.0fMB(lin)\n",
             getenv("KVF_DISK")?", floats on disk":"+RAM floats",
             ru.ru_maxrss/1e6, ru.ru_maxrss/1024.0); fflush(stdout); }
#else
    printf("peak working set after build: %.0f MB (%s)\n", peak_ws_mb(),
           (rerank_n>0 && getenv("KVF_DISK"))?"int8 graph; floats mmap'd to disk":
           (rerank_n>0?"int8 graph + RAM float store":"int8 graph only")); fflush(stdout);
#endif

    /* optional external float-exact gt (from competitor_bench) for apples-to-apples. */
    uint64_t *gtf=NULL;
    if(gtfile){
        FILE *gf=fopen(gtfile,"rb");
        if(gf){ gtf=malloc((size_t)nq*K*sizeof(uint64_t));
            size_t rd=fread(gtf,sizeof(uint64_t),(size_t)nq*K,gf); fclose(gf);
            if(rd!=(size_t)nq*K){ printf("WARN gtfile short read %zu\n",rd); free(gtf); gtf=NULL; }
            else printf("loaded float-exact gt from %s\n",gtfile);
        } else printf("WARN cannot open gtfile %s\n",gtfile);
        fflush(stdout);
    }
    /* internal SDC-exact gt only when no external gt is given (it is slow: O(nq*N)). */
    uint64_t *gt=NULL; float *gtsc=NULL;
    if(!gtf){
        gt=malloc((size_t)nq*K*sizeof(uint64_t)); gtsc=malloc((size_t)nq*K*sizeof(float));
        double te=now();
        for(int i=0;i<nq;i++) kci_vec_search_exact(v, query+(size_t)i*dim, K, T,
                                                   gt+(size_t)i*K, gtsc+(size_t)i*K);
        printf("exact gt for %d queries in %.1fs\n", nq, now()-te); fflush(stdout);
    }

    g_v=v; g_q=query; g_dim=dim;
    g_hres=malloc((size_t)nq*K*sizeof(uint64_t));
    g_hsc=malloc((size_t)nq*K*sizeof(float));

    int efs[]={40,100,200};
    for(int e=0;e<3;e++){ g_ef=efs[e];
        double qt1=run_threads(nq,1);
        long hit=0,tie=0,hitf=0;
        for(int i=0;i<nq;i++){
            float thr = gtsc ? gtsc[(size_t)i*K+K-1] : 0.0f;
            for(int a=0;a<K;a++){ uint64_t id=g_hres[(size_t)i*K+a];
                if(gtf){ for(int b=0;b<K;b++) if(gtf[(size_t)i*K+b]==id){hitf++;break;} }
                else { for(int b=0;b<K;b++) if(gt[(size_t)i*K+b]==id){hit++;break;}
                       if(g_hsc[(size_t)i*K+a]>=thr) tie++; }
            }
        }
        double qtT=run_threads(nq,T);
        if(gtf)
            printf("ef=%d recall@10(vsFloat)=%.2f%% QPS_1=%.0f QPS_%d=%.0f\n",
                   g_ef, 100.0*hitf/((double)nq*K), nq/qt1, T, nq/qtT);
        else
            printf("ef=%d recall@10=%.2f%% tie-recall=%.2f%% QPS_1=%.0f QPS_%d=%.0f\n",
                   g_ef, 100.0*hit/((double)nq*K), 100.0*tie/((double)nq*K),
                   nq/qt1, T, nq/qtT);
        fflush(stdout);
    }
#ifndef _WIN32
    { struct rusage ru; getrusage(RUSAGE_SELF,&ru);
      /* macOS ru_maxrss = bytes; Linux = KB. Print both interpretations. */
      double mb_mac = ru.ru_maxrss/1e6, mb_lin = ru.ru_maxrss/1024.0;
      printf("peak RSS: %.0fMB (macOS) / %.0fMB (linux)  [int8 graph pinned; floats %s]\n",
             mb_mac, mb_lin, getenv("KVF_DISK")?"mmap'd/evictable":"RAM-resident");
      fflush(stdout); }
#else
    printf("resident set now: %.0f MB (current) / %.0f MB (peak)\n", cur_ws_mb(), peak_ws_mb()); fflush(stdout);
#endif
    kci_vec_free(v);
    return 0;
}
