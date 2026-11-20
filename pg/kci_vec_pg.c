/*
 kci_vec_pg.c — KCI-Vec as a PostgreSQL extension (stage 1, .
 *
 * SQL functions over KCI-Vec's list index (kci_vec_build_ivf / kci_vec_search_ivf), not an index access method.
 *
 * Still not here: a real access method (WAL, buffers, MVCC inside the index), encryption of the image (it lives
 * where the tables live).
 */
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "commands/trigger.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/timestamp.h"
#include "utils/syscache.h"
#include "access/xact.h"
#include "access/xlog.h"
#include "common/pg_prng.h"
#include "pgstat.h"
#include "port/atomics.h"
#include "postmaster/bgworker.h"
#include "postmaster/interrupt.h"
#include "storage/latch.h"
#include "storage/proc.h"
#include "utils/guc.h"
#include "utils/snapmgr.h"
#include "utils/backend_status.h"
#include "utils/wait_event.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "catalog/pg_type.h"
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "kci_vec.h"

PG_MODULE_MAGIC;

#define KV_MAXIDX 16

/* a backend's view of one index */
typedef struct {
    char name[NAMEDATALEN];
    KciVec *v; KciVecSearcher *s;          /* attached image */
    struct stat st;                         /* of the file attached (ino, size, mtime: a new image is noticed by stat) */
    int64 img;
    pg_atomic_uint64 *img_p;                /* the generation in shared memory (NULL: not preloaded / table full) */
    instr_time polled;                      /* last catalog poll through SPI (standby / no preload) */
    int64 pcount;                           /* pending rows seen by that poll (with pmax: the reload key) */
    bool tracked;                           /* catalog row says changes are collected in kci_vec_pending */
    float gap; int minp;                    /* kci_vec_set_probe_gap, re-applied after a re-attach */
    /* pending changes since the image was built (only when tracked) */
    MemoryContext pctx;
    int64 pmax;                             /* max(seq) seen by the last catalog poll (standby / no preload) */
    int64 loaded_pmax, loaded_pcount;       /* the (max(seq), count) the replay was built for; rebuilt when the poll differs */
    pg_atomic_uint32 *gen_p;                /* the index's generation in shared memory (NULL: not preloaded / table full) */
    uint32 gen_seen, local_seen;            /* generations the replay was built for */
    bool need_reload;
    int nins, nexcl;
    bool trunc;
    int64 *ins_ids; float *ins_vecs;        /* inserted since the build: nins unit vectors, dim16 floats each (zero padded) */
    int dim16;                              /* row stride of ins_vecs: dim rounded up to 16 (whole SIMD vectors, no tail) */
    int metric;
    float *ins_n2;                          /* L2: |x|^2 of every pending insert (ins_vecs are then not unit vectors) */
    int64 *excl;                            /* sorted ids to drop from the image's answer: deleted or re-inserted */
} KvEntry;
static KvEntry kv_cache[KV_MAXIDX];
static int kv_ncache = 0;
static char *kv_schema_name = NULL;         /* schema of the extension's tables */

#define KV_SHM_MAX 256
typedef struct { Oid dbid; char name[NAMEDATALEN]; pg_atomic_uint32 gen; pg_atomic_uint64 img; } KvShmSlot;   /* img: 0 = not read yet */
typedef struct { LWLock *lock; int nslots; KvShmSlot slots[KV_SHM_MAX]; } KvShm;
static KvShm *kv_shm = NULL;
static shmem_request_hook_type prev_shmem_request_hook = NULL;
static shmem_startup_hook_type prev_shmem_startup_hook = NULL;
static uint32 kv_local_gen = 0;             /* bumped by this backend's own trigger firings: its uncommitted rows */
static pg_atomic_uint32 *kv_dirty[KV_SHM_MAX]; static int kv_ndirty = 0;   /* slots to bump at this transaction's commit */
static bool kv_xact_cb_registered = false;
static bool kv_dirty_local = false;

static void kv_shmem_request(void){
    if(prev_shmem_request_hook) prev_shmem_request_hook();
    RequestAddinShmemSpace(MAXALIGN(sizeof(KvShm)));
    RequestNamedLWLockTranche("kci_vec_pg", 1);
}
static void kv_shmem_startup(void){
    bool found;
    if(prev_shmem_startup_hook) prev_shmem_startup_hook();
    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
    kv_shm = ShmemInitStruct("kci_vec_pg", sizeof(KvShm), &found);
    if(!found){
        int i;
        kv_shm->lock = &(GetNamedLWLockTranche("kci_vec_pg"))->lock;
        kv_shm->nslots = 0;
        for(i = 0; i < KV_SHM_MAX; i++){ kv_shm->slots[i].dbid = InvalidOid; kv_shm->slots[i].name[0] = 0; pg_atomic_init_u32(&kv_shm->slots[i].gen, 0); pg_atomic_init_u64(&kv_shm->slots[i].img, 0); }
    }
    LWLockRelease(AddinShmemInitLock);
}
static char *kv_guc_databases = NULL;
static int kv_guc_poll_ms = 0;               /* standby / no preload: re-read the catalog at most every N ms (0 = every search) */
static int kv_guc_refresh_pending = 10000;   /* worker: refresh when this many pending rows wait */
static int kv_guc_refresh_max_age = 0;       /* worker: ... or when the oldest pending row is older than N seconds (0 = off) */
static int kv_guc_refresh_interval = 10;     /* worker: seconds between looks */
static int kv_guc_refresh_threads = 0;       /* worker: threads for the refresh / materialize (0 = the library's default) */
PGDLLEXPORT void kv_worker_main(Datum arg);
void _PG_init(void);
void _PG_init(void){
    DefineCustomStringVariable("kci_vec_pg.databases", "Databases whose tracked indexes a background worker refreshes (comma separated). Needs shared_preload_libraries.", NULL,
                               &kv_guc_databases, "", process_shared_preload_libraries_in_progress ? PGC_POSTMASTER : PGC_SIGHUP, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("kci_vec_pg.poll_ms", "Standby or no preload: re-read the catalog at most every this many ms (0 = every search).", NULL,
                            &kv_guc_poll_ms, 0, 0, 3600000, PGC_USERSET, GUC_UNIT_MS, NULL, NULL, NULL);
    DefineCustomIntVariable("kci_vec_pg.refresh_pending", "Worker: refresh an index when this many pending rows wait.", NULL,
                            &kv_guc_refresh_pending, 10000, 1, INT_MAX, PGC_SIGHUP, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("kci_vec_pg.refresh_max_age", "Worker: refresh when the oldest pending row is older than this many seconds (0 = off).", NULL,
                            &kv_guc_refresh_max_age, 0, 0, INT_MAX, PGC_SIGHUP, GUC_UNIT_S, NULL, NULL, NULL);
    DefineCustomIntVariable("kci_vec_pg.refresh_interval", "Worker: seconds between looks at the pending rows.", NULL,
                            &kv_guc_refresh_interval, 10, 1, 86400, PGC_SIGHUP, GUC_UNIT_S, NULL, NULL, NULL);
    DefineCustomIntVariable("kci_vec_pg.refresh_threads", "Worker: threads for a refresh or materialize (0 = library default).", NULL,
                            &kv_guc_refresh_threads, 0, 0, 1024, PGC_SIGHUP, 0, NULL, NULL, NULL);
    MarkGUCPrefixReserved("kci_vec_pg");
    if(!process_shared_preload_libraries_in_progress) return;    /* loaded by a function call: no shared memory, the SPI fallback */
    prev_shmem_request_hook = shmem_request_hook; shmem_request_hook = kv_shmem_request;
    prev_shmem_startup_hook = shmem_startup_hook; shmem_startup_hook = kv_shmem_startup;
    if(kv_guc_databases && kv_guc_databases[0]){           /* one worker per database named */
        char *list = pstrdup(kv_guc_databases), *tok, *save = NULL;
        for(tok = strtok_r(list, ", ", &save); tok; tok = strtok_r(NULL, ", ", &save)){
            BackgroundWorker w;
            memset(&w, 0, sizeof w);
            snprintf(w.bgw_name, BGW_MAXLEN, "kci_vec_pg refresh (%s)", tok);
            snprintf(w.bgw_type, BGW_MAXLEN, "kci_vec_pg refresh");
            w.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
            w.bgw_start_time = BgWorkerStart_ConsistentState;     /* RecoveryFinished never comes on a hot standby; the loop checks RecoveryInProgress */
            w.bgw_restart_time = 10;
            snprintf(w.bgw_library_name, BGW_MAXLEN, "kci_vec_pg");
            snprintf(w.bgw_function_name, BGW_MAXLEN, "kv_worker_main");
            strlcpy(w.bgw_extra, tok, BGW_EXTRALEN);
            w.bgw_notify_pid = 0;
            RegisterBackgroundWorker(&w);
        }
    }
}

/* the slot of (this database, name): found or allocated; NULL without shared memory or when the table is full */
static KvShmSlot *kv_shm_slot(const char *name){
    int i; KvShmSlot *g = NULL;
    if(!kv_shm) return NULL;
    LWLockAcquire(kv_shm->lock, LW_SHARED);
    for(i = 0; i < kv_shm->nslots; i++) if(kv_shm->slots[i].dbid == MyDatabaseId && strcmp(kv_shm->slots[i].name, name) == 0){ g = &kv_shm->slots[i]; break; }
    LWLockRelease(kv_shm->lock);
    if(g) return g;
    LWLockAcquire(kv_shm->lock, LW_EXCLUSIVE);
    for(i = 0; i < kv_shm->nslots; i++) if(kv_shm->slots[i].dbid == MyDatabaseId && strcmp(kv_shm->slots[i].name, name) == 0){ g = &kv_shm->slots[i]; break; }
    if(!g && kv_shm->nslots < KV_SHM_MAX){
        KvShmSlot *sl = &kv_shm->slots[kv_shm->nslots++];
        sl->dbid = MyDatabaseId; strlcpy(sl->name, name, NAMEDATALEN); pg_atomic_write_u32(&sl->gen, 1); pg_atomic_write_u64(&sl->img, 0);
        g = sl;
    }
    LWLockRelease(kv_shm->lock);
    return g;
}
static pg_atomic_uint32 *kv_shm_gen(const char *name){ KvShmSlot *sl = kv_shm_slot(name); return sl ? &sl->gen : NULL; }

typedef struct { char name[NAMEDATALEN]; int64 img; char newpath[MAXPGPATH], oldpath[MAXPGPATH]; } KvPub;
static KvPub kv_pub[KV_SHM_MAX]; static int kv_npub = 0;

static void kv_xact_cb(XactEvent ev, void *arg){
    int i;
    if(ev == XACT_EVENT_COMMIT || ev == XACT_EVENT_PARALLEL_COMMIT){
        for(i = 0; i < kv_ndirty; i++) pg_atomic_fetch_add_u32(kv_dirty[i], 1);
        kv_ndirty = 0; kv_dirty_local = false;
        for(i = 0; i < kv_npub; i++){
            KvShmSlot *sl = kv_shm_slot(kv_pub[i].name);
            if(sl) pg_atomic_write_u64(&sl->img, (uint64)kv_pub[i].img);
            if(kv_pub[i].oldpath[0]) unlink(kv_pub[i].oldpath);
        }
        kv_npub = 0;
    } else if(ev == XACT_EVENT_ABORT || ev == XACT_EVENT_PARALLEL_ABORT){
        if(kv_ndirty || kv_dirty_local) kv_local_gen++;
        kv_ndirty = 0; kv_dirty_local = false;
        for(i = 0; i < kv_npub; i++) if(kv_pub[i].newpath[0]) unlink(kv_pub[i].newpath);
        kv_npub = 0;
    }
}
static void kv_register_xact_cb(void){
    if(!kv_xact_cb_registered){ RegisterXactCallback(kv_xact_cb, NULL); kv_xact_cb_registered = true; }
}
/* remember what this transaction's build publishes at commit (see kv_xact_cb) */
static void kv_publish(const char *name, int64 img, const char *newpath, const char *oldpath){
    KvPub *pb;
    kv_register_xact_cb();
    if(kv_npub >= KV_SHM_MAX) ereport(ERROR, (errmsg("kci_vec: too many builds in one transaction")));
    pb = &kv_pub[kv_npub++];
    strlcpy(pb->name, name, NAMEDATALEN); pb->img = img;
    strlcpy(pb->newpath, newpath ? newpath : "", MAXPGPATH); strlcpy(pb->oldpath, oldpath ? oldpath : "", MAXPGPATH);
}

/* this transaction changed the pending rows (or the image) of name: this backend sees it at once, others at commit */
static void kv_mark_dirty(const char *name){
    pg_atomic_uint32 *g;
    int i;
    kv_local_gen++;
    kv_register_xact_cb();
    kv_dirty_local = true;
    g = kv_shm_gen(name);
    if(!g) return;
    for(i = 0; i < kv_ndirty; i++) if(kv_dirty[i] == g) return;
    if(kv_ndirty < KV_SHM_MAX) kv_dirty[kv_ndirty++] = g;
}

static void kv_check_name(const char *name){
    size_t n = strlen(name), i;
    if(n == 0 || n >= NAMEDATALEN) ereport(ERROR, (errmsg("kci_vec: index name must be 1..%d chars", NAMEDATALEN-1)));
    for(i = 0; i < n; i++){
        char c = name[i];
        if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'))
            ereport(ERROR, (errmsg("kci_vec: index name may use only letters, digits and _")));
    }
}

static char *kv_dir(void){
    char *dir = psprintf("%s/kci_vec", DataDir);
    if(mkdir(dir, S_IRWXU) != 0 && errno != EEXIST)
        ereport(ERROR, (errcode_for_file_access(), errmsg("kci_vec: could not create directory \"%s\": %m", dir)));
    return dir;
}
static char *kv_path(const char *name, int64 img){
    return psprintf("%s/%s.%016" INT64_MODIFIER "x.kvimg", kv_dir(), name, (uint64)img);
}
/* a new generation id: random, never 0 */
static int64 kv_new_img(void){
    int64 x;
    do x = (int64)(pg_prng_uint64(&pg_global_prng_state) >> 1); while(x == 0);
    return x;
}
/* remove every file of the index but the one of generation keep (0: all) */
static void kv_unlink_others(const char *name, int64 keep){
    char *dir = kv_dir(), *keepf = keep ? psprintf("%s.%016" INT64_MODIFIER "x.kvimg", name, (uint64)keep) : NULL;
    DIR *d = opendir(dir); struct dirent *de; size_t nl = strlen(name);
    if(!d) return;
    while((de = readdir(d)) != NULL){
        const char *f = de->d_name; size_t fl = strlen(f);
        if(fl == nl + 1 + 16 + 6 && strncmp(f, name, nl) == 0 && f[nl] == '.' && strcmp(f + fl - 6, ".kvimg") == 0 && (!keepf || strcmp(f, keepf) != 0))
            unlink(psprintf("%s/%s", dir, f));
    }
    closedir(d);
}

/* "schema.table" of the extension's own tables, from the schema this function lives in */
static const char *kv_tbl(FunctionCallInfo fcinfo, const char *tbl){
    if(!kv_schema_name){
        Oid ns;
        if(!fcinfo) ereport(ERROR, (errmsg("kci_vec: extension schema unknown")));   /* the worker sets it first (kv_worker_schema) */
        ns = get_func_namespace(fcinfo->flinfo->fn_oid);
        kv_schema_name = MemoryContextStrdup(TopMemoryContext, get_namespace_name(ns));
    }
    return quote_qualified_identifier(kv_schema_name, tbl);
}

static KvEntry *kv_find(const char *name){
    int i;
    for(i = 0; i < kv_ncache; i++) if(strcmp(kv_cache[i].name, name) == 0) return &kv_cache[i];
    return NULL;
}

static void kv_detach(KvEntry *e){
    if(e->s) kci_vec_searcher_free(e->s);
    if(e->v) kci_vec_free(e->v);
    e->s = NULL; e->v = NULL;
}

static void kv_pending_clear(KvEntry *e){
    if(e->pctx) MemoryContextReset(e->pctx);
    e->ins_ids = NULL; e->ins_vecs = NULL; e->excl = NULL; e->nins = e->nexcl = 0; e->pmax = -1; e->need_reload = true;
    e->trunc = false;
}

static void kv_entry_drop(KvEntry *e){
    kv_detach(e);
    if(e->pctx) MemoryContextDelete(e->pctx);
    *e = kv_cache[--kv_ncache];
}

static bool kv_catalog(FunctionCallInfo fcinfo, const char *name, int64 *img, bool *tracked, bool pending, int64 *pmax, int64 *pcount){
    bool found = false;
    Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(name) };
    char *sql = pending
        ? psprintf("SELECT i.img, i.tracked, (SELECT max(seq) FROM %s p WHERE p.name = i.name), (SELECT count(*) FROM %s p WHERE p.name = i.name) FROM %s i WHERE i.name = $1",
                   kv_tbl(fcinfo, "kci_vec_pending"), kv_tbl(fcinfo, "kci_vec_pending"), kv_tbl(fcinfo, "kci_vec_index"))
        : psprintf("SELECT img, tracked FROM %s WHERE name = $1", kv_tbl(fcinfo, "kci_vec_index"));
    *img = 0; *tracked = false; if(pmax) *pmax = -1; if(pcount) *pcount = 0;
    if(SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("kci_vec: SPI_connect failed")));
    if(SPI_execute_with_args(sql, 1, at, av, NULL, true, 1) == SPI_OK_SELECT && SPI_processed == 1){
        HeapTuple t = SPI_tuptable->vals[0]; TupleDesc td = SPI_tuptable->tupdesc; bool isnull; Datum d;
        found = true;
        d = SPI_getbinval(t, td, 1, &isnull); *img = isnull ? 0 : DatumGetInt64(d);
        d = SPI_getbinval(t, td, 2, &isnull); *tracked = !isnull && DatumGetBool(d);
        if(pending){
            d = SPI_getbinval(t, td, 3, &isnull); if(pmax) *pmax = isnull ? -1 : DatumGetInt64(d);
            d = SPI_getbinval(t, td, 4, &isnull); if(pcount) *pcount = isnull ? 0 : DatumGetInt64(d);
        }
    }
    SPI_finish();
    return found;
}
/* is the catalog read through SPI (a standby has no commits of its own to bump the shared slot; no preload has no slot)? */
static bool kv_spi_path(const KvEntry *e){ return e == NULL || e->img_p == NULL || RecoveryInProgress(); }
/* the current generation of name: the shared slot when it holds one (a primary with the preload), else the catalog. The
 * slot is filled from the catalog on first use. Also returns tracked (catalog) when read. */
static int64 kv_current_img(FunctionCallInfo fcinfo, const char *name, KvShmSlot *sl, bool *found, bool *tracked, bool *read_catalog){
    int64 img = 0; bool tr = false;
    *read_catalog = false;
    if(sl && !RecoveryInProgress()){ img = (int64)pg_atomic_read_u64(&sl->img); }
    if(img == 0){
        *found = kv_catalog(fcinfo, name, &img, &tr, false, NULL, NULL); *read_catalog = true;
        if(tracked) *tracked = tr;
        if(sl && *found && img && !RecoveryInProgress()) pg_atomic_write_u64(&sl->img, (uint64)img);
    } else *found = true;
    return img;
}

/* a poll is due when kci_vec_pg.poll_ms passed since the last one (or poll_ms is 0) */
static bool kv_poll_due(KvEntry *e){
    instr_time now;
    if(kv_guc_poll_ms <= 0) return true;
    INSTR_TIME_SET_CURRENT(now);
    if(INSTR_TIME_IS_ZERO(e->polled)) return true;
    INSTR_TIME_SUBTRACT(now, e->polled);
    return INSTR_TIME_GET_MILLISEC(now) >= (double)kv_guc_poll_ms;
}

/* the index by name: this server's file of the current generation, attached in this backend; re-attached when the
 * generation or the file changed. The generation comes from the shared slot (primary with preload) or the catalog
 * (standby, no preload: every search or every poll_ms); *pending_polled tells the caller the catalog poll also
 * refreshed e->pmax / e->pcount. */
static KvEntry *kv_get_ex(FunctionCallInfo fcinfo, const char *name, bool *pending_polled){
    KvEntry *e; struct stat st; char *path; KvShmSlot *sl;
    int64 img; bool found = false, tracked = false, read_cat = false;
    kv_check_name(name);
    if(pending_polled) *pending_polled = false;
    e = kv_find(name);
    sl = kv_shm_slot(name);
    if(e && kv_spi_path(e) && !kv_poll_due(e)) img = e->img;                        /* within poll_ms: trust what we have */
    else if(e && kv_spi_path(e) && e->tracked){                                       /* standby / no preload, tracked: one query for all */
        found = kv_catalog(fcinfo, name, &img, &tracked, true, &e->pmax, &e->pcount); read_cat = true;
        INSTR_TIME_SET_CURRENT(e->polled);
        if(pending_polled) *pending_polled = true;
        if(!found) img = 0;
    } else {
        img = kv_current_img(fcinfo, name, sl, &found, &tracked, &read_cat);
        if(e && kv_spi_path(e)) INSTR_TIME_SET_CURRENT(e->polled);
    }
    if(img == 0){
        if(e) kv_entry_drop(e);
        ereport(ERROR, (errmsg("kci_vec: no index \"%s\" (build it with kci_vec_build)", name)));
    }
    path = kv_path(name, img);
    if(stat(path, &st) != 0){
        if(e) kv_entry_drop(e);
        if(sl && !RecoveryInProgress() && !read_cat){     /* the slot may be behind the catalog (a build that crashed between commit and callback) */
            int64 img2; bool tr2;
            if(kv_catalog(fcinfo, name, &img2, &tr2, false, NULL, NULL) && img2 && img2 != img){ pg_atomic_write_u64(&sl->img, (uint64)img2); img = img2; path = kv_path(name, img); }
        }
        if(stat(path, &st) != 0)
            ereport(ERROR, (errmsg("kci_vec: index \"%s\" has no image on this server (generation %016" INT64_MODIFIER "x: built elsewhere, or restored from a dump)", name, (uint64)img),
                            errhint("Run SELECT kci_vec_materialize('%s') here (it reads the local table; works on a hot standby too).", name)));
    }
    if(e && e->v && e->img == img && e->st.st_ino == st.st_ino && e->st.st_size == st.st_size
       && e->st.st_mtim.tv_sec == st.st_mtim.tv_sec && e->st.st_mtim.tv_nsec == st.st_mtim.tv_nsec)
        return e;
    if(!e){
        if(kv_ncache >= KV_MAXIDX) ereport(ERROR, (errmsg("kci_vec: too many indexes open in this backend (%d)", KV_MAXIDX)));
        e = &kv_cache[kv_ncache++];
        memset(e, 0, sizeof *e);
        strlcpy(e->name, name, NAMEDATALEN);
        e->pctx = AllocSetContextCreate(TopMemoryContext, "kci_vec pending", ALLOCSET_DEFAULT_SIZES);
        e->pmax = -1;
    }
    kv_detach(e);
    kv_pending_clear(e);
    e->v = kci_vec_attach(path);
    if(!e->v) ereport(ERROR, (errmsg("kci_vec: could not attach \"%s\" (not an image, damaged, or written by another build)", path)));
    e->s = kci_vec_searcher_new(e->v);
    if(!e->s){ kv_detach(e); ereport(ERROR, (errmsg("kci_vec: out of memory (searcher)"))); }
    if(e->gap > 0) kci_vec_searcher_set_probe_gap(e->s, e->gap, e->minp);
    e->st = st; e->img = img;
    if(!read_cat){ int64 i2; kv_catalog(fcinfo, name, &i2, &tracked, false, NULL, NULL); }
    e->tracked = tracked;
    e->gen_p = sl ? &sl->gen : NULL; e->img_p = sl ? &sl->img : NULL;
    INSTR_TIME_SET_CURRENT(e->polled);
    if(pending_polled) *pending_polled = false;        /* just attached: the pending cache is empty, the search replays */
    return e;
}
static KvEntry *kv_get(FunctionCallInfo fcinfo, const char *name){ return kv_get_ex(fcinfo, name, NULL); }

/* float4[] (1-dim, no nulls) -> pointer + length */
static const float *kv_array_floats(ArrayType *a, int *len){
    if(ARR_NDIM(a) != 1 || ARR_HASNULL(a) || ARR_ELEMTYPE(a) != FLOAT4OID)
        ereport(ERROR, (errmsg("kci_vec: vectors must be 1-dimensional real[] without nulls")));
    *len = ARR_DIMS(a)[0];
    return (const float*)ARR_DATA_PTR(a);
}

/* ── pending changes (tracked indexes) ─────────────────────────────────────────────────────────────────────── */

static int kv_cmp_i64(const void *a, const void *b){ int64 x = *(const int64*)a, y = *(const int64*)b; return x < y ? -1 : x > y; }

/* filter for the image's search: 0 for an id deleted or re-inserted since the build */
typedef struct { const KvEntry *e; const int64 *allow; int nallow; uint64 *bits; int64 base; uint64 span; int64 *hk; uint32 hmask; } KvFilt;
static inline bool kv_in_allow(const KvFilt *f, int64 id){
    if(f->bits){ uint64 o = (uint64)(id - f->base); return o < f->span && (f->bits[o >> 6] >> (o & 63)) & 1; }
    if(f->hk){ uint32 h = (uint32)(((uint64)id * 0x9E3779B97F4A7C15ull) >> 32) & f->hmask;
        while(f->hk[h] != PG_INT64_MIN){ if(f->hk[h] == id) return true; h = (h + 1) & f->hmask; } return false; }
    return bsearch(&id, f->allow, f->nallow, sizeof(int64), kv_cmp_i64) != NULL;
}
static int kv_filt(uint64_t id, void *ctx){
    const KvFilt *f = ctx; int64 k = (int64)id;
    if(f->allow && !kv_in_allow(f, k)) return 0;
    return f->e->nexcl == 0 || bsearch(&k, f->e->excl, f->e->nexcl, sizeof(int64), kv_cmp_i64) == NULL;
}
static bool kv_allowed(const KvFilt *f, int64 id){ return !f->allow || kv_in_allow(f, id); }
/* the sorted unique ids in f->allow[0..nallow): choose and fill the bitmap or the hash */
static void kv_filt_index(KvFilt *f){
    int i;
    if(f->nallow == 0) return;
    {   uint64 span = (uint64)(f->allow[f->nallow - 1] - f->allow[0]) + 1;
        if(span <= (uint64)f->nallow * 64 && span <= ((uint64)1 << 26)){
            f->base = f->allow[0]; f->span = span; f->bits = (uint64*)palloc0(sizeof(uint64) * ((span + 63) / 64));
            for(i = 0; i < f->nallow; i++){ uint64 o = (uint64)(f->allow[i] - f->base); f->bits[o >> 6] |= (uint64)1 << (o & 63); }
            return; } }
    {   uint32 sz = 16; while(sz < (uint32)f->nallow * 2) sz <<= 1;
        f->hk = (int64*)palloc(sizeof(int64) * sz); f->hmask = sz - 1;
        for(i = 0; i < (int)sz; i++) f->hk[i] = PG_INT64_MIN;
        for(i = 0; i < f->nallow; i++){ uint32 h = (uint32)(((uint64)f->allow[i] * 0x9E3779B97F4A7C15ull) >> 32) & f->hmask;
            while(f->hk[h] != PG_INT64_MIN) h = (h + 1) & f->hmask;
            f->hk[h] = f->allow[i]; } }
}

/* small open-addressing map id -> slot in ins_ids (for the replay of the pending rows) */
typedef struct { int64 *key; int32 *val; uint32 mask; } KvMap;
static void kv_map_init(KvMap *m, uint32 n){ uint32 sz = 16; while(sz < n * 2) sz <<= 1; m->mask = sz - 1;
    m->key = palloc(sz * sizeof(int64)); m->val = palloc(sz * sizeof(int32)); memset(m->val, 0xFF, sz * sizeof(int32)); }
static int32 *kv_map_slot(KvMap *m, int64 k){ uint32 h = (uint32)(((uint64)k * 0x9E3779B97F4A7C15ull) >> 32) & m->mask;
    while(m->val[h] >= 0 && m->key[h] != k) h = (h + 1) & m->mask;
    m->key[h] = k; return &m->val[h]; }

/* replay kci_vec_pending for this index into e: inserted vectors (unit, padded to dim) and the ids to exclude */
static void kv_pending_load(FunctionCallInfo fcinfo, KvEntry *e, int64 pmax){
    int dim = kci_vec_dim(e->v), dim16 = (dim + 15) & ~15;
    int metric = kci_vec_metric(e->v);
    Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(e->name) };
    char *sql = psprintf("SELECT op, id, v FROM %s WHERE name = $1 ORDER BY seq", kv_tbl(fcinfo, "kci_vec_pending"));
    uint64 i, n;
    MemoryContext old;
    KvMap map; int64 *ids; float *vecs; bool *live; int nins = 0;
    int64 *dels; int ndel = 0;

    kv_pending_clear(e);
    if(SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("kci_vec: SPI_connect failed")));
    if(SPI_execute_with_args(sql, 1, at, av, NULL, true, 0) != SPI_OK_SELECT) ereport(ERROR, (errmsg("kci_vec: reading kci_vec_pending failed")));
    n = SPI_processed;
    old = MemoryContextSwitchTo(e->pctx);
    kv_map_init(&map, (uint32)n);
    ids = palloc(sizeof(int64) * (n + 1)); vecs = palloc(sizeof(float) * dim16 * (n + 1)); live = palloc(n + 1);
    dels = palloc(sizeof(int64) * (n + 1));
    for(i = 0; i < n; i++){
        HeapTuple t = SPI_tuptable->vals[i]; TupleDesc td = SPI_tuptable->tupdesc;
        bool n1, n2, n3;
        char op = DatumGetChar(SPI_getbinval(t, td, 1, &n1));
        int64 id = DatumGetInt64(SPI_getbinval(t, td, 2, &n2));
        Datum dv = SPI_getbinval(t, td, 3, &n3);
        int32 *slot;
        if(n1 || n2) continue;
        if(op == 'T'){
            int j; for(j = 0; j < nins; j++) live[j] = false;
            ndel = 0; e->trunc = true; continue; }
        slot = kv_map_slot(&map, id);
        if(op == 'I' && !n3){
            int len; const float *x = kv_array_floats(DatumGetArrayTypeP(dv), &len);
            float *dst; double nn = 0; int d;
            if(len > dim || len < dim - 3) ereport(ERROR, (errmsg("kci_vec: pending vector of %d dims for an index of %d dims (id " INT64_FORMAT ")", len, dim, id)));
            if(*slot < 0){ *slot = nins; ids[nins] = id; nins++; }
            dst = vecs + (size_t)(*slot) * dim16;
            memset(dst, 0, sizeof(float) * dim16); memcpy(dst, x, sizeof(float) * len);
            for(d = 0; d < dim; d++) nn += (double)dst[d] * dst[d];
            if(metric == 0 && nn > 0){ float s = (float)(1.0 / sqrt(nn)); for(d = 0; d < dim; d++) dst[d] *= s; }
            live[*slot] = true;
        } else {                                        /* 'D' (or an insert without a vector): gone */
            if(*slot >= 0) live[*slot] = false;
            dels[ndel++] = id;
        }
    }
    /* excluded from the image's answer: every id touched (deleted, or inserted again); kept inserts: the live ones */
    e->excl = palloc(sizeof(int64) * (ndel + nins + 1));
    e->nexcl = 0;
    for(i = 0; i < (uint64)ndel; i++) e->excl[e->nexcl++] = dels[i];
    for(i = 0; i < (uint64)nins; i++) e->excl[e->nexcl++] = ids[i];
    if(e->nexcl){ int w = 1, r; qsort(e->excl, e->nexcl, sizeof(int64), kv_cmp_i64);
        for(r = 1; r < e->nexcl; r++) if(e->excl[r] != e->excl[w-1]) e->excl[w++] = e->excl[r];
        e->nexcl = w; }
    e->ins_ids = palloc(sizeof(int64) * (nins + 1)); e->ins_vecs = palloc(sizeof(float) * dim16 * (nins + 4));   /* +4: the last group of 4 may read past nins */
    memset(e->ins_vecs + (size_t)nins * dim16, 0, sizeof(float) * dim16 * 4);
    e->nins = 0; e->dim16 = dim16; e->metric = metric;
    e->ins_n2 = palloc0(sizeof(float) * (nins + 4));
    for(i = 0; i < (uint64)nins; i++) if(live[i]){
        const float *src = vecs + i * dim16; double nn = 0; int d;
        for(d = 0; d < dim; d++) nn += (double)src[d] * src[d];
        e->ins_n2[e->nins] = (float)nn;
        e->ins_ids[e->nins] = ids[i]; memcpy(e->ins_vecs + (size_t)e->nins * dim16, src, sizeof(float) * dim16); e->nins++; }
    pfree(ids); pfree(vecs); pfree(live); pfree(dels); pfree(map.key); pfree(map.val);
    MemoryContextSwitchTo(old);
    SPI_finish();
    e->pmax = pmax; e->pcount = (int64)n; e->loaded_pmax = pmax; e->loaded_pcount = (int64)n;
}

typedef void (*KvDot4)(const float *q, const float *x, int stride, int n16, float out[4]);
static void kv_dot4_scalar(const float *q, const float *x, int stride, int n16, float out[4]){
    int r, d;
    for(r = 0; r < 4; r++){ const float *xr = x + (size_t)r * stride; float a = 0;
        for(d = 0; d < n16; d++) a += q[d] * xr[d];
        out[r] = a; }
}
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
__attribute__((target("avx2,fma")))
static void kv_dot4_avx2(const float *q, const float *x, int stride, int n16, float out[4]){
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    const float *x0 = x, *x1 = x + stride, *x2 = x + 2 * (size_t)stride, *x3 = x + 3 * (size_t)stride;
    int d, r;
    for(d = 0; d < n16; d += 8){ __m256 qv = _mm256_loadu_ps(q + d);
        a0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(x0 + d), a0); a1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(x1 + d), a1);
        a2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(x2 + d), a2); a3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(x3 + d), a3); }
    { __m256 acc[4] = { a0, a1, a2, a3 };
      for(r = 0; r < 4; r++){ __m128 lo = _mm256_castps256_ps128(acc[r]), hi = _mm256_extractf128_ps(acc[r], 1);
          lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo); out[r] = _mm_cvtss_f32(lo); } }
}
__attribute__((target("avx512f")))
static void kv_dot4_avx512(const float *q, const float *x, int stride, int n16, float out[4]){
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    const float *x0 = x, *x1 = x + stride, *x2 = x + 2 * (size_t)stride, *x3 = x + 3 * (size_t)stride;
    int d;
    for(d = 0; d < n16; d += 16){ __m512 qv = _mm512_loadu_ps(q + d);
        a0 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(x0 + d), a0); a1 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(x1 + d), a1);
        a2 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(x2 + d), a2); a3 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(x3 + d), a3); }
    out[0] = _mm512_reduce_add_ps(a0); out[1] = _mm512_reduce_add_ps(a1); out[2] = _mm512_reduce_add_ps(a2); out[3] = _mm512_reduce_add_ps(a3);
}
#endif
static KvDot4 kv_dot4 = NULL;
static const char *kv_dot4_name = "scalar";
static void kv_dot_init(void){
    kv_dot4 = kv_dot4_scalar;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    if(__builtin_cpu_supports("avx512f")){ kv_dot4 = kv_dot4_avx512; kv_dot4_name = "avx512"; }
    else if(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")){ kv_dot4 = kv_dot4_avx2; kv_dot4_name = "avx2"; }
#endif
}

/* keep the k best (score desc) in arr[0..n) */
static void kv_topk_push(int64 *ids, float *sc, int *n, int k, int64 id, float s){
    int i;
    if(*n < k){ i = (*n)++; }
    else { if(s <= sc[k-1]) return; i = k - 1; }
    while(i > 0 && sc[i-1] < s){ ids[i] = ids[i-1]; sc[i] = sc[i-1]; i--; }
    ids[i] = id; sc[i] = s;
}

/* ── build / refresh ───────────────────────────────────────────────────────────────────────────────────────── */

/* read rel(idcol, veccol) through a read-only cursor (the calling statement's snapshot), build the lists, write the
 * image; with pending rows seen under the same snapshot deleted afterwards (seqs: int8[] datum, 0 = none).
 * Returns the rows indexed. Called with SPI connected. */
static int64 kv_build_image(const char *path, Oid relid, const char *idcol, const char *veccol,
                            int nlist, int threads, int iters, int64 sample, int metric, double *secs){
    char *relname = DatumGetCString(DirectFunctionCall1(regclassout, ObjectIdGetDatum(relid)));
    char *sql = psprintf("SELECT (%s)::int8, (%s)::float4[] FROM %s", quote_identifier(idcol), quote_identifier(veccol), relname);
    KciVec *v = NULL; float *buf = NULL;
    int dim = 0, pdim = 0; int64 count = 0;
    Portal portal;
    float max_norm = 0;
    instr_time t0, t1;
    INSTR_TIME_SET_CURRENT(t0);
    if(metric != 0){
        Portal p0 = SPI_cursor_open_with_args("kci_vec_norm_cur", sql, 0, NULL, NULL, NULL, true, 0);
        double mx = 0;
        for(;;){
            uint64 i;
            SPI_cursor_fetch(p0, true, 20000);
            if(SPI_processed == 0) break;
            for(i = 0; i < SPI_processed; i++){
                bool n1, n2; int len, d; const float *x; double nn = 0;
                Datum dv = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &n2);
                (void)SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &n1);
                if(n1 || n2) continue;
                x = kv_array_floats(DatumGetArrayTypeP(dv), &len);
                for(d = 0; d < len; d++) nn += (double)x[d] * x[d];
                if(nn > mx) mx = nn;
            }
            SPI_freetuptable(SPI_tuptable);
            CHECK_FOR_INTERRUPTS();
        }
        SPI_cursor_close(p0);
        if(!(mx > 0) || !isfinite(mx)) ereport(ERROR, (errmsg("kci_vec: no rows in %s, or only zero vectors (L2 / inner product need a longest vector)", relname)));
        max_norm = (float)(sqrt(mx) * 1.0001);
    }
    portal = SPI_cursor_open_with_args("kci_vec_build_cur", sql, 0, NULL, NULL, NULL, true, 0);
    PG_TRY();
    {
        for(;;){
            uint64 i;
            SPI_cursor_fetch(portal, true, 20000);
            if(SPI_processed == 0) break;
            for(i = 0; i < SPI_processed; i++){
                HeapTuple t = SPI_tuptable->vals[i]; TupleDesc td = SPI_tuptable->tupdesc;
                bool n1, n2;
                Datum did = SPI_getbinval(t, td, 1, &n1);
                Datum dv = SPI_getbinval(t, td, 2, &n2);
                int len; const float *x;
                if(n1 || n2) continue;
                x = kv_array_floats(DatumGetArrayTypeP(dv), &len);
                if(!v){
                    dim = len; pdim = (dim + 3) & ~3;
                    if(dim <= 0) ereport(ERROR, (errmsg("kci_vec: empty vector in the first row")));
                    v = metric == 1 ? kci_vec_new_l2(pdim, 0, 0, max_norm) : metric == 2 ? kci_vec_new_ip(pdim, 0, 0, max_norm) : kci_vec_new(pdim, 0, 0);
                    if(!v) ereport(ERROR, (errmsg("kci_vec: kci_vec_new failed (dim %d, metric %d)", pdim, metric)));
                    kci_vec_keep_float(v, 1);
                    if(iters > 0 || sample > 0)
                        kci_vec_set_ivf_training(v, iters, sample <= 0 ? 0 : (sample >= UINT32_MAX ? UINT32_MAX : (uint32_t)sample));
                    buf = (float*)palloc0(sizeof(float) * pdim);
                }
                if(len != dim) ereport(ERROR, (errmsg("kci_vec: vector of %d dims in a table of %d dims (id " INT64_FORMAT ")", len, dim, DatumGetInt64(did))));
                memcpy(buf, x, sizeof(float) * dim);
                if(kci_vec_add(v, (uint64_t)DatumGetInt64(did), buf) < 0)
                    ereport(ERROR, (errmsg("kci_vec: kci_vec_add failed at id " INT64_FORMAT " (zero vector, or out of memory)", DatumGetInt64(did))));
                count++;
            }
            SPI_freetuptable(SPI_tuptable);
            CHECK_FOR_INTERRUPTS();
        }
        SPI_cursor_close(portal);
        if(!v) ereport(ERROR, (errmsg("kci_vec: no rows in %s", relname)));
        if(kci_vec_build_ivf(v, nlist, threads) != 0)
            ereport(ERROR, (errmsg("kci_vec: kci_vec_build_ivf failed (out of memory?)")));
        if(kci_vec_image_save(v, path) != 0)
            ereport(ERROR, (errcode_for_file_access(), errmsg("kci_vec: could not write \"%s\"", path)));
    }
    PG_CATCH();
    {
        if(v) kci_vec_free(v);
        PG_RE_THROW();
    }
    PG_END_TRY();
    kci_vec_free(v);
    INSTR_TIME_SET_CURRENT(t1); INSTR_TIME_SUBTRACT(t1, t0);
    if(secs) *secs = INSTR_TIME_GET_DOUBLE(t1);
    return count;
}

static int64 kv_refresh_diff(FunctionCallInfo fcinfo, const char *name, const char *oldpath, const char *path, int64 full_rows, double *secs, int64 *applied){
    KciVec *a = kci_vec_attach(oldpath), *w = NULL;
    int64 live = -1; int dim;
    volatile bool too_long = false;
    instr_time t0, t1;
    *applied = 0;
    if(!a){ elog(DEBUG1, "kci_vec: refresh of \"%s\": the image could not be attached, building anew", name); return -1; }
    INSTR_TIME_SET_CURRENT(t0);
    w = kci_vec_thaw(a);
    kci_vec_free(a);
    if(!w){ elog(DEBUG1, "kci_vec: refresh of \"%s\": the image could not be thawed, building anew", name); return -1; }
    dim = kci_vec_dim(w);
    PG_TRY();
    {
        Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(name) };
        char *sql = psprintf("SELECT op, id, v FROM %s WHERE name = $1 ORDER BY seq", kv_tbl(fcinfo, "kci_vec_pending"));
        float *buf = (float*)palloc0(sizeof(float) * dim);
        uint64 i, n;
        if(SPI_execute_with_args(sql, 1, at, av, NULL, true, 0) != SPI_OK_SELECT) ereport(ERROR, (errmsg("kci_vec: reading kci_vec_pending failed")));
        n = SPI_processed;
        for(i = 0; i < n; i++){
            HeapTuple t = SPI_tuptable->vals[i]; TupleDesc td = SPI_tuptable->tupdesc;
            bool n1, n2, n3;
            char op = DatumGetChar(SPI_getbinval(t, td, 1, &n1));
            int64 id = DatumGetInt64(SPI_getbinval(t, td, 2, &n2));
            Datum dv = SPI_getbinval(t, td, 3, &n3);
            if(n1 || n2) continue;
            kci_vec_delete(w, (uint64_t)id);                 /* -1 when the id is not in the image: fine */
            if(op == 'I' && !n3){
                int len; const float *x = kv_array_floats(DatumGetArrayTypeP(dv), &len);
                if(len > dim || len < dim - 3) ereport(ERROR, (errmsg("kci_vec: pending vector of %d dims for an index of %d dims (id " INT64_FORMAT ")", len, dim, id)));
                memset(buf, 0, sizeof(float) * dim); memcpy(buf, x, sizeof(float) * len);
                if(kci_vec_insert(w, (uint64_t)id, buf) < 0){
                    if(kci_vec_metric(w) != 0){ too_long = true; break; }
                    ereport(ERROR, (errmsg("kci_vec: kci_vec_insert failed at id " INT64_FORMAT " (zero vector, no floats in the image, or out of memory)", id)));
                }
            }
            if((i & 4095) == 4095) CHECK_FOR_INTERRUPTS();
        }
        SPI_freetuptable(SPI_tuptable);
        *applied = (int64)n;
        if(too_long){ live = -1; elog(DEBUG1, "kci_vec: refresh of \"%s\": a pending vector is longer than the image allows, building anew", name); }
        else {
        if(kci_vec_squeeze(w) < 0) ereport(ERROR, (errmsg("kci_vec: kci_vec_squeeze failed (out of memory?)")));
        live = (int64)kci_vec_count(w);
        if(full_rows > 0 && (live > 2 * full_rows || 2 * live < full_rows)){ live = -1;   /* drifted: the caller builds */
            elog(DEBUG1, "kci_vec: refresh of \"%s\": the row count drifted past half / double of the last full build, building anew", name); }
        else if(kci_vec_image_save(w, path) != 0)
            ereport(ERROR, (errcode_for_file_access(), errmsg("kci_vec: could not write \"%s\"", path)));
        }
    }
    PG_CATCH();
    {
        kci_vec_free(w);
        PG_RE_THROW();
    }
    PG_END_TRY();
    kci_vec_free(w);
    INSTR_TIME_SET_CURRENT(t1); INSTR_TIME_SUBTRACT(t1, t0);
    if(secs) *secs = INSTR_TIME_GET_DOUBLE(t1);
    return live;
}

static bool kv_pending_truncated(FunctionCallInfo fcinfo, const char *name){
    Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(name) };
    char *sql = psprintf("SELECT 1 FROM %s WHERE name = $1 AND op = 'T' LIMIT 1", kv_tbl(fcinfo, "kci_vec_pending"));
    if(SPI_execute_with_args(sql, 1, at, av, NULL, true, 1) != SPI_OK_SELECT) ereport(ERROR, (errmsg("kci_vec: reading kci_vec_pending failed")));
    return SPI_processed > 0;
}

/* the pending seqs visible to the calling statement's snapshot, as an int8[] datum (0 if none) */
static Datum kv_pending_seqs(FunctionCallInfo fcinfo, const char *name, int *n){
    Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(name) };
    char *sql = psprintf("SELECT seq FROM %s WHERE name = $1", kv_tbl(fcinfo, "kci_vec_pending"));
    Datum *d; uint64 i;
    *n = 0;
    if(SPI_execute_with_args(sql, 1, at, av, NULL, true, 0) != SPI_OK_SELECT) ereport(ERROR, (errmsg("kci_vec: reading kci_vec_pending failed")));
    if(SPI_processed == 0) return (Datum)0;
    d = palloc(sizeof(Datum) * SPI_processed);
    for(i = 0; i < SPI_processed; i++){ bool isnull; d[i] = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &isnull); }
    *n = (int)SPI_processed;
    return PointerGetDatum(construct_array_builtin(d, (int)SPI_processed, INT8OID));
}

static int kv_metric_of(const char *s){
    if(!s || !*s || pg_strcasecmp(s, "cosine") == 0) return 0;
    if(pg_strcasecmp(s, "l2") == 0) return 1;
    if(pg_strcasecmp(s, "ip") == 0) return 2;
    ereport(ERROR, (errmsg("kci_vec: metric must be 'cosine', 'l2' or 'ip', not \"%s\"", s)));
    return 0;
}
static const char *kv_metric_name(int m){ return m == 1 ? "l2" : m == 2 ? "ip" : "cosine"; }
/* Does kci_vec_index have the metric column yet? Not between installing this library and ALTER EXTENSION UPDATE to
 * 0.3: everything then works as in 0.2 (cosine), and only a build with another metric is refused. Called with SPI
 * connected. */
static bool kv_has_metric(FunctionCallInfo fcinfo){
    char *sql = psprintf("SELECT 1 FROM pg_catalog.pg_attribute WHERE attrelid = '%s'::pg_catalog.regclass AND attname = 'metric' AND NOT attisdropped", kv_tbl(fcinfo, "kci_vec_index"));
    return SPI_execute(sql, true, 1) == SPI_OK_SELECT && SPI_processed == 1;
}

/* build or refresh: lock the name, read the pending seqs and the table under one snapshot, write the image, delete
 * those pending rows, write the catalog row. is_refresh: the parameters come from the catalog; a tracked index is
 * refreshed from its pending rows alone (kv_refresh_diff) unless full is set or the diff declines. */
static int64 kv_do_build(FunctionCallInfo fcinfo, const char *name, Oid relid, const char *idcol, const char *veccol,
                         int nlist, int threads, int iters, int64 sample, int metric, bool is_refresh, bool full){
    int64 count = -1, applied = 0, full_rows = 0, cat_rows = 0, old_img = 0, img; int nseq; Datum seqs; double secs = 0;
    bool tracked = false, diff = false, has_metric;
    char *path, *oldpath = NULL;
    kv_check_name(name);
    if(RecoveryInProgress()) ereport(ERROR, (errmsg("kci_vec: cannot build or refresh on a standby (read only)"), errhint("Build on the primary; kci_vec_materialize makes this server's copy.")));
    if(SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("kci_vec: SPI_connect failed")));
    {   Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(name) };
        SPI_execute_with_args("SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext($1)::bigint)", 1, at, av, NULL, false, 0); }
    {   Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(name) };
        char *sql; bool isnull; int rc;
        has_metric = kv_has_metric(fcinfo);
        if(!has_metric && metric != 0) ereport(ERROR, (errmsg("kci_vec: metric '%s' needs extension version 0.3", kv_metric_name(metric)), errhint("Run ALTER EXTENSION kci_vec_pg UPDATE first.")));
        sql = psprintf("SELECT rel, id_col, vec_col, nlist, train_iters, train_sample, tracked, full_rows, rows, img%s FROM %s WHERE name = $1", has_metric ? ", metric" : "", kv_tbl(fcinfo, "kci_vec_index"));
        rc = SPI_execute_with_args(sql, 1, at, av, NULL, true, 1);
        if(rc == SPI_OK_SELECT && SPI_processed == 1){
            HeapTuple t = SPI_tuptable->vals[0]; TupleDesc td = SPI_tuptable->tupdesc;
            old_img = DatumGetInt64(SPI_getbinval(t, td, 10, &isnull)); if(isnull) old_img = 0;
            if(is_refresh){
                relid = DatumGetObjectId(SPI_getbinval(t, td, 1, &isnull));
                idcol = SPI_getvalue(t, td, 2); veccol = SPI_getvalue(t, td, 3);
                nlist = DatumGetInt32(SPI_getbinval(t, td, 4, &isnull)); if(isnull) nlist = 0;
                iters = DatumGetInt32(SPI_getbinval(t, td, 5, &isnull)); if(isnull) iters = 0;
                sample = DatumGetInt64(SPI_getbinval(t, td, 6, &isnull)); if(isnull) sample = 0;
                tracked = DatumGetBool(SPI_getbinval(t, td, 7, &isnull)) && !isnull;
                full_rows = DatumGetInt64(SPI_getbinval(t, td, 8, &isnull)); if(isnull) full_rows = 0;
                cat_rows = DatumGetInt64(SPI_getbinval(t, td, 9, &isnull)); if(isnull) cat_rows = 0;
                metric = has_metric ? kv_metric_of(SPI_getvalue(t, td, 11)) : 0; }
        } else if(is_refresh) ereport(ERROR, (errmsg("kci_vec: no index \"%s\" in kci_vec_index", name)));
    }
    if(old_img) oldpath = kv_path(name, old_img);
    img = kv_new_img(); path = kv_path(name, img);
    seqs = kv_pending_seqs(fcinfo, name, &nseq);
    if(is_refresh && !full && tracked){
        struct stat st;
        if(nseq == 0){ SPI_finish(); return cat_rows; }         /* nothing to fold in: the image stands */
        if(kv_pending_truncated(fcinfo, name)){}
        else if(oldpath && stat(oldpath, &st) == 0){            /* the current generation is here: fold the pending rows into it */
            count = kv_refresh_diff(fcinfo, name, oldpath, path, full_rows, &secs, &applied);
            diff = count >= 0;
        }
    }
    if(!diff) count = kv_build_image(path, relid, idcol, veccol, nlist, threads, iters, sample, metric, &secs);
    kv_publish(name, img, path, oldpath);                       /* at commit: the slot, the old file; at abort: the new file */
    if(nseq){
        Oid at[1] = { INT8ARRAYOID }; Datum av[1] = { seqs };
        char *sql = psprintf("DELETE FROM %s WHERE seq = ANY($1)", kv_tbl(fcinfo, "kci_vec_pending"));
        if(SPI_execute_with_args(sql, 1, at, av, NULL, false, 0) != SPI_OK_DELETE) ereport(ERROR, (errmsg("kci_vec: clearing kci_vec_pending failed")));
    }
    {   Oid at[9] = { TEXTOID, REGCLASSOID, TEXTOID, TEXTOID, INT4OID, INT4OID, INT8OID, INT8OID, TEXTOID };
        Datum av[9] = { CStringGetTextDatum(name), ObjectIdGetDatum(relid), CStringGetTextDatum(idcol), CStringGetTextDatum(veccol),
                        Int32GetDatum(nlist), Int32GetDatum(iters), Int64GetDatum(sample), Int64GetDatum(img),
                        CStringGetTextDatum(kv_metric_name(metric)) };
        /* a full build resets full_rows (the data the centroids were trained on) and the diff count; a diff keeps them */
        char *sql = psprintf("INSERT INTO %s (name, rel, id_col, vec_col, nlist, train_iters, train_sample, rows, built_at, build_secs, full_rows, diffs, img%s) "
                             "VALUES ($1, $2, $3, $4, $5, $6, $7, " INT64_FORMAT ", now(), %.3f, " INT64_FORMAT ", 0, $8%s) "
                             "ON CONFLICT (name) DO UPDATE SET rel = EXCLUDED.rel, id_col = EXCLUDED.id_col, vec_col = EXCLUDED.vec_col, "
                             "nlist = EXCLUDED.nlist, train_iters = EXCLUDED.train_iters, train_sample = EXCLUDED.train_sample, "
                             "rows = EXCLUDED.rows, built_at = EXCLUDED.built_at, build_secs = EXCLUDED.build_secs, img = EXCLUDED.img%s%s",
                             kv_tbl(fcinfo, "kci_vec_index"), has_metric ? ", metric" : "", count, secs, count, has_metric ? ", $9" : "",
                             has_metric ? ", metric = EXCLUDED.metric" : "",
                             diff ? ", diffs = kci_vec_index.diffs + 1" : ", full_rows = EXCLUDED.full_rows, diffs = 0");
        if(SPI_execute_with_args(sql, has_metric ? 9 : 8, at, av, NULL, false, 0) < 0) ereport(ERROR, (errmsg("kci_vec: writing kci_vec_index failed")));
    }
    SPI_finish();
    kv_mark_dirty(name);                                        /* others replay the pending rows again after the commit */
    {   KvEntry *e = kv_find(name); if(e) kv_entry_drop(e); }   /* this backend re-attaches on its next search */
    return count;
}

PG_FUNCTION_INFO_V1(kci_vec_pg_materialize);
Datum kci_vec_pg_materialize(PG_FUNCTION_ARGS){
    char *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int threads = PG_GETARG_INT32(1);
    Oid relid; char *idcol, *veccol, *path, *tmp; int nlist, iters, metric; int64 sample, img, count; double secs = 0;
    bool isnull; struct stat st;
    kv_check_name(name);
    if(SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("kci_vec: SPI_connect failed")));
    {   Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(name) };
        bool has_metric = kv_has_metric(fcinfo);
        char *sql = psprintf("SELECT rel, id_col, vec_col, nlist, train_iters, train_sample, img%s FROM %s WHERE name = $1", has_metric ? ", metric" : "", kv_tbl(fcinfo, "kci_vec_index"));
        if(SPI_execute_with_args(sql, 1, at, av, NULL, true, 1) != SPI_OK_SELECT || SPI_processed != 1)
            ereport(ERROR, (errmsg("kci_vec: no index \"%s\" in kci_vec_index", name)));
        {   HeapTuple t = SPI_tuptable->vals[0]; TupleDesc td = SPI_tuptable->tupdesc;
            relid = DatumGetObjectId(SPI_getbinval(t, td, 1, &isnull));
            idcol = SPI_getvalue(t, td, 2); veccol = SPI_getvalue(t, td, 3);
            nlist = DatumGetInt32(SPI_getbinval(t, td, 4, &isnull)); if(isnull) nlist = 0;
            iters = DatumGetInt32(SPI_getbinval(t, td, 5, &isnull)); if(isnull) iters = 0;
            sample = DatumGetInt64(SPI_getbinval(t, td, 6, &isnull)); if(isnull) sample = 0;
            img = DatumGetInt64(SPI_getbinval(t, td, 7, &isnull)); if(isnull) img = 0;
            metric = has_metric ? kv_metric_of(SPI_getvalue(t, td, 8)) : 0; }
    }
    if(img == 0){ SPI_finish(); ereport(ERROR, (errmsg("kci_vec: index \"%s\" has no generation in kci_vec_index (built by an older version): build it again", name))); }
    path = kv_path(name, img);
    if(stat(path, &st) == 0){ SPI_finish(); PG_RETURN_INT64(0); }
    tmp = psprintf("%s.%d.part", path, MyProcPid);              /* two sessions may materialize at once: separate temporaries, same content */
    count = kv_build_image(tmp, relid, idcol, veccol, nlist, threads, iters, sample, metric, &secs);
    if(rename(tmp, path) != 0){ unlink(tmp); ereport(ERROR, (errcode_for_file_access(), errmsg("kci_vec: could not rename \"%s\": %m", tmp))); }
    SPI_finish();
    kv_unlink_others(name, img);
    {   KvShmSlot *sl = kv_shm_slot(name); if(sl && !RecoveryInProgress()) pg_atomic_write_u64(&sl->img, (uint64)img); }
    {   KvEntry *e = kv_find(name); if(e) kv_entry_drop(e); }
    ereport(NOTICE, (errmsg("kci_vec: materialized \"%s\" generation %016" INT64_MODIFIER "x: " INT64_FORMAT " rows in %.1f s", name, (uint64)img, count, secs)));
    PG_RETURN_INT64(count);
}

PG_FUNCTION_INFO_V1(kci_vec_pg_build);
Datum kci_vec_pg_build(PG_FUNCTION_ARGS){
    char *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    Oid relid = PG_GETARG_OID(1);
    char *idcol = text_to_cstring(PG_GETARG_TEXT_PP(2));
    char *veccol = text_to_cstring(PG_GETARG_TEXT_PP(3));
    int metric = PG_NARGS() > 8 ? kv_metric_of(text_to_cstring(PG_GETARG_TEXT_PP(8))) : 0;
    PG_RETURN_INT64(kv_do_build(fcinfo, name, relid, idcol, veccol, PG_GETARG_INT32(4), PG_GETARG_INT32(5), PG_GETARG_INT32(6), PG_GETARG_INT64(7), metric, false, true));
}

/* kci_vec_refresh(name text, threads int, full boolean) -> bigint: fold the pending rows into the image (tracked: from
 * the pending rows alone; untracked or full: build again with the catalog's parameters) */
PG_FUNCTION_INFO_V1(kci_vec_pg_refresh);
Datum kci_vec_pg_refresh(PG_FUNCTION_ARGS){
    char *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    PG_RETURN_INT64(kv_do_build(fcinfo, name, InvalidOid, NULL, NULL, 0, PG_GETARG_INT32(1), 0, 0, 0, true, PG_GETARG_BOOL(2)));
}

/* ── search ────────────────────────────────────────────────────────────────────────────────────────────────── */

static Datum kv_search(FunctionCallInfo fcinfo, ArrayType *allowed){
    ReturnSetInfo *rsi = (ReturnSetInfo*)fcinfo->resultinfo;
    char *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    ArrayType *qa = PG_GETARG_ARRAYTYPE_P(1);
    int k = PG_GETARG_INT32(2), nprobe = PG_GETARG_INT32(3), rerank = PG_GETARG_INT32(4);
    bool polled;
    KvEntry *e = kv_get_ex(fcinfo, name, &polled);
    KvFilt flt = { e, NULL, 0, NULL, 0, 0, NULL, 0 };
    int dim = kci_vec_dim(e->v), qlen, n, i, metric;
    const float *q;
    float *qbuf = NULL;
    uint64_t *ids; float *sc;
    int64 *oid; float *osc; int on = 0;
    Datum values[2]; bool nulls[2] = {false, false};

    if(k <= 0 || k > 10000) ereport(ERROR, (errmsg("kci_vec: k must be 1..10000")));
    if(nprobe <= 0) nprobe = 32;
    if(rerank < k) rerank = k;
    q = kv_array_floats(qa, &qlen);
    if(qlen > dim || qlen < dim - 3) ereport(ERROR, (errmsg("kci_vec: query of %d dims for an index of %d dims", qlen, dim)));
    if(qlen != dim){ qbuf = (float*)palloc0(sizeof(float) * dim); memcpy(qbuf, q, sizeof(float) * qlen); q = qbuf; }
    if(allowed){                                     /* sorted, unique copy of the allowed ids */
        Datum *el; bool *nl; int ne, w = 0;
        int64 *a;
        if(ARR_ELEMTYPE(allowed) != INT8OID) ereport(ERROR, (errmsg("kci_vec: allowed must be a bigint[]")));
        deconstruct_array(allowed, INT8OID, sizeof(int64), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE, &el, &nl, &ne);   /* _builtin has no int8 in PG 16 */
        a = (int64*)palloc(sizeof(int64) * (ne + 1));
        for(i = 0; i < ne; i++) if(!nl[i]) a[w++] = DatumGetInt64(el[i]);
        if(w > 1){ int r; qsort(a, w, sizeof(int64), kv_cmp_i64); for(r = 1, n = 1; r < w; r++) if(a[r] != a[n-1]) a[n++] = a[r]; w = n; }
        flt.allow = a; flt.nallow = w; kv_filt_index(&flt);
    }

    if(e->tracked){                                  /* pending rows: replay them again when something changed */
        bool reload = e->need_reload || e->local_seen != kv_local_gen;   /* own uncommitted changes */
        uint32 g = 0;
        if(!kv_spi_path(e)){ g = pg_atomic_read_u32(e->gen_p); reload = reload || g != e->gen_seen; }   /* one shared word */
        else {                                                                                            /* standby / no preload: the catalog poll */
            if(!polled && kv_poll_due(e)){ int64 i2; bool t2; kv_catalog(fcinfo, name, &i2, &t2, true, &e->pmax, &e->pcount); INSTR_TIME_SET_CURRENT(e->polled); }
            reload = reload || e->pmax != e->loaded_pmax || e->pcount != e->loaded_pcount;
        }
        if(reload){ kv_pending_load(fcinfo, e, e->pmax); e->gen_seen = g; e->local_seen = kv_local_gen; e->need_reload = false; }
    }

    InitMaterializedSRF(fcinfo, 0);
    ids = (uint64_t*)palloc(sizeof(uint64_t) * k); sc = (float*)palloc(sizeof(float) * k);
    if(e->tracked && e->trunc) n = 0;
    else if(flt.allow) n = flt.nallow ? kci_vec_search_ivf_filtered(e->s, q, k, nprobe, rerank, kv_filt, &flt, ids, sc) : 0;
    else {
        n = kci_vec_search_ivf(e->s, q, k, nprobe, rerank, ids, sc);
        if(n > 0 && e->tracked && e->nexcl){
            for(i = 0; i < n; i++) if(!kv_filt(ids[i], &flt)) break;
            if(i < n) n = kci_vec_search_ivf_filtered(e->s, q, k, nprobe, rerank, kv_filt, &flt, ids, sc);
        }
    }
    if(n < 0) ereport(ERROR, (errmsg("kci_vec: index \"%s\" has no lists", name)));
    oid = (int64*)palloc(sizeof(int64) * k); osc = (float*)palloc(sizeof(float) * k);
    metric = kci_vec_metric(e->v);
    for(i = 0; i < n; i++) kv_topk_push(oid, osc, &on, k, (int64)ids[i], metric == 1 ? -sc[i] : sc[i]);
    if(e->tracked && e->nins){                       /* the inserts since the build: exact scores, merged in */
        double nn = 0; float qs, s4[4]; int d, j, r, dim16 = e->dim16;
        float *q16 = (float*)palloc0(sizeof(float) * dim16);
        memcpy(q16, q, sizeof(float) * dim);
        for(d = 0; d < dim; d++) nn += (double)q[d] * q[d];
        qs = nn > 0 ? (float)(1.0 / sqrt(nn)) : 0.0f;
        if(metric != 0) qs = 1.0f;
        if(!kv_dot4) kv_dot_init();
        for(j = 0; j < e->nins; j += 4){             /* the rows past nins are zero (allocated and cleared in the load) */
            kv_dot4(q16, e->ins_vecs + (size_t)j * dim16, dim16, dim16, s4);
            for(r = 0; r < 4 && j + r < e->nins; r++) if(kv_allowed(&flt, e->ins_ids[j + r]))
                kv_topk_push(oid, osc, &on, k, e->ins_ids[j + r], metric == 1 ? -((float)nn - 2.0f * s4[r] + e->ins_n2[j + r]) : s4[r] * qs);
        }
    }
    for(i = 0; i < on; i++){
        values[0] = Int64GetDatum(oid[i]);
        values[1] = Float4GetDatum(metric == 1 ? -osc[i] : osc[i]);
        tuplestore_putvalues(rsi->setResult, rsi->setDesc, values, nulls);
    }
    return (Datum)0;
}
PG_FUNCTION_INFO_V1(kci_vec_pg_search);
Datum kci_vec_pg_search(PG_FUNCTION_ARGS){ return kv_search(fcinfo, NULL); }
PG_FUNCTION_INFO_V1(kci_vec_pg_search_in);
Datum kci_vec_pg_search_in(PG_FUNCTION_ARGS){ return kv_search(fcinfo, PG_GETARG_ARRAYTYPE_P(5)); }

/* kci_vec_set_probe_gap(name text, gap real, min_probe int) -> int (this backend) */
PG_FUNCTION_INFO_V1(kci_vec_pg_set_probe_gap);
Datum kci_vec_pg_set_probe_gap(PG_FUNCTION_ARGS){
    char *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    float gap = PG_GETARG_FLOAT4(1);
    int minp = PG_GETARG_INT32(2);
    KvEntry *e = kv_get(fcinfo, name);
    e->gap = gap; e->minp = minp;
    PG_RETURN_INT32(kci_vec_searcher_set_probe_gap(e->s, gap, minp));
}

/* kci_vec_count(name text) -> bigint: rows in the image (not the pending ones) */
PG_FUNCTION_INFO_V1(kci_vec_pg_count);
Datum kci_vec_pg_count(PG_FUNCTION_ARGS){
    char *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    KvEntry *e = kv_get(fcinfo, name);
    PG_RETURN_INT64((int64)kci_vec_count(e->v));
}

/* ── tracking ──────────────────────────────────────────────────────────────────────────────────────────────── */

typedef struct { Oid relid, tid, tvec; SPIPlanPtr plan; } KvTrgPlan;
#define KV_TRG_PLANS 16
static KvTrgPlan kv_trg_plans[KV_TRG_PLANS]; static int kv_ntrg = 0;
static SPIPlanPtr kv_trg_plan(FunctionCallInfo fcinfo, Oid relid, Oid tid, Oid tvec){
    int i; KvTrgPlan *t; char *sql; SPIPlanPtr pl;
    Oid at[4] = { TEXTOID, CHAROID, tid, tvec };
    for(i = 0; i < kv_ntrg; i++) if(kv_trg_plans[i].relid == relid && kv_trg_plans[i].tid == tid && kv_trg_plans[i].tvec == tvec) return kv_trg_plans[i].plan;
    sql = psprintf("INSERT INTO %s (name, op, id, v) VALUES ($1, $2, ($3)::int8, ($4)::real[])", kv_tbl(fcinfo, "kci_vec_pending"));
    pl = SPI_prepare(sql, 4, at);
    if(!pl) ereport(ERROR, (errmsg("kci_vec: could not plan the write into kci_vec_pending: %s", SPI_result_code_string(SPI_result))));
    SPI_keepplan(pl);
    if(kv_ntrg < KV_TRG_PLANS) t = &kv_trg_plans[kv_ntrg++];
    else { t = &kv_trg_plans[0]; SPI_freeplan(t->plan); }      /* more tables than slots: recycle the first one */
    t->relid = relid; t->tid = tid; t->tvec = tvec; t->plan = pl;
    return pl;
}

PG_FUNCTION_INFO_V1(kci_vec_pg_track_trg);
Datum kci_vec_pg_track_trg(PG_FUNCTION_ARGS){
    TriggerData *td = (TriggerData*)fcinfo->context;
    TupleDesc tupdesc; int aid, avec; Oid tid, tvec;
    char *name, *idcol, *veccol;
    SPIPlanPtr pl;
    if(!CALLED_AS_TRIGGER(fcinfo)) ereport(ERROR, (errmsg("kci_vec_track_trg: not called as a trigger")));
    if(td->tg_trigger->tgnargs != 3) ereport(ERROR, (errmsg("kci_vec_track_trg: needs (name, id_col, vec_col)")));
    name = td->tg_trigger->tgargs[0]; idcol = td->tg_trigger->tgargs[1]; veccol = td->tg_trigger->tgargs[2];
    if(TRIGGER_FIRED_BY_TRUNCATE(td->tg_event)){
        Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(name) };
        if(SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("kci_vec: SPI_connect failed")));
        if(SPI_execute_with_args(psprintf("INSERT INTO %s (name, op, id) VALUES ($1, 'T', 0)", kv_tbl(fcinfo, "kci_vec_pending")), 1, at, av, NULL, false, 0) != SPI_OK_INSERT)
            ereport(ERROR, (errmsg("kci_vec: writing kci_vec_pending failed")));
        SPI_finish();
        kv_mark_dirty(name);
        return PointerGetDatum(NULL);
    }
    tupdesc = RelationGetDescr(td->tg_relation);
    aid = SPI_fnumber(tupdesc, idcol); avec = SPI_fnumber(tupdesc, veccol);
    if(aid <= 0 || avec <= 0) ereport(ERROR, (errmsg("kci_vec_track_trg: column \"%s\" or \"%s\" not in %s", idcol, veccol, RelationGetRelationName(td->tg_relation))));
    tid = TupleDescAttr(tupdesc, aid - 1)->atttypid; tvec = TupleDescAttr(tupdesc, avec - 1)->atttypid;
    if(SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("kci_vec: SPI_connect failed")));
    pl = kv_trg_plan(fcinfo, RelationGetRelid(td->tg_relation), tid, tvec);
    {
        Datum av[4]; char nulls[5] = "    ";
        bool isnull;
        av[0] = CStringGetTextDatum(name);
        if(TRIGGER_FIRED_BY_UPDATE(td->tg_event) || TRIGGER_FIRED_BY_DELETE(td->tg_event)){   /* the old row goes */
            av[1] = CharGetDatum('D');
            av[2] = heap_getattr(td->tg_trigtuple, aid, tupdesc, &isnull); if(isnull) nulls[2] = 'n';
            av[3] = (Datum)0; nulls[3] = 'n';
            if(nulls[2] != 'n' && SPI_execute_plan(pl, av, nulls, false, 0) != SPI_OK_INSERT) ereport(ERROR, (errmsg("kci_vec: writing kci_vec_pending failed")));
        }
        if(TRIGGER_FIRED_BY_INSERT(td->tg_event) || TRIGGER_FIRED_BY_UPDATE(td->tg_event)){   /* the new row comes */
            HeapTuple nt = TRIGGER_FIRED_BY_UPDATE(td->tg_event) ? td->tg_newtuple : td->tg_trigtuple;
            nulls[2] = nulls[3] = ' ';
            av[1] = CharGetDatum('I');
            av[2] = heap_getattr(nt, aid, tupdesc, &isnull); if(isnull) nulls[2] = 'n';
            av[3] = heap_getattr(nt, avec, tupdesc, &isnull); if(isnull) nulls[3] = 'n';
            if(nulls[2] != 'n' && nulls[3] != 'n' && SPI_execute_plan(pl, av, nulls, false, 0) != SPI_OK_INSERT) ereport(ERROR, (errmsg("kci_vec: writing kci_vec_pending failed")));
        }
    }
    SPI_finish();
    kv_mark_dirty(name);
    return PointerGetDatum(NULL);
}

/* kci_vec_track(name) / kci_vec_untrack(name): the row trigger on the index's table; the image's mtime is touched
 * so that every backend re-reads the catalog row */
static void kv_touch(const char *name, int64 img){
    char *path = kv_path(name, img);
    if(img && utimes(path, NULL) != 0 && errno != ENOENT)
        ereport(ERROR, (errcode_for_file_access(), errmsg("kci_vec: could not touch \"%s\": %m", path)));
}
static void kv_track(FunctionCallInfo fcinfo, const char *name, bool on){
    Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(name) };
    char *sql; Oid relid; char *idcol, *veccol, *relname; bool isnull; int64 img;
    kv_check_name(name);
    if(SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("kci_vec: SPI_connect failed")));
    sql = psprintf("SELECT rel, id_col, vec_col, img FROM %s WHERE name = $1", kv_tbl(fcinfo, "kci_vec_index"));
    if(SPI_execute_with_args(sql, 1, at, av, NULL, true, 1) != SPI_OK_SELECT || SPI_processed != 1)
        ereport(ERROR, (errmsg("kci_vec: no index \"%s\" in kci_vec_index (build it first)", name)));
    relid = DatumGetObjectId(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
    idcol = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2);
    veccol = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3);
    img = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4, &isnull)); if(isnull) img = 0;
    relname = DatumGetCString(DirectFunctionCall1(regclassout, ObjectIdGetDatum(relid)));
    sql = psprintf("DROP TRIGGER IF EXISTS %s ON %s", quote_identifier(psprintf("kci_vec_track_%s", name)), relname);
    if(SPI_execute(sql, false, 0) != SPI_OK_UTILITY) ereport(ERROR, (errmsg("kci_vec: dropping the trigger failed")));
    sql = psprintf("DROP TRIGGER IF EXISTS %s ON %s", quote_identifier(psprintf("kci_vec_truncate_%s", name)), relname);
    if(SPI_execute(sql, false, 0) != SPI_OK_UTILITY) ereport(ERROR, (errmsg("kci_vec: dropping the trigger failed")));
    if(on){
        sql = psprintf("CREATE TRIGGER %s AFTER INSERT OR UPDATE OR DELETE ON %s FOR EACH ROW EXECUTE FUNCTION %s(%s, %s, %s)",
                       quote_identifier(psprintf("kci_vec_track_%s", name)), relname, kv_tbl(fcinfo, "kci_vec_track_trg"),
                       quote_literal_cstr(name), quote_literal_cstr(idcol), quote_literal_cstr(veccol));
        if(SPI_execute(sql, false, 0) != SPI_OK_UTILITY) ereport(ERROR, (errmsg("kci_vec: creating the trigger failed")));
        sql = psprintf("CREATE TRIGGER %s AFTER TRUNCATE ON %s FOR EACH STATEMENT EXECUTE FUNCTION %s(%s, %s, %s)",
                       quote_identifier(psprintf("kci_vec_truncate_%s", name)), relname, kv_tbl(fcinfo, "kci_vec_track_trg"),
                       quote_literal_cstr(name), quote_literal_cstr(idcol), quote_literal_cstr(veccol));
        if(SPI_execute(sql, false, 0) != SPI_OK_UTILITY) ereport(ERROR, (errmsg("kci_vec: creating the truncate trigger failed")));
    }
    sql = psprintf("UPDATE %s SET tracked = %s WHERE name = $1", kv_tbl(fcinfo, "kci_vec_index"), on ? "true" : "false");
    if(SPI_execute_with_args(sql, 1, at, av, NULL, false, 0) != SPI_OK_UPDATE) ereport(ERROR, (errmsg("kci_vec: updating kci_vec_index failed")));
    if(!on){
        sql = psprintf("DELETE FROM %s WHERE name = $1", kv_tbl(fcinfo, "kci_vec_pending"));
        SPI_execute_with_args(sql, 1, at, av, NULL, false, 0);
    }
    SPI_finish();
    kv_touch(name, img);                                        /* every backend re-attaches and re-reads the catalog row */
    {   KvEntry *e = kv_find(name); if(e) e->tracked = on; }
}
PG_FUNCTION_INFO_V1(kci_vec_pg_track);
Datum kci_vec_pg_track(PG_FUNCTION_ARGS){ kv_track(fcinfo, text_to_cstring(PG_GETARG_TEXT_PP(0)), true); PG_RETURN_VOID(); }
PG_FUNCTION_INFO_V1(kci_vec_pg_untrack);
Datum kci_vec_pg_untrack(PG_FUNCTION_ARGS){ kv_track(fcinfo, text_to_cstring(PG_GETARG_TEXT_PP(0)), false); PG_RETURN_VOID(); }

/* kci_vec_drop(name text) -> void: trigger, catalog row, pending rows, this backend's attachment, the file */
PG_FUNCTION_INFO_V1(kci_vec_pg_drop);
Datum kci_vec_pg_drop(PG_FUNCTION_ARGS){
    char *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    Oid at[1] = { TEXTOID }; Datum av[1] = { CStringGetTextDatum(name) };
    KvEntry *e; char *sql;
    kv_check_name(name);
    if(SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("kci_vec: SPI_connect failed")));
    sql = psprintf("SELECT rel FROM %s WHERE name = $1", kv_tbl(fcinfo, "kci_vec_index"));
    if(SPI_execute_with_args(sql, 1, at, av, NULL, true, 1) == SPI_OK_SELECT && SPI_processed == 1){
        bool isnull; Oid relid = DatumGetObjectId(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
        if(!isnull && SearchSysCacheExists1(RELOID, ObjectIdGetDatum(relid))){
            char *relname = DatumGetCString(DirectFunctionCall1(regclassout, ObjectIdGetDatum(relid)));
            SPI_execute(psprintf("DROP TRIGGER IF EXISTS %s ON %s", quote_identifier(psprintf("kci_vec_track_%s", name)), relname), false, 0);
            SPI_execute(psprintf("DROP TRIGGER IF EXISTS %s ON %s", quote_identifier(psprintf("kci_vec_truncate_%s", name)), relname), false, 0);
        }
    }
    SPI_execute_with_args(psprintf("DELETE FROM %s WHERE name = $1", kv_tbl(fcinfo, "kci_vec_pending")), 1, at, av, NULL, false, 0);
    SPI_execute_with_args(psprintf("DELETE FROM %s WHERE name = $1", kv_tbl(fcinfo, "kci_vec_index")), 1, at, av, NULL, false, 0);
    SPI_finish();
    e = kv_find(name);
    if(e) kv_entry_drop(e);
    kv_unlink_others(name, 0);
    {   KvShmSlot *sl = kv_shm_slot(name); if(sl) pg_atomic_write_u64(&sl->img, 0); }
    PG_RETURN_VOID();
}

/* kci_vec_image_here(name text) -> boolean: is the file of the current generation on this server? */
PG_FUNCTION_INFO_V1(kci_vec_pg_image_here);
Datum kci_vec_pg_image_here(PG_FUNCTION_ARGS){
    char *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int64 img; bool tracked; struct stat st;
    kv_check_name(name);
    if(!kv_catalog(fcinfo, name, &img, &tracked, false, NULL, NULL) || img == 0) PG_RETURN_BOOL(false);
    PG_RETURN_BOOL(stat(kv_path(name, img), &st) == 0);
}

PG_FUNCTION_INFO_V1(kci_vec_pg_adopt_images);
Datum kci_vec_pg_adopt_images(PG_FUNCTION_ARGS){
    int done = 0; uint64 i, n; char **names; char *sql;
    if(RecoveryInProgress()) ereport(ERROR, (errmsg("kci_vec: cannot adopt images on a standby (read only)")));
    if(SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("kci_vec: SPI_connect failed")));
    sql = psprintf("SELECT name FROM %s WHERE img = 0 ORDER BY name", kv_tbl(fcinfo, "kci_vec_index"));
    if(SPI_execute(sql, false, 0) != SPI_OK_SELECT) ereport(ERROR, (errmsg("kci_vec: reading kci_vec_index failed")));
    n = SPI_processed;
    names = palloc(sizeof(char*) * (n + 1));
    for(i = 0; i < n; i++) names[i] = pstrdup(SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1));
    for(i = 0; i < n; i++){
        int64 img = kv_new_img();
        char *newpath = kv_path(names[i], img), *oldpath = psprintf("%s/%s.kvimg", kv_dir(), names[i]);
        struct stat st; bool have = stat(oldpath, &st) == 0;
        Oid at[2] = { INT8OID, TEXTOID }; Datum av[2] = { Int64GetDatum(img), CStringGetTextDatum(names[i]) };
        kv_check_name(names[i]);
        if(have && link(oldpath, newpath) != 0)
            ereport(ERROR, (errcode_for_file_access(), errmsg("kci_vec: could not link \"%s\" to \"%s\": %m", oldpath, newpath)));
        kv_publish(names[i], img, newpath, have ? oldpath : NULL);
        if(SPI_execute_with_args(psprintf("UPDATE %s SET img = $1 WHERE name = $2", kv_tbl(fcinfo, "kci_vec_index")), 2, at, av, NULL, false, 0) != SPI_OK_UPDATE)
            ereport(ERROR, (errmsg("kci_vec: updating kci_vec_index failed")));
        ereport(NOTICE, (errmsg("kci_vec: index \"%s\": generation %016" INT64_MODIFIER "x%s", names[i], (uint64)img,
                                have ? " (the 0.1 file renamed)" : " (no file on this server: kci_vec_materialize or kci_vec_build)")));
        done++;
    }
    SPI_finish();
    PG_RETURN_INT32(done);
}

static volatile sig_atomic_t kv_wk_sighup = false;
static void kv_wk_sighup_handler(SIGNAL_ARGS){ kv_wk_sighup = true; SetLatch(MyLatch); }

/* the extension's schema in this database, or NULL when the extension is not installed */
static bool kv_worker_schema(void){
    bool ok = false;
    if(SPI_connect() != SPI_OK_CONNECT) return false;
    if(SPI_execute("SELECT n.nspname FROM pg_catalog.pg_extension e JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'kci_vec_pg'", true, 1) == SPI_OK_SELECT && SPI_processed == 1){
        char *ns = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1);
        if(kv_schema_name) pfree(kv_schema_name);
        kv_schema_name = MemoryContextStrdup(TopMemoryContext, ns);
        ok = true;
    }
    SPI_finish();
    return ok;
}

/* one look: returns the number of actions taken */
static int kv_worker_once(bool standby){
    int acted = 0; uint64 i, n;
    char **names; bool *do_refresh, *do_mat;
    if(SPI_connect() != SPI_OK_CONNECT) return 0;
    {   char *sql = psprintf("SELECT i.name, i.tracked, i.img, (SELECT count(*) FROM %s p WHERE p.name = i.name), "
                             "(SELECT min(at) FROM %s p WHERE p.name = i.name), "
                             "EXISTS (SELECT 1 FROM %s p WHERE p.name = i.name AND p.op = 'T') FROM %s i ORDER BY i.name",
                             kv_tbl(NULL, "kci_vec_pending"), kv_tbl(NULL, "kci_vec_pending"), kv_tbl(NULL, "kci_vec_pending"), kv_tbl(NULL, "kci_vec_index"));
        if(SPI_execute(sql, true, 0) != SPI_OK_SELECT){ SPI_finish(); return 0; } }
    n = SPI_processed;
    names = palloc(sizeof(char*) * (n + 1)); do_refresh = palloc0(n + 1); do_mat = palloc0(n + 1);
    for(i = 0; i < n; i++){
        HeapTuple t = SPI_tuptable->vals[i]; TupleDesc td = SPI_tuptable->tupdesc; bool isnull, n3, n4, n6; struct stat st;
        char *name = SPI_getvalue(t, td, 1);
        bool tracked = DatumGetBool(SPI_getbinval(t, td, 2, &isnull)) && !isnull;
        int64 img = DatumGetInt64(SPI_getbinval(t, td, 3, &n3));
        int64 cnt = DatumGetInt64(SPI_getbinval(t, td, 4, &n4));
        Datum oldest = SPI_getbinval(t, td, 5, &isnull);
        bool trunc = DatumGetBool(SPI_getbinval(t, td, 6, &n6)) && !n6;
        if(n3) img = 0;
        if(n4) cnt = 0;
        names[i] = pstrdup(name);
        if(img && stat(kv_path(name, img), &st) != 0) do_mat[i] = true;
        else if(!standby && tracked && cnt > 0){
            if(cnt >= kv_guc_refresh_pending || trunc) do_refresh[i] = true;
            else if(kv_guc_refresh_max_age > 0 && !isnull){
                TimestampTz now = GetCurrentTimestamp();
                if(TimestampDifferenceExceeds(DatumGetTimestampTz(oldest), now, kv_guc_refresh_max_age * 1000)) do_refresh[i] = true; }
        }
    }
    for(i = 0; i < n; i++){
        if(do_mat[i]){
            char *sql = psprintf("SELECT %s(%s, %d)", kv_tbl(NULL, "kci_vec_materialize"), quote_literal_cstr(names[i]), kv_guc_refresh_threads);
            pgstat_report_activity(STATE_RUNNING, sql);
            if(SPI_execute(sql, true, 0) == SPI_OK_SELECT) acted++;
        } else if(do_refresh[i]){
            char *sql = psprintf("SELECT %s(%s, %d)", kv_tbl(NULL, "kci_vec_refresh"), quote_literal_cstr(names[i]), kv_guc_refresh_threads);
            pgstat_report_activity(STATE_RUNNING, sql);
            if(SPI_execute(sql, false, 0) == SPI_OK_SELECT){
                bool isnull; int64 rows = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
                ereport(LOG, (errmsg("kci_vec_pg worker: refreshed \"%s\": " INT64_FORMAT " rows", names[i], rows)));
                acted++; }
        }
    }
    SPI_finish();
    return acted;
}

PGDLLEXPORT void kv_worker_main(Datum arg){
    char dbname[BGW_EXTRALEN];
    strlcpy(dbname, MyBgworkerEntry->bgw_extra, BGW_EXTRALEN);
    pqsignal(SIGHUP, kv_wk_sighup_handler);
    pqsignal(SIGTERM, SignalHandlerForShutdownRequest);
    BackgroundWorkerUnblockSignals();
    BackgroundWorkerInitializeConnection(dbname, NULL, 0);
    ereport(LOG, (errmsg("kci_vec_pg worker started for database \"%s\"", dbname)));
    while(!ShutdownRequestPending){
        bool standby = RecoveryInProgress();
        if(kv_wk_sighup){ kv_wk_sighup = false; ProcessConfigFile(PGC_SIGHUP); }
        SetCurrentStatementStartTimestamp();
        StartTransactionCommand();
        PushActiveSnapshot(GetTransactionSnapshot());
        PG_TRY();
        {
            if(kv_worker_schema()) kv_worker_once(standby);
            PopActiveSnapshot();
            CommitTransactionCommand();
        }
        PG_CATCH();
        {
            MemoryContextSwitchTo(TopMemoryContext);
            EmitErrorReport();
            FlushErrorState();
            AbortCurrentTransaction();
        }
        PG_END_TRY();
        pgstat_report_activity(STATE_IDLE, NULL);
        pgstat_report_stat(true);
        (void)WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH, (long)kv_guc_refresh_interval * 1000L, PG_WAIT_EXTENSION);
        ResetLatch(MyLatch);
    }
    proc_exit(0);
}
