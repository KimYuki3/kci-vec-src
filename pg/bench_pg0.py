#!/usr/bin/env python3
"""Stage-0 bench KCI-Vec inside PostgreSQL (kci_vec_pg, SQL functions) against pgvector's HNSW on the same
machine and the same table. Every query goes through the same SQL round trip (prepared statement, one row set
per query), so the client cost is the same for both; the "round trip" line measures that floor."""
import sys, os, time, io, argparse, subprocess, platform
import numpy as np
import h5py
import psycopg2

K = 10
EF_SEARCH = [10, 20, 40, 60, 80, 120, 160, 240, 400, 600, 1000]
KCI_GRID = [(4, 30), (8, 40), (12, 60), (16, 80), (24, 100), (32, 120), (48, 160), (64, 200), (96, 300), (128, 400), (200, 600)]

def log(out, s):
    print(s, flush=True)
    if out:
        with open(out, "a") as f: f.write(s + "\n")

def fmt_vec(x):   # pgvector literal
    return "[" + ",".join("%.7g" % v for v in x) + "]"
def fmt_arr(x):   # real[] literal
    return "{" + ",".join("%.7g" % v for v in x) + "}"

def load_table(cur, conn, train, out):
    n, d = train.shape
    cur.execute("DROP TABLE IF EXISTS glove")
    cur.execute(f"CREATE TABLE glove(id bigint NOT NULL, v vector({d}) NOT NULL)")
    conn.commit()
    t0 = time.time()
    step = 50000
    for s in range(0, n, step):
        blk = train[s:s+step]
        buf = io.StringIO()
        for i in range(blk.shape[0]):
            buf.write(str(s + i)); buf.write("\t"); buf.write(fmt_vec(blk[i])); buf.write("\n")
        buf.seek(0)
        cur.copy_expert("COPY glove(id, v) FROM STDIN", buf)
        conn.commit()
        print(f"  loaded {s+blk.shape[0]}/{n}", end="\r", flush=True)
    cur.execute("ANALYZE glove")
    conn.commit()
    cur.execute("SELECT pg_size_pretty(pg_table_size('glove'))")
    log(out, f"table glove: {n} rows x {d} dims, {cur.fetchone()[0]}, COPY {time.time()-t0:.0f} s")

def run_queries(cur, prep, params_of, test, gt, nq, warm=200):
    """prep: EXECUTE statement name; params_of(i) -> tuple."""
    sql = f"EXECUTE {prep}(" + ",".join(["%s"] * len(params_of(0))) + ")"
    for i in range(min(warm, nq)):
        cur.execute(sql, params_of(i)); cur.fetchall()
    hit = 0
    t0 = time.perf_counter()
    for i in range(nq):
        cur.execute(sql, params_of(i))
        ids = [r[0] for r in cur.fetchall()]
        hit += len(set(ids[:K]) & set(gt[i].tolist()))
    dt = time.perf_counter() - t0
    return hit / (nq * K), nq / dt, dt / nq * 1000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5"); ap.add_argument("--load", action="store_true")
    ap.add_argument("--pgvector", action="store_true"); ap.add_argument("--kci", action="store_true")
    ap.add_argument("--nq", type=int, default=10000); ap.add_argument("--nlist", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=0); ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--gap", type=float, default=0.0); ap.add_argument("--min-probe", type=int, default=8)
    ap.add_argument("--out", default=None); ap.add_argument("--db", default="pg0")
    ap.add_argument("--m", type=int, default=16); ap.add_argument("--efc", type=int, default=200)
    a = ap.parse_args()

    with h5py.File(a.h5, "r") as f:
        train = np.array(f["train"], dtype=np.float32) if a.load else None
        test = np.array(f["test"], dtype=np.float32)
        gt = np.array(f["neighbors"], dtype=np.int64)[:, :K]
    nrm = np.linalg.norm(test, axis=1, keepdims=True); nrm[nrm == 0] = 1; test /= nrm
    if train is not None:
        nrm = np.linalg.norm(train, axis=1, keepdims=True); nrm[nrm == 0] = 1; train /= nrm
    nq = min(a.nq, test.shape[0]); test = test[:nq]; gt = gt[:nq]
    d = test.shape[1]

    conn = psycopg2.connect(dbname=a.db); conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SELECT version()"); pgver = cur.fetchone()[0].split(",")[0]
    cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'"); pgv = cur.fetchone()
    log(a.out, f"# {time.strftime('%Y-%m-%d %H:%M')} {os.path.basename(a.h5)} nq={nq} k={K} | {pgver} | pgvector {pgv[0] if pgv else '-'} | {platform.platform()}")
    try:
        top = subprocess.run(["ps", "-eo", "pcpu,args", "--sort=-pcpu"], capture_output=True, text=True).stdout.splitlines()[1:4]
        log(a.out, "# other load: " + " | ".join(t.strip()[:60] for t in top))
    except Exception: pass

    if a.load:
        load_table(cur, conn, train, a.out)
        del train

    vec_lit = [fmt_vec(x) for x in test]
    arr_lit = [fmt_arr(x) for x in test]

    # the floor: a prepared statement that does nothing, same client loop
    cur.execute("PREPARE p0(int) AS SELECT $1")
    _, qps0, ms0 = run_queries(cur, "p0", lambda i: (i,), test, gt, min(nq, 2000))
    log(a.out, f"round trip only: {qps0:,.0f} q/s ({ms0:.3f} ms/query)")

    if a.pgvector:
        cur.execute("SELECT 1 FROM pg_indexes WHERE indexname='glove_hnsw'")
        if not cur.fetchone():
            t0 = time.time()
            cur.execute(f"CREATE INDEX glove_hnsw ON glove USING hnsw (v vector_cosine_ops) WITH (m={a.m}, ef_construction={a.efc})")
            conn.commit()
            cur.execute("SELECT pg_size_pretty(pg_relation_size('glove_hnsw'))")
            log(a.out, f"pgvector hnsw build (m={a.m}, ef_construction={a.efc}, max_parallel_maintenance_workers from server): {time.time()-t0:.0f} s, index {cur.fetchone()[0]}")
        cur.execute(f"PREPARE ph(vector) AS SELECT id FROM glove ORDER BY v <=> $1 LIMIT {K}")
        cur.execute("SET enable_seqscan = off")
        for ef in EF_SEARCH:
            cur.execute(f"SET hnsw.ef_search = {ef}")
            rec, qps, ms = run_queries(cur, "ph", lambda i: (vec_lit[i],), test, gt, nq)
            log(a.out, f"pgvector hnsw ef_search={ef:<4d} recall@10 {rec:.4f}  {qps:8,.0f} q/s  {ms:.3f} ms")
        cur.execute("DEALLOCATE ph")

    if a.kci:
        # one index per (nlist, training); built once, reused by later runs (like the pgvector index)
        name = f"glove_n{a.nlist}_i{a.iters}_s{a.sample}"
        cur.execute("SELECT image_here FROM kci_vec_status() WHERE name = %s", (name,))
        r = cur.fetchone()
        if not (r and r[0]):
            t0 = time.time()
            cur.execute("SELECT kci_vec_build(%s, 'glove', 'id', 'v', %s, 8, %s, %s)", (name, a.nlist, a.iters, a.sample))
            n = cur.fetchone()[0]; conn.commit()
            cur.execute("SELECT pg_size_pretty(size) FROM kci_vec_status() s, pg_stat_file(current_setting('data_directory') || '/kci_vec/' || s.name || '.' || s.img || '.kvimg') WHERE s.name = %s", (name,))
            log(a.out, f"kci_vec build {name} (nlist={a.nlist}, 8 threads, iters={a.iters}, sample={a.sample}): {n} rows, {time.time()-t0:.0f} s, file {cur.fetchone()[0]}")
        tag = f"kci_vec {name}"
        if a.gap > 0:
            cur.execute("SELECT kci_vec_set_probe_gap(%s, %s, %s)", (name, a.gap, a.min_probe)); conn.commit()
            tag += f" gap={a.gap}/{a.min_probe}"
        cur.execute(f"PREPARE pk(real[], int, int) AS SELECT id FROM kci_vec_search('{name}', $1, {K}, $2, $3)")
        for nprobe, rerank in KCI_GRID:
            rec, qps, ms = run_queries(cur, "pk", lambda i: (arr_lit[i], nprobe, rerank), test, gt, nq)
            log(a.out, f"{tag} nprobe={nprobe:<3d} rerank={rerank:<3d} recall@10 {rec:.4f}  {qps:8,.0f} q/s  {ms:.3f} ms")
        cur.execute("DEALLOCATE pk")
    conn.rollback()

if __name__ == "__main__":
    main()
