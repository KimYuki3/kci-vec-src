# kci_vec_pg — KCI-Vec from inside PostgreSQL (prototype)

KCI-Vec as SQL functions: `CREATE EXTENSION kci_vec_pg`. The server, permissions, backups and replication are
PostgreSQL's own. **This is a prototype**: the functions and the catalog tables may still change.

It is **not written like pgvector**: there is no `CREATE INDEX` and no `<=>` operator; you call functions.
It is for **a PostgreSQL you run yourself**: managed services that do not let you install your own extensions
(RDS, Cloud SQL, Supabase and the like) cannot use it.

## Build and install

Needs `pg_config` (the package `postgresql-server-dev-NN`), gcc and the files of this repository.

```sh
cd pg && make && sudo make install            # or: make PG_CONFIG=/path/to/pg_config install
```

Optional, in `postgresql.conf` (then restart):

```
shared_preload_libraries = 'kci_vec_pg'      # cheaper change tracking; needed for the background worker
kci_vec_pg.databases = 'mydb'                # background worker: refreshes on the primary, builds missing files on a standby
kci_vec_pg.refresh_pending = 10000           # refresh when this many changes wait
```

Tested on PostgreSQL 15.19, 16.15 and 17.11 (Linux x86-64).

## Use

```sql
CREATE EXTENSION vector;                     -- optional: pgvector's vector type can be passed as it is
CREATE EXTENSION kci_vec_pg;

-- an index named 'docs' from table docs(id bigint, embedding vector(384) or real[])
SELECT kci_vec_build('docs', 'docs', 'id', 'embedding');
SELECT kci_vec_track('docs');                -- follow INSERT / UPDATE / DELETE / TRUNCATE from now on

-- the 10 nearest, nearest first
SELECT id, score FROM kci_vec_search('docs', '[0.1, ...]'::vector, 10);
-- only among these ids (your WHERE clause, evaluated in SQL)
SELECT id, score FROM kci_vec_search('docs', '[0.1, ...]'::vector, 10, 64, 200,
                                     ARRAY(SELECT id FROM docs WHERE tenant = 7));

SELECT * FROM kci_vec_status();              -- rows, pending changes, whether this server has the index file
SELECT kci_vec_refresh('docs');              -- fold the pending changes into the index
```

| function | what it does |
|---|---|
| `kci_vec_build(name, rel, id_col, vec_col, nlist=0, threads=0, train_iters=0, train_sample=0, metric='cosine')` | Builds the index from the table. `metric`: `'cosine'`, `'l2'` (Euclidean) or `'ip'` (inner product). Returns the rows indexed. |
| `kci_vec_search(name, q, k=10, nprobe=32, rerank=100)` | `(id bigint, score real)`, nearest first. `score`: cosine or inner product as they are (larger is nearer); for `'l2'` the squared distance (smaller is nearer). Larger `nprobe` / `rerank`: more accurate, slower. |
| `kci_vec_search(name, q, k, nprobe, rerank, allowed bigint[])` | The same among the ids in `allowed` only. |
| `kci_vec_track(name)` / `kci_vec_untrack(name)` | Start / stop following the table's changes. While tracked, a search sees them at once. |
| `kci_vec_refresh(name, threads=0, full=false)` | Folds the pending changes into the index. |
| `kci_vec_status()` | One row per index. |
| `kci_vec_materialize(name, threads=0)` | Builds this server's copy of the index file from its own table (a standby, or after restoring a dump). |
| `kci_vec_image_here(name)` | Is the current index file on this server? |
| `kci_vec_set_probe_gap(name, gap, min_probe=8)` | A search setting for this connection. |
| `kci_vec_count(name)` / `kci_vec_drop(name)` | Rows in the index / remove the index. |

The id column must cast to `bigint`, the vector column to `real[]`. Rows with a NULL id or vector are skipped.

## What to know

- **Changes of a tracked table show in the next search**, also inside your own open transaction. They are kept in an
  ordinary table until `kci_vec_refresh` (or the background worker) folds them in; a search gets slower as they pile up.
- **Standby and restore**: the index file is not replicated and not in `pg_dump`. A standby or a restored server
  refuses to search with an outdated or missing file and names `kci_vec_materialize`, which builds it from the local
  table (the background worker does this by itself on a standby). `kci_vec_build` and `kci_vec_refresh` run on the
  primary only.
- **The index file is not encrypted** (`$PGDATA/kci_vec/`). It is protected the way PostgreSQL's own table files are;
  use disk encryption. `DROP EXTENSION` does not remove the files: call `kci_vec_drop` first.
- **`'l2'` and `'ip'`**: the table is read twice at the build. If a later row is longer than the longest vector at
  the build, the next refresh builds the index anew. These metrics suit data whose vectors have similar lengths: a few
  vectors far longer than the rest make the search less accurate at the same settings.
- **Filters** are an array of allowed ids. A small set (around 1 % of the rows or less) needs a larger `nprobe`; for a
  few thousand ids an exact `ORDER BY` in SQL is the better tool.
- **Updating the extension**: install the new files, restart the server, then `ALTER EXTENSION kci_vec_pg UPDATE`.
  Existing indexes stay.
- Not there: `CREATE INDEX` and operators (so the planner cannot choose it by itself), replication of the index file,
  its encryption.

## Against pgvector's HNSW

Both inside the same PostgreSQL 16.15, on the same table, through the same SQL round trip (a prepared statement, one
query at a time, one connection). glove-100-angular (1,183,514 rows x 100, cosine), 10,000 queries, k = 10, recall@10
against the data set's own exact answers. pgvector 0.8.1, HNSW with m 16, ef_construction 200. Ryzen 7 8845HS, WSL2,
`shared_buffers` 6 GB, 2026-11-20, three alternating rounds, minimum to maximum.

| | recall@10 | queries / s |
|---|---|---|
| kci_vec_pg, nprobe 64 / rerank 200 | 0.9005 | 3,261–3,906 |
| pgvector HNSW, ef_search 240 | 0.8947 | 199–210 |
| kci_vec_pg, nprobe 200 / rerank 600 | 0.9632 | 1,776–2,206 |
| pgvector HNSW, ef_search 1000 (its upper limit) | 0.9612 | 58–59 |
| the SQL round trip alone | | 9,549–11,622 |

- The author's own measurement: one data set, one machine, one connection.
- **Not measured with this version**: `'l2'` and `'ip'` indexes, several connections at once, filtered searches,
  other data sets, PostgreSQL 15 / 17.
- pgvector's index was built once (373 s with 7 parallel workers, on 2026-11-08) and not rebuilt for this run;
  kci_vec_pg's index was rebuilt in every round (12 s with 8 threads).
- pgvector works where this extension does not: managed services, `CREATE INDEX`, operators the planner understands,
  replication of the index through WAL. With a filter that keeps about a tenth of the rows or more, pgvector's own
  `WHERE` was the faster of the two in an earlier measurement (2026-11-09), because this extension first has to be
  given the array of allowed ids.
- The script is `pg/bench_pg0.py`.
