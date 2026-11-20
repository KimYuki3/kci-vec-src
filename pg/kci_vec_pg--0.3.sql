-- kci_vec_pg 0.3 ; 0.2 was KCI-Vec's list index as SQL functions, one shared image per index (named by its
-- generation), changes of the table collected by triggers and merged into every search. 0.3: the metric
-- (cosine, L2, inner product) is chosen at kci_vec_build. From 0.1 / 0.2: ALTER EXTENSION kci_vec_pg UPDATE,
-- after installing the new .so and restarting.
\echo Use "CREATE EXTENSION kci_vec_pg" to load this file. \quit

-- the indexes (one row each) and the changes of tracked tables since their image was built
CREATE TABLE kci_vec_index (
    name         text PRIMARY KEY,
    rel          regclass NOT NULL,
    id_col       text NOT NULL,
    vec_col      text NOT NULL,
    nlist        int,
    train_iters  int,
    train_sample bigint,
    rows         bigint,
    built_at     timestamptz,
    build_secs   real,                   -- of the last build or refresh
    tracked      boolean NOT NULL DEFAULT false,
    full_rows    bigint,                 -- rows at the last full build (the centroids were trained on them)
    diffs        int NOT NULL DEFAULT 0, -- differential refreshes since then
    img          bigint NOT NULL DEFAULT 0,  -- generation of the current image: the file is <name>.<img hex>.kvimg
    metric       text NOT NULL DEFAULT 'cosine'  -- 'cosine', 'l2' or 'ip'
);
CREATE TABLE kci_vec_pending (
    seq   bigserial PRIMARY KEY,
    name  text NOT NULL,
    op    "char" NOT NULL,  -- 'I' inserted (v holds the vector), 'D' deleted, 'T' the table was truncated ; id 0)
    id    bigint NOT NULL,
    v     real[],
    at    timestamptz NOT NULL DEFAULT now()  -- when it was collected (the worker's refresh_max_age)
);
CREATE INDEX kci_vec_pending_name_seq ON kci_vec_pending (name, seq);
SELECT pg_catalog.pg_extension_config_dump('kci_vec_index', '');
SELECT pg_catalog.pg_extension_config_dump('kci_vec_pending', '');
SELECT pg_catalog.pg_extension_config_dump('kci_vec_pending_seq_seq', '');

-- Build the index "name" from rel(id_col bigint-castable, vec_col real[]-castable; pgvector's vector casts).
-- nlist 0 = 2*sqrt(rows); threads 0 = all cores; train_iters / train_sample 0 = defaults (10 / 250,000).
-- metric: 'cosine' (the default), 'l2' (Euclidean) or 'ip' (inner product). For 'l2' and 'ip' the table is read
-- twice (the longest vector has to be known first).
-- Writes $PGDATA/kci_vec/<name>.kvimg (shared by every backend) and the kci_vec_index row. Returns the rows indexed.
CREATE FUNCTION kci_vec_build(name text, rel regclass, id_col text, vec_col text,
                              nlist int DEFAULT 0, threads int DEFAULT 0,
                              train_iters int DEFAULT 0, train_sample bigint DEFAULT 0, metric text DEFAULT 'cosine')
RETURNS bigint AS 'MODULE_PATHNAME', 'kci_vec_pg_build' LANGUAGE C STRICT PARALLEL UNSAFE;

-- Fold the pending changes into the image. A tracked index: from the pending rows alone (the image's rows and
-- centroids kept; each new vector assigned and coded against the existing centroids), unless full is true or the
-- rows have drifted past half / double of the last full build; untracked: built again with kci_vec_index's parameters.
CREATE FUNCTION kci_vec_refresh(name text, threads int DEFAULT 0, full boolean DEFAULT false)
RETURNS bigint AS 'MODULE_PATHNAME', 'kci_vec_pg_refresh' LANGUAGE C STRICT PARALLEL UNSAFE;

-- This server's copy of the current image, built from the local table with the catalog's parameters; the
-- catalog is not written, so it runs on a hot standby and after a restore from a dump (the image file is
-- outside WAL and pg_dump). Returns the rows indexed, 0 when the file is already here.
CREATE FUNCTION kci_vec_materialize(name text, threads int DEFAULT 0)
RETURNS bigint AS 'MODULE_PATHNAME', 'kci_vec_pg_materialize' LANGUAGE C STRICT PARALLEL UNSAFE;

-- Is the file of the current generation on this server?
CREATE FUNCTION kci_vec_image_here(name text)
RETURNS boolean AS 'MODULE_PATHNAME', 'kci_vec_pg_image_here' LANGUAGE C STRICT STABLE PARALLEL RESTRICTED;

-- k nearest by the index's metric, nearest first: nprobe lists scanned, rerank candidates re-scored with the float
-- rows; with a tracked table the inserts since the build are scanned exactly and merged, deleted rows are left out.
-- score: cosine and inner product as they are (larger = nearer); 'l2': the squared distance (smaller = nearer).
CREATE FUNCTION kci_vec_search(name text, q real[], k int DEFAULT 10, nprobe int DEFAULT 32, rerank int DEFAULT 100)
RETURNS TABLE(id bigint, score real)
AS 'MODULE_PATHNAME', 'kci_vec_pg_search' LANGUAGE C STRICT STABLE PARALLEL RESTRICTED ROWS 10;
-- The same, restricted to the ids in `allowed` (a WHERE clause evaluated in SQL: ARRAY(SELECT id FROM t WHERE ...)).
-- The lists are scanned with that filter, so a selective set wants a larger nprobe. NULL allowed: no rows (STRICT).
CREATE FUNCTION kci_vec_search(name text, q real[], k int, nprobe int, rerank int, allowed bigint[])
RETURNS TABLE(id bigint, score real)
AS 'MODULE_PATHNAME', 'kci_vec_pg_search_in' LANGUAGE C STRICT STABLE PARALLEL RESTRICTED ROWS 10;

-- Collect the table's changes (a row trigger, and a statement trigger for TRUNCATE) / stop collecting (and forget the pending rows).
CREATE FUNCTION kci_vec_track_trg() RETURNS trigger AS 'MODULE_PATHNAME', 'kci_vec_pg_track_trg' LANGUAGE C;
CREATE FUNCTION kci_vec_track(name text) RETURNS void AS 'MODULE_PATHNAME', 'kci_vec_pg_track' LANGUAGE C STRICT PARALLEL UNSAFE;
CREATE FUNCTION kci_vec_untrack(name text) RETURNS void AS 'MODULE_PATHNAME', 'kci_vec_pg_untrack' LANGUAGE C STRICT PARALLEL UNSAFE;

CREATE FUNCTION kci_vec_set_probe_gap(name text, gap real, min_probe int DEFAULT 8)
RETURNS int AS 'MODULE_PATHNAME', 'kci_vec_pg_set_probe_gap' LANGUAGE C STRICT PARALLEL RESTRICTED;

CREATE FUNCTION kci_vec_count(name text)
RETURNS bigint AS 'MODULE_PATHNAME', 'kci_vec_pg_count' LANGUAGE C STRICT STABLE PARALLEL RESTRICTED;

CREATE FUNCTION kci_vec_drop(name text)
RETURNS void AS 'MODULE_PATHNAME', 'kci_vec_pg_drop' LANGUAGE C STRICT PARALLEL UNSAFE;

-- One row per index: what is built, how old, how many changes wait (refresh when `pending` grows), the oldest pending
-- row, whether the table was truncated since the build (the image is ignored until a refresh), and whether this server
-- has the file of the current generation (false on a standby or after a restore until kci_vec_materialize).
CREATE FUNCTION kci_vec_status()
RETURNS TABLE(name text, rel regclass, rows bigint, built_at timestamptz, build_secs real, tracked boolean,
              pending bigint, pending_inserts bigint, pending_deletes bigint, oldest_pending timestamptz, truncated boolean,
              full_rows bigint, diffs int, img text, image_here boolean)
LANGUAGE sql STABLE AS $$
  SELECT i.name, i.rel, i.rows, i.built_at, i.build_secs, i.tracked,
         count(p.seq), count(p.seq) FILTER (WHERE p.op = 'I'), count(p.seq) FILTER (WHERE p.op = 'D'), min(p.at),
         coalesce(bool_or(p.op = 'T'), false),
         i.full_rows, i.diffs, lpad(to_hex(i.img), 16, '0'), kci_vec_image_here(i.name)
  FROM kci_vec_index i LEFT JOIN kci_vec_pending p ON p.name = i.name
  GROUP BY i.name, i.rel, i.rows, i.built_at, i.build_secs, i.tracked, i.full_rows, i.diffs, i.img
  ORDER BY i.name
$$;

-- With pgvector installed: accept its vector type directly.
DO $do$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vector') THEN
    EXECUTE $f$
      CREATE FUNCTION kci_vec_search(name text, q vector, k int DEFAULT 10, nprobe int DEFAULT 32, rerank int DEFAULT 100)
      RETURNS TABLE(id bigint, score real) LANGUAGE sql STABLE PARALLEL RESTRICTED ROWS 10
      AS 'SELECT * FROM kci_vec_search($1, $2::real[], $3, $4, $5)'
    $f$;
    EXECUTE $f$
      CREATE FUNCTION kci_vec_search(name text, q vector, k int, nprobe int, rerank int, allowed bigint[])
      RETURNS TABLE(id bigint, score real) LANGUAGE sql STABLE PARALLEL RESTRICTED ROWS 10
      AS 'SELECT * FROM kci_vec_search($1, $2::real[], $3, $4, $5, $6)'
    $f$;
  END IF;
END
$do$;
