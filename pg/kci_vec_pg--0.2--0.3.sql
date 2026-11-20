-- kci_vec_pg 0.2 -> 0.3 the metric (cosine, L2, inner product) is chosen at kci_vec_build.
-- kci_vec_index.metric (existing indexes are cosine), kci_vec_build with a ninth argument. Order: install the
-- new files (make install), restart the server (the preloaded library), then ALTER EXTENSION kci_vec_pg
-- UPDATE; Between the restart and the update everything works as in 0.2 (the library sees that kci_vec_index
-- has no metric column yet and treats every index as cosine); only a build with 'l2' or 'ip' is refused. No
-- image is rebuilt.
\echo Use "ALTER EXTENSION kci_vec_pg UPDATE" to load this file. \quit

ALTER TABLE kci_vec_index ADD COLUMN IF NOT EXISTS metric text NOT NULL DEFAULT 'cosine';

DROP FUNCTION kci_vec_build(text, regclass, text, text, int, int, int, bigint);
CREATE FUNCTION kci_vec_build(name text, rel regclass, id_col text, vec_col text,
                              nlist int DEFAULT 0, threads int DEFAULT 0,
                              train_iters int DEFAULT 0, train_sample bigint DEFAULT 0, metric text DEFAULT 'cosine')
RETURNS bigint AS 'MODULE_PATHNAME', 'kci_vec_pg_build' LANGUAGE C STRICT PARALLEL UNSAFE;
