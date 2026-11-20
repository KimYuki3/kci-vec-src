-- kci_vec_pg 0.1 -> 0.2 . From the 0.1 that was pushed on (image generations were not there yet):
-- kci_vec_index.img (the image's generation; the file is <name>.<img hex>.kvimg), kci_vec_pending.at,
-- kci_vec_materialize / kci_vec_image_here, TRUNCATE of a tracked table (a statement trigger, op 'T'),
-- kci_vec_status with oldest_pending / truncated / img / image_here. Order: install the new files (make
-- install), restart the server (the preloaded library), then ALTER EXTENSION kci_vec_pg UPDATE; Between the
-- restart and the update a search fails (the new library wants kci_vec_index.img). Every existing image gets a
-- generation here; its 0.1 file <name>.kvimg is renamed to the new name. Tracked tables get the TRUNCATE
-- trigger.
\echo Use "ALTER EXTENSION kci_vec_pg UPDATE" to load this file. \quit

ALTER TABLE kci_vec_index ADD COLUMN IF NOT EXISTS img bigint NOT NULL DEFAULT 0;
ALTER TABLE kci_vec_pending ADD COLUMN IF NOT EXISTS at timestamptz NOT NULL DEFAULT now();

CREATE FUNCTION kci_vec_materialize(name text, threads int DEFAULT 0)
RETURNS bigint AS 'MODULE_PATHNAME', 'kci_vec_pg_materialize' LANGUAGE C STRICT PARALLEL UNSAFE;
CREATE FUNCTION kci_vec_image_here(name text)
RETURNS boolean AS 'MODULE_PATHNAME', 'kci_vec_pg_image_here' LANGUAGE C STRICT STABLE PARALLEL RESTRICTED;

DROP FUNCTION kci_vec_status();
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

-- a generation for every existing index; the 0.1 file <name>.kvimg becomes <name>.<generation>.kvimg
CREATE FUNCTION kci_vec_adopt_images() RETURNS int AS 'MODULE_PATHNAME', 'kci_vec_pg_adopt_images' LANGUAGE C STRICT PARALLEL UNSAFE;
SELECT kci_vec_adopt_images();
DROP FUNCTION kci_vec_adopt_images();

-- the TRUNCATE trigger on every tracked table (kci_vec_track makes both triggers again; the pending rows are kept)
SELECT kci_vec_track(name) FROM kci_vec_index WHERE tracked;
