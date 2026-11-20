-- test_pg1.sql stage-1 smoke test of kci_vec_pg. Run in a scratch database with pgvector installed: psql -d
-- pg0 -v ON_ERROR_STOP=1 -f pg/test_pg1.sql Checks: build / search / count / status, tracking (insert, delete,
-- update seen by a search without a rebuild), transaction visibility of pending rows, a second connection
-- seeing the same, refresh folding them in, TRUNCATE of the tracked table , untrack / drop, and the refusals
-- (unknown index, wrong dims, empty table).
\set ON_ERROR_STOP on
\set QUIET on
DROP EXTENSION IF EXISTS kci_vec_pg CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION kci_vec_pg;
DROP TABLE IF EXISTS t1;
CREATE TABLE t1(id bigint PRIMARY KEY, v vector(8));
INSERT INTO t1 SELECT i, ARRAY[random(),random(),random(),random(),random(),random(),random(),random()]::vector FROM generate_series(1, 5000) i;
\set QUIET off

\echo '--- build: 5000 rows'
SELECT kci_vec_build('t1', 't1', 'id', 'v', 16, 4) AS rows;
SELECT kci_vec_count('t1') AS image_rows;

\echo '--- search for row 7: first id must be 7, score ~1'
SELECT id, round(score::numeric, 4) AS score FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 7), 3, 16, 50);
\echo '--- the same through pgvector exact order (reference)'
SELECT id, round((1 - (v <=> (SELECT v FROM t1 WHERE id = 7)))::numeric, 4) AS cos FROM t1 ORDER BY v <=> (SELECT v FROM t1 WHERE id = 7) LIMIT 3;

\echo '--- track, then insert a near-copy of row 7 as id 99999: it must appear without a rebuild'
SELECT kci_vec_track('t1');
INSERT INTO t1 SELECT 99999, v FROM t1 WHERE id = 7;
SELECT id, round(score::numeric, 4) AS score FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 7), 3, 16, 50);

\echo '--- delete row 7: it must be gone from the answer'
DELETE FROM t1 WHERE id = 7;
SELECT id, round(score::numeric, 4) AS score FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 3, 16, 50);

\echo '--- update row 8 to the same vector: 8 must now be in the top 2'
UPDATE t1 SET v = (SELECT v FROM t1 WHERE id = 99999) WHERE id = 8;
SELECT id, round(score::numeric, 4) AS score FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 3, 16, 50);

\echo '--- status: 5000 rows in the image, pending inserts 2 (99999, 8) deletes 2 (7, 8 old)'
SELECT name, rows, tracked, pending, pending_inserts, pending_deletes FROM kci_vec_status();

\echo '--- an uncommitted insert is seen by its own session only; after ROLLBACK it is gone'
BEGIN;
INSERT INTO t1 SELECT 77777, v FROM t1 WHERE id = 99999;
SELECT id FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 3, 16, 50);
ROLLBACK;
SELECT id FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 3, 16, 50);
SELECT count(*) AS rolled_back_found_0 FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 10, 16, 50) WHERE id = 77777;

\echo '--- a second connection sees the same (attaches the same image, replays the same pending rows)'
\c
SELECT id, round(score::numeric, 4) AS score FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 3, 16, 50);

\echo '--- refresh (differential: tracked): pending folded into the image; answers unchanged; pending 0; rows 5000; diffs 1'
SELECT kci_vec_refresh('t1', 4) AS rows;
SELECT id, round(score::numeric, 4) AS score FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 3, 16, 50);
SELECT name, rows, tracked, pending, full_rows, diffs FROM kci_vec_status();
\echo '--- a refresh with nothing pending changes nothing (diffs still 1); more changes, a second differential refresh'
SELECT kci_vec_refresh('t1') AS rows;
CREATE TEMP TABLE keep150 AS SELECT v FROM t1 WHERE id = 150;
DELETE FROM t1 WHERE id BETWEEN 100 AND 199;
INSERT INTO t1 SELECT id + 200000, v FROM t1 WHERE id BETWEEN 300 AND 349;
UPDATE t1 SET v = (SELECT v FROM t1 WHERE id = 99999) WHERE id = 9;
SELECT id, round(score::numeric, 4) AS score FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 4, 16, 50);
SELECT kci_vec_refresh('t1') AS rows;
\echo '--- after it: rows 4950, diffs 2, same answer (99999, 8, 9 at 1.0), deleted 150 absent, 200300 found by its vector'
SELECT name, rows, pending, diffs FROM kci_vec_status();
SELECT id, round(score::numeric, 4) AS score FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 4, 16, 50);
SELECT count(*) AS deleted_found FROM kci_vec_search('t1', (SELECT v FROM keep150), 10, 16, 50) WHERE id = 150;
SELECT id FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 200300), 2, 16, 50) ORDER BY score DESC;
\echo '--- full refresh: rows 4950, diffs 0, full_rows 4950'
SELECT kci_vec_refresh('t1', 4, true) AS rows;
SELECT name, rows, pending, full_rows, diffs FROM kci_vec_status();
SELECT id, round(score::numeric, 4) AS score FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 4, 16, 50);

\echo '--- allowed ids (a WHERE clause in SQL): only even ids; 99999 / 9 odd -> 8 first; an id outside the set never returns'
SELECT id, round(score::numeric, 4) AS score FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 3, 16, 50, ARRAY(SELECT id FROM t1 WHERE id % 2 = 0));
SELECT count(*) AS odd_returned FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 50, 16, 200, ARRAY(SELECT id FROM t1 WHERE id % 2 = 0)) WHERE id % 2 = 1;
\echo '--- a set of 3 ids with the pgvector type; a pending insert inside / outside the set; an empty set'
INSERT INTO t1 SELECT 88888, v FROM t1 WHERE id = 99999;
SELECT id FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 10, 16, 50, ARRAY[88888, 4932, 10]::bigint[]) ORDER BY score DESC;
SELECT id FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 10, 16, 50, ARRAY[4932, 10]::bigint[]) ORDER BY score DESC;
SELECT count(*) AS empty_set FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 10, 16, 50, ARRAY[]::bigint[]);
DELETE FROM t1 WHERE id = 88888;

\echo '--- (2026-11-10) TRUNCATE of the tracked table, uncommitted: its session sees nothing from the image, only rows inserted after it; others see the table'
CREATE TEMP TABLE keep_t1 AS SELECT * FROM t1;
BEGIN;
TRUNCATE t1;
SELECT count(*) AS after_truncate FROM kci_vec_search('t1', (SELECT v FROM keep_t1 WHERE id = 99999), 10, 16, 50);
INSERT INTO t1 SELECT id, v FROM keep_t1 WHERE id IN (99999, 8, 9);
SELECT id FROM kci_vec_search('t1', (SELECT v FROM keep_t1 WHERE id = 99999), 10, 16, 50) ORDER BY id;
ROLLBACK;
SELECT count(*) AS after_rollback_10 FROM kci_vec_search('t1', (SELECT v FROM keep_t1 WHERE id = 99999), 10, 16, 50);
\echo '--- committed: status says truncated; the image rows are still counted but never answer; a refresh builds anew (diffs 0)'
TRUNCATE t1;
INSERT INTO t1 SELECT * FROM keep_t1 WHERE id <= 1000 OR id IN (99999, 200300);
SELECT truncated, pending, kci_vec_count('t1') AS image_rows_still FROM kci_vec_status() WHERE name = 't1';
SELECT id FROM kci_vec_search('t1', (SELECT v FROM keep_t1 WHERE id = 99999), 3, 16, 50) ORDER BY id;
SELECT count(*) AS old_rows_found_0 FROM kci_vec_search('t1', (SELECT v FROM keep_t1 WHERE id = 3000), 10, 16, 50) WHERE id > 1000 AND id NOT IN (99999, 200300);
SELECT kci_vec_refresh('t1') = (SELECT count(*) FROM t1) AS rows_match;
SELECT rows = (SELECT count(*) FROM t1) AS rows_match, truncated, pending, diffs, full_rows = rows AS full FROM kci_vec_status() WHERE name = 't1';
SELECT id FROM kci_vec_search('t1', (SELECT v FROM keep_t1 WHERE id = 99999), 3, 16, 50) ORDER BY id;
INSERT INTO t1 SELECT * FROM keep_t1 WHERE id > 1000 AND id NOT IN (99999, 200300);
SELECT kci_vec_refresh('t1', 4, true) AS rows_4950;
DROP TABLE keep_t1;

\echo '--- (2026-11-09) generations: the file is named by kci_vec_index.img; status shows it and whether it is here'
SELECT length(img) AS img_hex_len, image_here FROM kci_vec_status() WHERE name = 't1';
SELECT kci_vec_materialize('t1') AS already_here_0;
\echo '--- a restore / standby: the current generation has no file here -> a clear refusal, then kci_vec_materialize makes it'
SELECT 'rm -f ' || current_setting('data_directory') || '/kci_vec/t1.*.kvimg' AS rmcmd \gset
COPY (SELECT 1) TO PROGRAM :'rmcmd';
SELECT image_here FROM kci_vec_status() WHERE name = 't1';
\set ON_ERROR_STOP off
SELECT count(*) FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 3, 16, 50);
\set ON_ERROR_STOP on
SELECT kci_vec_materialize('t1') > 0 AS materialized;
SELECT image_here FROM kci_vec_status() WHERE name = 't1';
SELECT id FROM kci_vec_search('t1', (SELECT v FROM t1 WHERE id = 99999), 3, 16, 50) ORDER BY score DESC LIMIT 1;
SELECT kci_vec_materialize('t1') AS already_here_0;
\echo '--- a refresh replaces the generation and removes the old file (one file per index)'
INSERT INTO t1 SELECT 77777, v FROM t1 WHERE id = 10;
SELECT img AS before FROM kci_vec_status() WHERE name = 't1' \gset
SELECT kci_vec_refresh('t1') AS rows;
SELECT img <> :'before' AS new_generation, image_here, pending FROM kci_vec_status() WHERE name = 't1';
SELECT count(*) AS files FROM pg_ls_dir(current_setting('data_directory') || '/kci_vec') f WHERE f LIKE 't1.%';
DELETE FROM t1 WHERE id = 77777;

\echo '--- refusals'
\set ON_ERROR_STOP off
SELECT kci_vec_search('nope', ARRAY[1,2,3,4]::real[], 3);
SELECT kci_vec_search('t1', ARRAY[1,2,3]::real[], 3);
SELECT kci_vec_build('bad name', 't1', 'id', 'v');
CREATE TABLE t_empty(id bigint, v vector(8));
SELECT kci_vec_build('t_empty', 't_empty', 'id', 'v');
DROP TABLE t_empty;
SELECT kci_vec_materialize('nope');
\set ON_ERROR_STOP on

\echo '--- untrack: trigger gone, pending cleared; drop: file and rows gone'
SELECT kci_vec_untrack('t1');
SELECT count(*) AS triggers FROM pg_trigger WHERE tgname IN ('kci_vec_track_t1', 'kci_vec_truncate_t1');
SELECT kci_vec_drop('t1');
SELECT count(*) AS index_rows FROM kci_vec_index;
SELECT count(*) AS files_left FROM pg_ls_dir(current_setting('data_directory') || '/kci_vec') f WHERE f LIKE 't1.%';
\set ON_ERROR_STOP off
SELECT kci_vec_count('t1');
\set ON_ERROR_STOP on
DROP TABLE t1;

\echo '--- (2026-11-20) metric l2 / ip: vectors of different lengths; the top 5 must be pgvector''s exact top 5'
DROP TABLE IF EXISTS t3;
CREATE TABLE t3(id bigint PRIMARY KEY, v vector(8));
INSERT INTO t3 SELECT i, ARRAY[(random() - 0.5) * (1 + (i % 7)), (random() - 0.5) * (1 + (i % 7)), (random() - 0.5) * (1 + (i % 7)), (random() - 0.5) * (1 + (i % 7)), (random() - 0.5) * (1 + (i % 7)), (random() - 0.5) * (1 + (i % 7)), (random() - 0.5) * (1 + (i % 7)), (random() - 0.5) * (1 + (i % 7))]::vector FROM generate_series(1, 4000) i;
SELECT kci_vec_build('t3l2', 't3', 'id', 'v', 16, 4, 0, 0, 'l2') AS rows_l2, kci_vec_build('t3ip', 't3', 'id', 'v', 16, 4, 0, 0, 'ip') AS rows_ip;
SELECT name, metric FROM kci_vec_index WHERE name LIKE 't3%' ORDER BY name;
SELECT (SELECT array_agg(id ORDER BY score, id) FROM kci_vec_search('t3l2', (SELECT v FROM t3 WHERE id = 11), 5, 16, 200))
     = (SELECT array_agg(id) FROM (SELECT id FROM t3 ORDER BY v <-> (SELECT v FROM t3 WHERE id = 11), id LIMIT 5) x) AS l2_same_as_exact,
       (SELECT round(max(score)::numeric, 3) FROM kci_vec_search('t3l2', (SELECT v FROM t3 WHERE id = 11), 5, 16, 200))
     = (SELECT round(max(d * d)::numeric, 3) FROM (SELECT v <-> (SELECT v FROM t3 WHERE id = 11) AS d FROM t3 ORDER BY 1 LIMIT 5) x) AS l2_scores_are_squared_distances;
SELECT (SELECT array_agg(id ORDER BY score DESC, id) FROM kci_vec_search('t3ip', (SELECT v FROM t3 WHERE id = 11), 5, 16, 200))
     = (SELECT array_agg(id) FROM (SELECT id FROM t3 ORDER BY v <#> (SELECT v FROM t3 WHERE id = 11), id LIMIT 5) x) AS ip_same_as_exact;
\echo '--- l2 / ip, tracked: an inserted copy of row 11, a far longer vector, a deleted row; then the refresh (the long vector makes it build anew)'
SELECT kci_vec_track('t3l2'); SELECT kci_vec_track('t3ip');
INSERT INTO t3 SELECT 90011, v FROM t3 WHERE id = 11;
INSERT INTO t3 VALUES (90012, ARRAY[6,6,6,6,6,6,6,6]::vector);
DELETE FROM t3 WHERE id = 12;
-- rows 11 and 90011 are the same vector (distance 0 twice): compared as sets, their order among themselves is open
SELECT (SELECT array_agg(id ORDER BY id) FROM kci_vec_search('t3l2', (SELECT v FROM t3 WHERE id = 11), 5, 16, 200))
     = (SELECT array_agg(id ORDER BY id) FROM (SELECT id FROM t3 ORDER BY v <-> (SELECT v FROM t3 WHERE id = 11), id LIMIT 5) x) AS l2_pending_same_as_exact,
       (SELECT array_agg(id ORDER BY id) FROM kci_vec_search('t3ip', (SELECT v FROM t3 WHERE id = 11), 5, 16, 200))
     = (SELECT array_agg(id ORDER BY id) FROM (SELECT id FROM t3 ORDER BY v <#> (SELECT v FROM t3 WHERE id = 11), id LIMIT 5) x) AS ip_pending_same_as_exact,
       (SELECT id FROM kci_vec_search('t3ip', ARRAY[1,1,1,1,1,1,1,1]::real[], 1, 16, 200)) AS ip_top_is_the_long_one_90012;
SELECT kci_vec_refresh('t3l2') AS l2_rows, kci_vec_refresh('t3ip') AS ip_rows;
SELECT name, pending, diffs FROM kci_vec_status() WHERE name LIKE 't3%' ORDER BY name;
-- re-rank 1000 here: with one vector twice as long as the rest the first estimate is coarser (200 missed one of the 5)
SELECT (SELECT array_agg(id ORDER BY id) FROM kci_vec_search('t3l2', (SELECT v FROM t3 WHERE id = 11), 5, 16, 1000))
     = (SELECT array_agg(id ORDER BY id) FROM (SELECT id FROM t3 ORDER BY v <-> (SELECT v FROM t3 WHERE id = 11), id LIMIT 5) x) AS l2_after_refresh_same_as_exact,
       (SELECT id FROM kci_vec_search('t3ip', ARRAY[1,1,1,1,1,1,1,1]::real[], 1, 16, 200)) AS ip_top_after_refresh_90012,
       (SELECT id FROM kci_vec_search('t3l2', ARRAY[6,6,6,6,6,6,6,6]::real[], 1, 16, 200)) AS l2_nearest_to_the_long_one_90012;
\echo '--- l2: a short vector added after the refresh goes in by the differential refresh (diffs 1)'
INSERT INTO t3 VALUES (90013, ARRAY[0.01,0,0,0,0,0,0,0]::vector);
SELECT kci_vec_refresh('t3l2') AS l2_rows;
SELECT diffs, (SELECT id FROM kci_vec_search('t3l2', ARRAY[0.01,0,0,0,0,0,0,0]::real[], 1, 16, 1000)) AS nearest_90013 FROM kci_vec_status() WHERE name = 't3l2';
\set ON_ERROR_STOP off
SELECT kci_vec_build('t3bad', 't3', 'id', 'v', 16, 4, 0, 0, 'manhattan');
\set ON_ERROR_STOP on
SELECT kci_vec_untrack('t3l2'); SELECT kci_vec_untrack('t3ip'); SELECT kci_vec_drop('t3l2'); SELECT kci_vec_drop('t3ip');
DROP TABLE t3;
\echo '--- done'
