SET enable_seqscan = off;

-- Empty (UNLOGGED -> ambuildempty path) index has a valid meta page.
CREATE UNLOGGED TABLE tqhnsw_e (v vector(8));
CREATE INDEX tqhnsw_e_idx ON tqhnsw_e USING tqhnsw (v vector_l2_ops);
SELECT tqhnsw_test_meta('tqhnsw_e_idx');   -- dim=8 m=16 ef_construction=64 bits=4 metric=0 nvectors=0 entry_level=-1
DROP TABLE tqhnsw_e;

-- Bad options rejected.
CREATE TABLE tqhnsw_bad (v vector(4));
CREATE INDEX ON tqhnsw_bad USING tqhnsw (v vector_l2_ops) WITH (m = 1);              -- ERROR: out of range
CREATE INDEX ON tqhnsw_bad USING tqhnsw (v vector_l2_ops) WITH (ef_construction = 2); -- ERROR: out of range
DROP TABLE tqhnsw_bad;

-- Real build: 500 rows -> a graph with an entry point and level-0 neighbors.
CREATE TABLE tqhnsw_b (id int, v vector(8));
INSERT INTO tqhnsw_b SELECT g, ARRAY[g,g+1,g+2,g+3,g+4,g+5,g+6,g+7]::real[]::vector
  FROM generate_series(1, 500) g;
CREATE INDEX tqhnsw_b_idx ON tqhnsw_b USING tqhnsw (v vector_l2_ops) WITH (m = 8, ef_construction = 32);
-- nvectors is a stable count; entry_level varies run-to-run so it is omitted here.
SELECT regexp_replace(tqhnsw_test_meta('tqhnsw_b_idx'), ' entry_level=.*', '');
-- entry node has >0 level-0 neighbors and the graph has 500 nodes (stable predicates):
SELECT tqhnsw_test_graph('tqhnsw_b_idx');

-- Post-build insert: verify that inserting after a real build does not corrupt the
-- codebook page (element tuples must land on element pages, not block 1).
INSERT INTO tqhnsw_b VALUES (501, ARRAY[1,1,1,1,1,1,1,1]::real[]::vector);
-- The nearest neighbor of [1,1,...] is id=501 (exact match; distance=0).
-- With seqscan off the planner must use the tqhnsw index.
SET enable_seqscan = off;
SELECT id FROM tqhnsw_b ORDER BY v <-> ARRAY[1,1,1,1,1,1,1,1]::real[]::vector LIMIT 1;
EXPLAIN (COSTS OFF) SELECT id FROM tqhnsw_b ORDER BY v <-> ARRAY[1,1,1,1,1,1,1,1]::real[]::vector LIMIT 1;

DROP TABLE tqhnsw_b;

-- Parallel build: forced workers produce a usable index with correct recall.
SET max_parallel_maintenance_workers = 4;
SET min_parallel_table_scan_size = 1;
CREATE TABLE tqhnsw_par (id int, v vector(32));
INSERT INTO tqhnsw_par SELECT g,
  ARRAY[g%7,g%11,g%13,g%17,g%19,g%23,g%29,g%31,
        g%37,g%41,g%43,g%47,g%53,g%59,g%61,g%67,
        g%71,g%73,g%79,g%83,g%89,g%97,g%101,g%103,
        g%107,g%109,g%113,g%127,g%131,g%137,g%139,g%149]::real[]::vector(32)
  FROM generate_series(1, 2000) g;
CREATE INDEX tqhnsw_par_idx ON tqhnsw_par USING tqhnsw (v vector_l2_ops);
-- Verify the graph is fully built.
SELECT regexp_replace(tqhnsw_test_meta('tqhnsw_par_idx'), ' entry_level=.*', '');
SELECT tqhnsw_test_graph('tqhnsw_par_idx');
-- Every one of the first 10 query rows finds itself as the nearest neighbor.
SELECT count(*) = 10 AS all_self_match FROM (
  SELECT q.id, (SELECT p.id FROM tqhnsw_par p ORDER BY p.v <-> q.v LIMIT 1) AS nn
  FROM tqhnsw_par q WHERE q.id <= 10
) s WHERE nn = id;
RESET min_parallel_table_scan_size;
RESET max_parallel_maintenance_workers;
DROP TABLE tqhnsw_par;
