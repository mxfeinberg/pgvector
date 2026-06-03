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
DROP TABLE tqhnsw_b;
