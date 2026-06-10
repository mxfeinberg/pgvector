SET enable_seqscan = off;

CREATE TABLE tqivf_s (id int, v vector(4));
INSERT INTO tqivf_s SELECT g, ARRAY[g, 0, 0, 0]::real[]::vector FROM generate_series(1, 200) g;
CREATE INDEX tqivf_s_idx ON tqivf_s USING tqivf (v vector_l2_ops) WITH (lists = 10);

SET tqivf.probes = 10;
SET tqivf.rerank = 100;
-- The 5 nearest to [5,0,0,0] are {5,4,6,3,7} (distances 0,1,1,2,2 -> two
-- exact-distance ties).  Re-sort by id so the tie order is deterministic.
SELECT id FROM (
  SELECT id FROM tqivf_s ORDER BY v <-> '[5,0,0,0]' LIMIT 5
) q ORDER BY id;

CREATE TABLE tqivf_e (v vector(4));
CREATE INDEX tqivf_e_idx ON tqivf_e USING tqivf (v vector_l2_ops) WITH (lists = 2);
SELECT count(*) FROM (SELECT v FROM tqivf_e ORDER BY v <-> '[1,1,1,1]' LIMIT 5) q;

-- Inner product
CREATE TABLE tqivf_ip (id int, v vector(4));
INSERT INTO tqivf_ip SELECT g, ARRAY[g, 1, 0, 0]::real[]::vector FROM generate_series(1, 200) g;
CREATE INDEX ON tqivf_ip USING tqivf (v vector_ip_ops) WITH (lists = 10);
SET tqivf.probes = 10;
SELECT id FROM tqivf_ip ORDER BY v <#> '[10,0,0,0]' LIMIT 3;

-- Cosine: angles to the query differ per row (low ids are nearest), so the
-- expected order is meaningful rather than an arbitrary tie-break.
CREATE TABLE tqivf_cos (id int, v vector(4));
INSERT INTO tqivf_cos SELECT g, ARRAY[g, 1, 0, 0]::real[]::vector FROM generate_series(1, 200) g;
CREATE INDEX ON tqivf_cos USING tqivf (v vector_cosine_ops) WITH (lists = 10);
SELECT id FROM tqivf_cos ORDER BY v <=> '[0,1,0,0]' LIMIT 3;

DROP TABLE tqivf_ip, tqivf_cos;

-- Insert after build, then query (tail-chain path)
CREATE TABLE tqivf_ins (id int, v vector(4));
INSERT INTO tqivf_ins SELECT g, ARRAY[g,0,0,0]::real[]::vector FROM generate_series(1, 100) g;
CREATE INDEX ON tqivf_ins USING tqivf (v vector_l2_ops) WITH (lists = 5);
SET tqivf.probes = 5;
INSERT INTO tqivf_ins VALUES (999, '[3,0,0,0]');
-- ids 3 and 999 are both at distance 0 (a tie); re-sort by id.
SELECT id FROM (
  SELECT id FROM tqivf_ins ORDER BY v <-> '[3,0,0,0]' LIMIT 2
) q ORDER BY id;
DROP TABLE tqivf_ins;

-- Iterative scan: probes=1 with many tiny lists; relaxed_order must probe more
-- lists on demand to satisfy a LIMIT larger than one list holds.
CREATE TABLE tqivf_it (id int, v vector(4));
INSERT INTO tqivf_it SELECT g, ARRAY[g,0,0,0]::real[]::vector FROM generate_series(1, 200) g;
CREATE INDEX ON tqivf_it USING tqivf (v vector_l2_ops) WITH (lists = 100);
SET tqivf.probes = 1;
-- non-iterative: one tiny list cannot satisfy LIMIT 10
SET tqivf.iterative_scan = off;
SELECT count(*) < 10 AS one_list_insufficient
  FROM (SELECT id FROM tqivf_it ORDER BY v <-> '[100,0,0,0]' LIMIT 10) q;
-- iterative: probes more lists to reach the full LIMIT
SET tqivf.iterative_scan = relaxed_order;
SET tqivf.max_probes = 100;
SELECT count(*) AS iterative_count
  FROM (SELECT id FROM tqivf_it ORDER BY v <-> '[100,0,0,0]' LIMIT 10) q;
RESET tqivf.iterative_scan;
RESET tqivf.max_probes;
RESET tqivf.probes;
DROP TABLE tqivf_it;

DROP TABLE tqivf_s, tqivf_e;
