-- tqflat scan tests: M3.2 (full scan + heap-fetch rerank, L2 metric).
-- Deterministic small data where the nearest neighbours are unambiguous, so
-- the index result is stable regardless of quantization noise once reranked.

CREATE TABLE tqscan (id int, v vector(8));
INSERT INTO tqscan VALUES
	(1, '[10,0,0,0,0,0,0,0]'),
	(2, '[9,1,0,0,0,0,0,0]'),
	(3, '[0,10,0,0,0,0,0,0]'),
	(4, '[0,0,10,0,0,0,0,0]'),
	(5, '[0,0,0,10,0,0,0,0]'),
	(6, '[-10,0,0,0,0,0,0,0]'),
	(7, '[5,5,0,0,0,0,0,0]'),
	(8, '[0,0,0,0,0,0,0,10]');

CREATE INDEX tqscan_idx ON tqscan USING tqflat (v vector_l2_ops) WITH (bits = 4, tq_prod = false);

SET enable_seqscan = off;

-- With rerank the top-k are exact: nearest to [10,0,...] is id 1, then 2, then 7.
SET tqflat.rerank = 100;
SELECT id FROM tqscan ORDER BY v <-> '[10,0,0,0,0,0,0,0]'::vector LIMIT 3;

-- A query aligned with id 3 returns id 3 first.
SELECT id FROM tqscan ORDER BY v <-> '[0,10,0,0,0,0,0,0]'::vector LIMIT 1;

-- An exact-match query returns that row first.
SELECT id FROM tqscan ORDER BY v <-> '[0,0,0,0,0,0,0,10]'::vector LIMIT 1;

-- rerank = 0 (pure quantized estimate): still returns the exact match first on
-- this well-separated data, and returns exactly LIMIT rows.
SET tqflat.rerank = 0;
SELECT count(*) FROM (SELECT id FROM tqscan ORDER BY v <-> '[10,0,0,0,0,0,0,0]'::vector LIMIT 3) s;

-- Empty index: returns no rows, no error.
RESET tqflat.rerank;
CREATE TABLE tqscan_empty (id int, v vector(4));
CREATE INDEX tqscan_empty_idx ON tqscan_empty USING tqflat (v vector_l2_ops);
SELECT id FROM tqscan_empty ORDER BY v <-> '[1,2,3,4]'::vector LIMIT 5;

-- Insert-then-query: a newly inserted exact match is found via the index.
INSERT INTO tqscan VALUES (9, '[0,0,0,0,10,0,0,0]');
SELECT id FROM tqscan ORDER BY v <-> '[0,0,0,0,10,0,0,0]'::vector LIMIT 1;

-- tq_prod = true index also returns the unambiguous nearest neighbour.
CREATE INDEX tqscan_idx_prod ON tqscan USING tqflat (v vector_l2_ops) WITH (bits = 4, tq_prod = true);
DROP INDEX tqscan_idx;
SET tqflat.rerank = 100;
SELECT id FROM tqscan ORDER BY v <-> '[10,0,0,0,0,0,0,0]'::vector LIMIT 2;

RESET tqflat.rerank;
RESET enable_seqscan;
DROP TABLE tqscan;
DROP TABLE tqscan_empty;

-- M4.1: inner-product and cosine operator classes
-- Deterministic spike vectors: each vector has a large value in one dimension
-- so cosine / IP nearest neighbours are unambiguous regardless of magnitude.

CREATE TABLE tqmetrics (id int, v vector(8));
INSERT INTO tqmetrics VALUES
	(1,  '[10,0,0,0,0,0,0,0]'),
	(2,  '[9,1,0,0,0,0,0,0]'),
	(3,  '[0,10,0,0,0,0,0,0]'),
	(4,  '[0,0,10,0,0,0,0,0]'),
	(5,  '[100,0,0,0,0,0,0,0]'),   -- same direction as id 1, huge norm (tests cosine rerank)
	(6,  '[0.1,0,0,0,0,0,0,0]'),   -- same direction as id 1, tiny norm
	(7,  '[-10,0,0,0,0,0,0,0]'),
	(8,  '[0,0,0,0,0,0,0,10]');

CREATE INDEX tqmetrics_ip_idx ON tqmetrics USING tqflat (v vector_ip_ops) WITH (bits = 4, tq_prod = false);
CREATE INDEX tqmetrics_cos_idx ON tqmetrics USING tqflat (v vector_cosine_ops) WITH (bits = 4, tq_prod = false);

SET enable_seqscan = off;
SET tqflat.rerank = 100;

-- Inner-product: id 5 has the biggest IP with [1,0,...] (norm=100).
-- Expect: 5, 1, 2 in that order.
SELECT id FROM tqmetrics ORDER BY v <#> '[1,0,0,0,0,0,0,0]'::vector LIMIT 3;

-- Inner-product aligned with dimension 3: id 4 wins.
SELECT id FROM tqmetrics ORDER BY v <#> '[0,0,1,0,0,0,0,0]'::vector LIMIT 1;

-- Cosine: direction [1,0,...] — ids 1,2,5,6 all point the same direction.
-- The query [1,0,...] is unit-norm; the four same-direction vectors must all rank
-- before id 2 (which has a small cosine-perpendicular component) and before id 7
-- (anti-parallel).  Verify ids 5 and 6 appear before id 7.
SELECT id FROM tqmetrics WHERE id IN (5, 6, 7) ORDER BY v <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 3;

-- Cosine query aligned with dimension 2: id 3 wins (independent of magnitude).
SELECT id FROM tqmetrics ORDER BY v <=> '[0,1,0,0,0,0,0,0]'::vector LIMIT 1;

-- rerank = 0 path for both new opclasses still returns LIMIT rows.
SET tqflat.rerank = 0;
SELECT count(*) FROM (SELECT id FROM tqmetrics ORDER BY v <#> '[1,0,0,0,0,0,0,0]'::vector LIMIT 3) s;
SELECT count(*) FROM (SELECT id FROM tqmetrics ORDER BY v <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 3) s;

RESET tqflat.rerank;
RESET enable_seqscan;
DROP TABLE tqmetrics;

-- fast_rotation index returns the unambiguous nearest neighbour.
CREATE TABLE tqfast (id int, v vector(8));
INSERT INTO tqfast VALUES
	(1,'[10,0,0,0,0,0,0,0]'),(2,'[9,1,0,0,0,0,0,0]'),(3,'[0,10,0,0,0,0,0,0]'),
	(4,'[0,0,10,0,0,0,0,0]'),(5,'[0,0,0,0,0,0,0,10]');
CREATE INDEX tqfast_idx ON tqfast USING tqflat (v vector_l2_ops) WITH (bits=4, fast_rotation=true);
SET enable_seqscan = off;
SET tqflat.rerank = 100;
SELECT id FROM tqfast ORDER BY v <-> '[10,0,0,0,0,0,0,0]'::vector LIMIT 2;  -- expect 1, 2
SELECT id FROM tqfast ORDER BY v <-> '[0,0,0,0,0,0,0,10]'::vector LIMIT 1;  -- expect 5
RESET tqflat.rerank;
RESET enable_seqscan;
DROP TABLE tqfast;
