-- tqflat build tests: M2.1 (meta/side pages + loader), M2.2 (heap scan + data pages).
-- v4 blocked layout: bits MUST be 4, tq_prod MUST be false (QJL unsupported).
-- fast_rotation in {on (default), off} are both supported.
-- tqflat_test_meta returns 9 ints:
--   {dim, bits, metric, tqProd, nVectors, fastRotation, dimPadded, blockWidth, blockCount}
-- blockWidth is always 32; blockCount = ceil(nVectorsBuiltAtBuildTime / 32).

CREATE TABLE tqbuild (id serial, v vector(8));
INSERT INTO tqbuild (v)
	SELECT ('[' || array_to_string(array(SELECT (i * 7 + g) % 13 - 6 FROM generate_series(1, 8) i), ',') || ']')::vector
	FROM generate_series(1, 500) g;

-- (a) default (no options): bits = 4, tq_prod = off, fast_rotation = on.
-- 500 rows -> blockCount = ceil(500/32) = 16.
CREATE INDEX tqbuild_idxdef ON tqbuild USING tqflat (v vector_l2_ops);
SELECT tqflat_test_meta('tqbuild_idxdef'::regclass);  -- {8,4,0,0,500,1,8,32,16}
SELECT pg_relation_size('tqbuild_idxdef') > 0 AS idxdef_nonempty;

-- (b) explicit bits = 4 (same as default).
CREATE INDEX tqbuild_idx4 ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 4);
SELECT tqflat_test_meta('tqbuild_idx4'::regclass);    -- {8,4,0,0,500,1,8,32,16}
SELECT pg_relation_size('tqbuild_idx4') > 0 AS idx4_nonempty;

-- (c) bits = 4, fast_rotation = false (dense rotation path).
-- fastRotation = 0; dimPadded = dim = 8.
CREATE INDEX tqbuild_idxdense ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 4, fast_rotation = false);
SELECT tqflat_test_meta('tqbuild_idxdense'::regclass);  -- {8,4,0,0,500,0,8,32,16}
SELECT pg_relation_size('tqbuild_idxdense') > 0 AS idxdense_nonempty;

-- tq_prod = true is rejected by the blocked layout (QJL unsupported in v4).
\set ON_ERROR_STOP 0
CREATE INDEX tqbuild_idxprod ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 4, tq_prod = true);
\set ON_ERROR_STOP 1

-- Bad bits must error at reloption validation: only bits = 4 is valid.
\set ON_ERROR_STOP 0
CREATE INDEX tqbuild_idxbad1 ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 1);
CREATE INDEX tqbuild_idxbad2 ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 2);
CREATE INDEX tqbuild_idxbad3 ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 3);
CREATE INDEX tqbuild_idxbad5 ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 5);
\set ON_ERROR_STOP 1

-- M2.3: single-row insert (tqinsert) -- build then insert, nVectors must increase
-- while blockCount stays the same (inserts go to the row-major tail, not new blocks).
-- A single INSERT updates ALL indexes on the table simultaneously, so one batch of
-- 100 rows increments nVectors on every index from 500 to 600.
INSERT INTO tqbuild (v)
	SELECT ('[' || array_to_string(array(SELECT (i * 3 + g) % 7 - 3 FROM generate_series(1, 8) i), ',') || ']')::vector
	FROM generate_series(1, 100) g;

-- nVectors 500 -> 600 (5th element); blockCount stays 16 (9th element).
SELECT tqflat_test_meta('tqbuild_idxdef'::regclass);    -- {8,4,0,0,600,1,8,32,16}
SELECT tqflat_test_meta('tqbuild_idx4'::regclass);      -- {8,4,0,0,600,1,8,32,16}
-- dense path: nVectors 500 -> 600; blockCount stays 16; fastRotation = 0.
SELECT tqflat_test_meta('tqbuild_idxdense'::regclass);  -- {8,4,0,0,600,0,8,32,16}

-- index size must be non-zero after inserts
SELECT pg_relation_size('tqbuild_idx4') > 0 AS idx4_nonempty_after_insert;

DROP TABLE tqbuild;
